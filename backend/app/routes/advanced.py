"""Generate › Advanced: one request runs a plan, Krea 2 generation or masked edits, checks
and corrections (app/advanced). Separate from the built-in Krea 2 runs
(routes/comfy.py) and LoRA training, which it only reads: their connections, model files,
LoRA versions and compatibility rules."""

from __future__ import annotations

import secrets
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.advanced import compiler, engine, graphs
from app.advanced.intelligence import IntelligenceError
from app.comfy import anypaint, krea2
from app.comfy import enhance as pgupscaler
from app.comfy.generation import LoraUse, SourceImage, resolve_lora, source_input
from app.comfy.service import file_capability, lora_filename, owned
from app.db import get_db
from app.deps import api_error, current_user
from app.models import AdvancedJob, AIIntegration, ComfyConnection, Generation, Image, User
from app.services import storage
from app.services.generation import GenerationRejected
from app.services.safety import ADULT_OFF, PromptRefused, check_prompt, is_explicit
from app.services.serializers import image_json

router = APIRouter(prefix="/api/advanced", tags=["advanced"])

MAX_PROMPT = 2000


class AdvancedBody(BaseModel):
    connection_id: uuid.UUID
    mode: Literal["text-to-image", "edit"]
    variant: Literal["turbo", "raw"] = "turbo"
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
    image: SourceImage | None = None
    lora: LoraUse | None = None
    # Text to image.
    aspect_ratio: str = "3:4"
    resolution: str = "1K"
    seed: int | None = Field(default=None, ge=0, le=krea2.SEED_MAX)
    # Edit: the regions to redraw when there is no AI analysis.
    targets: list[str] = Field(default_factory=list, max_length=4)
    # Edit with a LoRA: the person in the photo becomes the LoRA's character (body and face
    # redrawn with it, the face measured against its photos). Off = the LoRA only helps.
    identity: bool = True
    # Edit: the background redrawn sharper, the same place (a new one is asked in the text).
    improve_background: bool = False
    # Edit: after a new background the whole photo once more, gently (off by default).
    unify_light: bool = False
    # Edit strength for every region (None = per region: hair 0.85, mouth 0.7, …).
    denoise: float | None = Field(default=None, ge=graphs.DENOISE_MIN, le=graphs.DENOISE_MAX)
    # Stages the user switched on.
    analysis: bool = True
    quality: bool = True
    corrections: int = Field(default=1, ge=0, le=3)
    adult: bool = False
    # A plan the user saw (and maybe edited) in the preview; None = make one in the job.
    spec: dict[str, Any] | None = None
    client_request_id: str | None = Field(default=None, max_length=64)


def _reject(exc: GenerationRejected | PromptRefused | IntelligenceError):
    status = getattr(exc, "status_code", None) or (
        422 if isinstance(exc, PromptRefused) else 409 if exc.code == "not_connected" else 502
    )
    return api_error(status, exc.code, exc.message)


async def _ai_connected(db: AsyncSession, user: User) -> bool:
    return (
        await db.scalar(select(AIIntegration.id).where(AIIntegration.user_id == user.id).limit(1))
    ) is not None


def _sam3_installed(conn: ComfyConnection) -> bool:
    files = ((conn.info or {}).get("enhance") or {}).get("files") or []
    return any(f.get("filename") == graphs.SAM3_FILE and f.get("installed") for f in files)


def _need_anypaint(caps: dict[str, Any], conn: ComfyConnection) -> None:
    """Region redraws need Krea 2 AnyPaint; without it the job is refused, never run another
    way."""
    state = caps["anypaint"]
    if state.get("missing_nodes"):
        raise api_error(
            409,
            "missing_nodes",
            f"Krea 2 AnyPaint's nodes are missing on {conn.name}. A GPU rented by the studio "
            "gets them when it starts: rent it again. On your own ComfyUI install "
            "alexw5702-afk/krea2-anypaint into custom_nodes.",
        )
    if state.get("missing_files"):
        raise api_error(
            409,
            "missing_models",
            f"Krea 2 AnyPaint is not installed on {conn.name}: install it in Advanced (229 MB).",
        )


def capabilities(conn: ComfyConnection, ai: bool) -> dict[str, Any]:
    """What this connection can do for Advanced, and what is missing (shown in the UI)."""
    info = conn.info or {}
    detection = info.get("krea2") or {}
    nodes = set(info.get("node_names") or [])
    variants = {
        key: [
            f["filename"] for f in (detection.get("variants") or {}).get(key, {}).get("files", [])
        ]
        for key in krea2.VARIANTS
    }
    base_ok = (
        bool(detection.get("supported"))
        and bool(detection.get("text_encoders"))
        and bool(detection.get("vaes"))
    )
    missing_t2i = [n for n in graphs.NODES_T2I if nodes and n not in nodes]
    missing_edit = [n for n in graphs.NODES_EDIT if nodes and n not in nodes]
    return {
        "ready": conn.status == "ready",
        "variants": {k: bool(v) and base_ok for k, v in variants.items()},
        "missing_nodes_text": missing_t2i,
        "missing_nodes_edit": missing_edit,
        "sam3": _sam3_installed(conn),
        # Region redraws run on Krea 2 AnyPaint (its nodes come with a rented GPU's start, its
        # LoRA with one click); never another method in its place.
        "anypaint": info.get("anypaint")
        or {
            "ready": False,
            "missing_nodes": list(anypaint.NODES),
            "missing_files": [f.filename for f in anypaint.FILES],
            "files": [],
        },
        "anypaint_file": {
            "filename": anypaint.LORA.filename,
            "size": anypaint.LORA.size,
        },
        "sam3_file": {
            "url": pgupscaler.SAM3.url,
            "folder": pgupscaler.SAM3.folder,
            "filename": pgupscaler.SAM3.filename,
            "size": pgupscaler.SAM3.size,
        },
        "ai": ai,
        "file_capability": file_capability(conn),
    }


@router.get("/capabilities")
async def get_capabilities(
    connection_id: uuid.UUID = Query(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conn = await owned(db, user.id, connection_id)
    if conn is None:
        raise api_error(404, "not_found", "That ComfyUI connection no longer exists.")
    return capabilities(conn, await _ai_connected(db, user))


async def _prepare(
    db: AsyncSession, user: User, body: AdvancedBody
) -> tuple[ComfyConnection, dict[str, Any], dict[str, Any] | None]:
    """Every check before a job starts; (connection, options, photo)."""
    conn = await owned(db, user.id, body.connection_id)
    if conn is None:
        raise api_error(404, "not_found", "That ComfyUI connection no longer exists.")
    if conn.status != "ready":
        raise api_error(409, "not_connected", f"{conn.name} is not ready yet.")
    ai = await _ai_connected(db, user)
    caps = capabilities(conn, ai)
    variant = krea2.VARIANTS[body.variant]
    detection = (conn.info or {}).get("krea2") or {}
    installed = [
        f["filename"]
        for f in (detection.get("variants") or {}).get(body.variant, {}).get("files", [])
    ]
    if not caps["variants"][body.variant] or not installed:
        raise api_error(
            409,
            "missing_models",
            f"{variant.name} is not installed on {conn.name}. Install it in Generate › Krea 2.",
        )
    if body.resolution not in variant.tiers:
        raise api_error(
            400, "invalid_parameter", f"{variant.name} supports {', '.join(variant.tiers)}."
        )
    missing = caps["missing_nodes_edit" if body.mode == "edit" else "missing_nodes_text"]
    if missing:
        raise api_error(
            409, "missing_nodes", f"{conn.name} is missing ComfyUI nodes: {', '.join(missing)}."
        )
    if body.mode == "text-to-image" and body.aspect_ratio not in krea2.ASPECT_RATIOS:
        raise api_error(400, "invalid_parameter", "Unsupported aspect ratio.")
    targets = [t for t in body.targets if t in compiler.TARGETS]
    source: dict[str, Any] | None = None
    if body.mode == "edit":
        if body.image is None:
            raise api_error(400, "invalid_parameter", "Add the photo to edit.")
        try:
            source = await source_input(db, user, body.image)
        except GenerationRejected as exc:
            raise _reject(exc) from exc
        identity = body.identity and body.lora is not None
        regions = (
            body.analysis
            or identity
            or body.improve_background
            or any(t != "whole" for t in targets)
            or bool(compiler.detected_targets(body.prompt))
            or bool(body.spec)
        )
        if not body.analysis and not regions and not body.spec:
            raise api_error(400, "invalid_parameter", "Choose what to change, or turn on analysis.")
        if regions and not caps["sam3"]:
            raise api_error(
                409,
                "missing_models",
                f"SAM 3.1 (finds hair, mouth, background …) is not installed on {conn.name}.",
            )
        if regions:
            _need_anypaint(caps, conn)
    elif body.lora is not None:
        # Text to image with a LoRA: a face short of the goal is redrawn with AnyPaint.
        _need_anypaint(caps, conn)
    if body.analysis and not ai:
        raise api_error(
            409,
            "ai_not_connected",
            "Analysis and the AI check use your AI assistant: connect one in Integrations, "
            "or switch analysis off.",
        )
    if body.corrections and not body.quality:
        raise api_error(400, "invalid_parameter", "Corrections need the quality check.")
    prompt = body.prompt.strip()
    try:
        check_prompt(prompt, allows_mature=True)
        if not body.adult and is_explicit(prompt):
            raise PromptRefused("adult_off", ADULT_OFF)
    except PromptRefused as exc:
        raise _reject(exc) from exc
    lora_meta: dict[str, Any] | None = None
    if body.lora is not None:
        try:
            lora, version = await resolve_lora(db, user, body.lora, body.variant)
        except GenerationRejected as exc:
            raise _reject(exc) from exc
        name = lora_filename(lora, version)
        listed = name in (detection.get("loras") or [])
        deployed = str(conn.id) in (version.deployments or {})
        if file_capability(conn) == "none" and not deployed and not listed:
            raise api_error(
                409,
                "missing_models",
                "This ComfyUI accepts no uploads, and the LoRA is not in its models/loras "
                f"folder. Copy it there as {name}.",
            )
        lora_meta = {
            "lora_id": str(lora.id),
            "version_id": str(version.id),
            "name": lora.name,
            "version": version.number,
            "trigger_word": lora.trigger_word,
            "strength": body.lora.strength,
            "insert_trigger": body.lora.insert_trigger,
            "file": name,
            "trained_on": lora.base_variant,
        }
    options = {
        "connection_name": conn.name,
        "connection_kind": conn.kind,
        "unet": installed[0],
        "text_encoder": detection["text_encoders"][0],
        "vae": detection["vaes"][0],
        "steps": variant.steps,
        "cfg": variant.cfg,
        "lora": lora_meta,
        "aspect_ratio": body.aspect_ratio,
        "resolution": body.resolution,
        "seed": body.seed,
        # The first attempt's seed (the user's, else one chosen now): "Load settings" can
        # repeat the request; corrections always take new seeds.
        "first_seed": body.seed if body.seed is not None else secrets.randbelow(krea2.SEED_MAX),
        "targets": targets,
        "identity": body.mode == "edit" and body.identity and lora_meta is not None,
        "improve_background": body.mode == "edit" and body.improve_background,
        "unify_light": body.mode == "edit" and body.unify_light,
        "denoise": body.denoise,
        "analysis": body.analysis,
        "quality": body.quality,
        "corrections": body.corrections if body.quality else 0,
        "adult": body.adult or bool(source and source.get("nsfw")),
    }
    return conn, options, source


class PlanBody(BaseModel):
    mode: Literal["text-to-image", "edit"]
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
    image: SourceImage | None = None
    lora: LoraUse | None = None
    variant: Literal["turbo", "raw"] = "turbo"
    analysis: bool = True
    targets: list[str] = Field(default_factory=list, max_length=4)
    identity: bool = True
    improve_background: bool = False
    adult: bool = False


@router.post("/plan")
async def plan(
    body: PlanBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """The plan before anything runs (nothing on the GPU): what is kept, what changes, the
    Krea 2 prompts, regions and checks. The user can edit it and start with it."""
    import asyncio

    from app.services import storage

    prompt = body.prompt.strip()
    try:
        check_prompt(prompt, allows_mature=True)
        if not body.adult and is_explicit(prompt):
            raise PromptRefused("adult_off", ADULT_OFF)
        photo = None
        if body.mode == "edit":
            if body.image is None:
                raise api_error(400, "invalid_parameter", "Add the photo to edit.")
            source = await source_input(db, user, body.image)
            photo = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        lora_note = None
        if body.lora is not None:
            lora, _version = await resolve_lora(db, user, body.lora, body.variant)
            lora_note = f"{lora.name} ({lora.trigger_word})"
        identity = body.mode == "edit" and body.identity and body.lora is not None
        if body.analysis:
            spec = await compiler.by_ai(
                db,
                user,
                mode=body.mode,
                prompt=prompt,
                photo=photo,
                lora_note=lora_note,
                identity=identity,
            )
        else:
            spec = compiler.by_rules(
                mode=body.mode,
                prompt=prompt,
                targets=body.targets,
                whole=not body.improve_background,
            )
        spec = compiler.validate(
            spec,
            user_prompt=prompt,
            adult=body.adult,
            identity=identity,
            improve_background=body.mode == "edit" and body.improve_background,
        )
    except (GenerationRejected, PromptRefused, IntelligenceError) as exc:
        raise _reject(exc) from exc
    return spec.model_dump()


ACTIVE_STEP = ("queued", "processing")


async def _steps(
    db: AsyncSession, user_id: uuid.UUID, job_ids: list[uuid.UUID]
) -> dict[str, list[Generation]]:
    """The ComfyUI runs (steps) of these jobs, with their images, oldest first."""
    if not job_ids:
        return {}
    rows = (
        await db.scalars(
            select(Generation)
            .where(
                Generation.user_id == user_id,
                Generation.provider == "comfyui",
                Generation.comfy_meta["advanced"]["job_id"].astext.in_([str(i) for i in job_ids]),
            )
            .options(selectinload(Generation.images), selectinload(Generation.tasks))
            .order_by(Generation.created_at)
        )
    ).all()
    out: dict[str, list[Generation]] = {}
    for gen in rows:
        out.setdefault((gen.comfy_meta or {})["advanced"]["job_id"], []).append(gen)
    return out


def job_json(
    job: AdvancedJob, result: Image | None, steps: list[Generation] | None = None
) -> dict[str, Any]:
    options = dict(job.options or {})
    for key in ("unet", "text_encoder", "vae"):
        options.pop(key, None)
    started = job.started_at
    ended = job.finished_at
    return {
        "id": str(job.id),
        "mode": job.mode,
        "variant": job.variant,
        "prompt": job.prompt,
        "source": {
            k: (job.source or {}).get(k)
            for k in ("kind", "id", "thumb_url", "url", "width", "height")
        }
        if job.source
        else None,
        "lora": options.get("lora"),
        "options": options,
        "spec": job.spec or None,
        "stages": job.stages or [],
        "reports": job.reports or [],
        "status": job.status,
        "error_code": job.error_code,
        "error_message": job.error_message,
        # The chosen image, like every image elsewhere (download, viewer, delete).
        "result": {**image_json(result), "image_id": str(result.id)}
        if result is not None
        else None,
        # Every image the request made (attempts, corrections), oldest first.
        "images": [
            image_json(image, gen)
            for gen in steps or []
            for image in sorted(gen.images, key=lambda i: i.index)
        ],
        "steps_running": sum(1 for gen in steps or [] if gen.status in ACTIVE_STEP),
        "seconds": round((ended - started).total_seconds(), 1) if started and ended else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


async def _owned_job(db: AsyncSession, user: User, job_id: uuid.UUID) -> AdvancedJob:
    job = await db.scalar(
        select(AdvancedJob).where(AdvancedJob.id == job_id, AdvancedJob.user_id == user.id)
    )
    if job is None:
        raise api_error(404, "not_found", "That request no longer exists.")
    return job


async def _json(
    db: AsyncSession, job: AdvancedJob, steps: list[Generation] | None = None
) -> dict[str, Any]:
    if steps is None:
        steps = (await _steps(db, job.user_id, [job.id])).get(str(job.id), [])
    result = next((i for gen in steps for i in gen.images if i.id == job.result_image_id), None)
    if result is None and job.result_image_id:
        result = await db.scalar(
            select(Image)
            .where(Image.id == job.result_image_id)
            .options(selectinload(Image.generation))
        )
    return job_json(job, result, steps)


@router.post("/jobs", status_code=201)
async def create_job(
    body: AdvancedBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    if body.client_request_id:
        existing = await db.scalar(
            select(AdvancedJob).where(
                AdvancedJob.user_id == user.id,
                AdvancedJob.client_request_id == body.client_request_id,
            )
        )
        if existing is not None:
            return await _json(db, existing)
    conn, options, source = await _prepare(db, user, body)
    spec: dict[str, Any] = {}
    if body.spec is not None:
        try:
            checked = compiler.Spec(**{**body.spec, "mode": body.mode, "source": "user"})
            checked = compiler.validate(
                checked,
                user_prompt=body.prompt,
                adult=options["adult"],
                identity=options["identity"],
                improve_background=options["improve_background"],
            )
        except ValidationError as exc:
            raise api_error(400, "invalid_parameter", "The plan is not valid.") from exc
        except PromptRefused as exc:
            raise _reject(exc) from exc
        spec = checked.model_dump()
    job = AdvancedJob(
        user_id=user.id,
        client_request_id=body.client_request_id,
        connection_id=conn.id,
        mode=body.mode,
        variant=body.variant,
        prompt=body.prompt.strip(),
        source=source,
        lora_version_id=uuid.UUID(options["lora"]["version_id"]) if options["lora"] else None,
        lora_strength=body.lora.strength if body.lora else 1.0,
        options=options,
        spec=spec,
        stages=[],
        reports=[],
        status="queued",
        cancel_requested=False,
    )
    db.add(job)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise api_error(409, "duplicate", "This request was already started.") from exc
    engine.start(job.id)
    return await _json(db, job)


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    jobs = (
        await db.scalars(
            select(AdvancedJob)
            .where(AdvancedJob.user_id == user.id)
            .order_by(AdvancedJob.created_at.desc())
            .limit(limit)
        )
    ).all()
    steps = await _steps(db, user.id, [j.id for j in jobs])
    return {"items": [await _json(db, j, steps.get(str(j.id), [])) for j in jobs]}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return await _json(db, await _owned_job(db, user, job_id))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    job = await _owned_job(db, user, job_id)
    if job.status in ("queued", "running"):
        job.cancel_requested = True
        await db.commit()
    return await _json(db, job)


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Removes the request with every image it made and its steps' history entries (as
    deleting a generation does). A photo it started from is the user's own and stays."""
    job = await _owned_job(db, user, job_id)
    steps = (await _steps(db, user.id, [job.id])).get(str(job.id), [])
    if job.status in ("queued", "running") or any(g.status in ACTIVE_STEP for g in steps):
        raise api_error(409, "in_progress", "Stop the request or wait until it has finished.")
    paths = [p for gen in steps for img in gen.images for p in (img.file_path, img.thumb_path)]
    await db.delete(job)
    for gen in steps:
        await db.execute(delete(Image).where(Image.generation_id == gen.id))
        await db.delete(gen)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}
