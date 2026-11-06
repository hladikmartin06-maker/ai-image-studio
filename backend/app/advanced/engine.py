"""Runs an Advanced job: plan → Krea 2 generation or masked edits → checks → corrections
(each only when needed or switched on). Krea 2 only: no upscaler (Enhance photo is a
separate tool since the owner's request, 2026-11-06).

Every GPU stage is an ordinary ComfyUI generation of the engine's own graph, run by the
existing per-connection queue (services/worker.py + comfy/runner.py) exactly like any Krea 2
run: one at a time on that GPU, its LoRA uploaded the usual way, never sent twice after a
restart. The engine only creates those runs, waits for them and reads their images; it
changes nothing in that code. A failed stage fails the job with that stage's message — no
other model is tried instead."""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.advanced import crop, graphs, quality
from app.advanced.compiler import Edit, Spec, by_ai, by_rules, face_edit, validate
from app.advanced.intelligence import IntelligenceError
from app.advanced.quality import FACE_MATCH, evaluate, failed, rank
from app.comfy import krea2
from app.db import SessionLocal
from app.models import (
    AdvancedJob,
    Generation,
    Image,
    LoraDatasetItem,
    ProviderTask,
    ReferenceImage,
    User,
)
from app.services import faces, storage
from app.services.safety import PromptRefused
from app.services.worker import worker

log = logging.getLogger("studio.advanced")

POLL = 1.0  # tests make it instant
_tasks: dict[uuid.UUID, asyncio.Task[None]] = {}


class StageFailed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Cancelled(Exception):
    pass


def start(job_id: uuid.UUID) -> None:
    task = asyncio.create_task(_run(job_id))
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))


async def drain() -> None:
    """Waits until every running job has ended (tests)."""
    while _tasks:  # noqa: ASYNC110
        await asyncio.sleep(0.01)


async def recover() -> None:
    """A job that was running when the app stopped is not continued (its finished stages and
    their images are kept): it ends as "interrupted"."""
    async with SessionLocal() as db:
        await db.execute(
            update(AdvancedJob)
            .where(AdvancedJob.status.in_(("queued", "running")))
            .values(
                status="interrupted",
                error_code="interrupted",
                error_message="The app stopped while this request ran; start it again.",
                finished_at=datetime.now(UTC),
            )
        )
        await db.commit()


# --- job record ---------------------------------------------------------------------------


async def _update(job_id: uuid.UUID, **values: Any) -> None:
    async with SessionLocal() as db:
        await db.execute(update(AdvancedJob).where(AdvancedJob.id == job_id).values(**values))
        await db.commit()


async def _stage(job_id: uuid.UUID, key: str, label: str, **values: Any) -> None:
    """Adds or updates one stage of the job's timeline."""
    async with SessionLocal() as db:
        job = await db.get(AdvancedJob, job_id)
        if job is None:
            return
        stages = [dict(s) for s in job.stages or []]
        now = datetime.now(UTC).isoformat()
        current = next((s for s in stages if s["key"] == key), None)
        if current is None:
            current = {"key": key, "label": label, "status": "running", "started_at": now}
            stages.append(current)
        current.update(values)
        if values.get("status") in ("completed", "failed", "skipped", "cancelled"):
            current["finished_at"] = now
        job.stages = stages
        await db.commit()


async def _cancel_requested(job_id: uuid.UUID) -> bool:
    async with SessionLocal() as db:
        return bool(
            await db.scalar(select(AdvancedJob.cancel_requested).where(AdvancedJob.id == job_id))
        )


# --- one GPU stage --------------------------------------------------------------------------


def _image_input(image: Image) -> dict[str, Any]:
    return {
        "kind": "image",
        "id": str(image.id),
        "file_path": image.file_path,
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
        "thumb_url": f"/api/images/{image.id}/thumb",
        "url": f"/api/images/{image.id}/file",
        "nsfw": image.nsfw,
    }


async def _gpu(
    job: AdvancedJob,
    *,
    key: str,
    label: str,
    workflow: dict[str, Any],
    prompt: str,
    settings: dict[str, Any],
    source: dict[str, Any] | None,
    model_key: str,
    family: str,
    lora: dict[str, Any] | None,
) -> Image:
    """One ComfyUI run of the engine's graph through the existing queue; its image."""
    await _stage(job.id, key, label, status="running")
    options = job.options or {}
    adult = bool(options.get("adult")) or bool(source and source.get("nsfw"))
    async with SessionLocal() as db:
        gen = Generation(
            user_id=job.user_id,
            model_key=model_key,
            provider="comfyui",
            mode="image-edit" if source else "text-to-image",
            provider_model_id=options.get("unet") or model_key,
            prompt=prompt,
            settings={**settings, **({"adult": True} if adult else {})},
            request_payload=workflow,
            reference_image_ids=[],
            image_count=1,
            status="queued",
            lora_version_id=uuid.UUID(lora["version_id"]) if lora else None,
            comfy_meta={
                "connection_id": str(job.connection_id),
                "connection_name": options.get("connection_name"),
                "connection_kind": options.get("connection_kind"),
                "family": family,
                "variant": job.variant,
                "model_file": options.get("unet"),
                "text_encoder": options.get("text_encoder"),
                "vae": options.get("vae"),
                "workflow": {"id": None, "name": f"Advanced: {label}"},
                "lora": lora,
                **({"input": source} if source else {}),
                "adult": adult,
                "advanced": {"job_id": str(job.id), "stage": key},
            },
            tasks=[ProviderTask(index=0, images_requested=1)],
        )
        db.add(gen)
        await db.commit()
        gen_id = gen.id
    await _stage(job.id, key, label, generation_id=str(gen_id))
    worker.enqueue(gen_id)
    while True:
        await asyncio.sleep(POLL)
        if await _cancel_requested(job.id):
            await _cancel_generation(gen_id)
            await _stage(job.id, key, label, status="cancelled")
            raise Cancelled()
        async with SessionLocal() as db:
            row = await db.scalar(
                select(Generation)
                .where(Generation.id == gen_id)
                .options(selectinload(Generation.images))
            )
        if row is None:
            raise StageFailed("not_found", "A step of this request was deleted.")
        if row.status in ("completed", "failed", "cancelled"):
            break
    if row.status != "completed" or not row.images:
        await _stage(job.id, key, label, status="failed", detail=row.error_message)
        raise StageFailed(
            row.error_code or "generation_failed", row.error_message or "The step failed."
        )
    image = row.images[0]
    await _stage(job.id, key, label, status="completed", image_id=str(image.id))
    return image


async def _cancel_generation(gen_id: uuid.UUID) -> None:
    from app.comfy.runner import request_cancel
    from app.services.worker import load_generation

    async with SessionLocal() as db:
        row = await db.get(Generation, gen_id)
        if row is None:
            return
        if row.status == "queued":
            row.status = "cancelled"
            row.completed_at = datetime.now(UTC)
            await db.commit()
            return
        if row.status == "processing":
            row.comfy_meta = {**(row.comfy_meta or {}), "cancel_requested": True}
            await db.commit()
    loaded = await load_generation(gen_id)
    if loaded is not None and loaded.status == "processing":
        await request_cancel(loaded)


# --- stages ---------------------------------------------------------------------------------


def _krea_args(job: AdvancedJob) -> dict[str, Any]:
    options = job.options or {}
    lora = options.get("lora")
    return {
        "variant": job.variant,
        "unet": options["unet"],
        "text_encoder": options["text_encoder"],
        "vae": options["vae"],
        "steps": int(options["steps"]),
        "cfg": float(options["cfg"]),
        "loras": [(lora["file"], float(lora["strength"]))] if lora else [],
    }


def _with_trigger(job: AdvancedJob, prompt: str) -> str:
    lora = (job.options or {}).get("lora")
    if lora and lora.get("insert_trigger", True):
        trigger = lora["trigger_word"]
        if trigger.lower() not in prompt.lower().split(", ")[0].lower().split():
            return f"{trigger}, {prompt}"
    return prompt


def _negative(job: AdvancedJob, spec: Spec, target: str | None = None) -> str:
    if not krea2.VARIANTS[job.variant].negative_prompt:
        return ""
    parts = [spec.negative] if spec.negative else []
    if target == "mouth":
        parts.append("open mouth, teeth, tongue")
    return ", ".join(p for p in parts if p)


def _seed(job: AdvancedJob, attempt: int) -> int:
    """The first attempt uses the request's seed (chosen once when it was made, so it can
    be repeated); every correction a new one."""
    options = job.options or {}
    seed = options.get("seed")
    if not isinstance(seed, int):
        seed = options.get("first_seed")
    if isinstance(seed, int) and attempt == 0:
        return seed
    return secrets.randbelow(krea2.SEED_MAX)


async def _generate(job: AdvancedJob, spec: Spec, key: str, attempt: int) -> Image:
    options = job.options or {}
    width, height = krea2.size_for(
        options.get("aspect_ratio", "3:4"), options.get("resolution", "1K")
    )
    seed = _seed(job, attempt)
    prompt = _with_trigger(job, spec.prompt)
    args = _krea_args(job)
    workflow = graphs.text_to_image(
        **args, prompt=prompt, negative=_negative(job, spec), width=width, height=height, seed=seed
    )
    return await _gpu(
        job,
        key=key,
        label="Krea 2" if attempt == 0 else f"Krea 2 · attempt {attempt + 1}",
        workflow=workflow,
        prompt=prompt,
        settings={
            "width": width,
            "height": height,
            "seed": seed,
            "steps": args["steps"],
            "cfg": args["cfg"],
            "engine": "advanced",
        },
        source=None,
        model_key=krea2.VARIANTS[job.variant].model_key,
        family=krea2.FAMILY,
        lora=options.get("lora"),
    )


async def _edit(
    job: AdvancedJob, spec: Spec, edit: Edit, source: dict[str, Any], key: str, attempt: int
) -> Image:
    options = job.options or {}
    width, height = krea2.size_for_source(
        source.get("width") or 1024, source.get("height") or 1024, options.get("resolution", "1K")
    )
    region = edit.region if edit.target != "whole" else None
    exclude = graphs.EXCLUDE.get(edit.target, ()) if region else ()
    # The LoRA character's face: in an edit with the switch on, and in text to image.
    identity = bool(options.get("identity")) or (
        job.mode == "text-to-image" and edit.target == "face" and bool(options.get("lora"))
    )
    # A colour change starts from the tinted region, so it needs less redrawing.
    colour = edit.color if region and edit.target in graphs.COLOUR_TARGETS else None
    default = graphs.DENOISE[edit.target]
    if colour:
        default = graphs.DENOISE_COLOUR.get(edit.target, default)
    elif edit.target == "face" and identity:
        default = graphs.DENOISE_IDENTITY_FACE
    elif edit.target == "body" and identity:
        default = graphs.DENOISE_IDENTITY_BODY
    base = edit.denoise or options.get("denoise") or default
    # A correction redraws stronger; the LoRA character's face again as gently (its pose and
    # expression stay; the passes add up).
    gentle = edit.target == "face" and identity
    denoise = float(base) if gentle else min(graphs.DENOISE_MAX, float(base) + 0.1 * attempt)
    # Face regions are redrawn in an enlarged square around the face.
    square = None
    if region and edit.target in crop.FACE_TARGETS and source.get("file_path"):
        box = await crop.face_box(storage.absolute(source["file_path"]))
        square = crop.plan_crop(box, width, height)
    seed = _seed(job, attempt)
    # The background without the LoRA (its studio backdrops turned places grey) and its
    # trigger, which would draw the character into it.
    lora = None if edit.target == "background" else options.get("lora")
    prompt = _with_trigger(job, edit.prompt) if lora else edit.prompt
    args = _krea_args(job)
    if lora is None:
        args["loras"] = []
    workflow = graphs.edit(
        **args,
        prompt=prompt,
        negative=_negative(job, spec, edit.target),
        width=width,
        height=height,
        seed=seed,
        denoise=denoise,
        region=region,
        exclude=exclude,
        soften=graphs.SOFTEN.get(edit.target) if region else None,
        grow=graphs.GROW.get(edit.target, graphs.GROW_DEFAULT),
        colour=colour,
        attempt=attempt,
        crop=square,
        crop_size=crop.CROP_SIZE,
    )
    return await _gpu(
        job,
        key=key,
        label=f"{edit.target}" + (f" · attempt {attempt + 1}" if attempt else ""),
        workflow=workflow,
        prompt=prompt,
        settings={
            "width": width,
            "height": height,
            "seed": seed,
            "steps": args["steps"],
            "cfg": args["cfg"],
            "denoise": denoise,
            "region": region or "whole photo",
            **({"exclude": list(exclude)} if exclude else {}),
            **({"color": colour, "tint": list(graphs.tint(colour, attempt))} if colour else {}),
            **({"face_crop": list(square)} if square else {}),
            "engine": "advanced",
        },
        source=source,
        model_key=krea2.VARIANTS[job.variant].model_key,
        family=krea2.FAMILY,
        lora=lora,
    )


# Face corrections of a person turned into the LoRA's character when the user set fewer (each
# is one face redraw in its crop, stronger, with a new seed; the most alike face is kept).
IDENTITY_TRIES = 2


async def _lora_photos(job: AdvancedJob) -> list[Path]:
    lora = (job.options or {}).get("lora")
    if not lora:
        return []
    async with SessionLocal() as db:
        rows = (
            await db.scalars(
                select(ReferenceImage)
                .join(LoraDatasetItem, LoraDatasetItem.reference_image_id == ReferenceImage.id)
                .where(
                    LoraDatasetItem.lora_id == uuid.UUID(lora["lora_id"]),
                    LoraDatasetItem.user_id == job.user_id,
                    LoraDatasetItem.excluded.is_(False),
                    LoraDatasetItem.split == "train",
                )
                .order_by(LoraDatasetItem.position)
                .limit(24)
            )
        ).all()
    return [storage.absolute(r.file_path) for r in rows]


async def _check(
    job: AdvancedJob, user: User, spec: Spec, image: Image, key: str
) -> dict[str, Any]:
    options = job.options or {}
    await _stage(job.id, key, "check", status="running")
    original = storage.absolute(job.source["file_path"]) if job.source else None
    # The person became the LoRA's character: the face is measured against the LoRA's photos.
    lora_identity = bool(options.get("identity"))
    async with SessionLocal() as db:
        report = await evaluate(
            db,
            user,
            spec=spec,
            result=storage.absolute(image.file_path),
            original=original,
            identity_refs=await _lora_photos(job) if lora_identity or not original else [],
            # Quality checks off: only the measured face of a LoRA character (on this Mac).
            use_ai=bool(options.get("analysis") and options.get("quality")),
            lora_identity=lora_identity,
        )
    report["image_id"] = str(image.id)
    await _stage(job.id, key, "check", status="completed", detail=report.get("error"))
    return report


async def _plan(job: AdvancedJob, user: User) -> Spec:
    options = job.options or {}
    if job.spec:
        spec = Spec(**job.spec)
        await _stage(job.id, "plan", "plan", status="completed", detail="your plan")
        return spec
    await _stage(job.id, "plan", "plan", status="running")
    if options.get("analysis"):
        photo = (
            await asyncio.to_thread(storage.absolute(job.source["file_path"]).read_bytes)
            if job.source
            else None
        )
        lora = options.get("lora")
        async with SessionLocal() as db:
            spec = await by_ai(
                db,
                user,
                mode=job.mode,
                prompt=job.prompt,
                photo=photo,
                lora_note=f"{lora['name']} ({lora['trigger_word']})" if lora else None,
                identity=bool(options.get("identity")),
            )
    else:
        spec = by_rules(
            mode=job.mode,
            prompt=job.prompt,
            targets=options.get("targets") or [],
            whole=not options.get("improve_background"),
        )
    spec = validate(
        spec,
        user_prompt=job.prompt,
        adult=bool(options.get("adult")),
        identity=bool(options.get("identity")),
        improve_background=bool(options.get("improve_background")),
    )
    await _update(job.id, spec=spec.model_dump())
    await _stage(job.id, "plan", "plan", status="completed", detail=spec.source)
    return spec


# --- the job -------------------------------------------------------------------------------


async def _run(job_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        job = await db.get(AdvancedJob, job_id)
        user = await db.get(User, job.user_id) if job else None
    if job is None or user is None or job.status != "queued":
        return
    await _update(job_id, status="running", started_at=datetime.now(UTC))
    options = job.options or {}
    produced: list[Image] = []
    try:
        spec = await _plan(job, user)
        # 1. generate, or apply the edits one after another
        if job.mode == "text-to-image":
            image = await _generate(job, spec, "generate", 0)
            produced.append(image)
            if options.get("lora"):
                image = await _lora_face(job, spec, image, produced)
        else:
            current = job.source or {}
            image = None
            for index, edit in enumerate(spec.edits):
                key = f"edit{index + 1}"
                if job.mode == "edit" and options.get("identity") and edit.target == "face":
                    image = await _identity_face(job, spec, edit, current, key, produced)
                else:
                    image = await _edit(job, spec, edit, current, key, 0)
                    produced.append(image)
                current = _image_input(image)
            assert image is not None
        attempts: list[dict[str, Any]] = [{"attempt": 0, "image": image, "report": None}]
        # 2. check, 3. correct what failed (at most N times; never endless). A person turned
        # into the LoRA's character always has the face measured and corrected.
        identity = job.mode == "edit" and bool(options.get("identity"))
        corrections = int(options.get("corrections", 0))
        if identity:
            corrections = max(corrections, IDENTITY_TRIES)
        if options.get("quality") or identity:
            report = await _check(job, user, spec, image, "check")
            report["attempt"] = 0
            attempts[0]["report"] = report
            await _update(job_id, reports=[report])
            reports = [report]
            for attempt in range(1, corrections + 1):
                # The LoRA character's face was already tried before the other edits.
                bad = [
                    c
                    for c in failed(report)
                    if c.get("kind") == "modify" and not (identity and c.get("id") == FACE_MATCH)
                ]
                if not bad:
                    break
                # The LoRA character's face alone: every try starts again from the same image
                # with a new seed and the most alike is kept (tries that build on each other
                # added up artifacts on the owner's selfie, 2026-11-05).
                again = identity and all(c.get("id") == FACE_MATCH for c in bad)
                start = attempts[0]["image"] if again else image
                image = await _correct(job, spec, start, bad, attempt)
                produced.append(image)
                report = await _check(job, user, spec, image, f"check{attempt + 1}")
                report["attempt"] = attempt
                attempts.append({"attempt": attempt, "image": image, "report": report})
                reports.append(report)
                await _update(job_id, reports=reports)
            best = max(attempts, key=lambda a: rank(a["report"]) if a["report"] else (0, 0, 0.0))
            image = best["image"]
        await _finish(job_id, image, produced)
    except Cancelled:
        await _update(job_id, status="cancelled", finished_at=datetime.now(UTC))
    except (StageFailed, IntelligenceError, PromptRefused) as exc:
        code = getattr(exc, "code", "failed")
        await _update(
            job_id,
            status="failed",
            error_code=code,
            error_message=getattr(exc, "message", str(exc)),
            finished_at=datetime.now(UTC),
        )
        await _fail_open_stage(job_id)
    except Exception:  # noqa: BLE001 - one broken job must not stop the app
        log.exception("Advanced job %s crashed", job_id)
        await _update(
            job_id,
            status="failed",
            error_code="engine_error",
            error_message="Unexpected error in the advanced engine.",
            finished_at=datetime.now(UTC),
        )
        await _fail_open_stage(job_id)


def _goal(baseline: float | None) -> float:
    return max(faces.MATCH_THRESHOLD, quality.IDENTITY_SHARE * (baseline or 0.0))


async def _lora_face(job: AdvancedJob, spec: Spec, image: Image, produced: list[Image]) -> Image:
    """Text to image with a LoRA: a face short of the goal is redrawn like an edit's LoRA
    face (in its enlarged crop: a face far from the camera is a few latent pixels, too few
    for the LoRA's features; the owner's hard prompts, 2026-11-06, were 42–53 % alike). The
    most alike face is kept, the generated one included. No face (seen from behind, too
    small to measure) = the image as generated."""
    refs = await _lora_photos(job)
    try:
        score, baseline = await quality.identity(storage.absolute(image.file_path), refs)
    except faces.FaceError:
        return image
    if score is None or score >= _goal(baseline):
        return image
    edit = face_edit(spec.prompt or job.prompt)
    return await _identity_face(
        job, spec, edit, _image_input(image), "edit1", produced, start=(score, image)
    )


async def _identity_face(
    job: AdvancedJob,
    spec: Spec,
    edit: Edit,
    source: dict[str, Any],
    key: str,
    produced: list[Image],
    start: tuple[float, Image] | None = None,
) -> Image:
    """The LoRA character's face, tried up to 1 + IDENTITY_TRIES times from the same photo
    with a new seed each, until one is as alike to the LoRA's photos as the goal; the most
    alike try (or `start`, the image it began from) is kept. Measured on this Mac (SFace)."""
    refs = await _lora_photos(job)
    best = start
    for attempt in range(IDENTITY_TRIES + 1):
        image = await _edit(
            job, spec, edit, source, key if attempt == 0 else f"{key}-try{attempt + 1}", attempt
        )
        produced.append(image)
        try:
            score, baseline = await quality.identity(storage.absolute(image.file_path), refs)
        except faces.FaceError:
            score, baseline = None, None
        value = score if score is not None else -1.0
        if best is None or value > best[0]:
            best = (value, image)
        if score is None or score >= _goal(baseline):
            break
    assert best is not None
    return best[1]


async def _correct(
    job: AdvancedJob, spec: Spec, image: Image, bad: list[dict[str, Any]], attempt: int
) -> Image:
    """Redraws only what failed: the edit of that region again (stronger, new seed), or for
    text to image a masked edit of that region; a failure without a region = a new seed."""
    targets = list(dict.fromkeys(c.get("target") for c in bad))
    if None in targets and job.mode == "text-to-image":
        return await _generate(job, spec, f"fix{attempt}", attempt)
    current = _image_input(image)
    result = image
    for target in [t for t in targets if t]:
        edit = next((e for e in spec.edits if e.target == target), None)
        if edit is None:
            edit = Edit(
                target=target,
                region=graphs.REGIONS.get(target, target),
                prompt=spec.prompt or job.prompt,
            )
        result = await _edit(job, spec, edit, current, f"fix{attempt}-{target}", attempt)
        current = _image_input(result)
    return result


async def _fail_open_stage(job_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        job = await db.get(AdvancedJob, job_id)
        if job is None:
            return
        stages = [dict(s) for s in job.stages or []]
        for stage in stages:
            if stage.get("status") == "running":
                stage["status"] = "failed"
                stage["finished_at"] = datetime.now(UTC).isoformat()
        job.stages = stages
        await db.commit()


async def _finish(job_id: uuid.UUID, image: Image, produced: list[Image]) -> None:
    """The chosen image is the result (in the library as usual); the steps' other images stay
    in History but leave the library."""
    others = [i.id for i in produced if i.id != image.id]
    async with SessionLocal() as db:
        if others:
            await db.execute(update(Image).where(Image.id.in_(others)).values(in_library=False))
        await db.execute(
            update(AdvancedJob)
            .where(AdvancedJob.id == job_id)
            .values(status="completed", result_image_id=image.id, finished_at=datetime.now(UTC))
        )
        await db.commit()
