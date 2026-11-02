"""Generation requests that run on a ComfyUI connection (Krea 2, built-in or custom
workflow, optional trained LoRA and AI model). They are ordinary Generation rows (provider
"comfyui"), so the Library, History, Favorites and live updates work unchanged."""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import MAX_PROMPT_CHARS
from app.comfy import enhance, krea2, workflows
from app.comfy.service import file_capability, lora_filename, owned
from app.models import (
    ComfyConnection,
    ComfyWorkflow,
    Generation,
    Image,
    Lora,
    LoraVersion,
    ProviderTask,
    ReferenceImage,
    User,
)
from app.services import personas
from app.services.generation import GenerationRejected
from app.services.safety import ADULT_OFF, PromptRefused, check_prompt, is_explicit
from app.services.worker import worker

PROVIDER = "comfyui"


class LoraUse(BaseModel):
    version_id: uuid.UUID
    strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    # Adds the trigger word to the prompt when it is not there yet.
    insert_trigger: bool = True


class PersonaText(BaseModel):
    id: uuid.UUID
    use_description: bool = True


class SourceImage(BaseModel):
    """A photo of the user's: "reference" = an upload or an AI model's photo, "image" = a
    generated image."""

    kind: str = Field(pattern="^(reference|image)$")
    id: uuid.UUID


class ComfyGenerateBody(BaseModel):
    connection_id: uuid.UUID
    variant: str = Field(pattern="^(turbo|raw)$")
    model_file: str | None = Field(default=None, max_length=300)
    workflow_id: uuid.UUID | None = None
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    negative_prompt: str = Field(default="", max_length=2000)
    aspect_ratio: str = "1:1"
    resolution: str = "1K"
    seed: int | None = Field(default=None, ge=0, le=krea2.SEED_MAX)
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    count: int = Field(default=1, ge=1, le=4)
    lora: LoraUse | None = None
    # Image to image (built-in Krea 2 workflow): the photo and how much of it is redrawn.
    image: SourceImage | None = None
    denoise: float = Field(
        default=krea2.DENOISE_DEFAULT, ge=krea2.DENOISE_MIN, le=krea2.DENOISE_MAX
    )
    # 18+ content (adults only): off = explicit prompts are refused, like on the SFW models;
    # on = allowed, and the images are marked 18+ (blurred when Settings say so).
    adult: bool = False
    # Run without any LoRA even if the AI model has one (base-model comparisons).
    persona: PersonaText | None = None
    client_request_id: str | None = Field(default=None, max_length=64)
    # LoRA testing: groups runs that are compared side by side.
    test_group: str | None = Field(default=None, max_length=64)
    test_label: str | None = Field(default=None, max_length=80)
    test_lora_id: uuid.UUID | None = None
    test_prompt: str | None = Field(default=None, max_length=4000)
    source_generation_id: uuid.UUID | None = None


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", text, re.IGNORECASE) is not None


async def resolve_lora(
    db: AsyncSession, user: User, use: LoraUse, variant: str
) -> tuple[Lora, LoraVersion]:
    version = await db.scalar(
        select(LoraVersion).where(LoraVersion.id == use.version_id, LoraVersion.user_id == user.id)
    )
    if version is None:
        raise GenerationRejected("not_found", "That LoRA version no longer exists.", 404)
    lora = await db.scalar(select(Lora).where(Lora.id == version.lora_id, Lora.user_id == user.id))
    if lora is None:
        raise GenerationRejected("not_found", "That LoRA no longer exists.", 404)
    if version.status != "completed" or not version.file_path:
        raise GenerationRejected(
            "lora_not_ready", f"{lora.name} v{version.number} has no finished file yet.", 409
        )
    ok, message = krea2.lora_compatibility(lora.architecture, lora.base_variant, variant)
    if not ok:
        raise GenerationRejected("lora_incompatible", message, 409)
    return lora, version


async def create_comfy_generation(
    db: AsyncSession, user: User, body: ComfyGenerateBody
) -> Generation:
    if body.client_request_id:
        existing = await db.scalar(
            select(Generation)
            .where(
                Generation.user_id == user.id,
                Generation.client_request_id == body.client_request_id,
            )
            .options(selectinload(Generation.tasks), selectinload(Generation.images))
        )
        if existing is not None:
            return existing

    conn: ComfyConnection | None = await owned(db, user.id, body.connection_id)
    if conn is None:
        raise GenerationRejected("not_found", "That ComfyUI connection no longer exists.", 404)
    if conn.status != "ready":
        raise GenerationRejected(
            "not_connected", f"{conn.name} is not ready. Test or deploy it in ComfyUI first.", 409
        )
    variant = krea2.VARIANTS[body.variant]
    detection = (conn.info or {}).get("krea2") or {}
    installed = [
        f["filename"]
        for f in (detection.get("variants") or {}).get(body.variant, {}).get("files", [])
    ]
    unet = body.model_file or (installed[0] if installed else None)
    if not unet or unet not in installed:
        raise GenerationRejected(
            "missing_models",
            f"{variant.name} is not installed on {conn.name}. Add its model file in ComfyUI "
            "(models/diffusion_models) and test the connection again.",
            409,
        )
    encoders = detection.get("text_encoders") or []
    vaes = detection.get("vaes") or []
    if not encoders or not vaes:
        raise GenerationRejected(
            "missing_models",
            f"{conn.name} is missing the Krea 2 text encoder or VAE: "
            + "; ".join(detection.get("missing") or []),
            409,
        )
    if not detection.get("supported", False):
        raise GenerationRejected(
            "missing_nodes", "This ComfyUI version has no Krea 2 support. Update ComfyUI.", 409
        )
    if body.resolution not in variant.tiers:
        raise GenerationRejected(
            "invalid_parameter", f"{variant.name} supports {', '.join(variant.tiers)}."
        )
    source: dict[str, Any] | None = None
    if body.image is not None:
        if body.workflow_id is not None:
            raise GenerationRejected(
                "invalid_parameter", "Image to image works with the built-in Krea 2 workflow."
            )
        known_nodes = set((conn.info or {}).get("node_names") or [])
        missing = [n for n in krea2.IMG2IMG_NODES if known_nodes and n not in known_nodes]
        if missing:
            raise GenerationRejected(
                "missing_nodes",
                f"{conn.name} is missing ComfyUI nodes: {', '.join(missing)}. Update ComfyUI.",
                409,
            )
        source = await source_input(db, user, body.image)
        width, height = krea2.size_for_source(
            source["width"] or 1024, source["height"] or 1024, body.resolution
        )
    else:
        try:
            width, height = krea2.size_for(body.aspect_ratio, body.resolution)
        except ValueError as exc:
            raise GenerationRejected("invalid_parameter", "Unsupported aspect ratio.") from exc

    prompt = body.prompt.strip()
    try:
        # Sexual content with minor-related terms: never, whatever the switch says.
        check_prompt(prompt, allows_mature=True)
        if body.negative_prompt:
            check_prompt(body.negative_prompt, allows_mature=True)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc
    if not body.adult and is_explicit(prompt):
        raise GenerationRejected("adult_off", ADULT_OFF, 422)

    # Trained LoRA (strict compatibility) and the AI model's description.
    lora_meta: dict[str, Any] | None = None
    loras: list[tuple[str, float]] = []
    final = prompt
    lora_version_id = None
    if body.lora is not None:
        lora, version = await resolve_lora(db, user, body.lora, body.variant)
        if file_capability(conn) == "none" and str(conn.id) not in (version.deployments or {}):
            # Files cannot be pushed here; it must already be listed by ComfyUI.
            listed = detection.get("loras") or []
            if lora_filename(lora, version) not in listed:
                raise GenerationRejected(
                    "missing_models",
                    "This ComfyUI accepts no uploads, and the LoRA is not in its models/loras "
                    f"folder. Copy it there as {lora_filename(lora, version)}.",
                    409,
                )
        loras.append((lora_filename(lora, version), body.lora.strength))
        if body.lora.insert_trigger and not _has_word(final, lora.trigger_word):
            final = f"{lora.trigger_word}, {final}"
        lora_version_id = version.id
        lora_meta = {
            "lora_id": str(lora.id),
            "version_id": str(version.id),
            "name": lora.name,
            "version": version.number,
            "trigger_word": lora.trigger_word,
            "strength": body.lora.strength,
            "file": lora_filename(lora, version),
            "trained_on": lora.base_variant,
        }
    persona_meta: dict[str, Any] = {}
    persona_id = None
    if body.persona is not None:
        persona = await personas.owned(db, user.id, body.persona.id)
        if persona is None:
            raise GenerationRejected("not_found", "The selected AI model no longer exists.", 404)
        persona_id = persona.id
        persona_meta = {
            "name": persona.name,
            "description_used": body.persona.use_description,
            "references_used": False,
            "reference_ids": [],
            "references_skipped": "unsupported",
        }
        if body.persona.use_description:
            final = personas.compose_prompt(final, persona, 0)
            try:
                personas.check_composed(final, persona, True)
                if not body.adult and is_explicit(final):
                    raise PromptRefused("adult_off", ADULT_OFF)
            except PromptRefused as exc:
                raise GenerationRejected(exc.code, exc.message, 422) from exc
        persona.last_used_at = datetime.now(UTC)

    seed = body.seed if body.seed is not None else secrets.randbelow(krea2.SEED_MAX)
    steps = body.steps or variant.steps
    cfg = body.cfg if body.cfg is not None else variant.cfg
    negative = body.negative_prompt.strip() if variant.negative_prompt else ""

    workflow_meta: dict[str, Any] = {"id": None, "name": "Krea 2 (built-in)"}
    if body.workflow_id is not None:
        wf_row = await db.scalar(
            select(ComfyWorkflow).where(
                ComfyWorkflow.id == body.workflow_id, ComfyWorkflow.user_id == user.id
            )
        )
        if wf_row is None or not wf_row.api_json:
            raise GenerationRejected(
                "not_found", "That workflow is missing or has no API-format version.", 404
            )
        known_nodes = set((conn.info or {}).get("node_names") or [])
        missing = [n for n in workflows.node_types(wf_row.api_json) if n not in known_nodes]
        if missing:
            raise GenerationRejected(
                "missing_nodes",
                f"{conn.name} is missing custom nodes for this workflow: {', '.join(missing)}.",
                409,
            )
        try:
            workflow = workflows.apply(
                wf_row.api_json,
                wf_row.bindings or {},
                {
                    "prompt": final,
                    "negative_prompt": negative if variant.negative_prompt else None,
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "width": width,
                    "height": height,
                    "batch": body.count,
                    "unet": unet,
                    "loras": loras,
                },
            )
        except workflows.WorkflowError as exc:
            raise GenerationRejected("invalid_parameter", str(exc)) from exc
        workflow_meta = {"id": str(wf_row.id), "name": wf_row.name}
    else:
        workflow = krea2.build_workflow(
            variant=body.variant,
            unet=unet,
            text_encoder=encoders[0],
            vae=vaes[0],
            prompt=final,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            batch=body.count,
            loras=loras,
            image=krea2.INPUT_PLACEHOLDER if source else None,
            denoise=body.denoise,
        )

    settings = {
        "aspect_ratio": "match" if source else body.aspect_ratio,
        "resolution": body.resolution,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": variant.sampler,
        "scheduler": variant.scheduler,
    }
    if negative:
        settings["negative_prompt"] = negative
    if source:
        settings.update(
            denoise=body.denoise,
            source_width=source["width"],
            source_height=source["height"],
        )
    # 18+: the switch, or a photo that is itself marked 18+.
    adult = body.adult or bool(source and source.get("nsfw"))
    if adult:
        settings["adult"] = True
    gen = Generation(
        user_id=user.id,
        client_request_id=body.client_request_id,
        model_key=variant.model_key,
        provider=PROVIDER,
        mode="image-edit" if source else "text-to-image",
        provider_model_id=unet,
        prompt=prompt,
        final_prompt=final if final != prompt else None,
        settings=settings,
        request_payload=workflow,
        reference_image_ids=[source["id"]] if source and source["kind"] == "reference" else [],
        image_count=body.count,
        status="queued",
        source_generation_id=body.source_generation_id,
        persona_id=persona_id,
        persona_meta=persona_meta,
        lora_version_id=lora_version_id,
        comfy_meta={
            "connection_id": str(conn.id),
            "connection_name": conn.name,
            "connection_kind": conn.kind,
            "family": krea2.FAMILY,
            "variant": body.variant,
            "model_file": unet,
            "text_encoder": encoders[0],
            "vae": vaes[0],
            "workflow": workflow_meta,
            "lora": lora_meta,
            "test_group": body.test_group,
            "test_label": body.test_label,
            "test_lora_id": str(body.test_lora_id) if body.test_lora_id else None,
            "test_prompt": body.test_prompt,
            # Image to image: the photo, uploaded to ComfyUI before each run (runner).
            **({"input": source} if source else {}),
            "adult": adult,
        },
        tasks=[ProviderTask(index=0, images_requested=body.count)],
    )
    db.add(gen)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        if body.client_request_id:
            existing = await db.scalar(
                select(Generation)
                .where(
                    Generation.user_id == user.id,
                    Generation.client_request_id == body.client_request_id,
                )
                .options(selectinload(Generation.tasks), selectinload(Generation.images))
            )
            if existing is not None:
                return existing
        raise
    worker.enqueue(gen.id)
    loaded = await db.scalar(
        select(Generation)
        .where(Generation.id == gen.id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
        .execution_options(populate_existing=True)
    )
    assert loaded is not None
    return loaded


async def repeat(
    db: AsyncSession, user: User, source: Generation, count: int | None, rid: str | None
):
    """Runs a ComfyUI generation again with the same workflow (same seed, LoRA file)."""
    meta = source.comfy_meta or {}
    conn = (
        await owned(db, user.id, uuid.UUID(meta["connection_id"]))
        if meta.get("connection_id")
        else None
    )
    if conn is None:
        raise GenerationRejected(
            "not_found", "The ComfyUI connection of this run was removed.", 404
        )
    if conn.status != "ready":
        raise GenerationRejected("not_connected", f"{conn.name} is not ready.", 409)
    workflow = dict(source.request_payload or {})
    if count and count != source.image_count:
        workflow = {k: {**v, "inputs": dict(v.get("inputs") or {})} for k, v in workflow.items()}
        for node in workflow.values():
            if "batch_size" in node["inputs"]:
                node["inputs"]["batch_size"] = count
            if node.get("class_type") == "RepeatLatentBatch":
                node["inputs"]["amount"] = count
    gen = Generation(
        user_id=user.id,
        client_request_id=rid,
        model_key=source.model_key,
        provider=PROVIDER,
        mode=source.mode,
        provider_model_id=source.provider_model_id,
        prompt=source.prompt,
        final_prompt=source.final_prompt,
        settings=dict(source.settings),
        request_payload=workflow,
        reference_image_ids=[],
        image_count=count or source.image_count,
        status="queued",
        source_generation_id=source.id,
        persona_id=source.persona_id,
        persona_meta=dict(source.persona_meta or {}),
        lora_version_id=source.lora_version_id,
        comfy_meta={**meta, "test_group": None, "test_label": None},
        tasks=[ProviderTask(index=0, images_requested=count or source.image_count)],
    )
    db.add(gen)
    await db.commit()
    worker.enqueue(gen.id)
    return await db.scalar(
        select(Generation)
        .where(Generation.id == gen.id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
        .execution_options(populate_existing=True)
    )


# --- Enhance photo (PGUpscaler) -------------------------------------------------------------


EnhanceInput = SourceImage


class EnhanceBody(BaseModel):
    connection_id: uuid.UUID
    image: EnhanceInput
    strength: float = Field(default=enhance.DEFAULT_STRENGTH)
    face_only: bool = False
    upscale: float = Field(default=enhance.DEFAULT_UPSCALE)
    seed: int | None = Field(default=None, ge=0, le=krea2.SEED_MAX)
    # In English, like the template's prompts: what the skin fix should make (empty = the
    # template's skin description) and what to fix with face only (empty = "face").
    prompt: str = Field(default="", max_length=enhance.MAX_PROMPT)
    face_target: str = Field(default=enhance.FACE_TARGET, max_length=enhance.MAX_FACE_TARGET)
    client_request_id: str | None = Field(default=None, max_length=64)


async def source_input(db: AsyncSession, user: User, ref: SourceImage) -> dict[str, Any]:
    """The user's own photo (never another user's): its file and a thumbnail URL."""
    if ref.kind == "reference":
        row = await db.scalar(
            select(ReferenceImage).where(
                ReferenceImage.id == ref.id, ReferenceImage.user_id == user.id
            )
        )
        base = f"/api/reference-images/{ref.id}"
    else:
        row = await db.scalar(select(Image).where(Image.id == ref.id, Image.user_id == user.id))
        base = f"/api/images/{ref.id}"
    if row is None:
        raise GenerationRejected("not_found", "That photo no longer exists.", 404)
    return {
        "kind": ref.kind,
        "id": str(ref.id),
        "file_path": row.file_path,
        "mime_type": row.mime_type,
        "width": row.width,
        "height": row.height,
        "thumb_url": f"{base}/thumb",
        "url": f"{base}/file",
        # A generated image may be marked 18+; the result is then marked too.
        "nsfw": getattr(row, "nsfw", None),
    }


async def create_enhance_generation(db: AsyncSession, user: User, body: EnhanceBody) -> Generation:
    """Enhance photo on a ComfyUI connection: fixes plastic AI skin and upscales (see
    comfy/enhance.py). An ordinary Generation row, so Library and History show it."""
    if body.client_request_id:
        existing = await db.scalar(
            select(Generation)
            .where(
                Generation.user_id == user.id,
                Generation.client_request_id == body.client_request_id,
            )
            .options(selectinload(Generation.tasks), selectinload(Generation.images))
        )
        if existing is not None:
            return existing
    if body.strength not in enhance.STRENGTHS:
        raise GenerationRejected(
            "invalid_parameter",
            "Strength is one of " + ", ".join(f"{s:g}" for s in enhance.STRENGTHS) + ".",
        )
    prompt = body.prompt.strip() or enhance.SKIN_PROMPT
    face_target = " ".join(body.face_target.split()) or enhance.FACE_TARGET
    try:
        for text in (prompt, face_target):
            check_prompt(text, allows_mature=True)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc
    if body.upscale not in enhance.UPSCALES:
        raise GenerationRejected("invalid_parameter", "Upscale is 1.5, 2 or 3.")
    conn: ComfyConnection | None = await owned(db, user.id, body.connection_id)
    if conn is None:
        raise GenerationRejected("not_found", "That ComfyUI connection no longer exists.", 404)
    if conn.status != "ready":
        raise GenerationRejected("not_connected", f"{conn.name} is not ready yet.", 409)
    detection = (conn.info or {}).get("enhance") or {}
    if detection.get("missing_nodes"):
        raise GenerationRejected(
            "missing_nodes",
            f"{conn.name} is missing ComfyUI nodes: {', '.join(detection['missing_nodes'])}. "
            "Update ComfyUI.",
            409,
        )
    if not detection.get("ready"):
        raise GenerationRejected(
            "missing_models",
            f"The {enhance.NAME} models are not installed on {conn.name}: "
            + ", ".join(detection.get("missing_files") or []),
            409,
        )
    source = await source_input(db, user, body.image)
    seed = body.seed if body.seed is not None else secrets.randbelow(krea2.SEED_MAX)
    workflow = enhance.build_workflow(
        image=enhance.INPUT_PLACEHOLDER,
        strength=body.strength,
        face_only=body.face_only,
        upscale=body.upscale,
        seed=seed,
        prompt=prompt,
        face_target=face_target,
    )
    gen = Generation(
        user_id=user.id,
        client_request_id=body.client_request_id,
        model_key=enhance.MODEL_KEY,
        provider=PROVIDER,
        mode="image-edit",
        provider_model_id=enhance.NAME,
        # The prompt the skin fix ran with (cards, History and "load settings" show it).
        prompt=prompt,
        settings={
            "strength": body.strength,
            "face_only": body.face_only,
            "upscale": body.upscale,
            "seed": seed,
            "source_width": source["width"],
            "source_height": source["height"],
            **(
                {"face_target": face_target}
                if body.face_only and face_target != enhance.FACE_TARGET
                else {}
            ),
        },
        request_payload=workflow,
        reference_image_ids=[source["id"]] if body.image.kind == "reference" else [],
        image_count=1,
        status="queued",
        comfy_meta={
            "connection_id": str(conn.id),
            "connection_name": conn.name,
            "connection_kind": conn.kind,
            "family": enhance.FAMILY,
            "workflow": {"id": None, "name": f"{enhance.NAME} (built-in)"},
            "input": source,
            "adult": bool(source.get("nsfw")),
        },
        tasks=[ProviderTask(index=0, images_requested=1)],
    )
    db.add(gen)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise
    worker.enqueue(gen.id)
    loaded = await db.scalar(
        select(Generation)
        .where(Generation.id == gen.id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
        .execution_options(populate_existing=True)
    )
    assert loaded is not None
    return loaded
