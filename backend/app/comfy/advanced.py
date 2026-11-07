"""Generate › Advanced, image to image: the user's photo is edited from a plain instruction by
Krea 2 Turbo with an in-context edit LoRA (comfy/k2edit.py), optionally with the owner's own
Krea 2 LoRA for the character. An ordinary Generation row (provider "comfyui"), queued on
the connection like Krea 2 and Enhance photo, so Library, History and the before/after view
work unchanged. Text to image in Advanced is the Krea 2 Turbo run (comfy/generation.py)."""

from __future__ import annotations

import secrets
import uuid
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.comfy import enhance, k2edit, krea2
from app.comfy.generation import (
    PROVIDER,
    LoraUse,
    SourceImage,
    resolve_lora,
    source_input,
)
from app.comfy.service import file_capability, lora_filename, owned
from app.models import ComfyConnection, Generation, ProviderTask, User
from app.services.generation import GenerationRejected
from app.services.safety import ADULT_OFF, PromptRefused, check_prompt, is_explicit
from app.services.worker import worker


class AdvancedBody(BaseModel):
    connection_id: uuid.UUID
    # The photo to edit; none = a new photo from the prompt (text to image).
    image: SourceImage | None = None
    # What to change in plain words ("make her smile", "change the background to a beach"),
    # or, without a photo, the photo to make.
    prompt: str = Field(min_length=1, max_length=k2edit.MAX_PROMPT)
    # Text to image only: the shape and how many.
    aspect_ratio: str = "3:4"
    count: int = Field(default=1, ge=1, le=4)
    # With a character LoRA: its face redrawn close up when it is small (k2edit.face_*).
    face_refine: bool = True
    # The owner's character LoRA (Krea 2, usable on Turbo).
    lora: LoraUse | None = None
    # Untouched parts keep the original's own pixels (k2edit.keep_unchanged).
    keep_unchanged: bool = True
    # 18+ content (adults only), as on Krea 2: off = explicit instructions are refused.
    adult: bool = False
    seed: int | None = Field(default=None, ge=0, le=krea2.SEED_MAX)
    client_request_id: str | None = Field(default=None, max_length=64)


async def _existing(db: AsyncSession, user: User, request_id: str | None) -> Generation | None:
    if not request_id:
        return None
    return await db.scalar(
        select(Generation)
        .where(Generation.user_id == user.id, Generation.client_request_id == request_id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
    )


async def create_advanced_generation(
    db: AsyncSession, user: User, body: AdvancedBody
) -> Generation:
    existing = await _existing(db, user, body.client_request_id)
    if existing is not None:
        return existing
    prompt = " ".join(body.prompt.split())
    try:
        check_prompt(prompt, allows_mature=True)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc
    if not body.adult and is_explicit(prompt):
        raise GenerationRejected("adult_off", ADULT_OFF, 422)
    conn: ComfyConnection | None = await owned(db, user.id, body.connection_id)
    if conn is None:
        raise GenerationRejected("not_found", "That ComfyUI connection no longer exists.", 404)
    if conn.status != "ready":
        raise GenerationRejected("not_connected", f"{conn.name} is not ready yet.", 409)
    detection = (conn.info or {}).get(k2edit.FAMILY) or {}
    if detection.get("missing_nodes"):
        raise GenerationRejected(
            "missing_nodes",
            f"{conn.name} is missing ComfyUI nodes: {', '.join(detection['missing_nodes'])}. "
            "Stop the GPU and start it again (the studio adds them when it starts).",
            409,
        )
    edit_ready = (detection.get("edit_loras") or {}).get(k2edit.DEFAULT_EDIT)
    if detection.get("missing_files") or not edit_ready:
        raise GenerationRejected(
            "missing_models", "Install the Advanced models on this GPU first (one click).", 409
        )
    if body.image is None and body.aspect_ratio not in krea2.ASPECT_RATIOS:
        raise GenerationRejected("invalid_parameter", "Unsupported aspect ratio.")
    source = await source_input(db, user, body.image) if body.image else None

    loras: list[tuple[str, float]] = []
    lora_meta = None
    lora_version_id = None
    final = prompt
    face = None
    if body.lora is not None:
        lora, version = await resolve_lora(db, user, body.lora, "turbo")
        if file_capability(conn) == "none" and str(conn.id) not in (version.deployments or {}):
            raise GenerationRejected(
                "missing_models",
                "This ComfyUI accepts no uploads; copy the LoRA into its models/loras folder as "
                f"{lora_filename(lora, version)}.",
                409,
            )
        loras.append((lora_filename(lora, version), body.lora.strength))
        if source:
            final = k2edit.with_character(prompt, lora.trigger_word, lora.class_word)
        elif not k2edit.names(prompt, lora.trigger_word):
            final = f"{lora.trigger_word}, {prompt}"
        if body.face_refine:
            face = {
                "lora": lora_filename(lora, version),
                "strength": body.lora.strength,
                "who": f"{lora.trigger_word} {lora.class_word}".strip(),
            }
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

    seed = body.seed if body.seed is not None else secrets.randbelow(krea2.SEED_MAX)
    count = 1 if source else body.count
    if source:
        width, height = krea2.size_for_source(source["width"], source["height"], "1K")
        workflow = k2edit.build_edit(
            image=enhance.INPUT_PLACEHOLDER,
            prompt=final,
            width=width,
            height=height,
            seed=seed,
            edit_lora=k2edit.DEFAULT_EDIT,
            loras=loras,
            **k2edit.settings_for(character=bool(loras)),
        )
    else:
        width, height = krea2.size_for(body.aspect_ratio, "1K")
        workflow = krea2.build_workflow(
            variant="turbo",
            unet=k2edit.TURBO.filename,
            text_encoder=krea2.TEXT_ENCODER.filename,
            vae=krea2.VAE.filename,
            prompt=final,
            negative_prompt="",
            width=width,
            height=height,
            seed=seed,
            steps=8,
            cfg=1.0,
            batch=count,
            loras=loras,
            prefix="ai-image-studio-advanced",
        )
    adult = body.adult or bool(source and source.get("nsfw"))
    settings: dict[str, Any] = {
        "seed": seed,
        "width": width,
        "height": height,
        **({"adult": True} if adult else {}),
        **({"face_refine": True} if face else {}),
    }
    if source:
        settings.update(
            keep_unchanged=body.keep_unchanged,
            source_width=source["width"],
            source_height=source["height"],
        )
    else:
        settings.update(aspect_ratio=body.aspect_ratio, resolution="1K")
    gen = Generation(
        user_id=user.id,
        client_request_id=body.client_request_id,
        model_key=k2edit.MODEL_KEY,
        provider=PROVIDER,
        mode="image-edit" if source else "text-to-image",
        provider_model_id=k2edit.TURBO.filename,
        prompt=prompt,
        final_prompt=final if final != prompt else None,
        settings=settings,
        request_payload=workflow,
        reference_image_ids=[source["id"]] if source and source["kind"] == "reference" else [],
        image_count=count,
        status="queued",
        lora_version_id=lora_version_id,
        comfy_meta={
            "connection_id": str(conn.id),
            "connection_name": conn.name,
            "connection_kind": conn.kind,
            "family": k2edit.FAMILY,
            "variant": "turbo",
            "model_file": k2edit.TURBO.filename,
            "workflow": {"id": None, "name": k2edit.NAME},
            "lora": lora_meta,
            "edit_lora": k2edit.EDIT_LORAS[k2edit.DEFAULT_EDIT].filename if source else None,
            "keep_unchanged": bool(source) and body.keep_unchanged,
            **({"input": source} if source else {}),
            "face": face,
            "adult": adult,
        },
        tasks=[ProviderTask(index=0, images_requested=count)],
    )
    db.add(gen)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = await _existing(db, user, body.client_request_id)
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


# --- the instruction in clear English (optional, the user's own AI provider) ---------------
#
# The edit model reads English: in the studio's tests a Slovak "smile and look at the camera"
# changed nothing, while the same in English worked (docs/PROVIDERS.md, Advanced). The user's
# words go to the AI provider connected in Integrations › AI assistant (cheap tier); the
# result is shown for checking before anything runs.
REWRITE_SYSTEM = {
    "edit": (
        "You rewrite a user's photo-editing request (any language) as one clear English "
        "instruction for an instruction-based image editor, at most 60 words. Output only "
        "the instruction. "
        "Keep the user's intent exactly and add nothing they did not ask for. Be concrete: "
        "name the body parts, pose (what each arm, hand and leg does, how much of the person "
        "is visible), facial expression, clothing (the new outfit and what is gone), "
        "background, and where the light falls. Keep words such as trigger words verbatim. "
        "End with: Keep everything else unchanged. If the request involves anyone under 18, "
        "answer exactly: REFUSED"
    ),
    "text": (
        "You rewrite a user's image idea (any language) as one detailed English prompt for a "
        "photorealistic text-to-image model, at most 80 words. Output only the prompt. Keep "
        "the user's intent "
        "and add nothing they did not ask for beyond photographic detail: subject, pose, "
        "expression, clothing, setting, light, camera and framing. Keep words such as trigger "
        "words verbatim. If the request involves anyone under 18, answer exactly: REFUSED"
    ),
}


class RewriteBody(BaseModel):
    text: str = Field(min_length=1, max_length=k2edit.MAX_PROMPT)
    mode: str = Field(default="edit", pattern="^(edit|text)$")


async def rewrite_prompt(db: AsyncSession, user: User, body: RewriteBody) -> dict[str, Any]:
    from app.ai.providers.base import AIError, ChatRequest, ChatTurn
    from app.ai.registry import AI_PROVIDERS, get_adapter, resolve_tier_models
    from app.models import AIIntegration, AIUsageEvent
    from app.security import decrypt_secret

    text = " ".join(body.text.split())
    try:
        check_prompt(text, allows_mature=True)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc
    rows = {
        r.provider: r
        for r in await db.scalars(select(AIIntegration).where(AIIntegration.user_id == user.id))
    }
    provider = next((p for p in AI_PROVIDERS if p in rows), None)
    if provider is None:
        raise GenerationRejected(
            "not_connected", "Connect an AI provider in Integrations › AI assistant first.", 409
        )
    integration = rows[provider]
    tiers = resolve_tier_models(
        AI_PROVIDERS[provider], integration.models or [], integration.tier_models or {}
    )
    model = tiers["low"]["model"] or tiers["balanced"]["model"]
    if not model:
        raise GenerationRejected(
            "model_not_found", "Choose a model in Integrations › AI assistant.", 409
        )
    api_key = (
        decrypt_secret(integration.encrypted_key, associated=f"{user.id}:ai:{provider}")
        if integration.encrypted_key
        else ""
    )
    request = ChatRequest(
        model=str(model),
        system=REWRITE_SYSTEM[body.mode],
        messages=[ChatTurn("user", text)],
        # Room for the model's own reasoning on effort-aware models; the answer is short.
        max_tokens=1500,
        effort="low",
    )
    parts: list[str] = []
    usage = None
    try:
        async for event in get_adapter(AI_PROVIDERS[provider]).stream(
            api_key, integration.base_url, request
        ):
            if event.done:
                usage = event.usage
            else:
                parts.append(event.text)
    except AIError as exc:
        raise GenerationRejected(exc.code, exc.message, 502) from exc
    out = " ".join("".join(parts).split()).strip().strip('"')
    if usage is not None:
        db.add(
            AIUsageEvent(
                user_id=user.id,
                provider=provider,
                model=str(model),
                tier="low",
                purpose="advanced",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                estimated=usage.estimated,
            )
        )
        await db.commit()
    if not out or out == "REFUSED":
        raise GenerationRejected("content_blocked", "This request cannot be used.", 422)
    try:
        check_prompt(out, allows_mature=True)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc
    return {
        "prompt": out[: k2edit.MAX_PROMPT],
        "provider": provider,
        "model": str(model),
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }
