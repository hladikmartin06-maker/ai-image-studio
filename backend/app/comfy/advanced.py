"""Generate › Advanced, image to image: the user's photo is edited from a plain instruction by
Krea 2 Turbo with an in-context edit LoRA (comfy/k2edit.py), optionally with the owner's own
Krea 2 LoRA for the character. An ordinary Generation row (provider "comfyui"), queued on
the connection like Krea 2 and Enhance photo, so Library, History and the before/after view
work unchanged. Text to image in Advanced is the Krea 2 Turbo run (comfy/generation.py)."""

from __future__ import annotations

import re
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
    # "auto": text to image at 2K (real skin texture, a closer likeness than 1K; ~20 s
    # instead of 6 s on an L40S); an edit at 1K (at 2K a character edit restaged a selfie:
    # another pose and framing; the edit LoRAs are trained at about 1 MP).
    resolution: str = Field(default="auto", pattern="^(auto|1K|2K)$")
    # With a character LoRA: its face redrawn close up when it is small (k2edit.face_*).
    face_refine: bool = True
    # With a character LoRA: other faces of her (a mirror reflection) are changed too, in a
    # second pass (k2edit.other_face_instruction).
    all_faces: bool = True
    # Edit only the face: the face is cropped, edited at 1024 px and put back; the rest of
    # the photo stays exactly as it was (expressions, makeup, a smile on a small face).
    face_only: bool = False
    # What the edit works on: "photo" (all of it), "face" (= face_only), "head" (glasses,
    # earrings, a hat: the head edited close up), "hair" (SAM finds it), "body" (body_part).
    area: str = Field(default="photo", pattern="^(photo|face|head|hair|clothes|body)$")
    # How far the edit may go: "strong" for a new pose, or hard expressions with Face only.
    strength: str = Field(default="normal", pattern="^(normal|strong)$")
    # Body: change the shape of one part only (k2edit.BODY_PARTS: bust, butt, waist, …).
    body_part: str | None = Field(default=None, max_length=20)
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


async def _hair(db: AsyncSession, user: User, lora: Any) -> str:
    """The hair of the AI model the LoRA belongs to (its appearance), e.g. "long straight
    black"; empty when not filled in."""
    from app.services import personas

    if not lora.persona_id:
        return ""
    persona = await personas.owned(db, user.id, lora.persona_id)
    look = (persona.appearance or {}) if persona else {}
    return " ".join(str(look.get(k) or "").strip() for k in ("hairstyle", "hair_color")).strip()


def _size_of(data: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    return Image.open(io.BytesIO(data)).size


async def _other_face(source: dict[str, Any], who: str, note: str) -> str | None:
    """The second-pass instruction when the photo shows another face (her reflection)."""
    import asyncio

    from app.services import faces, storage

    try:
        data = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        upright = await asyncio.to_thread(k2edit.upright, data)
        others = await faces.other_faces(upright)
        main = await faces.face_box(upright)
        size = await asyncio.to_thread(
            lambda: __import__("PIL.Image").Image.open(__import__("io").BytesIO(upright)).size
        )
    except Exception:  # noqa: BLE001 - no face models: no second pass
        return None
    if not others or not main:
        return None
    x, y, w, h = main
    main_side = k2edit.place((x + w / 2) / size[0], (y + h / 2) / size[1])
    side = k2edit.place(*others[0])
    if side == main_side:
        return None
    return k2edit.other_face_instruction(who, side, main_side, note)


async def _face_square(source: dict[str, Any], area: str = "face") -> tuple[int, int, int] | None:
    """Where the face (or the head, area "head") is in the user's photo (YuNet on this Mac),
    as the square to edit."""
    import asyncio
    import io

    from PIL import Image

    from app.services import faces, storage

    try:
        data = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        upright = await asyncio.to_thread(k2edit.upright, data)
        size = Image.open(io.BytesIO(upright)).size
        box = await faces.face_box(upright)
    except Exception as exc:  # noqa: BLE001 - face models missing or an unreadable photo
        raise GenerationRejected(
            "face_check_failed", "The face could not be found on this Mac right now.", 409
        ) from exc
    if area == "head" and box:
        return k2edit.head_square(box, *size)  # None for a big face: the whole photo
    square = k2edit.face_square(box, *size, any_size=True) if box else None
    if square is None:
        raise GenerationRejected(
            "no_face", "No face was found in this photo (or it is too small).", 422
        )
    return square


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
    area = "face" if body.face_only else "body" if body.body_part else body.area
    if area == "body" and not body.body_part:
        raise GenerationRejected("invalid_parameter", "Choose the body part to change.")
    if body.body_part is not None and body.body_part not in k2edit.BODY_PARTS:
        raise GenerationRejected("invalid_parameter", "Unknown body part.")
    if body.body_part and body.face_only:
        raise GenerationRejected("invalid_parameter", "Choose either Face only or Body.")
    if area != "photo" and not detection.get("sam"):
        raise GenerationRejected(
            "missing_models",
            "Face, head, hair and body edits need SAM 3.1: install the Advanced models again "
            "(one click).",
            409,
        )
    source = await source_input(db, user, body.image) if body.image else None
    square = None
    if source and area in ("face", "head"):
        square = await _face_square(source, area)
        if square is None:  # head: a big face is edited on the whole photo
            area = "photo"

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
            final = k2edit.with_character(
                prompt, lora.trigger_word, lora.class_word, await _hair(db, user, lora)
            )
        elif not k2edit.names(prompt, lora.trigger_word):
            final = f"{lora.trigger_word}, {prompt}"
        if body.face_refine:
            face = {
                "lora": lora_filename(lora, version),
                "strength": body.lora.strength,
                "who": f"{lora.trigger_word} {lora.class_word}".strip(),
                # Edits only: the instruction, so the expression it asked for stays.
                "note": prompt if source else "",
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
    tier = body.resolution
    if tier == "auto":
        tier = "1K" if source else "2K"
    count = 1 if source else body.count
    if square:
        width = height = k2edit.FACE_PX
        face = None  # the face is already edited close up
        note = k2edit.FACE_ONLY_NOTE if area == "face" else k2edit.HEAD_NOTE
        final = f"{final.rstrip()} {note}"
    elif source:
        width, height = krea2.size_for_source(source["width"], source["height"], tier)
    extra = None
    if source and loras and area == "photo" and body.all_faces:
        who = face["who"] if face else f"{lora_meta['trigger_word']} woman"
        look = await _hair(db, user, lora)
        note = f"{prompt} Give her {look} hair." if look else prompt
        extra = await _other_face(source, who, note)
    # Her character on the whole photo: her body (pose) is held, her hair and face are free
    # (k2edit.HOLD_BOOST; PROVIDERS.md) — not when a new pose is asked for ("Strong", or the
    # words): held, "sit with her legs crossed" left her as she sat.
    hold = bool(
        source
        and loras
        and area == "photo"
        and detection.get("sam")
        and body.strength != "strong"
        and not k2edit.asks_for_pose(prompt)
    )
    fallback = None
    if area == "body":
        final = f"{final.rstrip()} {k2edit.KEEP_CLOTHES}"
    if area == "clothes" or (area == "body" and body.strength != "strong") or hold:
        final = f"{final.rstrip()} {k2edit.KEEP_POSE}"
    if source:
        edit_args: dict[str, Any] = dict(
            image=enhance.INPUT_PLACEHOLDER,
            prompt=final,
            width=width,
            height=height,
            seed=seed,
            edit_lora=k2edit.DEFAULT_EDIT,
            loras=[(f, s * k2edit.BODY_LORA) for f, s in loras] if area == "body" else loras,
            region=(
                body.body_part if area == "body" else area if area in ("hair", "clothes") else None
            ),
            keep_hands=bool(square),
            hold_body=hold,
            # Clothes: everything but the clothes held (the pose stays).
            hold_rest=area == "clothes",
            **k2edit.settings_for(
                character=bool(loras), strength=body.strength, face=area == "face"
            ),
        )
        workflow = k2edit.build_edit(**edit_args)
        if area == "body" and body.strength != "strong":
            # Body: the part let go (a visible change); if the person moves (her face is
            # elsewhere: a woman lying on a lounger stood up), the run is redone with
            # everything but the part held (k2edit.moved, runner._advanced_finish).
            fallback = k2edit.build_edit(**{**edit_args, "hold_rest": True})
    else:
        width, height = krea2.size_for(body.aspect_ratio, tier)
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
            change=body.strength,
            face_only=area == "face",
            area=area,
            **({"body_part": body.body_part} if body.body_part else {}),
            keep_unchanged=body.keep_unchanged,
            source_width=source["width"],
            source_height=source["height"],
        )
    else:
        settings.update(aspect_ratio=body.aspect_ratio)
    settings["resolution"] = tier if not square else "1K"
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
            # A second pass for another face of her (k2edit.other_face_instruction).
            "extra_pass": extra,
            "face_only": list(square) if square and area == "face" else None,
            "head_crop": list(square) if square and area == "head" else None,
            "pose_fallback": fallback,
            # Hair and Body (no character LoRA): the face comes back exactly (k2edit.hold_face).
            "hold_face": area in ("hair", "body") and not loras,
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
        "instruction for an instruction-based image editor, at most 60 words. Keep the "
        "user's intent exactly and add nothing they did not ask for. Be concrete: name the "
        "body parts, pose (what each arm, hand and leg does, how much of the person is "
        "visible), facial expression, clothing (the new outfit and what is gone), "
        "background, and where the light falls; for a body change say how much (e.g. "
        "'much larger') and say 'bust' rather than 'breasts'. Keep words such as trigger "
        "words verbatim. End with: Keep everything else unchanged. "
        'Answer with JSON only: {"instruction": "...", "target": "...", "part": "..."}. '
        'target is "face" when only the facial expression changes (eyes, gaze, mouth, a '
        'smile, a wink), "head" for things added on the head or face (glasses, sunglasses, '
        'a hat or cap, earrings, makeup, piercings), "hair" for the hair colour or style, '
        '"clothes" when only the clothes change, '
        '"body" when the shape or size of the body changes (bust, buttocks, hips, waist, '
        'weight, muscles), else "photo" (several things, pose, clothes, background, light). '
        'With "body", part is one of "figure" (the whole figure), "bust", "butt", "waist", '
        '"belly", "legs", "arms". '
        "If the request involves anyone under 18, answer exactly: REFUSED"
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


def parse_rewrite(text: str) -> tuple[str, str | None, str | None]:
    """The instruction and its target from the model's JSON answer; plain text (or broken
    JSON) is taken as the instruction alone."""
    import json

    raw = text.strip()
    found = re.search(r"\{.*\}", raw, re.DOTALL)
    if found:
        try:
            data = json.loads(found.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("instruction"), str):
            target = data.get("target")
            part = data.get("part")
            return (
                " ".join(data["instruction"].split()),
                target if target in ("photo", "face", "head", "hair", "clothes", "body") else None,
                part if target == "body" and part in k2edit.BODY_PARTS else None,
            )
    return " ".join(raw.split()).strip().strip('"'), None, None


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
    out, target, part = (
        parse_rewrite("".join(parts))
        if body.mode == "edit"
        else (" ".join("".join(parts).split()).strip().strip('"'), None, None)
    )
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
        # What the edit should work on: "photo", "face" (Face only) or "body" (body shape).
        "target": target,
        "body_part": part,
        "provider": provider,
        "model": str(model),
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }
