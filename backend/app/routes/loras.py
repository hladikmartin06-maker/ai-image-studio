"""LoRA studio: LoRA projects, dataset, captions, training runs, versions, files, tests."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.comfy import krea2
from app.comfy.client import ComfyError
from app.comfy.generation import ComfyGenerateBody, LoraUse, create_comfy_generation
from app.comfy.service import owned as owned_connection
from app.comfy.service import place_lora
from app.db import get_db
from app.deps import api_error, current_user
from app.gpu.registry import GPU_PROVIDERS
from app.lora import aitoolkit, dataset, runner
from app.lora.captions import CaptionError, apply_template, describe
from app.lora.presets import ARCHITECTURES, TrainingConfig, catalog
from app.lora.safetensors_meta import (
    InvalidSafetensors,
    lora_architecture,
    lora_variant,
    read_header,
)
from app.lora.serial import ACTIVE, item_json, lora_json, version_json
from app.models import (
    Generation,
    GPUIntegration,
    Image,
    Lora,
    LoraDatasetItem,
    LoraVersion,
    Persona,
    ReferenceImage,
    User,
    UserSecret,
)
from app.routes.secrets import hf_token
from app.services import faces, huggingface, personas, references, storage, uploads
from app.services.generation import GenerationRejected
from app.services.safety import mentions_minors
from app.services.serializers import generation_json

router = APIRouter(prefix="/api/loras", tags=["loras"])

TRIGGER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")
MAX_DATASET = 300


# --- helpers -------------------------------------------------------------------------------


async def _owned(db: AsyncSession, user: User, lora_id: uuid.UUID) -> Lora:
    lora = await db.scalar(select(Lora).where(Lora.id == lora_id, Lora.user_id == user.id))
    if lora is None:
        raise api_error(404, "not_found", "LoRA not found.")
    return lora


async def _version(db: AsyncSession, user: User, lora: Lora, version_id: uuid.UUID) -> LoraVersion:
    v = await db.scalar(
        select(LoraVersion).where(
            LoraVersion.id == version_id,
            LoraVersion.lora_id == lora.id,
            LoraVersion.user_id == user.id,
        )
    )
    if v is None:
        raise api_error(404, "not_found", "Version not found.")
    return v


async def _items(db: AsyncSession, lora: Lora) -> list[LoraDatasetItem]:
    return list(
        (
            await db.scalars(
                select(LoraDatasetItem)
                .where(LoraDatasetItem.lora_id == lora.id, LoraDatasetItem.user_id == lora.user_id)
                .options(selectinload(LoraDatasetItem.reference))
                .order_by(LoraDatasetItem.position, LoraDatasetItem.created_at)
            )
        ).all()
    )


async def _versions(db: AsyncSession, lora: Lora) -> list[LoraVersion]:
    return list(
        (
            await db.scalars(
                select(LoraVersion)
                .where(LoraVersion.lora_id == lora.id, LoraVersion.user_id == lora.user_id)
                .order_by(LoraVersion.number.desc())
            )
        ).all()
    )


def _views(items: list[LoraDatasetItem]) -> list[dataset.ItemView]:
    return [
        dataset.ItemView(
            id=str(i.id),
            width=i.reference.width,
            height=i.reference.height,
            crop=i.crop,
            caption=i.caption,
            excluded=i.excluded,
            split=i.split,
            dhash=i.dhash,
        )
        for i in items
    ]


async def _persona_name(db: AsyncSession, lora: Lora) -> str | None:
    if not lora.persona_id:
        return None
    return await db.scalar(
        select(Persona.name).where(Persona.id == lora.persona_id, Persona.user_id == lora.user_id)
    )


async def _detail(db: AsyncSession, lora: Lora) -> dict[str, Any]:
    items = await _items(db, lora)
    versions = await _versions(db, lora)
    return {
        **lora_json(
            lora,
            versions,
            persona_name=await _persona_name(db, lora),
            dataset_count=len(items),
        ),
        "dataset": [item_json(i) for i in items],
        "analysis": dataset.analyze(_views(items), lora.trigger_word),
        "versions": [version_json(v) for v in versions],
    }


def _check_words(*texts: str) -> None:
    if any(mentions_minors(t) for t in texts if t):
        raise api_error(
            422, "content_blocked", "LoRAs are for fictional adults. Remove words about minors."
        )


# --- catalog ---------------------------------------------------------------------------------


@router.get("/catalog")
async def lora_catalog(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    connected = (
        await db.scalars(select(GPUIntegration.provider).where(GPUIntegration.user_id == user.id))
    ).all()
    hf = await db.scalar(
        select(UserSecret.id).where(UserSecret.user_id == user.id, UserSecret.name == "huggingface")
    )
    return {
        **catalog(),
        "gpu_providers": [
            {"key": k, "name": v["name"], "connected": k in connected}
            for k, v in GPU_PROVIDERS.items()
        ],
        "hf_token_set": hf is not None,
        "krea2": {"architecture": krea2.ARCHITECTURE},
    }


# --- LoRA projects ---------------------------------------------------------------------------


class LoraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    trigger_word: str = Field(min_length=2, max_length=40)
    class_word: str = Field(default="", max_length=40)
    architecture: Literal["krea2"] = "krea2"
    base_variant: Literal["raw", "turbo"] = "raw"
    persona_id: uuid.UUID | None = None
    caption_template: str = Field(default="{trigger} {class}, {caption}", max_length=300)
    adult_confirmed: bool = False


class LoraCopy(BaseModel):
    """A new LoRA project with another LoRA's photos (and captions), e.g. on another base."""

    name: str = Field(min_length=1, max_length=80)
    trigger_word: str = Field(min_length=2, max_length=40)
    class_word: str = Field(default="", max_length=40)
    base_variant: Literal["raw", "turbo"] = "raw"
    persona_id: uuid.UUID | None = None
    captions: bool = True
    adult_confirmed: bool = False


class LoraPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    class_word: str | None = Field(default=None, max_length=40)
    caption_template: str | None = Field(default=None, max_length=300)
    tags: list[str] | None = Field(default=None, max_length=20)
    notes: str | None = Field(default=None, max_length=4000)
    persona_id: uuid.UUID | None = None
    preferred_version_id: uuid.UUID | None = None
    base_variant: Literal["raw", "turbo"] | None = None


async def _check_new_lora(db: AsyncSession, user: User, body: LoraCreate | LoraCopy) -> str:
    """Checks a new LoRA's confirmation, words, trigger and AI model; returns the trigger."""
    if not body.adult_confirmed:
        raise api_error(
            400, "adult_confirmation_required", "Confirm that the character is a fictional adult."
        )
    trigger = body.trigger_word.strip()
    if not TRIGGER_RE.match(trigger):
        raise api_error(
            400, "invalid_trigger", "Trigger word: one word of letters, digits, - or _ (2–40)."
        )
    _check_words(body.name, body.class_word, trigger)
    exists = await db.scalar(
        select(Lora.id).where(
            Lora.user_id == user.id, func.lower(Lora.trigger_word) == trigger.lower()
        )
    )
    if exists:
        raise api_error(409, "trigger_taken", "Another LoRA already uses this trigger word.")
    if body.persona_id and await personas.owned(db, user.id, body.persona_id) is None:
        raise api_error(404, "not_found", "AI model not found.")
    return trigger


@router.get("")
async def list_loras(
    q: str | None = Query(default=None, max_length=120),
    usable: bool = False,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Lora).where(Lora.user_id == user.id)
    if q and q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(Lora.name.ilike(like) | Lora.trigger_word.ilike(like))
    loras = (await db.scalars(stmt.order_by(Lora.updated_at.desc()).limit(300))).all()
    ids = [lo.id for lo in loras]
    versions = (
        (await db.scalars(select(LoraVersion).where(LoraVersion.lora_id.in_(ids)))).all()
        if ids
        else []
    )
    counts = (
        dict(
            (
                await db.execute(
                    select(LoraDatasetItem.lora_id, func.count(LoraDatasetItem.id))
                    .where(LoraDatasetItem.lora_id.in_(ids))
                    .group_by(LoraDatasetItem.lora_id)
                )
            ).all()
        )
        if ids
        else {}
    )
    names = dict(
        (
            await db.execute(
                select(Persona.id, Persona.name).where(
                    Persona.user_id == user.id,
                    Persona.id.in_([lo.persona_id for lo in loras if lo.persona_id]),
                )
            )
        ).all()
    )
    items = []
    for lora in loras:
        own = [v for v in versions if v.lora_id == lora.id]
        data = lora_json(
            lora, own, persona_name=names.get(lora.persona_id), dataset_count=counts.get(lora.id, 0)
        )
        if usable and not data["usable_versions"]:
            continue
        items.append(data)
    return {"items": items, "total": len(items)}


@router.post("", status_code=201)
async def create_lora(
    body: LoraCreate, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    trigger = await _check_new_lora(db, user, body)
    lora = Lora(
        user_id=user.id,
        name=body.name.strip(),
        trigger_word=trigger,
        class_word=body.class_word.strip(),
        architecture=body.architecture,
        base_variant=body.base_variant,
        persona_id=body.persona_id,
        caption_template=body.caption_template,
        tags=[],
    )
    db.add(lora)
    await db.commit()
    return await _detail(db, lora)


@router.post("/{lora_id}/copy", status_code=201)
async def copy_lora(
    lora_id: uuid.UUID,
    body: LoraCopy,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """New LoRA project with this one's dataset: the same photos (shared, not duplicated),
    held-out and excluded marks, crops and, if asked, captions with the new trigger word.
    Versions are not copied: the new project is trained on its own base model."""
    source = await _owned(db, user, lora_id)
    if source.architecture not in ARCHITECTURES:
        raise api_error(409, "not_trainable", "This LoRA's model family cannot be trained here.")
    items = await _items(db, source)
    if not items:
        raise api_error(409, "empty_dataset", "This LoRA has no photos to copy.")
    trigger = await _check_new_lora(db, user, body)
    lora = Lora(
        user_id=user.id,
        name=body.name.strip(),
        trigger_word=trigger,
        class_word=body.class_word.strip(),
        architecture=source.architecture,
        base_variant=body.base_variant,
        persona_id=body.persona_id,
        caption_template=source.caption_template,
        tags=list(source.tags or []),
    )
    db.add(lora)
    await db.flush()
    old = re.compile(rf"(?<![\w]){re.escape(source.trigger_word)}(?![\w])", re.IGNORECASE)
    for position, item in enumerate(items):
        caption = old.sub(lambda _m: trigger, item.caption) if body.captions else ""
        db.add(
            LoraDatasetItem(
                lora_id=lora.id,
                user_id=user.id,
                reference_image_id=item.reference_image_id,
                caption=caption,
                caption_source=item.caption_source if caption else "empty",
                split=item.split,
                excluded=item.excluded,
                crop=dict(item.crop) if item.crop else None,
                dhash=item.dhash,
                position=position,
            )
        )
    await db.commit()
    return await _detail(db, lora)


@router.get("/{lora_id}")
async def get_lora(
    lora_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return await _detail(db, await _owned(db, user, lora_id))


@router.patch("/{lora_id}")
async def update_lora(
    lora_id: uuid.UUID,
    body: LoraPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    _check_words(body.name or "", body.class_word or "", body.notes or "")
    if body.name:
        lora.name = body.name.strip()
    if body.class_word is not None:
        lora.class_word = body.class_word.strip()
    if body.caption_template is not None:
        lora.caption_template = body.caption_template
    if body.tags is not None:
        lora.tags = sorted({t.strip()[:30] for t in body.tags if t.strip()})
    if body.notes is not None:
        lora.notes = body.notes
    if "persona_id" in body.model_fields_set:
        if body.persona_id and await personas.owned(db, user.id, body.persona_id) is None:
            raise api_error(404, "not_found", "AI model not found.")
        lora.persona_id = body.persona_id
    if body.base_variant and lora.source == "trained":
        trained = await db.scalar(
            select(func.count(LoraVersion.id)).where(LoraVersion.lora_id == lora.id)
        )
        if trained:
            raise api_error(
                409, "has_versions", "The base model is fixed once a version was trained."
            )
        lora.base_variant = body.base_variant
    if "preferred_version_id" in body.model_fields_set:
        if body.preferred_version_id:
            v = await _version(db, user, lora, body.preferred_version_id)
            if v.status != "completed":
                raise api_error(409, "not_ready", "Only a finished version can be preferred.")
        lora.preferred_version_id = body.preferred_version_id
    lora.updated_at = datetime.now(UTC)
    await db.commit()
    return await _detail(db, lora)


@router.delete("/{lora_id}")
async def delete_lora(
    lora_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Deletes the LoRA, its versions (files, checkpoints, samples) and dataset links; the
    dataset images themselves are kept if an AI model or another LoRA still uses them."""
    lora = await _owned(db, user, lora_id)
    versions = await _versions(db, lora)
    if any(v.status in ACTIVE for v in versions):
        raise api_error(409, "in_progress", "Cancel the running training first.")
    items = await _items(db, lora)
    ref_ids = [i.reference_image_id for i in items]
    await db.delete(lora)
    await db.flush()
    paths = await references.delete_unused(db, user.id, ref_ids)
    await db.commit()
    storage.delete_files(*paths)
    folder = storage.root() / str(user.id) / "loras" / str(lora_id)
    await asyncio.to_thread(shutil.rmtree, folder, True)
    return {"ok": True}


# --- dataset ---------------------------------------------------------------------------------


async def _add_item(db: AsyncSession, lora: Lora, ref: ReferenceImage) -> bool:
    exists = await db.scalar(
        select(LoraDatasetItem.id).where(
            LoraDatasetItem.lora_id == lora.id, LoraDatasetItem.reference_image_id == ref.id
        )
    )
    if exists:
        return False
    count = await db.scalar(
        select(func.count(LoraDatasetItem.id)).where(LoraDatasetItem.lora_id == lora.id)
    )
    if (count or 0) >= MAX_DATASET:
        raise api_error(409, "limit_reached", f"A dataset holds up to {MAX_DATASET} images.")
    data = await asyncio.to_thread(storage.absolute(ref.thumb_path).read_bytes)
    db.add(
        LoraDatasetItem(
            lora_id=lora.id,
            user_id=lora.user_id,
            reference_image_id=ref.id,
            dhash=await asyncio.to_thread(dataset.dhash, data),
            position=(count or 0),
        )
    )
    await db.flush()
    return True


@router.post("/{lora_id}/dataset/upload", status_code=201)
async def upload_dataset_image(
    lora_id: uuid.UUID,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    data = await file.read(references.max_bytes() + 1)
    try:
        ref, _ = await references.store(db, user.id, data, file.filename or "")
    except references.ReferenceRejected as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
    added = await _add_item(db, lora, ref)
    await db.commit()
    return {"added": int(added), "lora": await _detail(db, lora)}


class AddDatasetBody(BaseModel):
    library_image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=200)
    reference_image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=200)
    persona_id: uuid.UUID | None = None


@router.post("/{lora_id}/dataset/add")
async def add_dataset_images(
    lora_id: uuid.UUID,
    body: AddDatasetBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """From the Image Library (copied), from uploads, or every image of an AI model."""
    lora = await _owned(db, user, lora_id)
    added = 0
    refs: list[ReferenceImage] = []
    if body.persona_id:
        persona = await personas.owned(db, user.id, body.persona_id)
        if persona is None:
            raise api_error(404, "not_found", "AI model not found.")
        refs += personas.ordered_references(persona)
    if body.reference_image_ids:
        refs += (
            await db.scalars(
                select(ReferenceImage).where(
                    ReferenceImage.user_id == user.id,
                    ReferenceImage.id.in_(body.reference_image_ids),
                )
            )
        ).all()
    try:
        for image in (
            (
                await db.scalars(
                    select(Image).where(
                        Image.user_id == user.id, Image.id.in_(body.library_image_ids)
                    )
                )
            ).all()
            if body.library_image_ids
            else []
        ):
            ext = image.file_path.rsplit(".", 1)[-1]
            ref, _ = await references.from_library_file(
                db, user.id, image.file_path, f"library-{image.id}.{ext}"
            )
            refs.append(ref)
    except references.ReferenceRejected as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
    for ref in refs:
        added += int(await _add_item(db, lora, ref))
    await db.commit()
    return {"added": added, "lora": await _detail(db, lora)}


class ItemPatch(BaseModel):
    caption: str | None = Field(default=None, max_length=2000)
    split: Literal["train", "validation"] | None = None
    excluded: bool | None = None
    crop: dict[str, float] | None = None


def _check_crop(crop: dict[str, float]) -> dict[str, float]:
    try:
        x, y, w, h = (float(crop[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise api_error(400, "invalid_parameter", "Crop needs x, y, w and h.") from exc
    if not (0 <= x < 1 and 0 <= y < 1 and 0.05 <= w <= 1 and 0.05 <= h <= 1):
        raise api_error(400, "invalid_parameter", "Crop is outside the image.")
    if x + w > 1.0001 or y + h > 1.0001:
        raise api_error(400, "invalid_parameter", "Crop is outside the image.")
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "w": round(min(w, 1 - x), 4),
        "h": round(min(h, 1 - y), 4),
    }


async def _item(db: AsyncSession, lora: Lora, item_id: uuid.UUID) -> LoraDatasetItem:
    item = await db.scalar(
        select(LoraDatasetItem)
        .where(LoraDatasetItem.id == item_id, LoraDatasetItem.lora_id == lora.id)
        .options(selectinload(LoraDatasetItem.reference))
    )
    if item is None:
        raise api_error(404, "not_found", "Dataset image not found.")
    return item


@router.patch("/{lora_id}/dataset/{item_id}")
async def update_item(
    lora_id: uuid.UUID,
    item_id: uuid.UUID,
    body: ItemPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    item = await _item(db, lora, item_id)
    if body.caption is not None:
        _check_words(body.caption)
        item.caption = " ".join(body.caption.split())
        item.caption_source = "manual" if item.caption else "empty"
    if body.split:
        item.split = body.split
    if body.excluded is not None:
        item.excluded = body.excluded
    if "crop" in body.model_fields_set:
        item.crop = _check_crop(body.crop) if body.crop else None
    await db.commit()
    return item_json(item)


class BulkBody(BaseModel):
    item_ids: list[uuid.UUID] = Field(min_length=1, max_length=MAX_DATASET)
    action: Literal[
        "include", "exclude", "train", "validation", "clear_captions", "add_trigger", "remove"
    ]


@router.post("/{lora_id}/dataset/bulk")
async def bulk_items(
    lora_id: uuid.UUID,
    body: BulkBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    items = [i for i in await _items(db, lora) if i.id in set(body.item_ids)]
    removed_refs: list[uuid.UUID] = []
    pattern = re.compile(rf"(?<![\w]){re.escape(lora.trigger_word)}(?![\w])", re.IGNORECASE)
    for item in items:
        if body.action == "include":
            item.excluded = False
        elif body.action == "exclude":
            item.excluded = True
        elif body.action in ("train", "validation"):
            item.split = body.action
        elif body.action == "clear_captions":
            item.caption = ""
            item.caption_source = "empty"
        elif body.action == "add_trigger" and item.caption and not pattern.search(item.caption):
            item.caption = f"{lora.trigger_word}, {item.caption}"
        elif body.action == "remove":
            removed_refs.append(item.reference_image_id)
            await db.delete(item)
    await db.flush()
    paths = await references.delete_unused(db, user.id, removed_refs)
    await db.commit()
    storage.delete_files(*paths)
    return await _detail(db, lora)


FACE_REFERENCES = 24


@router.post("/{lora_id}/dataset/faces")
async def dataset_faces(
    lora_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Is it the same face on every photo? For each photo in use: a face found or not, and
    its median similarity to the other photos' faces (runs on this Mac)."""
    lora = await _owned(db, user, lora_id)
    items = [i for i in await _items(db, lora) if not i.excluded]
    if not items:
        raise api_error(409, "no_photos", "This LoRA has no photos to check.")
    try:
        result = await faces.consistency([storage.absolute(i.reference.file_path) for i in items])
    except faces.FaceError as exc:
        raise api_error(502, "face_model", exc.message) from exc
    return {
        "threshold": faces.MATCH_THRESHOLD,
        "items": [{"item_id": str(i.id), **r} for i, r in zip(items, result, strict=True)],
    }


class CaptionBody(BaseModel):
    tier: Literal["low", "balanced", "high"] = "low"
    overwrite: bool = False


@router.post("/{lora_id}/dataset/{item_id}/caption")
async def caption_item(
    lora_id: uuid.UUID,
    item_id: uuid.UUID,
    body: CaptionBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Automatic caption with the user's AI provider (one paid request per image)."""
    lora = await _owned(db, user, lora_id)
    item = await _item(db, lora, item_id)
    if item.caption and not body.overwrite:
        return item_json(item)
    data = await asyncio.to_thread(storage.absolute(item.reference.file_path).read_bytes)
    try:
        text, provider, model = await describe(db, user, data, body.tier)
    except CaptionError as exc:
        status = 409 if exc.code == "not_connected" else 502
        raise api_error(status, exc.code, exc.message) from exc
    item.caption = apply_template(lora.caption_template, text, lora.trigger_word, lora.class_word)
    item.caption_source = "ai"
    await db.commit()
    return {**item_json(item), "captioned_by": {"provider": provider, "model": model}}


# --- training --------------------------------------------------------------------------------


class TrainBody(BaseModel):
    preset: Literal["character_identity", "photorealistic", "fast_experiment", "high_quality"]
    config: TrainingConfig
    provider: Literal["runpod", "modal"]
    gpu_type: str = Field(min_length=1, max_length=80)
    gpu_count: int = Field(default=1, ge=1, le=8)
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    image: str | None = Field(default=None, max_length=250)
    auto_cleanup: bool = True
    notes: str = Field(default="", max_length=2000)
    client_request_id: str | None = Field(default=None, max_length=64)
    confirm_billing: bool = False
    # The chosen GPU's memory as the provider lists it (GPU picker); decides full precision.
    gpu_memory_gb: int | None = Field(default=None, ge=1, le=4096)


def _package(lora: Lora, items: list[LoraDatasetItem], folder_rel: str) -> dict[str, Any]:
    """Writes the training copy (crop applied) + captions; returns the manifest."""
    files: list[tuple[str, bytes, str]] = []
    manifest: list[dict[str, Any]] = []
    for index, item in enumerate(i for i in items if not i.excluded and i.split == "train"):
        data, ext = dataset.pack_image(
            item.reference.file_path, item.reference.mime_type, item.crop
        )
        caption = item.caption.strip() or f"{lora.trigger_word} {lora.class_word}".strip()
        stem = f"{index + 1:04d}.{ext}"
        files.append((stem, data, caption))
        manifest.append(
            {
                "item_id": str(item.id),
                "reference_id": str(item.reference_image_id),
                "file": stem,
                "caption": caption,
                "crop": item.crop,
            }
        )
    dataset.write_pack(storage.absolute(folder_rel), files)
    return {"items": manifest, "source": "trained"}


@router.post("/{lora_id}/versions", status_code=201)
async def start_training(
    lora_id: uuid.UUID,
    body: TrainBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    if body.client_request_id:
        existing = await db.scalar(
            select(LoraVersion).where(
                LoraVersion.user_id == user.id,
                LoraVersion.client_request_id == body.client_request_id,
            )
        )
        if existing is not None:
            return version_json(existing)
    if not body.confirm_billing:
        raise api_error(400, "confirmation_required", "Confirm that the provider will bill you.")
    if lora.source != "trained":
        raise api_error(409, "imported", "Imported LoRAs cannot be trained.")
    if body.config.base_variant != lora.base_variant:
        raise api_error(409, "variant_mismatch", "The base model of this LoRA is fixed.")
    if body.image and not re.fullmatch(r"[a-z0-9][a-z0-9._/:@-]{1,250}", body.image, re.I):
        raise api_error(400, "invalid_parameter", "Enter a container image such as repo/name:tag.")
    connected = await db.scalar(
        select(GPUIntegration.id).where(
            GPUIntegration.user_id == user.id, GPUIntegration.provider == body.provider
        )
    )
    if connected is None:
        raise api_error(409, "not_connected", "Connect this GPU provider in Integrations first.")
    variant = ARCHITECTURES[lora.architecture]["variants"][lora.base_variant]
    if variant["gated"]:
        token = await hf_token(db, user)
        if token is None:
            raise api_error(
                409,
                "hf_token_required",
                "Add your Hugging Face token in Integrations › Hugging Face first.",
            )
        # Checked before any GPU is rented; when Hugging Face cannot be reached the run goes
        # ahead and the trainer reports a download error itself.
        access, reason = await huggingface.repo_access(token, variant["name_or_path"], refresh=True)
        if access == "needs_license":
            raise api_error(
                409,
                "hf_license_required",
                f"Accept the Krea 2 license once on https://huggingface.co/"
                f"{variant['name_or_path']} (button “Agree and access repository”), "
                "then start again." + (f" Hugging Face says: {reason}" if reason else ""),
            )
        if access == "invalid_token":
            raise api_error(
                409,
                "hf_token_invalid",
                "Hugging Face rejected your token. Add a new one in Integrations › Hugging Face.",
            )
    if any(v.status in ACTIVE for v in await _versions(db, lora)):
        raise api_error(409, "in_progress", "This LoRA is already training.")
    items = await _items(db, lora)
    analysis = dataset.analyze(_views(items), lora.trigger_word)
    if not analysis["ready"]:
        raise api_error(
            409,
            "dataset_not_ready",
            f"The dataset needs at least {dataset.MIN_IMAGES} usable images of at least "
            f"{dataset.MIN_EDGE_HARD} px.",
        )
    _check_words(*body.config.validation_prompts)
    full_gb = ARCHITECTURES[lora.architecture].get("full_precision_gb")
    if full_gb and body.gpu_memory_gb and body.gpu_memory_gb >= full_gb:
        # A big GPU is used fully: the model in full precision, no memory saving. If it runs
        # out of memory anyway, runner.memory_fallback turns savings on one at a time.
        body.config.quantize = False
        body.config.low_vram = False
    start: LoraVersion | None = None
    if body.config.continue_from:
        # Continue an earlier version of this LoRA: its file is the starting point, and its
        # size (rank/alpha) is kept so the weights fit.
        start = await db.scalar(
            select(LoraVersion).where(
                LoraVersion.id == uuid.UUID(body.config.continue_from),
                LoraVersion.lora_id == lora.id,
                LoraVersion.user_id == user.id,
            )
        )
        if start is None:
            raise api_error(404, "not_found", "That version of this LoRA no longer exists.")
        if start.status != "completed" or not start.file_path:
            raise api_error(409, "not_ready", f"v{start.number} has no finished LoRA file.")
        if not (start.config or {}).get("rank") or not (start.config or {}).get("alpha"):
            raise api_error(409, "unknown_size", f"The size of v{start.number} is unknown.")
        body.config.rank = int(start.config["rank"])
        body.config.alpha = int(start.config["alpha"])
    number = (
        await db.scalar(select(func.max(LoraVersion.number)).where(LoraVersion.lora_id == lora.id))
        or 0
    ) + 1
    version_id = uuid.uuid4()
    folder = f"{user.id}/loras/{lora.id}/{version_id}"
    manifest = await asyncio.to_thread(_package, lora, items, f"{folder}/dataset")
    run = aitoolkit.run_name(str(lora.id), number)
    paths = aitoolkit.run_paths(run)
    prompts = body.config.validation_prompts or aitoolkit.default_sample_prompts(
        lora.trigger_word, lora.class_word
    )
    trainer_config = aitoolkit.build(
        dataset_dir=paths["dataset_abs"],
        output_dir=paths["output_abs"],
        architecture=lora.architecture,
        name=run,
        trigger_word=lora.trigger_word,
        class_word=lora.class_word,
        config=body.config,
        sample_prompts=prompts,
        start_lora=paths["start_abs"] if start else None,
    )
    config = body.config.model_dump()
    config.update({"image": body.image, "cloud_type": body.cloud_type})
    if start:
        config["continue_from_number"] = start.number
    version = LoraVersion(
        id=version_id,
        lora_id=lora.id,
        user_id=user.id,
        number=number,
        client_request_id=body.client_request_id,
        preset=body.preset,
        config=config,
        trainer_config=trainer_config,
        status="queued",
        stage_detail="Queued",
        gpu_provider=body.provider,
        gpu_type=body.gpu_type,
        gpu_count=body.gpu_count,
        auto_cleanup=body.auto_cleanup,
        progress={"total": body.config.steps},
        dataset_manifest=manifest,
        notes=body.notes,
    )
    db.add(version)
    lora.updated_at = datetime.now(UTC)
    await db.commit()
    return version_json(version)


class CancelBody(BaseModel):
    stop_gpu: bool = True


@router.post("/{lora_id}/versions/{version_id}/cancel")
async def cancel_training(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    body: CancelBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    await runner.cancel(v.id, body.stop_gpu)
    await db.refresh(v)
    return version_json(v)


class VersionPatch(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)


@router.patch("/{lora_id}/versions/{version_id}")
async def update_version(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    body: VersionPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    if body.notes is not None:
        v.notes = body.notes
    await db.commit()
    return version_json(v)


@router.delete("/{lora_id}/versions/{version_id}")
async def delete_version(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    if v.status in ACTIVE:
        raise api_error(409, "in_progress", "Cancel the training first.")
    dependants = (
        await db.scalars(
            select(LoraVersion.number).where(
                LoraVersion.lora_id == lora.id,
                LoraVersion.dataset_manifest["shares_file_of"].astext == str(v.id),
            )
        )
    ).all()
    if dependants:
        raise api_error(
            409,
            "in_use",
            "Versions made from its checkpoints use its files: delete "
            + ", ".join(f"v{n}" for n in dependants)
            + " first.",
        )
    if lora.preferred_version_id == v.id:
        lora.preferred_version_id = None
    shared = (v.dataset_manifest or {}).get("shares_file_of")
    folder = storage.root() / f"{user.id}/loras/{lora.id}/{v.id}"
    await db.delete(v)
    await db.commit()
    if not shared:
        await asyncio.to_thread(shutil.rmtree, folder, True)
    return {"ok": True}


@router.post("/{lora_id}/versions/{version_id}/checkpoints/{step}/as-version", status_code=201)
async def checkpoint_as_version(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    step: int,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """An earlier checkpoint as its own version (e.g. before it started to overtrain)."""
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    checkpoint = next((c for c in v.checkpoints or [] if c["step"] == step), None)
    if checkpoint is None:
        raise api_error(404, "not_found", "Checkpoint not found.")
    number = (
        await db.scalar(select(func.max(LoraVersion.number)).where(LoraVersion.lora_id == lora.id))
        or 0
    ) + 1
    sha = await asyncio.to_thread(_sha256, checkpoint["file_path"])
    copy = LoraVersion(
        lora_id=lora.id,
        user_id=user.id,
        number=number,
        preset=v.preset,
        config=v.config,
        trainer_config=v.trainer_config,
        status="completed",
        gpu_provider=v.gpu_provider,
        gpu_type=v.gpu_type,
        gpu_count=v.gpu_count,
        progress={
            "step": step,
            "total": step,
            "loss": [p for p in (v.progress or {}).get("loss", []) if p[0] <= step],
        },
        file_path=checkpoint["file_path"],
        file_size=checkpoint.get("size"),
        file_sha256=sha,
        checkpoints=[],
        dataset_manifest={
            **(v.dataset_manifest or {}),
            "shares_file_of": str(v.id),
            "checkpoint_step": step,
        },
        notes=f"Checkpoint {step} of v{v.number}",
        completed_at=datetime.now(UTC),
    )
    db.add(copy)
    await db.commit()
    return version_json(copy)


def _sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with storage.absolute(relative).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@router.get("/{lora_id}/versions/{version_id}/logs")
async def version_logs(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    offset: int = Query(default=0, ge=0),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    path = storage.absolute(f"{runner.version_dir(v)}/train.log")

    def read() -> tuple[str, int]:
        if not path.exists():
            return "", 0
        size = path.stat().st_size
        start = offset if offset <= size else 0
        start = max(start, size - 200_000)
        with path.open("rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace"), size

    text, size = await asyncio.to_thread(read)
    return {"text": text, "next_offset": size}


@router.get("/{lora_id}/versions/{version_id}/file")
async def version_file(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    step: int | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    relative = v.file_path
    if step is not None:
        relative = next((c["file_path"] for c in v.checkpoints or [] if c["step"] == step), None)
    if not relative:
        raise api_error(404, "not_found", "No file.")
    slug = re.sub(r"[^a-z0-9]+", "-", lora.name.lower()).strip("-") or "lora"
    name = f"{slug}-v{v.number}" + (f"-step{step}" if step is not None else "") + ".safetensors"
    return FileResponse(
        storage.absolute(relative), media_type="application/octet-stream", filename=name
    )


@router.get("/{lora_id}/versions/{version_id}/likeness")
async def version_likeness(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Face likeness of the validation images to the LoRA's photos, per step, and the best
    step. Measured on this Mac and kept with the version until new images arrive."""
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    progress = dict(v.progress or {})
    samples = [s for s in progress.get("samples", []) if s.get("step") is not None]
    if not samples:
        return {"reference_faces": 0, "steps": [], "best_step": None, "samples": 0}
    refs = [
        storage.absolute(i.reference.file_path)
        for i in await _items(db, lora)
        if not i.excluded and i.split == "train"
    ][:FACE_REFERENCES]
    cached = progress.get("likeness")
    if (
        cached
        and cached.get("samples") == len(samples)
        and cached.get("refs") == len(refs)
        and "baseline" in cached
    ):
        return cached
    if not refs:
        raise api_error(409, "no_photos", "This LoRA has no photos to compare with.")
    folder = storage.absolute(runner.version_dir(v)) / "samples"
    try:
        result = await faces.likeness([(int(s["step"]), folder / s["file"]) for s in samples], refs)
    except faces.FaceError as exc:
        raise api_error(502, "face_model", exc.message) from exc
    result["samples"] = len(samples)
    result["refs"] = len(refs)
    result["threshold"] = faces.MATCH_THRESHOLD
    fresh = await db.get(LoraVersion, v.id, populate_existing=True)
    if fresh is not None:
        progress = dict(fresh.progress or {})
        progress["likeness"] = result
        fresh.progress = progress
        await db.commit()
    return result


@router.get("/{lora_id}/versions/{version_id}/samples/{name}")
async def version_sample(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    name: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    if not re.fullmatch(r"[\w.-]+\.(jpg|jpeg|png|webp)", name):
        raise api_error(404, "not_found", "Not found.")
    path = storage.absolute(f"{runner.version_dir(v)}/samples/{name}")
    if not await asyncio.to_thread(path.exists):
        raise api_error(404, "not_found", "Not found.")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


class DeployBody(BaseModel):
    connection_id: uuid.UUID


@router.post("/{lora_id}/versions/{version_id}/deploy")
async def deploy_version(
    lora_id: uuid.UUID,
    version_id: uuid.UUID,
    body: DeployBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Copies the LoRA file into a ComfyUI's models/loras (local folder or studio GPU)."""
    lora = await _owned(db, user, lora_id)
    v = await _version(db, user, lora, version_id)
    conn = await owned_connection(db, user.id, body.connection_id)
    if conn is None:
        raise api_error(404, "not_found", "ComfyUI connection not found.")
    try:
        name = await place_lora(db, conn, lora, v, force=True)
    except ComfyError as exc:
        raise api_error(400 if exc.code == "unsupported" else 502, exc.code, exc.message) from exc
    return {"filename": name, "version": version_json(v)}


# --- import ------------------------------------------------------------------------------------


class ImportBody(BaseModel):
    upload_id: str = Field(min_length=32, max_length=32)
    name: str = Field(min_length=1, max_length=80)
    trigger_word: str = Field(min_length=2, max_length=40)
    class_word: str = Field(default="", max_length=40)
    # When the file's metadata does not name its base model, the user states it.
    declared_architecture: Literal["krea2"] | None = None
    declared_variant: Literal["raw", "turbo"] | None = None
    persona_id: uuid.UUID | None = None
    adult_confirmed: bool = False


@router.post("/import", status_code=201)
async def import_lora(
    body: ImportBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    if not body.adult_confirmed:
        raise api_error(
            400, "adult_confirmation_required", "Confirm that the character is a fictional adult."
        )
    trigger = body.trigger_word.strip()
    if not TRIGGER_RE.match(trigger):
        raise api_error(
            400, "invalid_trigger", "Trigger word: one word of letters, digits, - or _."
        )
    _check_words(body.name, body.class_word, trigger)
    if await db.scalar(
        select(Lora.id).where(
            Lora.user_id == user.id, func.lower(Lora.trigger_word) == trigger.lower()
        )
    ):
        raise api_error(409, "trigger_taken", "Another LoRA already uses this trigger word.")
    try:
        path, _ = await uploads.take(user.id, body.upload_id)
        keys, metadata = await asyncio.to_thread(read_header, path)
    except (uploads.UploadError, InvalidSafetensors) as exc:
        raise api_error(400, "invalid_file", str(exc)) from exc
    arch, evidence = lora_architecture(keys, metadata)
    if arch is None and not any("lora" in k.lower() for k in keys):
        await uploads.discard(user.id, body.upload_id)
        raise api_error(400, "not_a_lora", evidence)
    if arch is not None and arch != "krea2":
        await uploads.discard(user.id, body.upload_id)
        raise api_error(
            409, "lora_incompatible", f"{evidence} Only Krea 2 LoRAs can be used in the studio."
        )
    declared = arch is None
    if declared and body.declared_architecture is None:
        raise api_error(
            409,
            "architecture_unknown",
            f"{evidence} Choose its base model explicitly if you know it.",
        )
    variant = lora_variant(metadata) or body.declared_variant
    if variant is None:
        raise api_error(409, "variant_unknown", "Choose whether it was trained on RAW or Turbo.")
    lora = Lora(
        user_id=user.id,
        name=body.name.strip(),
        trigger_word=trigger,
        class_word=body.class_word.strip(),
        architecture="krea2",
        base_variant=variant,
        source="imported",
        persona_id=body.persona_id,
        tags=[],
    )
    db.add(lora)
    await db.flush()
    version = LoraVersion(
        lora_id=lora.id,
        user_id=user.id,
        number=1,
        status="completed",
        config={},
        trainer_config={},
        dataset_manifest={
            "source": "imported",
            "evidence": evidence if not declared else "Declared by you",
            "metadata_keys": sorted(metadata)[:40],
        },
        completed_at=datetime.now(UTC),
    )
    db.add(version)
    await db.flush()
    relative = f"{user.id}/loras/{lora.id}/{version.id}/imported.safetensors"
    dest = storage.absolute(relative)

    def move() -> tuple[int, str]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)
        return dest.stat().st_size, _sha256(relative)

    size, sha = await asyncio.to_thread(move)
    await uploads.discard(user.id, body.upload_id)
    version.file_path = relative
    version.file_size = size
    version.file_sha256 = sha
    lora.preferred_version_id = version.id
    await db.commit()
    return await _detail(db, lora)


# --- testing ---------------------------------------------------------------------------------


class TestBody(BaseModel):
    connection_id: uuid.UUID
    variant: Literal["turbo", "raw"] = "turbo"
    prompts: list[str] = Field(min_length=1, max_length=4)
    seeds: list[int] = Field(min_length=1, max_length=4)
    version_ids: list[uuid.UUID] = Field(min_length=1, max_length=3)
    include_base: bool = True
    strength: float = Field(default=1.0, ge=-2, le=2)
    aspect_ratio: str = "3:4"
    resolution: str = "1K"
    # Krea 2's "18+ content" switch for these runs (e.g. the body consistency test).
    adult: bool = False
    confirm_count: int = Field(ge=1, le=48)


class DatasetGenerateBody(BaseModel):
    connection_id: uuid.UUID
    version_id: uuid.UUID
    variant: Literal["turbo", "raw"] = "turbo"
    # One run per prompt (different backgrounds, outfits, angles, light, …).
    prompts: list[str] = Field(min_length=1, max_length=30)
    count: int = Field(default=1, ge=1, le=4)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    aspect_ratio: str = "3:4"
    resolution: str = "1K"
    adult: bool = False
    confirm_count: int = Field(ge=1, le=120)


@router.post("/{lora_id}/dataset/generate", status_code=201)
async def generate_dataset(
    lora_id: uuid.UUID,
    body: DatasetGenerateBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """New photos for the dataset: Krea 2 with this LoRA, one run per prompt (they wait in
    the GPU's queue). The best results are then added with "Add to LoRA"."""
    lora = await _owned(db, user, lora_id)
    prompts = [p.strip() for p in body.prompts if p.strip()]
    if not prompts:
        raise api_error(400, "invalid_parameter", "Write at least one prompt.")
    images = len(prompts) * body.count
    if images != body.confirm_count:
        raise api_error(400, "confirmation_required", f"This is {images} images.")
    group = uuid.uuid4().hex
    subject = f"{lora.trigger_word} {lora.class_word}".strip()
    created = 0
    for prompt in prompts:
        _check_words(prompt)
        # The trigger word brings the character in; added when the prompt lacks it.
        has_trigger = re.search(rf"(?<![\w]){re.escape(lora.trigger_word)}(?![\w])", prompt, re.I)
        text = prompt if has_trigger else f"photo of {subject}, {prompt}"
        req = ComfyGenerateBody(
            connection_id=body.connection_id,
            variant=body.variant,
            prompt=text,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            count=body.count,
            lora=LoraUse(version_id=body.version_id, strength=body.strength, insert_trigger=False),
            adult=body.adult,
            dataset_lora_id=lora.id,
            batch_group=group,
        )
        try:
            await create_comfy_generation(db, user, req)
        except GenerationRejected as exc:
            raise api_error(exc.status_code, exc.code, exc.message) from exc
        created += 1
    return {"group": group, "runs": created, "images": images}


@router.get("/{lora_id}/dataset/generated")
async def generated_for_dataset(
    lora_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """The runs of the dataset generator for this LoRA (newest first), with their images."""
    lora = await _owned(db, user, lora_id)
    rows = (
        await db.scalars(
            select(Generation)
            .where(
                Generation.user_id == user.id,
                Generation.comfy_meta["dataset_lora_id"].astext == str(lora.id),
            )
            .options(selectinload(Generation.tasks), selectinload(Generation.images))
            .order_by(Generation.created_at.desc())
            .limit(120)
        )
    ).all()
    return {"items": [generation_json(g) for g in rows]}


@router.post("/{lora_id}/tests", status_code=201)
async def run_test(
    lora_id: uuid.UUID,
    body: TestBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Same prompts and seeds through the base model and one or more versions (side by side)."""
    lora = await _owned(db, user, lora_id)
    runs = len(body.prompts) * len(body.seeds) * (len(body.version_ids) + int(body.include_base))
    if runs != body.confirm_count:
        raise api_error(400, "confirmation_required", f"This test is {runs} generations.")
    group = uuid.uuid4().hex
    created = []
    for prompt in body.prompts:
        _check_words(prompt)
        for seed in body.seeds:
            labels: list[tuple[str, uuid.UUID | None]] = []
            if body.include_base:
                labels.append(("base", None))
            for vid in body.version_ids:
                labels.append((str(vid), vid))
            for label, vid in labels:
                text = (
                    prompt
                    if vid
                    else re.sub(
                        rf"(?<![\w]){re.escape(lora.trigger_word)}(?![\w]),?\s*",
                        "",
                        prompt,
                        flags=re.I,
                    ).strip()
                    or prompt
                )
                req = ComfyGenerateBody(
                    connection_id=body.connection_id,
                    variant=body.variant,
                    prompt=text,
                    aspect_ratio=body.aspect_ratio,
                    resolution=body.resolution,
                    seed=seed,
                    count=1,
                    lora=LoraUse(version_id=vid, strength=body.strength) if vid else None,
                    adult=body.adult,
                    test_group=group,
                    test_label=label,
                    test_lora_id=lora.id,
                    test_prompt=prompt,
                )
                try:
                    gen = await create_comfy_generation(db, user, req)
                except GenerationRejected as exc:
                    raise api_error(exc.status_code, exc.code, exc.message) from exc
                created.append(gen)
    return {"group": group, "count": len(created)}


@router.get("/{lora_id}/tests")
async def list_tests(
    lora_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    lora = await _owned(db, user, lora_id)
    rows = (
        await db.scalars(
            select(Generation)
            .where(
                Generation.user_id == user.id,
                Generation.provider == "comfyui",
                Generation.comfy_meta["test_lora_id"].astext == str(lora.id),
            )
            .options(selectinload(Generation.tasks), selectinload(Generation.images))
            .order_by(Generation.created_at.desc())
            .limit(300)
        )
    ).all()
    groups: dict[str, dict[str, Any]] = {}
    for gen in rows:
        meta = gen.comfy_meta or {}
        group = groups.setdefault(
            meta.get("test_group") or "", {"group": meta.get("test_group"), "runs": []}
        )
        group["runs"].append(generation_json(gen))
    out = []
    for g in groups.values():
        g["runs"].sort(key=lambda r: r["created_at"])
        g["created_at"] = g["runs"][0]["created_at"] if g["runs"] else None
        out.append(g)
    out.sort(key=lambda g: g["created_at"] or "", reverse=True)
    return {"items": out[:20]}
