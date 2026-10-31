"""The image library: browsing, search, favorites, downloads and bulk actions."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager
from starlette.background import BackgroundTask

from app.catalog import MODELS, PROVIDERS
from app.comfy.krea2 import MODEL_KEYS
from app.db import get_db
from app.deps import api_error, current_user
from app.models import Generation, Image, User
from app.routes.paging import after, encode_cursor
from app.services import preferences, storage
from app.services.serializers import image_json

router = APIRouter(prefix="/api/images", tags=["images"])

MAX_BULK = 200
MAX_ARCHIVE = 100
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class ImagePatch(BaseModel):
    is_favorite: bool | None = None
    in_library: bool | None = None


class BulkBody(BaseModel):
    ids: list[uuid.UUID] = Field(min_length=1, max_length=MAX_BULK)
    action: Literal["favorite", "unfavorite", "delete", "save"]


def _search_conditions(q: str) -> list:
    text = q.strip()
    like = f"%{text}%"
    conditions = [
        Generation.prompt.ilike(like),
        Generation.provider_model_id.ilike(like),
        Image.mime_type.ilike(like),
    ]
    lowered = text.lower()
    model_keys = [k for k, m in MODELS.items() if lowered in m.name.lower() or lowered in k]
    if model_keys:
        conditions.append(Generation.model_key.in_(model_keys))
    providers = [k for k, p in PROVIDERS.items() if lowered in p["name"].lower() or lowered in k]
    if providers:
        conditions.append(Generation.provider.in_(providers))
    size = re.fullmatch(r"(\d{2,5})\s*[x×*]\s*(\d{2,5})", lowered)
    if size:
        conditions.append(
            (Image.width == int(size.group(1))) & (Image.height == int(size.group(2)))
        )
    if lowered.isdigit():
        conditions.append(Generation.settings["seed"].astext == lowered)
    return conditions


async def _owned(db: AsyncSession, user: User, image_id: uuid.UUID) -> Image:
    image = await db.scalar(
        select(Image)
        .join(Image.generation)
        .options(contains_eager(Image.generation))
        .where(Image.id == image_id, Image.user_id == user.id)
    )
    if image is None:
        raise api_error(404, "not_found", "Image not found.")
    return image


@router.get("")
async def list_images(
    q: str | None = Query(default=None, max_length=200),
    model: str | None = None,
    provider: str | None = None,
    favorite: bool | None = None,
    status: Literal["complete", "partial"] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    scope: Literal["library", "all", "unsaved"] = "library",
    generation_id: uuid.UUID | None = None,
    persona: uuid.UUID | None = None,
    lora_version: uuid.UUID | None = None,
    sort: Literal["newest", "oldest", "favorited"] = "newest",
    cursor: str | None = None,
    limit: int = Query(default=48, ge=1, le=120),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conditions = [Image.user_id == user.id]
    if scope == "library":
        conditions.append(Image.in_library.is_(True))
    elif scope == "unsaved":
        conditions.append(Image.in_library.is_(False))
    if generation_id:
        conditions.append(Image.generation_id == generation_id)
    if persona:
        conditions.append(Generation.persona_id == persona)
    if lora_version:
        conditions.append(Generation.lora_version_id == lora_version)
    if model:
        keys = [m for m in model.split(",") if m in MODELS or m in MODEL_KEYS]
        if keys:
            conditions.append(Generation.model_key.in_(keys))
    if provider:
        keys = [p for p in provider.split(",") if p in PROVIDERS]
        if keys:
            conditions.append(Generation.provider.in_(keys))
    if favorite is not None:
        conditions.append(Image.is_favorite.is_(favorite))
    if status == "complete":
        conditions.append(Generation.error_message.is_(None))
    elif status == "partial":
        conditions.append(Generation.error_message.is_not(None))
    if date_from:
        conditions.append(Image.created_at >= date_from)
    if date_to:
        conditions.append(Image.created_at < date_to)
    if q and q.strip():
        conditions.append(or_(*_search_conditions(q)))

    base = select(Image).join(Image.generation).where(*conditions)
    total = await db.scalar(
        select(func.count()).select_from(Image).join(Image.generation).where(*conditions)
    )

    column = Image.favorited_at if sort == "favorited" else Image.created_at
    descending = sort != "oldest"
    if sort == "favorited":
        base = base.where(Image.favorited_at.is_not(None))
    if cursor:
        base = base.where(after(column, Image.id, cursor, descending))
    order = (column.desc(), Image.id.desc()) if descending else (column.asc(), Image.id.asc())
    rows = (
        await db.scalars(
            base.options(contains_eager(Image.generation)).order_by(*order).limit(limit + 1)
        )
    ).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit and page:
        last = page[-1]
        next_cursor = encode_cursor(
            last.favorited_at if sort == "favorited" else last.created_at, last.id
        )
    return {"items": [image_json(i) for i in page], "next_cursor": next_cursor, "total": total}


@router.get("/archive")
async def archive(
    ids: str = Query(min_length=1, max_length=MAX_ARCHIVE * 40),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """A ZIP of the original files (stored, not recompressed)."""
    try:
        wanted = [uuid.UUID(x) for x in ids.split(",") if x][:MAX_ARCHIVE]
    except ValueError as exc:
        raise api_error(400, "bad_ids", "Invalid image ids.") from exc
    rows = (
        await db.scalars(
            select(Image)
            .join(Image.generation)
            .options(contains_eager(Image.generation))
            .where(Image.user_id == user.id, Image.id.in_(wanted))
            .order_by(Image.created_at)
        )
    ).all()
    if not rows:
        raise api_error(404, "not_found", "No images to download.")
    prefs = await preferences.load(db, user.id)
    names = [_download_name(img, prefs.download_name) for img in rows]
    paths = [storage.absolute(img.file_path) for img in rows]

    def build() -> str:
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        used: set[str] = set()
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, path in zip(names, paths, strict=True):
                unique, n = name, 2
                while unique in used:
                    stem, ext = os.path.splitext(name)
                    unique, n = f"{stem}-{n}{ext}", n + 1
                used.add(unique)
                if path.exists():
                    zf.write(path, unique)
        return tmp

    tmp = await asyncio.to_thread(build)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return FileResponse(
        tmp,
        media_type="application/zip",
        filename=f"ai-image-studio-{stamp}.zip",
        background=BackgroundTask(os.unlink, tmp),
    )


@router.post("/bulk")
async def bulk(
    body: BulkBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    ids = list(dict.fromkeys(body.ids))
    owned = Image.user_id == user.id
    if body.action in ("favorite", "unfavorite"):
        fav = body.action == "favorite"
        result = await db.execute(
            update(Image)
            .where(owned, Image.id.in_(ids), Image.is_favorite.is_(not fav))
            .values(is_favorite=fav, favorited_at=datetime.now(UTC) if fav else None)
        )
        await db.commit()
        return {"ok": True, "changed": result.rowcount}
    if body.action == "save":
        result = await db.execute(
            update(Image).where(owned, Image.id.in_(ids)).values(in_library=True)
        )
        await db.commit()
        return {"ok": True, "changed": result.rowcount}
    rows = (await db.scalars(select(Image).where(owned, Image.id.in_(ids)))).all()
    paths = [p for img in rows for p in (img.file_path, img.thumb_path)]
    await db.execute(delete(Image).where(owned, Image.id.in_([r.id for r in rows])))
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True, "changed": len(rows)}


@router.get("/{image_id}")
async def get_image(
    image_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return image_json(await _owned(db, user, image_id))


@router.patch("/{image_id}")
async def patch_image(
    image_id: uuid.UUID,
    body: ImagePatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    image = await _owned(db, user, image_id)
    if body.is_favorite is not None and body.is_favorite != image.is_favorite:
        image.is_favorite = body.is_favorite
        image.favorited_at = datetime.now(UTC) if body.is_favorite else None
    if body.in_library is not None:
        image.in_library = body.in_library
    await db.commit()
    return image_json(image)


@router.delete("/{image_id}")
async def delete_image(
    image_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    image = await _owned(db, user, image_id)
    paths = (image.file_path, image.thumb_path)
    await db.delete(image)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}


def _slug(text: str, words: int = 6) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", text.lower())[:words]
    return "-".join(parts)[:60] or "image"


def _download_name(image: Image, style: str) -> str:
    ext = EXTENSIONS.get(image.mime_type, "png")
    if style == "id":
        return f"{image.id}.{ext}"
    stamp = image.created_at.strftime("%Y%m%d-%H%M%S")
    if style == "date":
        return f"{stamp}-{image.index + 1}.{ext}"
    return f"{_slug(image.generation.prompt)}-{stamp}-{image.index + 1}.{ext}"


@router.get("/{image_id}/file")
async def file(
    image_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    image = await _owned(db, user, image_id)
    return FileResponse(
        storage.absolute(image.file_path),
        media_type=image.mime_type,
        headers={"Cache-Control": "private, max-age=604800, immutable"},
    )


@router.get("/{image_id}/thumb")
async def thumb(
    image_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    image = await _owned(db, user, image_id)
    return FileResponse(
        storage.absolute(image.thumb_path),
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=604800, immutable"},
    )


@router.get("/{image_id}/download")
async def download(
    image_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """The original file as the provider delivered it (full quality and resolution)."""
    image = await _owned(db, user, image_id)
    prefs = await preferences.load(db, user.id)
    return FileResponse(
        storage.absolute(image.file_path),
        media_type=image.mime_type,
        filename=_download_name(image, prefs.download_name),
    )
