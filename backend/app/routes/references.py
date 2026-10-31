"""Reference image uploads for editing/reference-based generation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db
from app.deps import api_error, current_user
from app.models import ReferenceImage, User
from app.services import storage
from app.services.serializers import reference_json

router = APIRouter(prefix="/api/reference-images", tags=["references"])

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
MIN_EDGE = 64
MAX_EDGE = 8192


async def _owned(db: AsyncSession, user: User, ref_id: uuid.UUID) -> ReferenceImage:
    ref = await db.scalar(
        select(ReferenceImage).where(ReferenceImage.id == ref_id, ReferenceImage.user_id == user.id)
    )
    if ref is None:
        raise api_error(404, "not_found", "Reference image not found.")
    return ref


@router.post("", status_code=201)
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise api_error(413, "file_too_large", f"Images up to {get_settings().max_upload_mb} MB.")
    if not data:
        raise api_error(400, "empty_file", "The file is empty.")
    try:
        mime, _, width, height = storage.inspect(data)
    except storage.InvalidImage as exc:
        raise api_error(415, "invalid_image", str(exc)) from exc
    if mime not in ALLOWED_MIME:
        raise api_error(415, "invalid_image", "Only PNG, JPEG and WebP images are supported.")
    if min(width, height) < MIN_EDGE or max(width, height) > MAX_EDGE:
        raise api_error(
            400, "invalid_dimensions", f"Images must be {MIN_EDGE} to {MAX_EDGE} pixels per side."
        )
    ref_id = uuid.uuid4()
    stored = await storage.store_reference(user.id, ref_id, data)
    ref = ReferenceImage(
        id=ref_id,
        user_id=user.id,
        file_path=stored.file_path,
        thumb_path=stored.thumb_path,
        original_name=(file.filename or "")[:255],
        mime_type=stored.mime_type,
        width=stored.width,
        height=stored.height,
        size_bytes=stored.size_bytes,
        provider_urls={},
    )
    db.add(ref)
    await db.commit()
    return reference_json(ref)


@router.get("")
async def recent(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = (
        await db.scalars(
            select(ReferenceImage)
            .where(ReferenceImage.user_id == user.id)
            .order_by(ReferenceImage.created_at.desc())
            .limit(24)
        )
    ).all()
    return {"items": [reference_json(r) for r in rows]}


@router.get("/{ref_id}")
async def get_one(
    ref_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return reference_json(await _owned(db, user, ref_id))


@router.get("/{ref_id}/file")
async def file(
    ref_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    ref = await _owned(db, user, ref_id)
    return FileResponse(
        storage.absolute(ref.file_path),
        media_type=ref.mime_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{ref_id}/thumb")
async def thumb(
    ref_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    ref = await _owned(db, user, ref_id)
    return FileResponse(
        storage.absolute(ref.thumb_path),
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete("/{ref_id}")
async def remove(
    ref_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Deletes the upload. Past generations keep their record but can no longer be
    regenerated with this reference."""
    ref = await _owned(db, user, ref_id)
    paths = (ref.file_path, ref.thumb_path)
    await db.delete(ref)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}
