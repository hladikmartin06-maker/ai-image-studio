"""Reference image uploads for editing/reference-based generation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.models import Persona, PersonaImage, ReferenceImage, User
from app.services import references, storage
from app.services.serializers import reference_json

router = APIRouter(prefix="/api/reference-images", tags=["references"])


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
    data = await file.read(references.max_bytes() + 1)
    try:
        ref, _ = await references.store(db, user.id, data, file.filename or "")
    except references.ReferenceRejected as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
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
    persona = await db.scalar(
        select(Persona.name)
        .join(PersonaImage, PersonaImage.persona_id == Persona.id)
        .where(PersonaImage.reference_image_id == ref.id, Persona.user_id == user.id)
        .limit(1)
    )
    if persona is not None:
        raise api_error(
            409,
            "reference_in_use",
            f"This image belongs to the AI model “{persona}”. Remove it there.",
        )
    paths = (ref.file_path, ref.thumb_path)
    await db.delete(ref)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}
