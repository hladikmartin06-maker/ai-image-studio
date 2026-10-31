"""AI models: saved fictional characters with a description and reference images."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.models import Image, Persona, ReferenceImage, User
from app.services import personas, references, storage
from app.services.personas import (
    MAX_PERSONA_IMAGES,
    PersonaCreate,
    PersonaPatch,
    PersonaRejected,
    persona_json,
)

router = APIRouter(prefix="/api/personas", tags=["ai-models"])


class AddImagesBody(BaseModel):
    # Generated images from the library (copied into the collection).
    library_image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_PERSONA_IMAGES)
    # Reference images that were already uploaded (e.g. on the Generate page).
    reference_image_ids: list[uuid.UUID] = Field(
        default_factory=list, max_length=MAX_PERSONA_IMAGES
    )


def _rejected(exc: PersonaRejected | references.ReferenceRejected):
    return api_error(exc.status_code, exc.code, exc.message)


async def _owned(db: AsyncSession, user: User, persona_id: uuid.UUID) -> Persona:
    persona = await personas.owned(db, user.id, persona_id)
    if persona is None:
        raise api_error(404, "not_found", "AI model not found.")
    return persona


async def _json(db: AsyncSession, user: User, persona: Persona) -> dict:
    counts = await personas.generation_counts(db, user.id, [persona.id])
    return persona_json(persona, counts.get(persona.id, 0))


async def _reload(db: AsyncSession, user: User, persona_id: uuid.UUID) -> dict:
    # personas.owned() loads with populate_existing, so the collection is read fresh.
    return await _json(db, user, await _owned(db, user, persona_id))


@router.get("")
async def list_personas(
    q: str | None = Query(default=None, max_length=120),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every AI model of the user, recently used first (a user has dozens, not thousands)."""
    stmt = select(Persona).where(Persona.user_id == user.id)
    if q and q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(Persona.name.ilike(like), Persona.description.ilike(like)))
    rows = (
        await db.scalars(
            personas.with_images(
                stmt.order_by(
                    Persona.last_used_at.desc().nulls_last(), Persona.updated_at.desc()
                ).limit(500)
            )
        )
    ).all()
    counts = await personas.generation_counts(db, user.id, [p.id for p in rows])
    return {
        "items": [persona_json(p, counts.get(p.id, 0)) for p in rows],
        "total": len(rows),
        "max_images": MAX_PERSONA_IMAGES,
    }


@router.post("", status_code=201)
async def create(
    body: PersonaCreate, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    if not body.adult_confirmed:
        raise api_error(
            400,
            "adult_confirmation_required",
            "Confirm that the AI model is a fictional adult (18+).",
        )
    name = body.name.strip()
    if not name:
        raise api_error(400, "invalid_parameter", "Enter a name.")
    try:
        personas.check_persona_fields(name, body.description, body.appearance)
    except PersonaRejected as exc:
        raise _rejected(exc) from exc
    persona = Persona(
        user_id=user.id,
        name=name,
        description=body.description.strip(),
        notes=body.notes.strip(),
        appearance=body.appearance.model_dump(),
    )
    db.add(persona)
    await db.commit()
    return await _reload(db, user, persona.id)


@router.get("/{persona_id}")
async def get_one(
    persona_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return await _json(db, user, await _owned(db, user, persona_id))


@router.patch("/{persona_id}")
async def update(
    persona_id: uuid.UUID,
    body: PersonaPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    persona = await _owned(db, user, persona_id)
    name = body.name.strip() if body.name is not None else persona.name
    if not name:
        raise api_error(400, "invalid_parameter", "Enter a name.")
    description = body.description.strip() if body.description is not None else persona.description
    appearance = body.appearance or personas.Appearance.model_validate(persona.appearance or {})
    try:
        personas.check_persona_fields(name, description, appearance)
    except PersonaRejected as exc:
        raise _rejected(exc) from exc
    persona.name = name
    persona.description = description
    persona.appearance = appearance.model_dump()
    if body.notes is not None:
        persona.notes = body.notes.strip()

    linked = {link.reference_image_id: link for link in persona.images}
    if "cover_reference_id" in body.model_fields_set:
        if body.cover_reference_id is not None and body.cover_reference_id not in linked:
            raise api_error(400, "invalid_parameter", "That image is not part of this AI model.")
        persona.cover_reference_id = body.cover_reference_id
    if body.image_order is not None:
        order = list(dict.fromkeys(r for r in body.image_order if r in linked))
        rest = sorted(
            (link for ref_id, link in linked.items() if ref_id not in order),
            key=lambda link: link.position,
        )
        for position, link in enumerate([linked[r] for r in order] + rest):
            link.position = position
    persona.updated_at = datetime.now(UTC)
    await db.commit()
    return await _reload(db, user, persona.id)


@router.delete("/{persona_id}")
async def remove(
    persona_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Deletes the AI model and its reference images (unless another AI model uses them).
    Generated images stay in the library."""
    persona = await _owned(db, user, persona_id)
    ref_ids = [link.reference_image_id for link in persona.images]
    await db.delete(persona)
    await db.flush()
    paths = await references.delete_unused(db, user.id, ref_ids)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}


@router.post("/{persona_id}/duplicate", status_code=201)
async def duplicate(
    persona_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """A copy that shares the same image files (nothing is copied or uploaded again)."""
    source = await _owned(db, user, persona_id)
    suffix = " (copy)"
    copy = Persona(
        user_id=user.id,
        name=source.name[: personas.NAME_MAX - len(suffix)] + suffix,
        description=source.description,
        notes=source.notes,
        appearance=dict(source.appearance or {}),
        cover_reference_id=source.cover_reference_id,
        images=[],
    )
    db.add(copy)
    await db.flush()
    for link in sorted(source.images, key=lambda i: i.position):
        await personas.link_reference(db, copy, link.reference)
    await db.commit()
    return await _reload(db, user, copy.id)


@router.post("/{persona_id}/images", status_code=201)
async def upload_image(
    persona_id: uuid.UUID,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    persona = await _owned(db, user, persona_id)
    if len(persona.images) >= MAX_PERSONA_IMAGES:
        raise api_error(
            409, "limit_reached", f"An AI model holds up to {MAX_PERSONA_IMAGES} reference images."
        )
    data = await file.read(references.max_bytes() + 1)
    try:
        ref, _ = await references.store(db, user.id, data, file.filename or "")
        added = await personas.link_reference(db, persona, ref)
    except (references.ReferenceRejected, PersonaRejected) as exc:
        raise _rejected(exc) from exc
    persona.updated_at = datetime.now(UTC)
    await db.commit()
    return {"persona": await _reload(db, user, persona.id), "added": int(added)}


@router.post("/{persona_id}/images/add")
async def add_images(
    persona_id: uuid.UUID,
    body: AddImagesBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Adds library images (copied as references) or already uploaded references."""
    persona = await _owned(db, user, persona_id)
    added = 0
    try:
        if body.library_image_ids:
            images = (
                await db.scalars(
                    select(Image).where(
                        Image.user_id == user.id, Image.id.in_(body.library_image_ids)
                    )
                )
            ).all()
            if len(images) != len(set(body.library_image_ids)):
                raise api_error(404, "not_found", "An image no longer exists.")
            order = {image_id: i for i, image_id in enumerate(body.library_image_ids)}
            for image in sorted(images, key=lambda i: order[i.id]):
                ext = image.file_path.rsplit(".", 1)[-1]
                ref, _ = await references.from_library_file(
                    db, user.id, image.file_path, f"library-{image.id}.{ext}"
                )
                added += await personas.link_reference(db, persona, ref)
        if body.reference_image_ids:
            refs = (
                await db.scalars(
                    select(ReferenceImage).where(
                        ReferenceImage.user_id == user.id,
                        ReferenceImage.id.in_(body.reference_image_ids),
                    )
                )
            ).all()
            if len(refs) != len(set(body.reference_image_ids)):
                raise api_error(404, "not_found", "A reference image no longer exists.")
            for ref in refs:
                added += await personas.link_reference(db, persona, ref)
    except (references.ReferenceRejected, PersonaRejected) as exc:
        await db.rollback()
        raise _rejected(exc) from exc
    persona.updated_at = datetime.now(UTC)
    await db.commit()
    return {"persona": await _reload(db, user, persona.id), "added": added}


@router.delete("/{persona_id}/images/{reference_id}")
async def remove_image(
    persona_id: uuid.UUID,
    reference_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Removes an image from the collection and deletes the file unless another AI model
    still uses it or a running generation needs it."""
    persona = await _owned(db, user, persona_id)
    link = next((i for i in persona.images if i.reference_image_id == reference_id), None)
    if link is None:
        raise api_error(404, "not_found", "That image is not part of this AI model.")
    persona.images.remove(link)
    if persona.cover_reference_id == reference_id:
        persona.cover_reference_id = None
    persona.updated_at = datetime.now(UTC)
    await db.flush()
    paths = await references.delete_unused(db, user.id, [reference_id])
    await db.commit()
    storage.delete_files(*paths)
    return await _reload(db, user, persona.id)
