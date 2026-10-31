"""Generation jobs and the generation history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import Select, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import MAX_IMAGES_PER_GENERATION, MODELS
from app.db import get_db
from app.deps import api_error, current_user
from app.models import Generation, Image, ReferenceImage, User
from app.routes.paging import after, encode_cursor
from app.services import storage
from app.services.generation import (
    GenerationRejected,
    GenerationRequest,
    Replay,
    create_generation,
)
from app.services.serializers import generation_json

router = APIRouter(prefix="/api", tags=["generations"])

STATUSES = ("queued", "processing", "completed", "failed", "cancelled")


class RegenerateBody(BaseModel):
    count: int | None = Field(default=None, ge=1, le=MAX_IMAGES_PER_GENERATION)
    client_request_id: str | None = Field(default=None, max_length=64)


def _rejected(exc: GenerationRejected):
    return api_error(exc.status_code, exc.code, exc.message)


def _with_children(stmt: Select) -> Select:
    return stmt.options(selectinload(Generation.tasks), selectinload(Generation.images))


async def _owned(db: AsyncSession, user: User, gen_id: uuid.UUID) -> Generation:
    gen = await db.scalar(
        _with_children(
            select(Generation).where(Generation.id == gen_id, Generation.user_id == user.id)
        )
    )
    if gen is None:
        raise api_error(404, "not_found", "Generation not found.")
    return gen


@router.post("/generations", status_code=201)
async def create(
    body: GenerationRequest, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    try:
        gen = await create_generation(db, user, body)
    except GenerationRejected as exc:
        raise _rejected(exc) from exc
    return generation_json(gen)


@router.get("/generations")
async def list_generations(
    q: str | None = Query(default=None, max_length=200),
    status: str | None = Query(default=None),
    model: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    persona: uuid.UUID | None = Query(default=None),
    sort: str = Query(default="newest", pattern="^(newest|oldest)$"),
    active: bool = Query(default=False),
    cursor: str | None = None,
    limit: int = Query(default=30, ge=1, le=100),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conditions = [Generation.user_id == user.id]
    if q:
        like = f"%{q.strip()}%"
        conditions.append(
            or_(Generation.prompt.ilike(like), Generation.provider_model_id.ilike(like))
        )
    if active:
        conditions.append(Generation.status.in_(("queued", "processing")))
    elif status:
        wanted = [s for s in status.split(",") if s in STATUSES]
        if wanted:
            conditions.append(Generation.status.in_(wanted))
    if model and model in MODELS:
        conditions.append(Generation.model_key == model)
    if provider:
        conditions.append(Generation.provider == provider)
    if persona:
        conditions.append(Generation.persona_id == persona)

    total = await db.scalar(select(func.count()).select_from(Generation).where(*conditions))
    descending = sort == "newest"
    stmt = select(Generation).where(*conditions)
    if cursor:
        stmt = stmt.where(after(Generation.created_at, Generation.id, cursor, descending))
    order = (
        (Generation.created_at.desc(), Generation.id.desc())
        if descending
        else (Generation.created_at.asc(), Generation.id.asc())
    )
    rows = (await db.scalars(_with_children(stmt.order_by(*order).limit(limit + 1)))).all()
    page = rows[:limit]
    next_cursor = (
        encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit and page else None
    )
    return {"items": [generation_json(g) for g in page], "next_cursor": next_cursor, "total": total}


@router.get("/generations/{gen_id}")
async def get_generation(
    gen_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return generation_json(await _owned(db, user, gen_id))


@router.post("/generations/{gen_id}/regenerate", status_code=201)
async def regenerate(
    gen_id: uuid.UUID,
    body: RegenerateBody | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Runs a past request again with the same model, provider, prompt, settings and
    references. With a fixed seed the provider can reproduce the result."""
    source = await _owned(db, user, gen_id)
    body = body or RegenerateBody()
    existing_refs = (
        await db.scalars(
            select(ReferenceImage.id).where(
                ReferenceImage.user_id == user.id,
                ReferenceImage.id.in_([uuid.UUID(r) for r in source.reference_image_ids]),
            )
        )
    ).all()
    if len(existing_refs) != len(source.reference_image_ids):
        raise api_error(
            409,
            "reference_missing",
            "A reference image of this generation was deleted, so it cannot be repeated.",
        )
    request = GenerationRequest(
        model=source.model_key,
        provider=source.provider,
        prompt=source.prompt,
        count=body.count or source.image_count,
        settings=dict(source.settings),
        reference_image_ids=[uuid.UUID(r) for r in source.reference_image_ids],
        client_request_id=body.client_request_id,
        source_generation_id=source.id,
    )
    # Same prompt as sent last time (with the AI model's description of that moment).
    replay = Replay(
        final_prompt=source.final_prompt,
        persona_id=source.persona_id,
        persona_meta=dict(source.persona_meta or {}),
    )
    try:
        gen = await create_generation(db, user, request, replay=replay)
    except GenerationRejected as exc:
        raise _rejected(exc) from exc
    return generation_json(gen)


@router.post("/generations/{gen_id}/cancel")
async def cancel(
    gen_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Only a job that has not been sent yet can be cancelled: neither provider documents
    cancelling a request once it was accepted."""
    result = await db.execute(
        update(Generation)
        .where(
            Generation.id == gen_id, Generation.user_id == user.id, Generation.status == "queued"
        )
        .values(status="cancelled", completed_at=datetime.now(UTC))
    )
    await db.commit()
    if result.rowcount == 0:
        await _owned(db, user, gen_id)
        raise api_error(
            409,
            "cannot_cancel",
            "This request was already sent to the provider and cannot be cancelled.",
        )
    gen = await _owned(db, user, gen_id)
    from app.services.events import broker

    broker.publish(user.id, "generation", generation_json(gen))
    return generation_json(gen)


@router.delete("/generations/{gen_id}")
async def delete_generation(
    gen_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Removes the history entry together with its images."""
    gen = await _owned(db, user, gen_id)
    if gen.status in ("queued", "processing"):
        raise api_error(409, "in_progress", "Wait until the generation has finished.")
    paths = [p for img in gen.images for p in (img.file_path, img.thumb_path)]
    await db.execute(delete(Image).where(Image.generation_id == gen.id))
    await db.delete(gen)
    await db.commit()
    storage.delete_files(*paths)
    return {"ok": True}


@router.get("/prompts/history")
async def prompt_history(
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=30, ge=1, le=100),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recently used prompts, newest first, without duplicates."""
    stmt = (
        select(Generation.prompt, func.max(Generation.created_at).label("last_used"))
        .where(Generation.user_id == user.id)
        .group_by(Generation.prompt)
        .order_by(func.max(Generation.created_at).desc())
        .limit(limit)
    )
    if q:
        stmt = stmt.where(Generation.prompt.ilike(f"%{q.strip()}%"))
    rows = (await db.execute(stmt)).all()
    return {"items": [{"prompt": r.prompt, "last_used": r.last_used.isoformat()} for r in rows]}
