"""Health, catalog, preferences, dashboard statistics, storage and live events."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import catalog_json
from app.config import get_settings
from app.db import get_db
from app.deps import current_user
from app.models import Generation, Image, LoraDatasetItem, PersonaImage, ReferenceImage, User
from app.services import preferences, storage
from app.services.events import broker
from app.services.preferences import PreferencesPatch

router = APIRouter(prefix="/api")

APP_ID = "ai-image-studio"
VERSION = "1.0.0"


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Used by scripts/start.sh and the Mac app: `app` tells this studio apart from any
    other program that might be using the same port."""
    try:
        await db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # noqa: BLE001
        database = "unavailable"
    return {"status": "ok", "app": APP_ID, "version": VERSION, "database": database}


@router.get("/catalog")
async def catalog(_: User = Depends(current_user)):
    return catalog_json()


@router.get("/settings")
async def get_preferences(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    return (await preferences.load(db, user.id)).model_dump()


@router.patch("/settings")
async def patch_preferences(
    body: PreferencesPatch, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    return (await preferences.save(db, user.id, body)).model_dump()


def _zone(tz: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


@router.get("/stats/spending")
async def spending(
    tz: str | None = Query(default=None, max_length=64),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """This month's costs, only from prices and reports the providers gave (estimates)."""
    from app.services import spending as spending_service

    return await spending_service.summary(db, user.id, _zone(tz))


@router.get("/stats")
async def stats(
    tz: str | None = Query(default=None, max_length=64),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    zone = _zone(tz)
    now_local = datetime.now(zone)
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    mine_img = Image.user_id == user.id
    mine_gen = Generation.user_id == user.id

    images_total, images_today, favorites, bytes_total = (
        await db.execute(
            select(
                func.count(Image.id),
                func.count(Image.id).filter(Image.created_at >= today_start),
                func.count(Image.id).filter(Image.is_favorite.is_(True)),
                func.coalesce(func.sum(Image.size_bytes), 0),
            ).where(mine_img)
        )
    ).one()

    gen_row = (
        await db.execute(
            select(
                func.count(Generation.id),
                func.count(Generation.id).filter(Generation.status == "completed"),
                func.count(Generation.id).filter(Generation.status == "failed"),
                func.count(Generation.id).filter(Generation.status.in_(("queued", "processing"))),
                func.count(Generation.id).filter(Generation.created_at >= today_start),
                func.max(Generation.created_at),
            ).where(mine_gen)
        )
    ).one()

    image_counts = (
        select(Image.generation_id, func.count(Image.id).label("n"))
        .where(mine_img)
        .group_by(Image.generation_id)
        .subquery()
    )
    by_provider = (
        await db.execute(
            select(
                Generation.provider,
                func.count(Generation.id),
                func.coalesce(func.sum(image_counts.c.n), 0),
                func.max(Generation.created_at),
                func.count(Generation.id).filter(Generation.status == "failed"),
            )
            .outerjoin(image_counts, image_counts.c.generation_id == Generation.id)
            .where(mine_gen)
            .group_by(Generation.provider)
        )
    ).all()
    by_model = (
        await db.execute(
            select(
                Generation.model_key,
                func.count(Generation.id),
                func.coalesce(func.sum(image_counts.c.n), 0),
            )
            .outerjoin(image_counts, image_counts.c.generation_id == Generation.id)
            .where(mine_gen)
            .group_by(Generation.model_key)
        )
    ).all()

    # Images per local day for the last 14 days (zero-filled).
    days = 14
    start_local = (now_local - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    local_day = func.date(func.timezone(str(zone), Image.created_at))
    per_day_rows = (
        await db.execute(
            select(local_day, func.count(Image.id))
            .where(mine_img, Image.created_at >= start_local.astimezone(UTC))
            .group_by(local_day)
        )
    ).all()
    per_day = {str(d): n for d, n in per_day_rows}
    activity = []
    for i in range(days):
        day = (start_local + timedelta(days=i)).date().isoformat()
        activity.append({"date": day, "images": per_day.get(day, 0)})

    return {
        "images_total": images_total,
        "images_today": images_today,
        "favorites_total": favorites,
        "storage_bytes": int(bytes_total),
        "generations": {
            "total": gen_row[0],
            "completed": gen_row[1],
            "failed": gen_row[2],
            "active": gen_row[3],
            "today": gen_row[4],
            "last_at": gen_row[5].isoformat() if gen_row[5] else None,
        },
        "providers": [
            {
                "provider": p,
                "generations": g,
                "images": int(n),
                "failed": f,
                "last_used_at": last.isoformat() if last else None,
            }
            for p, g, n, last, f in by_provider
        ],
        "models": [{"model": m, "generations": g, "images": int(n)} for m, g, n in by_model],
        "activity": activity,
    }


@router.get("/storage")
async def storage_info(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    usage = await asyncio.to_thread(storage.user_usage, user.id)
    counts = (
        await db.execute(
            select(
                func.count(Image.id),
                func.count(Image.id).filter(Image.in_library.is_(False)),
                func.count(Image.id).filter(Image.is_favorite.is_(True)),
            ).where(Image.user_id == user.id)
        )
    ).one()
    refs = await db.scalar(
        select(func.count(ReferenceImage.id)).where(ReferenceImage.user_id == user.id)
    )
    persona_refs = await db.scalar(
        select(func.count(func.distinct(PersonaImage.reference_image_id))).where(
            PersonaImage.user_id == user.id
        )
    )
    failed = await db.scalar(
        select(func.count(Generation.id)).where(
            Generation.user_id == user.id, Generation.status.in_(("failed", "cancelled"))
        )
    )
    return {
        "location": str(get_settings().storage_dir),
        "bytes": usage,
        "bytes_total": sum(usage.values()),
        "images": counts[0],
        "unsaved_images": counts[1],
        "favorites": counts[2],
        "reference_images": refs,
        # Of those, the ones that belong to AI models (kept by "remove unused").
        "persona_reference_images": persona_refs,
        "failed_generations": failed,
    }


class CleanupBody(BaseModel):
    action: Literal["unused_references", "failed_generations", "unsaved_images"]


@router.post("/storage/cleanup")
async def storage_cleanup(
    body: CleanupBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    removed = 0
    if body.action == "failed_generations":
        result = await db.execute(
            delete(Generation).where(
                Generation.user_id == user.id, Generation.status.in_(("failed", "cancelled"))
            )
        )
        removed = result.rowcount
        await db.commit()
    elif body.action == "unsaved_images":
        rows = (
            await db.scalars(
                select(Image).where(Image.user_id == user.id, Image.in_library.is_(False))
            )
        ).all()
        paths = [p for r in rows for p in (r.file_path, r.thumb_path)]
        await db.execute(delete(Image).where(Image.id.in_([r.id for r in rows])))
        await db.commit()
        storage.delete_files(*paths)
        removed = len(rows)
    else:
        # References not used by any queued/processing generation.
        busy = (
            await db.scalars(
                select(Generation.reference_image_ids).where(
                    Generation.user_id == user.id,
                    Generation.status.in_(("queued", "processing")),
                )
            )
        ).all()
        keep = {ref for ids in busy for ref in ids}
        # References of AI models are never "unused".
        rows = (
            await db.scalars(
                select(ReferenceImage).where(
                    ReferenceImage.user_id == user.id,
                    ~exists().where(PersonaImage.reference_image_id == ReferenceImage.id),
                    ~exists().where(LoraDatasetItem.reference_image_id == ReferenceImage.id),
                )
            )
        ).all()
        doomed = [r for r in rows if str(r.id) not in keep]
        paths = [p for r in doomed for p in (r.file_path, r.thumb_path)]
        await db.execute(
            delete(ReferenceImage).where(ReferenceImage.id.in_([r.id for r in doomed]))
        )
        await db.commit()
        storage.delete_files(*paths)
        removed = len(doomed)
    return {"ok": True, "removed": removed}


@router.get("/events")
async def events(request: Request, user: User = Depends(current_user)):
    """Server-sent events with live generation updates for the signed-in user."""
    queue = broker.subscribe(user.id)

    async def stream():
        try:
            yield "retry: 3000\n\n"
            yield f"event: ready\ndata: {json.dumps({'at': datetime.now(UTC).isoformat()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
        finally:
            broker.unsubscribe(user.id, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
