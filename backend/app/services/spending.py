"""What the studio's work cost this month — only from numbers the providers gave:

- WaveSpeed / Atlas Cloud: the price the user confirmed before each run (WaveSpeed's own
  estimate; Atlas has no price API, so its runs are only counted).
- RunPod / Modal: the hours of the GPUs the studio started × the hourly price they started
  at (RunPod: the pod's own rate; Modal: its rates or list price) — an estimate.
- Modal: also its real cost from Modal's billing report (the whole workspace, up to the last
  full hour).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import PROVIDERS
from app.gpu import service as gpu_service
from app.gpu.base import GPUError
from app.gpu.registry import GPU_PROVIDERS, get_gpu_adapter
from app.models import Generation, GPUIntegration, GPUWorkload


def month_start(zone: ZoneInfo, now: datetime | None = None) -> datetime:
    local = (now or datetime.now(UTC)).astimezone(zone)
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


async def summary(db: AsyncSession, user_id: uuid.UUID, zone: ZoneInfo) -> dict[str, Any]:
    now = datetime.now(UTC)
    start = month_start(zone, now)
    hosted: list[dict[str, Any]] = []
    rows = (
        await db.execute(
            select(Generation.provider, Generation.cost_estimate).where(
                Generation.user_id == user_id,
                Generation.created_at >= start,
                Generation.status == "completed",
                Generation.provider.in_(list(PROVIDERS)),
            )
        )
    ).all()
    for key, info in PROVIDERS.items():
        mine = [cost for provider, cost in rows if provider == key]
        priced = [c for c in mine if c is not None]
        hosted.append(
            {
                "provider": key,
                "name": info["name"],
                "supported": bool(info["supports_prices"]),
                "runs": len(mine),
                "priced_runs": len(priced),
                "estimate": round(sum(priced), 4) if priced else (0.0 if not mine else None),
            }
        )

    gpu: list[dict[str, Any]] = []
    connected = set(
        (
            await db.scalars(
                select(GPUIntegration.provider).where(GPUIntegration.user_id == user_id)
            )
        ).all()
    )
    workloads = (await db.scalars(select(GPUWorkload).where(GPUWorkload.user_id == user_id))).all()
    for key, info in GPU_PROVIDERS.items():
        hours = priced_hours = 0.0
        estimate = 0.0
        for w in workloads:
            if w.provider != key:
                continue
            end = w.stopped_at or (now if w.status in ("running", "starting") else w.created_at)
            begin = max(w.created_at, start)
            if end <= begin:
                continue
            h = (end - begin).total_seconds() / 3600
            hours += h
            rate = (w.meta or {}).get("cost_per_hour")
            if isinstance(rate, int | float):
                priced_hours += h
                estimate += h * float(rate)
        entry: dict[str, Any] = {
            "provider": key,
            "name": info.get("name", key),
            "connected": key in connected,
            "gpu_hours": round(hours, 2),
            "unpriced_hours": round(hours - priced_hours, 2),
            "estimate": round(estimate, 4) if hours else 0.0,
            "billed": None,
            "billed_error": None,
        }
        adapter = get_gpu_adapter(key)
        if key in connected and hasattr(adapter, "spent_since"):
            try:
                creds = await gpu_service.load_creds(db, user_id, key)
                entry["billed"] = round(await adapter.spent_since(creds, start), 4)
            except GPUError as exc:
                entry["billed_error"] = exc.message
        gpu.append(entry)

    known = [h["estimate"] for h in hosted if h["estimate"] is not None]
    known += [g["billed"] if g["billed"] is not None else g["estimate"] for g in gpu]
    return {
        "since": start.isoformat(),
        "currency": "USD",
        "hosted": hosted,
        "gpu": gpu,
        "total": round(sum(known), 4),
    }
