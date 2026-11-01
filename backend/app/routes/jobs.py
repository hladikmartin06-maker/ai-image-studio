"""One list of every job: image generations (hosted and ComfyUI), LoRA training runs and
ComfyUI deployments, with normalized statuses."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.deps import current_user
from app.gpu.service import run_cost
from app.lora.serial import version_json
from app.models import ComfyConnection, Generation, Lora, LoraVersion, User
from app.services.serializers import iso, model_name, provider_name

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

GEN_STATUS = {
    "queued": "queued",
    "processing": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}
TRAIN_STATUS = {
    "queued": "queued",
    "preparing": "preparing",
    "uploading": "preparing",
    "running": "running",
    "downloading": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "interrupted": "interrupted",
}


def _elapsed(start: datetime | None, end: datetime | None) -> float | None:
    if start is None:
        return None
    return round(((end or datetime.now(UTC)) - start).total_seconds(), 1)


def _generation(gen: Generation) -> dict[str, Any]:
    task = gen.tasks[0] if gen.tasks else None
    progress = (task.progress if task else None) or {}
    status = GEN_STATUS.get(gen.status, gen.status)
    if task and task.error_code == "interrupted":
        status = "interrupted"
    meta = gen.comfy_meta or {}
    percent = None
    if progress.get("max"):
        percent = round(100 * (progress.get("value") or 0) / progress["max"])
    return {
        "id": str(gen.id),
        "type": "generation",
        "name": gen.prompt[:120],
        "model": model_name(gen.model_key),
        "model_key": gen.model_key,
        "lora": (meta.get("lora") or {}).get("name"),
        "provider": gen.provider,
        "provider_name": provider_name(gen.provider),
        "gpu": meta.get("connection_name"),
        "status": status,
        "queue_position": progress.get("queue_position"),
        "progress": percent,
        "created_at": iso(gen.created_at),
        "started_at": iso(gen.started_at),
        "completed_at": iso(gen.completed_at),
        "elapsed_seconds": _elapsed(gen.started_at, gen.completed_at),
        "cost": None,
        "error": gen.error_message,
        "link": f"/library/history?open={gen.id}",
        "artifacts": len(gen.images) if "images" in gen.__dict__ else 0,
    }


def _training(v: LoraVersion, lora: Lora) -> dict[str, Any]:
    data = version_json(v)
    step, total = data.get("step"), data.get("total_steps")
    return {
        "id": str(v.id),
        "type": "training",
        "name": f"{lora.name} v{v.number}",
        "model": f"Krea 2 {lora.base_variant.upper()}" if lora.architecture == "krea2" else "",
        "model_key": f"{lora.architecture}-{lora.base_variant}",
        "lora": lora.name,
        "provider": v.gpu_provider,
        "provider_name": {"runpod": "RunPod", "modal": "Modal"}.get(v.gpu_provider or "", "—"),
        "gpu": f"{v.gpu_count}× {v.gpu_type}" if v.gpu_type else None,
        "status": TRAIN_STATUS.get(v.status, v.status),
        "stage": v.status,
        "stage_detail": v.stage_detail,
        "queue_position": None,
        "progress": round(100 * step / total) if step and total else None,
        "step": step,
        "total_steps": total,
        "created_at": iso(v.created_at),
        "started_at": iso(v.started_at),
        "completed_at": iso(v.completed_at),
        "elapsed_seconds": _elapsed(v.started_at, v.completed_at),
        "cost": data.get("cost_final") or data.get("cost_estimate"),
        "cost_is_final": data.get("cost_final") is not None,
        "error": v.error_message,
        "link": f"/loras/{lora.id}?version={v.id}",
        "artifacts": (1 if v.file_path else 0) + len(v.checkpoints or []),
    }


def _deployment(conn: ComfyConnection) -> dict[str, Any]:
    d = conn.deploy or {}
    status = {
        "starting": "preparing",
        "creating": "preparing",
        "booting": "preparing",
        "installing": "preparing",
        "ready": "running",
        "stopped": "completed",
        "error": "failed",
    }.get(d.get("stage") or "", "queued")
    started = datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None
    end = None if status in ("preparing", "running") else conn.updated_at
    return {
        "id": str(conn.id),
        "type": "deployment",
        "name": f"ComfyUI · {conn.name}",
        "model": "ComfyUI",
        "model_key": "comfyui",
        "lora": None,
        "provider": conn.gpu_provider,
        "provider_name": {"runpod": "RunPod", "modal": "Modal"}.get(conn.gpu_provider or "", "—"),
        "gpu": f"{d.get('gpu_count', 1)}× {d.get('gpu_type', '')}",
        "status": status,
        "stage": d.get("stage"),
        "queue_position": None,
        "progress": None,
        "created_at": iso(conn.created_at),
        "started_at": d.get("started_at"),
        "completed_at": None,
        "elapsed_seconds": _elapsed(started, end),
        "cost": run_cost(d.get("cost_per_hour"), started, end or datetime.now(UTC)),
        "cost_is_final": False,
        "error": conn.last_error,
        "link": "/generate?engine=comfy",
        "artifacts": 0,
    }


@router.get("")
async def list_jobs(
    type: Literal["all", "generation", "training", "deployment"] = "all",
    status: str | None = Query(default=None, max_length=100),
    provider: str | None = Query(default=None, max_length=32),
    model: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=100, ge=1, le=300),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    items: list[dict[str, Any]] = []
    if type in ("all", "generation"):
        gens = (
            await db.scalars(
                select(Generation)
                .where(Generation.user_id == user.id)
                .options(selectinload(Generation.tasks), selectinload(Generation.images))
                .order_by(Generation.created_at.desc())
                .limit(limit)
            )
        ).all()
        items += [_generation(g) for g in gens]
    if type in ("all", "training"):
        rows = (
            await db.execute(
                select(LoraVersion, Lora)
                .join(Lora, Lora.id == LoraVersion.lora_id)
                .where(LoraVersion.user_id == user.id, Lora.user_id == user.id)
                .where(LoraVersion.gpu_provider.is_not(None))
                .order_by(LoraVersion.created_at.desc())
                .limit(limit)
            )
        ).all()
        items += [_training(v, lo) for v, lo in rows]
    if type in ("all", "deployment"):
        conns = (
            await db.scalars(
                select(ComfyConnection).where(
                    ComfyConnection.user_id == user.id, ComfyConnection.auth_type == "agent"
                )
            )
        ).all()
        items += [_deployment(c) for c in conns]
    if status:
        wanted = set(status.split(","))
        items = [i for i in items if i["status"] in wanted]
    if provider:
        items = [i for i in items if i["provider"] == provider]
    if model:
        items = [i for i in items if i["model_key"] == model]
    items.sort(key=lambda i: i["created_at"] or "", reverse=True)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"items": items[:limit], "counts": counts}
