"""Shared GPU helpers for the studio's own workflows (ComfyUI deployments, LoRA training):
credentials, workload records and cost of a run."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.gpu.base import GPUError, Workload, WorkloadSpec
from app.gpu.registry import GPU_PROVIDERS, get_gpu_adapter
from app.models import GPUIntegration, GPUWorkload
from app.security import decrypt_secret

# Driver CUDA versions the studio's images run on (allowedCudaVersions). RunPod matches them
# exactly against free machines, so only the ones its catalog lists as free are sent; these
# lists are used as they are only when the catalog cannot be read.
CUDA12 = ["12.8", "12.9"]
CUDA13 = ["13.0", "13.1", "13.2", "13.3"]


async def offer(
    db: AsyncSession, user_id: uuid.UUID, provider: str, gpu_type: str, cloud: str
) -> tuple[list[str] | None, float | None]:
    """RunPod's catalog for this GPU and cloud right now: the CUDA versions with a free
    machine and the list price per hour; (None, None) when unknown (Modal, or the catalog
    could not be read)."""
    adapter = get_gpu_adapter(provider)
    if not hasattr(adapter, "offer"):
        return None, None
    creds = await load_creds(db, user_id, provider)
    return await adapter.offer(creds, gpu_type, cloud)


def assoc(user_id: uuid.UUID, provider: str) -> str:
    return f"{user_id}:gpu:{provider}"


async def load_creds(db: AsyncSession, user_id: uuid.UUID, provider: str) -> dict[str, str]:
    if provider not in GPU_PROVIDERS:
        raise GPUError("invalid_request", "Unknown GPU provider")
    row = await db.scalar(
        select(GPUIntegration).where(
            GPUIntegration.user_id == user_id, GPUIntegration.provider == provider
        )
    )
    if row is None:
        raise GPUError("invalid_credentials", "Connect this GPU provider in Integrations first")
    return json.loads(
        decrypt_secret(row.encrypted_credentials, associated=assoc(user_id, provider))
    )


async def launch(
    db: AsyncSession,
    user_id: uuid.UUID,
    provider: str,
    spec: WorkloadSpec,
    purpose: str,
) -> Workload:
    """Creates the workload at the provider and records it in the GPU history."""
    creds = await load_creds(db, user_id, provider)
    workload = await get_gpu_adapter(provider).create_workload(creds, spec)
    # The workload exists and bills from here on: always record it (an id can only repeat
    # in test doubles, so an existing record is updated instead of failing).
    row = await db.scalar(
        select(GPUWorkload).where(
            GPUWorkload.provider == provider, GPUWorkload.external_id == workload.id
        )
    )
    if row is None:
        row = GPUWorkload(provider=provider, external_id=workload.id)
        db.add(row)
    row.user_id = user_id
    row.kind = workload.kind
    row.name = spec.name
    row.gpu_type = spec.gpu_type
    row.gpu_count = spec.gpu_count
    row.image = spec.image
    row.status = workload.status
    row.stopped_at = None
    row.meta = {"ports": spec.ports, "purpose": purpose}
    await db.commit()
    return workload


async def mark_workload(
    db: AsyncSession, user_id: uuid.UUID, provider: str, external_id: str, status: str, at: datetime
) -> None:
    row = await db.scalar(
        select(GPUWorkload).where(
            GPUWorkload.user_id == user_id,
            GPUWorkload.provider == provider,
            GPUWorkload.external_id == external_id,
        )
    )
    if row is not None:
        row.status = status
        if status in ("stopped", "exited", "terminated") and row.stopped_at is None:
            row.stopped_at = at
        await db.commit()


def run_cost(
    cost_per_hour: float | None, started: datetime | None, ended: datetime
) -> float | None:
    """Estimated cost from the provider's hourly rate and the time the GPU ran."""
    if cost_per_hour is None or started is None:
        return None
    hours = max(0.0, (ended - started).total_seconds() / 3600)
    return round(cost_per_hour * hours, 4)
