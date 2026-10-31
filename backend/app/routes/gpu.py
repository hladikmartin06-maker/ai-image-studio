"""GPU compute (Modal, RunPod): credentials, GPU catalog, workloads, endpoints and usage.

Anything that starts paid compute requires `confirm_billing: true`, which the UI sends only
after the user confirmed a dialog that says the provider will bill the account.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.gpu.base import GPUError, WorkloadSpec
from app.gpu.modal_provider import ModalProvider
from app.gpu.registry import GPU_PROVIDERS, get_gpu_adapter
from app.models import GPUIntegration, GPUWorkload, User
from app.security import decrypt_secret, encrypt_secret, mask_key
from app.services.serializers import iso

router = APIRouter(prefix="/api/gpu", tags=["gpu"])

_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:@-]{1,250}$", re.IGNORECASE)


class CredentialsBody(BaseModel):
    api_key: str | None = Field(default=None, max_length=300)
    token_id: str | None = Field(default=None, max_length=200)
    token_secret: str | None = Field(default=None, max_length=300)


class WorkloadBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    image: str = Field(min_length=3, max_length=250)
    gpu_type: str = Field(min_length=1, max_length=80)
    gpu_count: int = Field(default=1, ge=1, le=8)
    ports: list[int] = Field(default_factory=list, max_length=8)
    env: dict[str, str] = Field(default_factory=dict)
    cloud_type: str = Field(default="SECURE", pattern="^(SECURE|COMMUNITY)$")
    container_disk_gb: int = Field(default=20, ge=5, le=500)
    volume_gb: int = Field(default=0, ge=0, le=2000)
    command: list[str] = Field(default_factory=list, max_length=40)
    timeout_minutes: int = Field(default=60, ge=1, le=1440)
    confirm_billing: bool = False

    @field_validator("image")
    @classmethod
    def _image(cls, value: str) -> str:
        if not _IMAGE.match(value.strip()):
            raise ValueError("Enter a container image such as repository/name:tag")
        return value.strip()

    @field_validator("ports")
    @classmethod
    def _ports(cls, value: list[int]) -> list[int]:
        if any(p < 1 or p > 65535 for p in value):
            raise ValueError("Ports must be between 1 and 65535")
        return sorted(set(value))

    @field_validator("env")
    @classmethod
    def _env(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 30 or any(not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,63}", k) for k in value):
            raise ValueError("Environment variable names: A-Z, 0-9 and _ (max 30)")
        return value


class ConfirmBody(BaseModel):
    confirm_billing: bool = False


class ScaleBody(BaseModel):
    workers_min: int = Field(ge=0, le=100)
    workers_max: int = Field(ge=0, le=100)
    confirm_billing: bool = False


def _assoc(user: User, provider: str) -> str:
    return f"{user.id}:gpu:{provider}"


def _spec(provider: str) -> dict:
    spec = GPU_PROVIDERS.get(provider)
    if spec is None:
        raise api_error(404, "unknown_provider", "Unknown GPU provider.")
    return spec


def _fail(exc: GPUError):
    status = {"invalid_credentials": 400, "not_found": 404, "unsupported": 400}.get(exc.code, 502)
    if exc.code in ("invalid_request", "insufficient_credits", "rate_limited"):
        status = 400 if exc.code != "rate_limited" else 429
    return api_error(status, exc.code, exc.message)


async def _row(db: AsyncSession, user: User, provider: str) -> GPUIntegration | None:
    return await db.scalar(
        select(GPUIntegration).where(
            GPUIntegration.user_id == user.id, GPUIntegration.provider == provider
        )
    )


async def _creds(db: AsyncSession, user: User, provider: str) -> dict[str, str]:
    _spec(provider)
    row = await _row(db, user, provider)
    if row is None:
        raise api_error(409, "not_connected", "Connect this GPU provider in Integrations first.")
    return json.loads(decrypt_secret(row.encrypted_credentials, associated=_assoc(user, provider)))


def _provider_json(key: str, row: GPUIntegration | None) -> dict:
    spec = GPU_PROVIDERS[key]
    return {
        **spec,
        "status": "not_connected" if row is None else row.status,
        "masked_key": mask_key(row.key_last4) if row else None,
        "last_error": row.last_error if row else None,
        "last_checked_at": iso(row.last_checked_at) if row else None,
    }


@router.get("/providers")
async def list_providers(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = {
        r.provider: r
        for r in (await db.scalars(select(GPUIntegration).where(GPUIntegration.user_id == user.id)))
    }
    return {"providers": [_provider_json(k, rows.get(k)) for k in GPU_PROVIDERS]}


@router.put("/providers/{provider}")
async def save_credentials(
    provider: str,
    body: CredentialsBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    spec = _spec(provider)
    creds: dict[str, str] = {}
    for field in spec["credential_fields"]:
        value = (getattr(body, field) or "").strip()
        if len(value) < 6 or any(c.isspace() for c in value):
            raise api_error(400, "invalid_key", f"Enter a valid {field.replace('_', ' ')}.")
        creds[field] = value
    try:
        await get_gpu_adapter(provider).check(creds)
    except GPUError as exc:
        raise _fail(exc) from exc
    row = await _row(db, user, provider)
    if row is None:
        row = GPUIntegration(user_id=user.id, provider=provider)
        db.add(row)
    secret_part = creds.get("api_key") or creds.get("token_secret") or ""
    row.encrypted_credentials = encrypt_secret(json.dumps(creds), associated=_assoc(user, provider))
    row.key_last4 = secret_part[-4:]
    row.status = "connected"
    row.last_error = None
    row.last_checked_at = datetime.now(UTC)
    await db.commit()
    return _provider_json(provider, row)


@router.post("/providers/{provider}/test")
async def test_credentials(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    creds = await _creds(db, user, provider)
    row = await _row(db, user, provider)
    assert row is not None
    ok, message = True, None
    try:
        await get_gpu_adapter(provider).check(creds)
    except GPUError as exc:
        ok, message = False, exc.message
    row.status = "connected" if ok else "error"
    row.last_error = message
    row.last_checked_at = datetime.now(UTC)
    await db.commit()
    return {"ok": ok, "message": message, "provider": _provider_json(provider, row)}


@router.delete("/providers/{provider}")
async def disconnect(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _spec(provider)
    row = await _row(db, user, provider)
    if row is not None:
        await db.delete(row)
        await db.commit()
    return _provider_json(provider, None)


@router.get("/{provider}/gpu-types")
async def gpu_types(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    creds = await _creds(db, user, provider)
    try:
        items = await get_gpu_adapter(provider).gpu_types(creds)
    except GPUError as exc:
        raise _fail(exc) from exc
    return {"items": [g.json() for g in items]}


async def _records(db: AsyncSession, user: User, provider: str) -> dict[str, GPUWorkload]:
    rows = (
        await db.scalars(
            select(GPUWorkload).where(
                GPUWorkload.user_id == user.id, GPUWorkload.provider == provider
            )
        )
    ).all()
    return {r.external_id: r for r in rows}


@router.get("/{provider}/workloads")
async def list_workloads(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    creds = await _creds(db, user, provider)
    records = await _records(db, user, provider)
    adapter = get_gpu_adapter(provider)
    try:
        if isinstance(adapter, ModalProvider):
            info = {
                k: {
                    "name": r.name,
                    "gpu_type": r.gpu_type,
                    "gpu_count": r.gpu_count,
                    "image": r.image,
                    "ports": (r.meta or {}).get("ports", []),
                    "created_at": iso(r.created_at),
                }
                for k, r in records.items()
            }
            items = await adapter.list_workloads(creds, info)
        else:
            items = await adapter.list_workloads(creds)
    except GPUError as exc:
        raise _fail(exc) from exc
    # Keep the local history in step with the provider.
    live = {w.id: w for w in items}
    now = datetime.now(UTC)
    for external_id, record in records.items():
        workload = live.get(external_id)
        status = workload.status if workload else "terminated"
        if record.status != status:
            record.status = status
            if status in ("stopped", "exited", "terminated") and record.stopped_at is None:
                record.stopped_at = now
            if status == "running":
                record.stopped_at = None
    await db.commit()
    return {
        "items": [{**w.json(), "from_studio": w.id in records} for w in items],
        "supports_stop_start": GPU_PROVIDERS[provider]["supports_stop_start"],
    }


@router.post("/{provider}/workloads", status_code=201)
async def create_workload(
    provider: str,
    body: WorkloadBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    if not body.confirm_billing:
        raise api_error(400, "confirmation_required", "Confirm that the provider will bill you.")
    creds = await _creds(db, user, provider)
    spec = WorkloadSpec(
        name=body.name.strip(),
        image=body.image,
        gpu_type=body.gpu_type,
        gpu_count=body.gpu_count,
        ports=body.ports,
        env=body.env,
        cloud_type=body.cloud_type,  # type: ignore[arg-type]
        container_disk_gb=body.container_disk_gb,
        volume_gb=body.volume_gb,
        command=body.command,
        timeout_minutes=body.timeout_minutes,
    )
    try:
        workload = await get_gpu_adapter(provider).create_workload(creds, spec)
    except GPUError as exc:
        raise _fail(exc) from exc
    db.add(
        GPUWorkload(
            user_id=user.id,
            provider=provider,
            external_id=workload.id,
            kind=workload.kind,
            name=spec.name,
            gpu_type=spec.gpu_type,
            gpu_count=spec.gpu_count,
            image=spec.image,
            status=workload.status,
            meta={"ports": spec.ports},
        )
    )
    await db.commit()
    return workload.json()


async def _action(provider: str, workload_id: str, action: str, user: User, db: AsyncSession):
    creds = await _creds(db, user, provider)
    adapter = get_gpu_adapter(provider)
    try:
        await getattr(adapter, action)(creds, workload_id)
    except GPUError as exc:
        raise _fail(exc) from exc
    records = await _records(db, user, provider)
    record = records.get(workload_id)
    if record is not None:
        now = datetime.now(UTC)
        record.status = {"start": "running", "stop": "stopped", "terminate": "terminated"}[action]
        record.stopped_at = None if action == "start" else (record.stopped_at or now)
        await db.commit()
    return {"ok": True}


@router.post("/{provider}/workloads/{workload_id}/start")
async def start_workload(
    provider: str,
    workload_id: str,
    body: ConfirmBody | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    if body is None or not body.confirm_billing:
        raise api_error(400, "confirmation_required", "Confirm that the provider will bill you.")
    return await _action(provider, workload_id, "start", user, db)


@router.post("/{provider}/workloads/{workload_id}/stop")
async def stop_workload(
    provider: str,
    workload_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _action(provider, workload_id, "stop", user, db)


@router.delete("/{provider}/workloads/{workload_id}")
async def terminate_workload(
    provider: str,
    workload_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _action(provider, workload_id, "terminate", user, db)


@router.get("/{provider}/endpoints")
async def list_endpoints(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    creds = await _creds(db, user, provider)
    try:
        items = await get_gpu_adapter(provider).list_endpoints(creds)
    except GPUError as exc:
        raise _fail(exc) from exc
    return {
        "items": [e.json() for e in items],
        "supported": GPU_PROVIDERS[provider]["supports_endpoints"],
    }


@router.patch("/{provider}/endpoints/{endpoint_id}")
async def scale_endpoint(
    provider: str,
    endpoint_id: str,
    body: ScaleBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.workers_min > body.workers_max:
        raise api_error(400, "invalid_request", "Minimum workers cannot exceed the maximum.")
    if body.workers_min > 0 and not body.confirm_billing:
        raise api_error(
            400, "confirmation_required", "Always-on workers are billed even without requests."
        )
    creds = await _creds(db, user, provider)
    try:
        await get_gpu_adapter(provider).scale_endpoint(
            creds, endpoint_id, body.workers_min, body.workers_max
        )
    except GPUError as exc:
        raise _fail(exc) from exc
    return {"ok": True}


@router.delete("/{provider}/endpoints/{endpoint_id}")
async def delete_endpoint(
    provider: str,
    endpoint_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    creds = await _creds(db, user, provider)
    try:
        await get_gpu_adapter(provider).delete_endpoint(creds, endpoint_id)
    except GPUError as exc:
        raise _fail(exc) from exc
    return {"ok": True}


@router.get("/{provider}/usage")
async def usage(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    creds = await _creds(db, user, provider)
    try:
        account = await get_gpu_adapter(provider).account(creds)
    except GPUError as exc:
        account = {"balance": None, "spend_per_hour": None, "source": None, "error": exc.message}
    records = list((await _records(db, user, provider)).values())
    now = datetime.now(UTC)
    gpu_hours = 0.0
    for r in records:
        end = r.stopped_at or (now if r.status in ("running", "starting") else r.created_at)
        gpu_hours += max(0.0, (end - r.created_at).total_seconds() / 3600) * r.gpu_count
    return {
        **account,
        "studio_workloads": len(records),
        "studio_running": sum(1 for r in records if r.status in ("running", "starting")),
        "studio_gpu_hours": round(gpu_hours, 2),
        "recent": [
            {
                "id": r.external_id,
                "name": r.name,
                "gpu_type": r.gpu_type,
                "gpu_count": r.gpu_count,
                "status": r.status,
                "created_at": iso(r.created_at),
                "stopped_at": iso(r.stopped_at),
            }
            for r in sorted(records, key=lambda r: r.created_at, reverse=True)[:20]
        ],
    }
