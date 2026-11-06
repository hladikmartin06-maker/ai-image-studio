"""ComfyUI: connections (local, remote, on RunPod / Modal), deployment by the studio,
model files, workflows, Krea 2 generation."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.comfy import deploy, enhance, krea2, workflows
from app.comfy.client import ComfyError
from app.comfy.generation import (
    ComfyGenerateBody,
    EnhanceBody,
    create_comfy_generation,
    create_enhance_generation,
)
from app.comfy.service import (
    agent_for,
    assoc,
    check_url,
    comfy_subdir,
    file_capability,
    inspect,
    owned,
)
from app.db import get_db
from app.deps import api_error, current_user
from app.gpu import service as gpu_service
from app.gpu.agent_client import AgentError
from app.gpu.base import GPUError
from app.gpu.registry import GPU_PROVIDERS, get_gpu_adapter
from app.lora.safetensors_meta import InvalidSafetensors, read_header
from app.models import ComfyConnection, ComfyWorkflow, User
from app.security import encrypt_secret, mask_key
from app.services import uploads
from app.services.generation import GenerationRejected
from app.services.serializers import generation_json, iso

router = APIRouter(prefix="/api/comfy", tags=["comfyui"])

MODEL_FOLDERS = ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints")


def conn_json(conn: ComfyConnection) -> dict[str, Any]:
    d = dict(conn.deploy or {})
    d.pop("agent_token", None)
    info = dict(conn.info or {})
    info.pop("node_names", None)
    return {
        "id": str(conn.id),
        "name": conn.name,
        "kind": conn.kind,
        "base_url": conn.base_url if conn.auth_type != "agent" else d.get("agent_url"),
        "auth_type": conn.auth_type,
        "masked_secret": mask_key(conn.secret_last4)
        if conn.secret_last4 and conn.auth_type != "agent"
        else None,
        "models_dir": conn.models_dir,
        "gpu_provider": conn.gpu_provider,
        "workload_id": conn.workload_id,
        "status": conn.status,
        "last_error": conn.last_error,
        "last_checked_at": iso(conn.last_checked_at),
        "info": info,
        "deploy": d,
        # Spent so far and time left in the current session (estimate).
        "session": deploy.session_estimate(d) if conn.auth_type == "agent" else None,
        "file_capability": file_capability(conn),
        "created_at": iso(conn.created_at),
    }


def _fail(exc: ComfyError):
    status = {"unauthorized": 400, "not_found": 404, "unsupported": 400, "in_progress": 409}.get(
        exc.code, 502
    )
    if exc.code in ("invalid_workflow", "missing_models"):
        status = 400
    return api_error(status, exc.code, exc.message)


def _gpu_fail(exc: GPUError):
    status = 400 if exc.code in ("invalid_credentials", "invalid_request") else 502
    if exc.code in ("unavailable", "price_changed"):
        status = 409
    return api_error(status, exc.code, exc.message)


async def _owned(db: AsyncSession, user: User, conn_id: uuid.UUID) -> ComfyConnection:
    conn = await owned(db, user.id, conn_id)
    if conn is None:
        raise api_error(404, "not_found", "ComfyUI connection not found.")
    return conn


async def _test(db: AsyncSession, conn: ComfyConnection) -> None:
    try:
        conn.info = await inspect(conn)
        conn.status = "ready"
        conn.last_error = None
    except ComfyError as exc:
        conn.status = "error" if conn.status != "deploying" else conn.status
        conn.last_error = exc.message
    conn.last_checked_at = datetime.now(UTC)
    await db.commit()


@router.get("/catalog")
async def catalog(_: User = Depends(current_user)):
    return {
        "krea2": krea2.catalog_json(),
        "enhance": enhance.catalog_json(),
        "deploy_images": deploy.DEFAULT_IMAGES,
        "model_folders": list(MODEL_FOLDERS),
    }


# --- connections ----------------------------------------------------------------------------


class ConnectionBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["local", "remote", "runpod", "modal"]
    base_url: str = Field(min_length=1, max_length=400)
    auth_type: Literal["none", "bearer", "basic"] = "none"
    secret: str | None = Field(default=None, max_length=500)
    models_dir: str | None = Field(default=None, max_length=500)
    workload_id: str | None = Field(default=None, max_length=120)


class ConnectionPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    base_url: str | None = Field(default=None, max_length=400)
    auth_type: Literal["none", "bearer", "basic"] | None = None
    secret: str | None = Field(default=None, max_length=500)
    models_dir: str | None = Field(default=None, max_length=500)
    auto_stop_minutes: int | None = Field(default=None, ge=0, le=1440)


def _set_secret(conn: ComfyConnection, auth_type: str, secret: str | None) -> None:
    if auth_type == "none":
        conn.encrypted_secret = None
        conn.secret_last4 = None
        return
    if not secret:
        if conn.encrypted_secret:
            return
        raise ValueError("Enter the token or user:password.")
    if auth_type == "basic" and ":" not in secret:
        raise ValueError("Basic authentication needs user:password.")
    conn.encrypted_secret = encrypt_secret(secret.strip(), associated=assoc(conn))
    conn.secret_last4 = secret.strip()[-4:]


@router.get("/connections")
async def list_connections(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = (
        await db.scalars(
            select(ComfyConnection)
            .where(ComfyConnection.user_id == user.id)
            .order_by(ComfyConnection.created_at)
        )
    ).all()
    return {"items": [conn_json(c) for c in rows]}


@router.post("/connections", status_code=201)
async def add_connection(
    body: ConnectionBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    try:
        url = check_url(body.base_url)
    except ValueError as exc:
        raise api_error(400, "invalid_url", str(exc)) from exc
    conn = ComfyConnection(
        id=uuid.uuid4(),
        user_id=user.id,
        name=body.name.strip(),
        kind=body.kind,
        base_url=url,
        auth_type=body.auth_type,
        models_dir=(body.models_dir or "").strip() or None if body.kind == "local" else None,
        gpu_provider=body.kind if body.kind in ("runpod", "modal") else None,
        workload_id=body.workload_id,
        status="new",
        info={},
        deploy={},
    )
    try:
        _set_secret(conn, body.auth_type, body.secret)
    except ValueError as exc:
        raise api_error(400, "invalid_parameter", str(exc)) from exc
    db.add(conn)
    await db.commit()
    await _test(db, conn)
    return conn_json(conn)


@router.patch("/connections/{conn_id}")
async def update_connection(
    conn_id: uuid.UUID,
    body: ConnectionPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conn = await _owned(db, user, conn_id)
    if body.name:
        conn.name = body.name.strip()
    if conn.auth_type != "agent":
        if body.base_url:
            try:
                conn.base_url = check_url(body.base_url)
            except ValueError as exc:
                raise api_error(400, "invalid_url", str(exc)) from exc
        if body.auth_type:
            try:
                _set_secret(conn, body.auth_type, body.secret)
            except ValueError as exc:
                raise api_error(400, "invalid_parameter", str(exc)) from exc
            conn.auth_type = body.auth_type
        if body.models_dir is not None and conn.kind == "local":
            conn.models_dir = body.models_dir.strip() or None
    if body.auto_stop_minutes is not None and conn.auth_type == "agent":
        conn.deploy = {**(conn.deploy or {}), "auto_stop_minutes": body.auto_stop_minutes or None}
    await db.commit()
    if conn.auth_type != "agent":
        await _test(db, conn)
    return conn_json(conn)


@router.post("/connections/{conn_id}/test")
async def test_connection(
    conn_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    conn = await _owned(db, user, conn_id)
    if conn.auth_type == "agent" and conn.status in ("deploying", "stopped"):
        return conn_json(conn)
    await _test(db, conn)
    return conn_json(conn)


@router.delete("/connections/{conn_id}")
async def remove_connection(
    conn_id: uuid.UUID,
    terminate: bool = Query(default=False),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conn = await _owned(db, user, conn_id)
    if terminate and conn.auth_type == "agent":
        try:
            await deploy.terminate(db, conn)
        except GPUError as exc:
            raise _gpu_fail(exc) from exc
    await db.delete(conn)
    await db.commit()
    return {"ok": True}


# --- deployment on RunPod / Modal ---------------------------------------------------------


class DeployBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    provider: Literal["runpod", "modal"]
    gpu_type: str = Field(min_length=1, max_length=80)
    gpu_count: int = Field(default=1, ge=1, le=8)
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    # Small enough that many machines fit (ComfyUI and PyTorch are in the image; the volume
    # holds ComfyUI's copy, Krea 2 Turbo and RAW and LoRAs).
    volume_gb: int = Field(default=80, ge=40, le=2000)
    container_disk_gb: int = Field(default=30, ge=20, le=500)
    auto_stop_minutes: int | None = Field(default=30, ge=0, le=1440)
    # The session: the GPU stops after this long, or once the estimated spending reaches the
    # limit (USD), whichever comes first.
    session_minutes: int = Field(default=120, ge=15, le=1440)
    spend_limit: float | None = Field(default=None, gt=0, le=10000)
    # The hourly price the owner confirmed; nothing starts at another price.
    confirmed_price: float | None = Field(default=None, ge=0)
    confirm_billing: bool = False


@router.post("/deployments", status_code=201)
async def deploy_comfyui(
    body: DeployBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    if not body.confirm_billing:
        raise api_error(400, "confirmation_required", "Confirm that the provider will bill you.")
    conn = ComfyConnection(
        id=uuid.uuid4(),
        user_id=user.id,
        name=body.name.strip(),
        kind=body.provider,
        base_url="",
        auth_type="agent",
        gpu_provider=body.provider,
        status="deploying",
        info={},
        deploy={
            "provider": body.provider,
            "gpu_type": body.gpu_type,
            "gpu_count": body.gpu_count,
            "cloud_type": body.cloud_type,
            "volume_gb": body.volume_gb,
            "container_disk_gb": body.container_disk_gb,
            "auto_stop_minutes": body.auto_stop_minutes or None,
            "session_minutes": body.session_minutes,
            "spend_limit": body.spend_limit,
            "confirmed_price": body.confirmed_price,
        },
    )
    if deploy.creates_in_background(conn):
        # Created in the background (Modal): the price is checked now, while the owner waits.
        try:
            await deploy.check_price(db, conn)
        except GPUError as exc:
            raise _gpu_fail(exc) from exc
    db.add(conn)
    await db.commit()
    if deploy.creates_in_background(conn):
        deploy.mark_creating(conn)
        await db.commit()
        deploy.create_later(conn.id)
        return conn_json(conn)
    try:
        await deploy.launch(db, conn)
    except GPUError as exc:
        await db.delete(conn)
        await db.commit()
        raise _gpu_fail(exc) from exc
    return conn_json(conn)


class ConfirmBody(BaseModel):
    confirm_billing: bool = False


@router.post("/connections/{conn_id}/start")
async def start_deployment(
    conn_id: uuid.UUID,
    body: ConfirmBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conn = await _owned(db, user, conn_id)
    if conn.auth_type != "agent":
        raise api_error(400, "unsupported", "Only ComfyUI deployed by the studio can be started.")
    if not body.confirm_billing:
        raise api_error(400, "confirmation_required", "Confirm that the provider will bill you.")
    if deploy.outdated(conn):
        raise api_error(409, "outdated", deploy.OUTDATED)
    try:
        await deploy.start(db, conn)
    except GPUError as exc:
        raise _gpu_fail(exc) from exc
    return conn_json(conn)


@router.post("/connections/{conn_id}/stop")
async def stop_deployment(
    conn_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    conn = await _owned(db, user, conn_id)
    if conn.auth_type != "agent" or not conn.workload_id:
        raise api_error(400, "unsupported", "Only ComfyUI deployed by the studio can be stopped.")
    try:
        await deploy.stop(db, conn)
    except GPUError as exc:
        raise _gpu_fail(exc) from exc
    return conn_json(conn)


@router.get("/workloads")
async def pick_workloads(
    provider: Literal["runpod", "modal"] = Query(),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Running pods / sandboxes with exposed ports, to connect an existing ComfyUI."""
    try:
        creds = await gpu_service.load_creds(db, user.id, provider)
        items = await get_gpu_adapter(provider).list_workloads(creds)
    except GPUError as exc:
        raise _gpu_fail(exc) from exc
    return {
        "items": [w.json() for w in items if w.status == "running" and w.urls],
        "provider": GPU_PROVIDERS[provider]["name"],
    }


@router.get("/connections/{conn_id}/logs")
async def deployment_logs(
    conn_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    conn = await _owned(db, user, conn_id)
    return {"log": (conn.deploy or {}).get("log_tail", "")}


# --- model files ---------------------------------------------------------------------------


class DownloadBody(BaseModel):
    url: str = Field(min_length=10, max_length=1000)
    folder: Literal["diffusion_models", "text_encoders", "vae", "loras", "checkpoints"]
    filename: str = Field(min_length=3, max_length=200)


def _copy_into(models_dir: str, folder: str, name: str, source: Path) -> None:
    dest = Path(models_dir).expanduser() / "models" / folder / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _safe_name(name: str) -> str:
    clean = Path(name).name
    if not re.fullmatch(r"[\w.\- ]+\.(safetensors|sft|ckpt|pt|gguf)", clean):
        raise api_error(400, "invalid_parameter", "Use a model file name such as name.safetensors.")
    return clean


@router.post("/connections/{conn_id}/models/download")
async def download_model(
    conn_id: uuid.UUID,
    body: DownloadBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Downloads a model file directly on the GPU from Hugging Face (the token from
    Integrations › Hugging Face is used for gated files)."""
    conn = await _owned(db, user, conn_id)
    name = _safe_name(body.filename)
    if not re.match(r"https://huggingface\.co/", body.url):
        raise api_error(
            400, "invalid_url", "Use a Hugging Face link (https://huggingface.co/…/resolve/…)."
        )
    try:
        job = await deploy.agent_job(
            conn, "download", {"url": body.url, "folder": body.folder, "filename": name}
        )
    except ComfyError as exc:
        raise _fail(exc) from exc
    return {"job": job}


class InstallBody(BaseModel):
    # A Krea 2 variant, "pgupscaler" for Enhance photo or "anypaint" for Advanced.
    variant: Literal["turbo", "raw", "pgupscaler", "anypaint", "pose", "skin", "seedvr2"]
    # One of the variant's verified files (e.g. fp8 or int8); default the first.
    filename: str | None = Field(default=None, max_length=200)


@router.post("/connections/{conn_id}/models/install")
async def install_model(
    conn_id: uuid.UUID,
    body: InstallBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Downloads a Krea 2 model with its text encoder and VAE on the rented GPU (files
    already there are skipped); progress appears in deploy.model_job."""
    conn = await _owned(db, user, conn_id)
    try:
        await deploy.install_model(db, conn, body.variant, body.filename)
    except ComfyError as exc:
        raise _fail(exc) from exc
    return conn_json(conn)


@router.get("/connections/{conn_id}/jobs/{job_id}")
async def agent_job_status(
    conn_id: uuid.UUID,
    job_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conn = await _owned(db, user, conn_id)
    agent = agent_for(conn)
    if agent is None or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job_id):
        raise api_error(404, "not_found", "Unknown job.")
    try:
        job = await agent.job(job_id)
        text, _ = await agent.logs(job_id, max(0, int(job.get("log_size") or 0) - 4000))
    except AgentError as exc:
        raise api_error(502, exc.code, exc.message) from exc
    return {"job": job, "log": text}


class UploadModelBody(BaseModel):
    upload_id: str = Field(min_length=32, max_length=32)
    folder: Literal["diffusion_models", "text_encoders", "vae", "loras", "checkpoints"]


@router.post("/connections/{conn_id}/models/upload")
async def upload_model(
    conn_id: uuid.UUID,
    body: UploadModelBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Sends a file uploaded from this Mac (chunked) into the ComfyUI models folder."""
    conn = await _owned(db, user, conn_id)
    try:
        path, filename = await uploads.take(user.id, body.upload_id)
    except uploads.UploadError as exc:
        raise api_error(400, "invalid_upload", str(exc)) from exc
    name = _safe_name(filename)
    if name.endswith(".safetensors"):
        try:
            await asyncio.to_thread(read_header, path)
        except InvalidSafetensors as exc:
            await uploads.discard(user.id, body.upload_id)
            raise api_error(400, "invalid_file", str(exc)) from exc
    capability = file_capability(conn)
    try:
        if capability == "agent":
            agent = agent_for(conn)
            assert agent is not None
            await agent.upload_file(f"{comfy_subdir(conn)}/models/{body.folder}/{name}", path)
        elif capability == "folder":
            await asyncio.to_thread(_copy_into, conn.models_dir or "", body.folder, name, path)
        else:
            raise api_error(
                400,
                "unsupported",
                "This ComfyUI has no file upload API. Copy the file into its models folder.",
            )
    except AgentError as exc:
        raise api_error(502, exc.code, f"Upload to the GPU failed: {exc.message}") from exc
    finally:
        await uploads.discard(user.id, body.upload_id)
    await _test(db, conn)
    return conn_json(conn)


# --- workflows -------------------------------------------------------------------------------


def workflow_json(row: ComfyWorkflow) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "name": row.name,
        "family": row.family,
        "has_api": row.api_json is not None,
        "has_ui": row.ui_json is not None,
        "bindings": row.bindings,
        "node_types": row.node_types,
        "runnable": row.api_json is not None
        and "sampler" in (row.bindings or {})
        and "positive" in (row.bindings or {}),
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


class WorkflowBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    family: Literal["krea2"] | None = "krea2"
    data: dict[str, Any]


@router.get("/workflows")
async def list_workflows(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = (
        await db.scalars(
            select(ComfyWorkflow)
            .where(ComfyWorkflow.user_id == user.id)
            .order_by(ComfyWorkflow.updated_at.desc())
        )
    ).all()
    return {"items": [workflow_json(r) for r in rows]}


@router.post("/workflows", status_code=201)
async def import_workflow(
    body: WorkflowBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    if len(json.dumps(body.data)) > 5_000_000:
        raise api_error(413, "too_large", "Workflow files up to 5 MB.")
    try:
        kind = workflows.kind_of(body.data)
    except workflows.WorkflowError as exc:
        raise api_error(400, "invalid_workflow", str(exc)) from exc
    clean, removed = workflows.scrub(body.data)
    row = ComfyWorkflow(
        user_id=user.id,
        name=body.name.strip(),
        family=body.family,
        api_json=clean if kind == "api" else None,
        ui_json=clean if kind == "ui" else None,
        bindings=workflows.detect_bindings(clean) if kind == "api" else {},
        node_types=workflows.node_types(clean) if kind == "api" else [],
    )
    db.add(row)
    await db.commit()
    return {**workflow_json(row), "format": kind, "removed_secrets": removed}


class WorkflowPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    family: Literal["krea2"] | None = None
    api_data: dict[str, Any] | None = None


@router.patch("/workflows/{wf_id}")
async def update_workflow(
    wf_id: uuid.UUID,
    body: WorkflowPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.scalar(
        select(ComfyWorkflow).where(ComfyWorkflow.id == wf_id, ComfyWorkflow.user_id == user.id)
    )
    if row is None:
        raise api_error(404, "not_found", "Workflow not found.")
    if body.name:
        row.name = body.name.strip()
    if "family" in body.model_fields_set:
        row.family = body.family
    if body.api_data is not None:
        try:
            if workflows.kind_of(body.api_data) != "api":
                raise workflows.WorkflowError("Upload the API-format export here.")
        except workflows.WorkflowError as exc:
            raise api_error(400, "invalid_workflow", str(exc)) from exc
        clean, _ = workflows.scrub(body.api_data)
        row.api_json = clean
        row.bindings = workflows.detect_bindings(clean)
        row.node_types = workflows.node_types(clean)
    await db.commit()
    return workflow_json(row)


@router.delete("/workflows/{wf_id}")
async def delete_workflow(
    wf_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    await db.execute(
        delete(ComfyWorkflow).where(ComfyWorkflow.id == wf_id, ComfyWorkflow.user_id == user.id)
    )
    await db.commit()
    return {"ok": True}


@router.get("/workflows/{wf_id}/export")
async def export_workflow(
    wf_id: str,
    format: Literal["api", "ui"] = "api",
    variant: Literal["turbo", "raw"] = "turbo",
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    if wf_id == "builtin-krea2":
        spec = krea2.VARIANTS[variant]
        width, height = krea2.size_for("1:1", "1K")
        data: dict[str, Any] | None = krea2.build_workflow(
            variant=variant,
            unet=spec.known_files[0].filename if spec.known_files else "krea2_raw.safetensors",
            text_encoder=krea2.TEXT_ENCODER.filename,
            vae=krea2.VAE.filename,
            prompt="a portrait photo, soft window light",
            negative_prompt="",
            width=width,
            height=height,
            seed=1,
            steps=spec.steps,
            cfg=spec.cfg,
            batch=1,
            loras=[],
        )
        name = f"krea2-{variant}-api"
    else:
        try:
            uid = uuid.UUID(wf_id)
        except ValueError as exc:
            raise api_error(404, "not_found", "Workflow not found.") from exc
        row = await db.scalar(
            select(ComfyWorkflow).where(ComfyWorkflow.id == uid, ComfyWorkflow.user_id == user.id)
        )
        if row is None:
            raise api_error(404, "not_found", "Workflow not found.")
        data = row.api_json if format == "api" else row.ui_json
        name = re.sub(r"[^a-z0-9]+", "-", row.name.lower()).strip("-") + f"-{format}"
    if data is None:
        raise api_error(404, "not_found", f"No {format.upper()}-format version stored.")
    clean, _ = workflows.scrub(data)
    return JSONResponse(
        clean, headers={"Content-Disposition": f'attachment; filename="{name}.json"'}
    )


# --- generation ------------------------------------------------------------------------------


@router.post("/generate", status_code=201)
async def generate(
    body: ComfyGenerateBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    try:
        gen = await create_comfy_generation(db, user, body)
    except GenerationRejected as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
    return generation_json(gen)


@router.post("/enhance", status_code=201)
async def enhance_photo(
    body: EnhanceBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Enhance photo (PGUpscaler): fixes plastic AI skin and upscales one of the user's own
    photos on a ComfyUI connection."""
    try:
        gen = await create_enhance_generation(db, user, body)
    except GenerationRejected as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
    return generation_json(gen)
