"""ComfyUI deployed by the studio on the user's RunPod or Modal account.

The workload runs the studio agent (gpu/agent). Stages, advanced by the GPU runner:
  starting   -> the container is booting; wait until the agent answers on its port
  installing -> agent job "comfyui": installs ComfyUI once (kept on the volume), PyTorch for
                the GPU's driver and ComfyUI's packages (every new container), runs ComfyUI
                on 127.0.0.1
  ready      -> ComfyUI answers through the agent (/comfy/*); the user then downloads a
                Krea 2 model (agent job "models", progress in deploy["model_job"]);
                optional auto-stop when idle
  stopped / error (a failed setup stops the GPU, so it is not billed)
ComfyUI itself is never exposed: only the token-protected agent port is.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from app.comfy import krea2
from app.comfy.client import ComfyError
from app.comfy.service import agent_for, assoc, client_for, inspect
from app.db import SessionLocal
from app.gpu import service as gpu_service
from app.gpu.agent import AGENT_PORT, BOOT, agent_env, new_token
from app.gpu.agent_client import AgentClient, AgentError
from app.gpu.base import GPUError, WorkloadSpec
from app.gpu.registry import get_gpu_adapter
from app.models import ComfyConnection, Generation, UserSecret
from app.security import decrypt_secret, encrypt_secret

log = logging.getLogger("studio.deploy")

# RunPod's PyTorch image with PyTorch 2.13 for CUDA 13.0 and Python 3.12 (runpod/containers,
# official-templates/pytorch). ComfyUI needs PyTorch 2.7+, cu130 recommended (its README).
# The setup script installs a matching PyTorch when the image's does not fit the driver.
DEFAULT_IMAGES = {
    "runpod": "runpod/pytorch:1.5.1-cu1300-torch2130-ubuntu2404",
    "modal": "nvidia/cuda:12.4.1-devel-ubuntu22.04",
}
# RunPod hosts whose driver supports CUDA 12.8 or newer (allowedCudaVersions).
RUNPOD_CUDA = ["13.0", "12.9", "12.8"]
BOOT_TIMEOUT = timedelta(minutes=30)
INSTALL_TIMEOUT = timedelta(minutes=120)


def now() -> datetime:
    return datetime.now(UTC)


def model_downloads(variants: list[str], filename: str | None = None) -> list[dict[str, str]]:
    """Text encoder, VAE and the diffusion model of each variant (the chosen file, else the
    first): only files whose download URL is verified; others are added by URL by the user."""
    if not variants:
        return []
    out = [
        {
            "url": krea2.TEXT_ENCODER.url,
            "folder": "text_encoders",
            "filename": krea2.TEXT_ENCODER.filename,
        },
        {"url": krea2.VAE.url, "folder": "vae", "filename": krea2.VAE.filename},
    ]
    for key in variants:
        known = krea2.VARIANTS[key].known_files
        if known:
            f = next((k for k in known if k.filename == filename), known[0])
            out.append({"url": f.url, "folder": f.folder, "filename": f.filename})
    return out


async def hf_token(db, user_id: uuid.UUID) -> str | None:
    row = await db.scalar(
        select(UserSecret).where(UserSecret.user_id == user_id, UserSecret.name == "huggingface")
    )
    if row is None:
        return None
    return decrypt_secret(row.encrypted_value, associated=f"{user_id}:secret:huggingface")


def spec_for(conn: ComfyConnection, token: str, hf: str | None) -> WorkloadSpec:
    d = conn.deploy or {}
    provider = d["provider"]
    secret_env = {"STUDIO_AGENT_TOKEN": token}
    if hf:
        secret_env["HF_TOKEN"] = hf
    env = agent_env(token)
    env.pop("STUDIO_AGENT_TOKEN")
    return WorkloadSpec(
        name=f"ais-comfyui-{str(conn.id)[:8]}",
        image=d.get("image") or DEFAULT_IMAGES[provider],
        gpu_type=d["gpu_type"],
        gpu_count=int(d.get("gpu_count") or 1),
        ports=[AGENT_PORT],
        env=env,
        secret_env=secret_env,
        cloud_type=d.get("cloud_type") or "SECURE",
        container_disk_gb=int(d.get("container_disk_gb") or 50),
        volume_gb=int(d.get("volume_gb") or 120) if provider == "runpod" else 0,
        command=["bash", "-c", BOOT],
        timeout_minutes=int(d.get("timeout_minutes") or 720),
        volume_name=f"ais-comfyui-{str(conn.id)[:12]}" if provider == "modal" else None,
        image_python="3.12" if provider == "modal" and not d.get("image") else None,
        image_apt=["git", "curl"] if provider == "modal" and not d.get("image") else [],
        allowed_cuda_versions=RUNPOD_CUDA if provider == "runpod" else [],
    )


async def launch(db, conn: ComfyConnection) -> None:
    """Creates the GPU workload (billing starts) and moves the connection to "starting"."""
    token = new_token()
    conn.encrypted_secret = encrypt_secret(token, associated=assoc(conn))
    conn.secret_last4 = token[-4:]
    hf = await hf_token(db, conn.user_id)
    spec = spec_for(conn, token, hf)
    workload = await gpu_service.launch(db, conn.user_id, conn.deploy["provider"], spec, "comfyui")
    conn.workload_id = workload.id
    conn.gpu_provider = conn.deploy["provider"]
    conn.status = "deploying"
    conn.last_error = None
    conn.deploy = {
        **conn.deploy,
        "stage": "starting",
        "agent_url": None,
        "job_id": None,
        "started_at": now().isoformat(),
        "cost_per_hour": workload.cost_per_hour,
        "log_tail": "",
    }
    await db.commit()


def _since(value: str | None) -> timedelta:
    if not value:
        return timedelta(0)
    return now() - datetime.fromisoformat(value)


async def _fail(db, conn: ComfyConnection, message: str, stop_gpu: bool = False) -> None:
    """A failed setup: with stop_gpu the GPU is stopped (RunPod; Modal: ended) so it is not
    billed while nothing runs on it. "Try again" starts it again."""
    if stop_gpu and conn.workload_id and conn.gpu_provider:
        try:
            creds = await gpu_service.load_creds(db, conn.user_id, conn.gpu_provider)
            adapter = get_gpu_adapter(conn.gpu_provider)
            if conn.gpu_provider == "runpod":
                await adapter.stop(creds, conn.workload_id)
                status = "stopped"
            else:
                await adapter.terminate(creds, conn.workload_id)
                status = "terminated"
            await gpu_service.mark_workload(
                db, conn.user_id, conn.gpu_provider, conn.workload_id, status, now()
            )
            message += " The GPU was stopped, so it is not billed."
        except GPUError as exc:
            log.warning("could not stop GPU after a failed setup: %s", exc.message)
    conn.status = "error"
    conn.last_error = message[:1000]
    conn.deploy = {**(conn.deploy or {}), "stage": "error"}
    await db.commit()


async def advance(conn_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        conn = await db.get(ComfyConnection, conn_id)
        if conn is None or conn.auth_type != "agent":
            return
        d = dict(conn.deploy or {})
        stage = d.get("stage")
        provider = conn.gpu_provider or d.get("provider")
        if stage in ("starting", "installing"):
            try:
                creds = await gpu_service.load_creds(db, conn.user_id, provider)
                adapter = get_gpu_adapter(provider)
                workload = await adapter.workload(creds, conn.workload_id)
            except GPUError as exc:
                if exc.code in ("invalid_credentials",):
                    await _fail(db, conn, exc.message)
                return
            if workload is None or workload.status in ("terminated", "exited", "stopped"):
                await _fail(db, conn, "The GPU workload stopped before ComfyUI was ready.")
                return
            if workload.cost_per_hour is not None:
                d["cost_per_hour"] = workload.cost_per_hour
        if stage == "starting":
            url = d.get("agent_url") or await adapter.port_url(creds, conn.workload_id, AGENT_PORT)
            if url:
                d["agent_url"] = url
                conn.deploy = dict(d)
                await db.commit()
                agent = agent_for(conn)
                assert agent is not None
                try:
                    await agent.health()
                except AgentError:
                    agent = None
                if agent is not None:
                    job_id = f"comfyui-{int(now().timestamp())}"
                    try:
                        await agent.start_job(
                            job_id, "comfyui", {"models": model_downloads(d.get("variants") or [])}
                        )
                    except AgentError as exc:
                        await _fail(
                            db, conn, f"Could not start ComfyUI: {exc.message}", stop_gpu=True
                        )
                        return
                    d.update(stage="installing", job_id=job_id, installing_at=now().isoformat())
                    conn.deploy = dict(d)
                    await db.commit()
                    return
            if _since(d.get("started_at")) > BOOT_TIMEOUT:
                await _fail(
                    db, conn, "The GPU container did not start within 30 minutes.", stop_gpu=True
                )
            else:
                conn.deploy = dict(d)
                await db.commit()
            return
        if stage == "installing":
            agent = agent_for(conn)
            if agent is None:
                await _fail(db, conn, "The agent address is missing.")
                return
            try:
                job = await agent.job(d["job_id"])
                text, _ = await agent.logs(
                    d["job_id"], max(0, int(job.get("log_size") or 0) - 6000)
                )
                d["log_tail"] = text[-6000:]
            except AgentError as exc:
                if exc.code == "not_found":  # the container restarted: start again
                    d.update(stage="starting", agent_url=None)
                conn.deploy = dict(d)
                await db.commit()
                return
            if job.get("status") in ("failed", "cancelled"):
                conn.deploy = dict(d)
                await _fail(db, conn, setup_error(d.get("log_tail") or ""), stop_gpu=True)
                return
            try:
                await client_for(conn).system_stats()
            except ComfyError:
                if _since(d.get("installing_at")) > INSTALL_TIMEOUT:
                    conn.deploy = dict(d)
                    await _fail(db, conn, "ComfyUI did not start within 2 hours.", stop_gpu=True)
                    return
                conn.deploy = dict(d)
                await db.commit()
                return
            try:
                conn.info = await inspect(conn)
            except ComfyError as exc:
                conn.deploy = dict(d)
                await _fail(db, conn, exc.message)
                return
            d.update(stage="ready", ready_at=now().isoformat(), last_activity=now().isoformat())
            conn.deploy = dict(d)
            conn.status = "ready"
            conn.last_error = None
            conn.last_checked_at = now()
            await db.commit()
            return
        if stage == "ready" and (d.get("model_job") or {}).get("status") == "running":
            await _advance_model_job(db, conn, d)
            return
        if stage == "ready" and d.get("auto_stop_minutes"):
            await _auto_stop(db, conn, d)


async def _auto_stop(db, conn: ComfyConnection, d: dict[str, Any]) -> None:
    active = await db.scalar(
        select(func.count(Generation.id)).where(
            Generation.user_id == conn.user_id,
            Generation.provider == "comfyui",
            Generation.status.in_(("queued", "processing")),
            Generation.comfy_meta["connection_id"].astext == str(conn.id),
        )
    )
    if active:
        d["last_activity"] = now().isoformat()
        conn.deploy = dict(d)
        await db.commit()
        return
    last = await db.scalar(
        select(func.max(Generation.completed_at)).where(
            Generation.user_id == conn.user_id,
            Generation.comfy_meta["connection_id"].astext == str(conn.id),
        )
    )
    reference = max(
        [t for t in (last, datetime.fromisoformat(d.get("last_activity") or d["ready_at"])) if t]
    )
    if now() - reference < timedelta(minutes=int(d["auto_stop_minutes"])):
        return
    await stop(db, conn, reason="idle")


def setup_error(log_tail: str) -> str:
    """A short reason for a failed setup, from the end of its log."""
    if "needs 12.8 or newer" in log_tail:
        return (
            "This GPU's NVIDIA driver is too old for ComfyUI. Remove this ComfyUI and rent "
            "the GPU again."
        )
    if "No space left on device" in log_tail:
        return "The GPU's disk is full. Remove this ComfyUI and rent the GPU again."
    return "ComfyUI setup failed. See the log."


# --- Krea 2 model files on a deployed ComfyUI -------------------------------------------------

_CURL = re.compile(r"^\s*(\d{1,3})\s+[\d.]+[kMGT]?\s+(\d{1,3})\s+[\d.]+[kMGT]?\s")
_FILE = re.compile(r"\[studio\] file (\d+)/(\d+)")


def download_progress(log: str, sizes: list[float | None]) -> int:
    """Overall percent of a "models" job from its "[studio] file i/n" markers and curl's
    progress meter (files already on the volume are skipped and count as done)."""
    marks = list(_FILE.finditer(log))
    if not marks:
        return 0
    last = marks[-1]
    index, total = int(last.group(1)), int(last.group(2))
    percent = 0
    for line in re.split(r"[\r\n]+", log[last.end() :]):
        m = _CURL.match(line)
        if m:
            percent = int(m.group(2))
    weights = [float(x) for x in sizes] if sizes and all(sizes) else [1.0] * total
    if len(weights) != total:
        weights = [1.0] * total
    done = sum(weights[: index - 1]) + weights[index - 1] * min(percent, 100) / 100
    return max(0, min(99, round(100 * done / sum(weights))))


def _download_error(log_tail: str) -> str:
    if re.search(r"error: \(?(401|403)\)?|returned error: 40[13]", log_tail):
        return (
            "Hugging Face refused the download. Check your token and the license in "
            "Integrations › Hugging Face."
        )
    if "No space left on device" in log_tail:
        return "The GPU's disk is full."
    return "The download failed. Try again."


async def install_model(db, conn: ComfyConnection, variant: str, filename: str | None) -> None:
    """Starts downloading a Krea 2 model (and its text encoder and VAE) on the GPU."""
    if conn.auth_type != "agent" or conn.status != "ready":
        raise ComfyError("unsupported", "Connect the GPU first.")
    d = dict(conn.deploy or {})
    if (d.get("model_job") or {}).get("status") == "running":
        raise ComfyError("in_progress", "A download is already running.")
    spec = krea2.VARIANTS.get(variant)
    if spec is None or not spec.known_files:
        raise ComfyError(
            "unsupported",
            "There is no verified download for this model yet. Add its file under Manage.",
        )
    files = model_downloads([variant], filename)
    sizes = {k.filename: k.size_gb for k in spec.known_files}
    sizes[krea2.TEXT_ENCODER.filename] = krea2.TEXT_ENCODER.size_gb
    sizes[krea2.VAE.filename] = krea2.VAE.size_gb
    agent = agent_for(conn)
    if agent is None:
        raise ComfyError("unreachable", "The agent address is missing.")
    job_id = f"models-{uuid.uuid4().hex[:10]}"
    try:
        await agent.start_job(job_id, "models", {"models": files})
    except AgentError as exc:
        raise ComfyError("unreachable", exc.message) from exc
    d["model_job"] = {
        "id": job_id,
        "variant": variant,
        "filename": files[-1]["filename"],
        "files": [f["filename"] for f in files],
        "sizes": [sizes.get(f["filename"]) for f in files],
        "status": "running",
        "progress": 0,
        "error": None,
        "started_at": now().isoformat(),
    }
    d["last_activity"] = now().isoformat()
    conn.deploy = d
    await db.commit()


async def _advance_model_job(db, conn: ComfyConnection, d: dict[str, Any]) -> None:
    job_info = dict(d["model_job"])
    agent = agent_for(conn)
    d["last_activity"] = now().isoformat()  # a download keeps the GPU from auto-stopping
    try:
        if agent is None:
            raise AgentError("not_found", "The agent address is missing.")
        job = await agent.job(job_info["id"])
        text, _ = await agent.logs(job_info["id"], max(0, int(job.get("log_size") or 0) - 4000))
    except AgentError as exc:
        if exc.code == "not_found":
            job_info.update(status="failed", error="The GPU restarted. Download again.")
            d["model_job"] = job_info
            conn.deploy = dict(d)
            await db.commit()
        return
    job_info["progress"] = download_progress(text, job_info.get("sizes") or [])
    if job.get("status") == "completed":
        job_info.update(status="completed", progress=100)
        try:
            conn.info = await inspect(conn)
            conn.last_checked_at = now()
        except ComfyError as exc:
            job_info["error"] = exc.message
        seen = ((conn.info or {}).get("krea2") or {}).get("variants", {}).get(job_info["variant"])
        if not (seen or {}).get("installed"):
            job_info.update(
                status="failed",
                progress=0,
                error="The download finished but ComfyUI does not see the model. Try again.",
            )
    elif job.get("status") in ("failed", "cancelled"):
        job_info.update(status="failed", error=_download_error(text[-1500:]))
    d["model_job"] = job_info
    conn.deploy = dict(d)
    await db.commit()


async def stop(db, conn: ComfyConnection, reason: str = "user") -> None:
    """RunPod: stop the pod (its volume keeps ComfyUI and models). Modal: terminate the
    sandbox (its Volume keeps them)."""
    provider = conn.gpu_provider
    creds = await gpu_service.load_creds(db, conn.user_id, provider)
    adapter = get_gpu_adapter(provider)
    if provider == "runpod":
        await adapter.stop(creds, conn.workload_id)
        status = "stopped"
    else:
        await adapter.terminate(creds, conn.workload_id)
        status = "terminated"
    await gpu_service.mark_workload(db, conn.user_id, provider, conn.workload_id, status, now())
    conn.status = "stopped"
    conn.deploy = {
        **(conn.deploy or {}),
        "stage": "stopped",
        "stopped_by": reason,
        "agent_url": None if provider == "modal" else (conn.deploy or {}).get("agent_url"),
    }
    await db.commit()


async def start(db, conn: ComfyConnection) -> None:
    provider = conn.gpu_provider
    if provider == "runpod" and conn.workload_id:
        creds = await gpu_service.load_creds(db, conn.user_id, provider)
        adapter = get_gpu_adapter(provider)
        try:
            await adapter.start(creds, conn.workload_id)
        except GPUError:
            # "Try again" after a failure whose GPU could not be stopped: it still runs.
            workload = await adapter.workload(creds, conn.workload_id)
            if workload is None or workload.status != "running":
                raise
        await gpu_service.mark_workload(
            db, conn.user_id, provider, conn.workload_id, "running", now()
        )
        conn.status = "deploying"
        conn.last_error = None
        conn.deploy = {
            **(conn.deploy or {}),
            "stage": "starting",
            "job_id": None,
            "model_job": None,
            "started_at": now().isoformat(),
            "log_tail": "",
        }
        await db.commit()
        return
    # Modal (or a RunPod pod that is gone): a new workload on the same volume.
    await launch(db, conn)


async def terminate(db, conn: ComfyConnection) -> None:
    if not conn.workload_id or not conn.gpu_provider:
        return
    creds = await gpu_service.load_creds(db, conn.user_id, conn.gpu_provider)
    try:
        await get_gpu_adapter(conn.gpu_provider).terminate(creds, conn.workload_id)
    except GPUError as exc:
        if exc.code != "not_found":
            raise
    await gpu_service.mark_workload(
        db, conn.user_id, conn.gpu_provider, conn.workload_id, "terminated", now()
    )


async def agent_job(conn: ComfyConnection, kind: str, args: dict[str, Any]) -> dict[str, Any]:
    agent = agent_for(conn)
    if agent is None:
        raise ComfyError("unsupported", "Only ComfyUI deployed by the studio runs GPU-side jobs.")
    job_id = f"{kind}-{uuid.uuid4().hex[:10]}"
    try:
        return await agent.start_job(job_id, kind, args)
    except AgentError as exc:
        raise ComfyError("unreachable", exc.message) from exc


def agent_client(conn: ComfyConnection) -> AgentClient | None:
    return agent_for(conn)
