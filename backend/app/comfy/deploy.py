"""ComfyUI deployed by the studio on the user's RunPod or Modal account.

The GPU runs RunPod's official ComfyUI image (runpod/comfyui), which already holds ComfyUI and
PyTorch: nothing is installed when the GPU starts. The start command writes the studio agent
(gpu/agent) to a file, starts it and hands over to the image's /start.sh, which copies
ComfyUI into /workspace on the first start and runs it on port 8188 (private; only the
agent's token-protected port is exposed). Stages, advanced by the GPU runner (~every 10 s):
  starting -> the container is booting; wait until the agent answers on its port
  booting  -> the agent answers; wait until ComfyUI itself answers through the agent
  ready    -> the user installs a Krea 2 model with one click: the agent downloads the files
              from Hugging Face directly on the GPU (agent job "models", progress in
              deploy["model_job"]); optional auto-stop when idle
  stopped / error (no answer from ComfyUI in time stops the GPU, so it is not billed)
"""

from __future__ import annotations

import asyncio
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
from app.gpu.agent import AGENT_PORT, COMFY_AGENT_PORT, COMFY_BOOT, agent_env, new_token
from app.gpu.agent_client import AgentClient, AgentError
from app.gpu.base import GPUError, WorkloadSpec
from app.gpu.modal_provider import SANDBOX_CPU, SANDBOX_MEMORY_MB, hourly_estimate
from app.gpu.registry import get_gpu_adapter
from app.models import ComfyConnection, Generation, UserSecret
from app.security import decrypt_secret, encrypt_secret

log = logging.getLogger("studio.deploy")

# RunPod's ComfyUI images (github.com/runpod-workers/comfyui-base, tags on Docker Hub):
# cuda12.8 = PyTorch for CUDA 12.8, the default; cuda13.0 = PyTorch for CUDA 13.0, for GPUs
# the CUDA 12 image does not support (Blackwell, RTX 5090) and GPUs that are only offered on
# hosts with a CUDA 13 driver right now. allowedCudaVersions keeps each image on hosts whose
# driver fits it.
IMAGE_CUDA12 = "runpod/comfyui:cuda12.8"
IMAGE_CUDA13 = "runpod/comfyui:cuda13.0"
BLACKWELL = re.compile(r"5090|5080|blackwell|\bB200\b|\bB300\b", re.IGNORECASE)
DEFAULT_IMAGES = {"runpod": IMAGE_CUDA12, "modal": IMAGE_CUDA12}
# Where the image keeps ComfyUI, relative to /workspace.
COMFY_SUBDIR = "runpod-slim/ComfyUI"
# How long ComfyUI may take to answer after the GPU was started; then the GPU is stopped.
# Modal's first start also pulls the image and copies ComfyUI onto its Volume.
BOOT_LIMIT = {"runpod": timedelta(minutes=30), "modal": timedelta(minutes=45)}
POLL_EVERY = timedelta(seconds=10)
MODAL_VOLUME = "ais-comfyui"
# A session: from the start request until the GPU is stopped. Its time limit (default 2 h)
# and optional spending limit stop the GPU, like inactivity does.
DEFAULT_SESSION_MINUTES = 120


def session_minutes(d: dict[str, Any]) -> int:
    return int(d.get("session_minutes") or DEFAULT_SESSION_MINUTES)


def session_estimate(d: dict[str, Any]) -> dict[str, Any] | None:
    """Spent so far and time left in the current session: billable seconds since the start
    request x the GPU's hourly rate. An estimate; the cloud's own billing counts."""
    started = d.get("session_started_at")
    if not started or d.get("stage") in ("stopped", "error"):
        return None
    seconds = max(0.0, (now() - datetime.fromisoformat(started)).total_seconds())
    rate = d.get("cost_per_hour")
    spent = round(float(rate) * seconds / 3600, 4) if rate is not None else None
    return {
        "per_hour": rate,
        "spent": spent,
        "seconds": round(seconds),
        "seconds_left": max(0, round(session_minutes(d) * 60 - seconds)),
        "spend_limit": d.get("spend_limit"),
    }


# deploy["setup"]: how the GPU was set up (1 = older studio versions that installed ComfyUI).
SETUP = 2
OUTDATED = (
    "This GPU was set up by an older version of the studio and cannot start ComfyUI. Delete "
    "it (the GPU and its disk are removed) and rent a GPU again: it is ready in a few minutes."
)


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


def image_plan(
    provider: str, gpu_type: str, free: list[str] | None = None
) -> list[tuple[str, list[str]]]:
    """The images to try in order, each with the driver CUDA versions to ask for. `free` is
    what RunPod's catalog lists as free for this GPU and cloud (None = unknown): the CUDA 12.8
    image on 12.8/12.9 machines, the CUDA 13 image only when the GPU is offered with CUDA 13
    only right now (or is a Blackwell GPU, which the CUDA 12 image does not support)."""
    if provider == "modal":
        return [(IMAGE_CUDA12, [])]
    cuda12 = gpu_service.CUDA12 if free is None else [v for v in free if v in gpu_service.CUDA12]
    cuda13 = gpu_service.CUDA13 if free is None else [v for v in free if v.startswith("13.")]
    plan: list[tuple[str, list[str]]] = []
    if cuda12 and not BLACKWELL.search(gpu_type):
        plan.append((IMAGE_CUDA12, cuda12))
    if cuda13:
        plan.append((IMAGE_CUDA13, cuda13))
    return plan


def spec_for(
    conn: ComfyConnection, token: str, hf: str | None, image: str, cuda: list[str]
) -> WorkloadSpec:
    d = conn.deploy or {}
    provider = d["provider"]
    secret_env = {"STUDIO_AGENT_TOKEN": token}
    if hf:
        secret_env["HF_TOKEN"] = hf
    # The agent finds ComfyUI in $STUDIO_WORKSPACE/runpod-slim/ComfyUI (COMFY_SUBDIR).
    env = agent_env(token, port=COMFY_AGENT_PORT)
    env.pop("STUDIO_AGENT_TOKEN")
    return WorkloadSpec(
        name=f"ais-comfyui-{str(conn.id)[:8]}",
        image=image,
        gpu_type=d["gpu_type"],
        gpu_count=int(d.get("gpu_count") or 1),
        ports=[COMFY_AGENT_PORT],
        env=env,
        secret_env=secret_env,
        cloud_type=d.get("cloud_type") or "SECURE",
        container_disk_gb=int(d.get("container_disk_gb") or 30),
        # RunPod: a pod volume at /workspace keeps ComfyUI and the models when it is stopped.
        volume_gb=int(d.get("volume_gb") or 80) if provider == "runpod" else 0,
        entrypoint=["/bin/bash", "-c"],
        command=[COMFY_BOOT],
        # Modal: the sandbox ends itself after the session (+10 min, at most 24 h).
        timeout_minutes=min(session_minutes(d) + 10, 1440),
        # Modal: one named Volume at /workspace keeps ComfyUI and every downloaded model for
        # all of the user's Modal GPUs, so only the first start is slow.
        volume_name=MODAL_VOLUME if provider == "modal" else None,
        cpu=SANDBOX_CPU if provider == "modal" else None,
        memory_mb=SANDBOX_MEMORY_MB if provider == "modal" else None,
        clear_entrypoint=provider == "modal",
        allowed_cuda_versions=cuda,
        tags={"studio-connection": str(conn.id)},
    )


async def _create(db, conn: ComfyConnection, token: str, hf: str | None):
    """Creates the pod with the first image a free machine fits. The CUDA versions come from
    RunPod's catalog (it matches them exactly, so a version without a free machine fails). A
    request is followed by the next one only when RunPod answered that no machine is free
    (no pod was created) — same GPU and cloud, so the same price."""
    d = conn.deploy
    provider = d["provider"]
    cloud = d.get("cloud_type") or "SECURE"
    free, price = await gpu_service.offer(db, conn.user_id, provider, d["gpu_type"], cloud)
    confirmed = d.get("confirmed_price")
    if price is not None and confirmed is not None and abs(float(confirmed) - price) > 0.0005:
        # Nothing starts at a price the owner did not confirm.
        raise GPUError("price_changed", f"now {price:.3f} USD per hour")
    plan = image_plan(provider, d["gpu_type"], free)
    if not plan:
        raise GPUError(
            "unavailable",
            f"RunPod has no free {d['gpu_type']} with a recent driver (CUDA 12.8 or newer) in "
            f"{cloud.title()} Cloud right now",
        )
    last: GPUError | None = None
    for image, cuda in plan:
        spec = spec_for(conn, token, hf, image, cuda)
        try:
            workload = await gpu_service.launch(db, conn.user_id, provider, spec, "comfyui")
            return workload, image, cuda
        except GPUError as exc:
            last = exc
            if exc.code == "unavailable":
                continue  # no free machine for this image: try the next one
            raise
    assert last is not None
    raise last


async def launch(db, conn: ComfyConnection) -> None:
    """Creates the GPU workload (billing starts) and moves the connection to "starting"."""
    token = new_token()
    conn.encrypted_secret = encrypt_secret(token, associated=assoc(conn))
    conn.secret_last4 = token[-4:]
    # Saved before the GPU exists: a Modal sandbox whose creation outlived the app is taken
    # over with it (see _adopt).
    await db.commit()
    hf = await hf_token(db, conn.user_id)
    workload, image, cuda = await _create(db, conn, token, hf)
    conn.workload_id = workload.id
    conn.gpu_provider = conn.deploy["provider"]
    conn.status = "deploying"
    conn.last_error = None
    conn.deploy = {
        **conn.deploy,
        "setup": SETUP,
        "image": image,
        "cuda": cuda,
        "agent_port": COMFY_AGENT_PORT,
        "comfy_dir": COMFY_SUBDIR,
        "stage": "starting",
        "agent_url": None,
        "job_id": None,
        "model_job": None,
        "started_at": now().isoformat(),
        "session_started_at": conn.deploy.get("session_started_at") or now().isoformat(),
        "polled_at": None,
        "cost_per_hour": workload.cost_per_hour
        if workload.cost_per_hour is not None
        else hourly_estimate(conn.deploy["gpu_type"])
        if conn.deploy["provider"] == "modal"
        else None,
        "cost_is_estimate": conn.deploy["provider"] == "modal",
        "log_tail": "",
        "log_offset": 0,
    }
    await db.commit()


# Modal builds the image when the sandbox is created (the first time it pulls RunPod's
# ComfyUI image, many minutes), so Modal sandboxes are created in the background and the
# studio shows the progress; RunPod answers a create within seconds.
SLOW_CREATE = {"modal"}
_creating: set[uuid.UUID] = set()
_tasks: set[asyncio.Task[None]] = set()


def creates_in_background(conn: ComfyConnection) -> bool:
    return (conn.deploy or {}).get("provider") in SLOW_CREATE


def mark_creating(conn: ComfyConnection) -> None:
    """Stage "creating": the GPU is being created in the background (create_later)."""
    conn.status = "deploying"
    conn.last_error = None
    conn.deploy = {
        **(conn.deploy or {}),
        "setup": SETUP,
        "agent_port": COMFY_AGENT_PORT,
        "comfy_dir": COMFY_SUBDIR,
        "stage": "creating",
        "agent_url": None,
        "model_job": None,
        "started_at": now().isoformat(),
        "session_started_at": now().isoformat(),
        "polled_at": None,
        "log_tail": "",
        "log_offset": 0,
    }


def create_later(conn_id: uuid.UUID) -> None:
    """Creates the GPU of a connection in stage "creating" in the background."""
    if conn_id in _creating:
        return
    _creating.add(conn_id)

    async def run() -> None:
        try:
            async with SessionLocal() as db:
                conn = await db.get(ComfyConnection, conn_id)
                if conn is None or (conn.deploy or {}).get("stage") != "creating":
                    return
                try:
                    await launch(db, conn)
                except GPUError as exc:
                    await _fail(db, conn, exc.message)
        except Exception:  # noqa: BLE001 - reported in the log; the stage times out
            log.exception("creating the GPU for %s failed", conn_id)
        finally:
            _creating.discard(conn_id)

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _adopt(db, conn: ComfyConnection, d: dict[str, Any]) -> None:
    """Stage "creating" with no creation running in this app (it was restarted): take over a
    sandbox that was created for this connection, else create it (none exists, so nothing is
    billed twice), or give up after the start limit."""
    provider = d["provider"]
    adapter = get_gpu_adapter(provider)
    found: list[str] = []
    if hasattr(adapter, "find_workloads"):
        try:
            creds = await gpu_service.load_creds(db, conn.user_id, provider)
            found = await adapter.find_workloads(creds, {"studio-connection": str(conn.id)})
        except GPUError:
            return  # try again on the next pass
    if found:
        conn.workload_id = found[0]
        conn.gpu_provider = provider
        d.update(stage="starting", agent_url=None)
        conn.deploy = dict(d)
        await db.commit()
        return
    limit = BOOT_LIMIT.get(provider, BOOT_LIMIT["runpod"])
    if _since(d.get("started_at")) > limit:
        await _fail(db, conn, "The GPU could not be created in time. Try again.")
        return
    create_later(conn.id)


def _since(value: str | None) -> timedelta:
    if not value:
        return timedelta(0)
    return now() - datetime.fromisoformat(value)


async def _fail(db, conn: ComfyConnection, message: str, stop_gpu: bool = False) -> None:
    """A failed start: with stop_gpu the GPU is stopped (RunPod; Modal: ended) so it is not
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
            log.warning("could not stop GPU after a failed start: %s", exc.message)
    conn.status = "error"
    conn.last_error = message[:1000]
    conn.deploy = {**(conn.deploy or {}), "stage": "error"}
    await db.commit()


async def _read_boot_log(agent: AgentClient, d: dict[str, Any]) -> None:
    """Appends the new part of the container's start log (ComfyUI's output) to log_tail."""
    try:
        text, offset = await agent.boot_log(int(d.get("log_offset") or 0))
    except AgentError:
        return
    if text:
        d["log_tail"] = ((d.get("log_tail") or "") + text)[-8000:]
    d["log_offset"] = offset


async def advance(conn_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        conn = await db.get(ComfyConnection, conn_id)
        if conn is None or conn.auth_type != "agent":
            return
        d = dict(conn.deploy or {})
        stage = d.get("stage")
        provider = conn.gpu_provider or d.get("provider")
        if stage == "creating":
            if conn.id not in _creating:
                await _adopt(db, conn, d)
            return
        if stage in ("starting", "booting", "ready") and await _session_limits(db, conn, d):
            return
        if stage in ("starting", "booting", "installing"):
            if d.get("setup") != SETUP:
                await _fail(db, conn, OUTDATED, stop_gpu=True)
                return
            if d.get("polled_at") and _since(d["polled_at"]) < POLL_EVERY:
                return
            d["polled_at"] = now().isoformat()
            await _advance_start(db, conn, d, provider)
            return
        if stage == "ready" and (d.get("model_job") or {}).get("status") == "running":
            await _advance_model_job(db, conn, d)
            return
        if stage == "ready" and d.get("auto_stop_minutes"):
            await _auto_stop(db, conn, d)


async def _session_limits(db, conn: ComfyConnection, d: dict[str, Any]) -> bool:
    """Stops the GPU at the session's time limit or spending limit. True when it stopped."""
    est = session_estimate(d)
    if est is None:
        return False
    reason = None
    if est["seconds_left"] <= 0:
        reason = "time_limit"
    elif d.get("spend_limit") and est["spent"] is not None and est["spent"] >= d["spend_limit"]:
        reason = "spend_limit"
    if reason is None:
        return False
    try:
        await stop(db, conn, reason=reason)
    except GPUError as exc:
        log.warning("could not stop GPU at its %s: %s", reason, exc.message)
        return False
    return True


async def _advance_start(db, conn: ComfyConnection, d: dict[str, Any], provider: str) -> None:
    limit = BOOT_LIMIT.get(provider, BOOT_LIMIT["runpod"])
    try:
        creds = await gpu_service.load_creds(db, conn.user_id, provider)
        adapter = get_gpu_adapter(provider)
        workload = await adapter.workload(creds, conn.workload_id)
    except GPUError as exc:
        if exc.code == "invalid_credentials":
            await _fail(db, conn, exc.message)
        else:
            conn.deploy = dict(d)
            await db.commit()
        return
    if workload is None or workload.status in ("terminated", "exited", "stopped", "error"):
        await _fail(db, conn, "The GPU stopped before ComfyUI was ready.")
        return
    if workload.cost_per_hour is not None:
        d["cost_per_hour"] = workload.cost_per_hour

    if d.get("stage") == "starting":
        port = int(d.get("agent_port") or AGENT_PORT)
        url = d.get("agent_url") or await adapter.port_url(creds, conn.workload_id, port)
        if url:
            d["agent_url"] = url
            conn.deploy = dict(d)
            agent = agent_for(conn)
            try:
                health = await agent.health() if agent else None
            except AgentError:
                health = None
            if health:
                d.update(stage="booting", agent_at=now().isoformat())
                if health.get("gpus"):
                    d["gpu"] = health["gpus"][0]

    if d.get("stage") == "booting":
        conn.deploy = dict(d)
        agent = agent_for(conn)
        if agent is not None:
            await _read_boot_log(agent, d)
        try:
            await client_for(conn).system_stats()
            conn.info = await inspect(conn)
        except ComfyError:
            pass  # not answering yet (the first start copies ComfyUI onto the volume)
        else:
            d.update(stage="ready", ready_at=now().isoformat(), last_activity=now().isoformat())
            conn.deploy = dict(d)
            conn.status = "ready"
            conn.last_error = None
            conn.last_checked_at = now()
            await db.commit()
            return

    conn.deploy = dict(d)
    if _since(d.get("started_at")) > limit:
        minutes = int(limit.total_seconds() // 60)
        reason = (
            "ComfyUI did not answer"
            if d.get("stage") == "booting"
            else "The GPU container did not start"
        )
        await _fail(db, conn, f"{reason} within {minutes} minutes.", stop_gpu=True)
        return
    await db.commit()


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


# --- Krea 2 model files on a deployed ComfyUI -------------------------------------------------

DOWNLOAD_ERRORS = {
    "denied": "Hugging Face refused the download. Check your token and the license in "
    "Integrations › Hugging Face.",
    "not_found": "The file is no longer on Hugging Face.",
    "no_space": "The GPU's disk is full.",
    "not_allowed": "Only downloads from huggingface.co are allowed.",
    "cancelled": "The download was cancelled.",
}


def model_progress(job: dict[str, Any], sizes_gb: list[float | None]) -> int:
    """Overall percent of a "models" job: bytes done of all files (a file's real size once
    its download started, else the catalog size; skipped files count as done)."""
    files = job.get("files") or []
    if not files:
        return 0
    done = total = 0.0
    for i, f in enumerate(files):
        estimate = (sizes_gb[i] if i < len(sizes_gb) else None) or 1.0
        size = float(f.get("size") or estimate * 1e9)
        total += size
        done += min(float(f.get("done") or 0), size)
    return max(0, min(99, round(100 * done / total))) if total else 0


async def install_model(db, conn: ComfyConnection, variant: str, filename: str | None) -> None:
    """Starts downloading a Krea 2 model (and its text encoder and VAE) on the GPU, straight
    from Hugging Face. Only once ComfyUI answers (its first-start copy is done by then)."""
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
        "current": None,
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
    except AgentError as exc:
        if exc.code == "not_found":
            job_info.update(status="failed", error="The GPU restarted. Download again.")
            d["model_job"] = job_info
            conn.deploy = dict(d)
            await db.commit()
        return
    job_info["progress"] = model_progress(job, job_info.get("sizes") or [])
    current = next((f for f in job.get("files") or [] if f.get("status") == "downloading"), None)
    job_info["current"] = current.get("name") if current else None
    if job.get("status") == "completed":
        job_info.update(status="completed", progress=100, current=None)
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
        error = job.get("error") or {}
        job_info.update(
            status="failed",
            current=None,
            error=DOWNLOAD_ERRORS.get(str(error.get("code")), "The download failed. Try again."),
        )
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
    est = session_estimate(conn.deploy or {})
    spent_total = float((conn.deploy or {}).get("spent_total") or 0) + float(
        (est or {}).get("spent") or 0
    )
    conn.deploy = {
        **(conn.deploy or {}),
        "stage": "stopped",
        "stopped_by": reason,
        "last_session": est,
        "spent_total": round(spent_total, 4),
        "session_started_at": None,
        "agent_url": None if provider == "modal" else (conn.deploy or {}).get("agent_url"),
    }
    await db.commit()


def outdated(conn: ComfyConnection) -> bool:
    return conn.auth_type == "agent" and (conn.deploy or {}).get("setup") != SETUP


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
            "session_started_at": now().isoformat(),
            "polled_at": None,
            # A new container: the agent's address stays, its start log begins again.
            "log_tail": "",
            "log_offset": 0,
        }
        await db.commit()
        return
    # Modal (or a RunPod pod that is gone): a new workload on the same volume.
    if creates_in_background(conn):
        mark_creating(conn)
        await db.commit()
        create_later(conn.id)
        return
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
