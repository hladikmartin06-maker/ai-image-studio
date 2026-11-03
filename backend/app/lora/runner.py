"""Training runs on the user's RunPod / Modal account, step by step (restart-safe: every
step reads and writes the state in the database).

  queued      -> create the GPU workload (ai-toolkit image + studio agent); billing starts
  preparing   -> wait until the agent answers (image pull, boot)
  uploading   -> send the packaged dataset + ai-toolkit config, start `python run.py`
  running     -> step/loss from the trainer's own output, logs, validation samples
  downloading -> fetch the final LoRA and the kept checkpoints into local storage
  completed | failed | cancelled | interrupted
Artifacts are copied to the Mac before the workload is removed (auto-cleanup), because a
file in a stopped container is not durable.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal
from app.gpu import service as gpu_service
from app.gpu.agent import AGENT_PORT, BOOT, agent_env, new_token
from app.gpu.agent_client import AgentClient, AgentError
from app.gpu.base import GPUError, WorkloadSpec
from app.gpu.modal_provider import (
    CAPACITY_PAUSES,
    CAPACITY_WAIT,
    SANDBOX_CPU,
    SANDBOX_MEMORY_MB,
    hourly_estimate,
)
from app.gpu.registry import get_gpu_adapter
from app.lora.aitoolkit import run_name, run_paths
from app.lora.presets import ARCHITECTURES
from app.lora.serial import ACTIVE, version_json
from app.models import Lora, LoraVersion, UserSecret
from app.security import decrypt_secret, encrypt_secret
from app.services import storage
from app.services.events import broker

log = logging.getLogger("studio.training")

BOOT_TIMEOUT = timedelta(minutes=30)
# Out of GPU memory, in torch's and ai-toolkit's own words ("CUDA out of memory",
# OutOfMemoryError, "OOM during training step 3 times in a row").
_OOM = re.compile(r"out of memory|OutOfMemoryError|OOM during training", re.IGNORECASE)
# After an out-of-memory stop, training starts again on the same GPU (already running and
# billed, models and latents already there) with more memory saving, at most this often.
MAX_MEMORY_RETRIES = 2
OOM_MESSAGE = (
    "The GPU ran out of memory, even with memory saving on. Train at a lower resolution "
    "(e.g. 768 px only) or choose a GPU with more memory."
)
MAX_SAMPLES = 60
TRAINING_VOLUME = "ai-image-studio-training"


def now() -> datetime:
    return datetime.now(UTC)


def token_assoc(v: LoraVersion) -> str:
    return f"{v.user_id}:train:{v.id}"


def version_dir(v: LoraVersion) -> str:
    return f"{v.user_id}/loras/{v.lora_id}/{v.id}"


def publish(v: LoraVersion) -> None:
    broker.publish(v.user_id, "training", version_json(v))


async def _hf_token(db, user_id: uuid.UUID) -> str | None:
    row = await db.scalar(
        select(UserSecret).where(UserSecret.user_id == user_id, UserSecret.name == "huggingface")
    )
    if row is None:
        return None
    return decrypt_secret(row.encrypted_value, associated=f"{user_id}:secret:huggingface")


def _agent(v: LoraVersion) -> AgentClient | None:
    if not v.agent_url or not v.encrypted_agent_token:
        return None
    return AgentClient(
        v.agent_url, decrypt_secret(v.encrypted_agent_token, associated=token_assoc(v))
    )


async def _set(db, v: LoraVersion, **values: Any) -> None:
    for key, value in values.items():
        setattr(v, key, value)
    await db.commit()
    publish(v)


async def _cleanup(db, v: LoraVersion) -> None:
    """Removes the GPU workload (after artifacts were saved, or on failure)."""
    if not v.workload_id or not v.gpu_provider or v.gpu_stopped_at:
        return
    try:
        creds = await gpu_service.load_creds(db, v.user_id, v.gpu_provider)
        await get_gpu_adapter(v.gpu_provider).terminate(creds, v.workload_id)
    except GPUError as exc:
        if exc.code != "not_found":
            v.stage_detail = f"Could not remove the GPU workload: {exc.message}"[:300]
            await db.commit()
            return
    v.gpu_stopped_at = now()
    await gpu_service.mark_workload(
        db, v.user_id, v.gpu_provider, v.workload_id, "terminated", v.gpu_stopped_at
    )
    await db.commit()


async def _fail(db, v: LoraVersion, code: str, message: str, status: str = "failed") -> None:
    v.status = status
    v.error_code = code
    v.error_message = message[:2000]
    v.completed_at = now()
    await db.commit()
    if v.auto_cleanup:
        await _cleanup(db, v)
    publish(v)


def _job_id(v: LoraVersion, run: str) -> str:
    """The agent job of this version's current training attempt."""
    return str((v.progress or {}).get("job_id") or f"train-{run}")


def memory_fallback(trainer_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    """The same ai-toolkit config with more memory saving, and what was changed: one thing
    at a time, the cheapest for quality first — gradient checkpointing (slower, same result),
    then batch 1 with the same effective batch, then 8-bit quantization, then low-VRAM mode;
    None when nothing is left to save."""
    config = copy.deepcopy(trainer_config)
    process = config["config"]["process"][0]
    train, model = process["train"], process["model"]
    if not train.get("gradient_checkpointing"):
        train["gradient_checkpointing"] = True
        return config, ["gradient_checkpointing"]
    if int(train.get("batch_size") or 1) > 1:
        batch = int(train["batch_size"])
        train["batch_size"] = 1
        train["gradient_accumulation"] = int(train.get("gradient_accumulation") or 1) * batch
        return config, ["batch_size"]
    if not model.get("quantize"):
        model["quantize"] = model["quantize_te"] = True
        return config, ["quantize"]
    if not model.get("low_vram"):
        model["low_vram"] = True
        return config, ["low_vram"]
    return None


async def _retry_with_less_memory(db, v: LoraVersion, agent: AgentClient, run: str) -> bool:
    """Starts the training again on the same GPU with more memory saving (the settings shown
    for this version change with it). False when there is nothing left to try."""
    progress = dict(v.progress or {})
    retries = list(progress.get("memory_retries") or [])
    fallback = memory_fallback(v.trainer_config or {})
    if len(retries) >= MAX_MEMORY_RETRIES or fallback is None:
        return False
    trainer_config, changes = fallback
    paths = run_paths(run)
    job_id = f"train-{run}-m{len(retries) + 1}"
    try:
        await agent.upload_bytes(paths["config_rel"], json.dumps(trainer_config, indent=2).encode())
        await agent.start_job(job_id, "train", {"config": paths["config_rel"]})
    except AgentError as exc:
        log.warning("Training could not be started again: %s", exc.code)
        return False
    train = trainer_config["config"]["process"][0]["train"]
    model = trainer_config["config"]["process"][0]["model"]
    config = dict(v.config or {})
    config.update(
        gradient_checkpointing=bool(train.get("gradient_checkpointing")),
        quantize=bool(model.get("quantize")),
        low_vram=bool(model.get("low_vram")),
        batch_size=int(train.get("batch_size") or 1),
        gradient_accumulation=int(train.get("gradient_accumulation") or 1),
    )
    retries.append({"at_step": progress.get("step"), "changes": changes, "at": now().isoformat()})
    progress.update(memory_retries=retries, job_id=job_id, log_offset=0, step=None, loss=[])
    v.trainer_config = trainer_config
    v.config = config
    v.progress = progress
    v.stage_detail = "The GPU ran out of memory: training again with more memory saving"
    note = f"\n--- Out of GPU memory: starting again with {', '.join(changes)} ---\n"
    path = storage.absolute(version_dir(v)) / "train.log"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(note)

    await asyncio.to_thread(write)
    await db.commit()
    publish(v)
    return True


def _spec(v: LoraVersion, lora: Lora, token: str, hf: str | None) -> WorkloadSpec:
    arch = ARCHITECTURES[lora.architecture]
    secret_env = {"STUDIO_AGENT_TOKEN": token}
    if hf:
        secret_env["HF_TOKEN"] = hf
    env = agent_env(token, {"HF_HOME": "/workspace/hf-cache", "STUDIO_AITK_DIR": "/app/ai-toolkit"})
    env.pop("STUDIO_AGENT_TOKEN")
    cfg = v.config or {}
    return WorkloadSpec(
        name=f"ais-train-{run_name(str(lora.id), v.number)}"[:60],
        image=cfg.get("image") or arch["trainer_image"],
        gpu_type=v.gpu_type,
        gpu_count=v.gpu_count,
        ports=[AGENT_PORT],
        env=env,
        secret_env=secret_env,
        cloud_type=cfg.get("cloud_type") or "SECURE",
        container_disk_gb=50,
        volume_gb=100 if v.gpu_provider == "runpod" else 0,
        command=["bash", "-c", BOOT],
        timeout_minutes=1440,
        # Modal: one volume keeps the Hugging Face cache between runs (no re-download).
        volume_name=TRAINING_VOLUME if v.gpu_provider == "modal" else None,
        # Modal's default sandbox is tiny: give the trainer the studio's 4 cores and 32 GiB.
        cpu=SANDBOX_CPU if v.gpu_provider == "modal" else None,
        memory_mb=SANDBOX_MEMORY_MB if v.gpu_provider == "modal" else None,
        # ai-toolkit's image is built for CUDA 13: only machines whose driver supports it
        # (the free ones from RunPod's catalog, see _launch).
        allowed_cuda_versions=list(gpu_service.CUDA13) if v.gpu_provider == "runpod" else [],
    )


async def _wait_for_capacity(db, v: LoraVersion, exc: GPUError) -> bool:
    """Modal has no free GPU of this type right now: the version waits (nothing exists or is
    billed) and the next ticks ask again after a pause. False = give up."""
    if exc.code != "unavailable" or v.gpu_provider != "modal":
        return False
    progress = dict(v.progress or {})
    first = progress.get("capacity_since") or now().isoformat()
    if now() - datetime.fromisoformat(first) >= CAPACITY_WAIT:
        exc.detail = (
            f"Modal had no free {v.gpu_type} for {int(CAPACITY_WAIT.total_seconds() // 60)} "
            "minutes; nothing was billed. Pick another GPU and start again"
        )
        return False
    tries = int(progress.get("capacity_tries") or 0) + 1
    pause = CAPACITY_PAUSES[min(tries, len(CAPACITY_PAUSES)) - 1]
    progress.update(
        capacity_since=first,
        capacity_tries=tries,
        retry_at=(now() + timedelta(seconds=pause)).isoformat(),
    )
    v.progress = progress
    v.stage_detail = (
        f"Modal has no free {v.gpu_type} right now. Trying again (attempt {tries + 1}); "
        "nothing is billed while waiting."
    )
    await db.commit()
    publish(v)
    return True


async def _launch(db, v: LoraVersion, lora: Lora) -> None:
    retry_at = (v.progress or {}).get("retry_at")
    if retry_at and now() < datetime.fromisoformat(retry_at):
        return  # waiting for a free GPU at Modal
    token = new_token()
    v.encrypted_agent_token = encrypt_secret(token, associated=token_assoc(v))
    v.stage_detail = "Starting the GPU"
    await db.commit()
    hf = await _hf_token(db, v.user_id)
    spec = _spec(v, lora, token, hf)
    try:
        if v.gpu_provider == "runpod":
            # RunPod matches CUDA versions exactly: ask only for free CUDA 13 machines.
            free, _ = await gpu_service.offer(
                db, v.user_id, "runpod", v.gpu_type or "", spec.cloud_type
            )
            if free is not None:
                spec.allowed_cuda_versions = [x for x in free if x.startswith("13.")]
                if not spec.allowed_cuda_versions:
                    raise GPUError(
                        "unavailable",
                        f"RunPod has no free {v.gpu_type} with a CUDA 13 driver in "
                        f"{spec.cloud_type.title()} Cloud right now",
                    )
        workload = await gpu_service.launch(
            db, v.user_id, v.gpu_provider or "", spec, "lora-training"
        )
    except GPUError as exc:
        if await _wait_for_capacity(db, v, exc):
            return
        await _fail(db, v, exc.code, exc.message)
        return
    progress = dict(v.progress or {})
    for key in ("capacity_since", "capacity_tries", "retry_at"):
        progress.pop(key, None)
    v.progress = progress
    await _set(
        db,
        v,
        status="preparing",
        workload_id=workload.id,
        # Modal reports no rate: its list-price estimate (GPU + CPU + memory).
        cost_per_hour=workload.cost_per_hour
        if workload.cost_per_hour is not None
        else hourly_estimate(v.gpu_type or "")
        if v.gpu_provider == "modal"
        else None,
        gpu_started_at=now(),
        started_at=v.started_at or now(),
        stage_detail="Waiting for the container (the first start pulls a large image)",
    )


async def _workload_alive(db, v: LoraVersion) -> bool | None:
    try:
        creds = await gpu_service.load_creds(db, v.user_id, v.gpu_provider or "")
        workload = await get_gpu_adapter(v.gpu_provider or "").workload(creds, v.workload_id or "")
    except GPUError:
        return None  # unknown right now
    if workload is None or workload.status in ("terminated", "exited", "stopped", "error"):
        return False
    if workload.cost_per_hour is not None and v.cost_per_hour is None:
        v.cost_per_hour = workload.cost_per_hour
    return True


async def _wait_agent(db, v: LoraVersion) -> None:
    alive = await _workload_alive(db, v)
    if alive is False:
        await _fail(db, v, "interrupted", "The GPU workload stopped while starting.", "interrupted")
        return
    if not v.agent_url:
        try:
            creds = await gpu_service.load_creds(db, v.user_id, v.gpu_provider or "")
            url = await get_gpu_adapter(v.gpu_provider or "").port_url(
                creds, v.workload_id or "", AGENT_PORT
            )
        except GPUError:
            url = None
        if url:
            v.agent_url = url
            await db.commit()
    agent = _agent(v)
    if agent is not None:
        try:
            health = await agent.health()
        except AgentError:
            health = None
        if health:
            if not health.get("aitoolkit"):
                await _fail(
                    db, v, "trainer_missing", "ai-toolkit was not found in the container image."
                )
                return
            progress = dict(v.progress or {})
            progress["gpu"] = (health.get("gpus") or [None])[0]
            await _set(db, v, status="uploading", progress=progress, stage_detail="Uploading")
            return
    if v.gpu_started_at and now() - v.gpu_started_at > BOOT_TIMEOUT:
        await _fail(db, v, "timeout", "The GPU container did not start within 30 minutes.")


async def _upload_and_start(db, v: LoraVersion, lora: Lora) -> None:
    agent = _agent(v)
    if agent is None:
        await _fail(db, v, "agent_missing", "The GPU agent address is missing.")
        return
    run = run_name(str(lora.id), v.number)
    paths = run_paths(run)
    local = storage.absolute(version_dir(v)) / "dataset"
    files = sorted(await asyncio.to_thread(lambda: [p for p in local.iterdir() if p.is_file()]))
    start: Path | None = None
    if (v.config or {}).get("continue_from"):
        # Continue training: the earlier version's LoRA file goes to the GPU first.
        source = await db.get(LoraVersion, uuid.UUID(v.config["continue_from"]))
        start = storage.absolute(source.file_path) if source and source.file_path else None
        if start is None or not await asyncio.to_thread(start.is_file):
            await _fail(
                db, v, "not_found", "The LoRA file to continue from is no longer on this Mac."
            )
            return
    try:
        if start is not None:
            v.stage_detail = "Uploading the LoRA to continue from"
            await db.commit()
            publish(v)
            await agent.upload_file(paths["start_rel"], start)
        for index, path in enumerate(files):
            v.stage_detail = f"Uploading dataset {index + 1}/{len(files)}"
            if index % 5 == 0:
                await db.commit()
                publish(v)
            await agent.upload_file(f"{paths['dataset_rel']}/{path.name}", path)
        await agent.upload_bytes(
            paths["config_rel"], json.dumps(v.trainer_config, indent=2).encode()
        )
        await agent.start_job(f"train-{run}", "train", {"config": paths["config_rel"]})
    except AgentError as exc:
        if exc.code in ("unreachable", "agent_error"):
            return  # retried on the next tick
        await _fail(db, v, "upload_failed", exc.message)
        return
    await _set(db, v, status="running", stage_detail="Training", started_at=v.started_at or now())


_SAMPLE_STEP = re.compile(r"__0*(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$")


async def _append_log(v: LoraVersion, agent: AgentClient, job_id: str) -> str:
    offset = int((v.progress or {}).get("log_offset") or 0)
    text, next_offset = await agent.logs(job_id, offset)
    if text:
        path = storage.absolute(version_dir(v)) / "train.log"

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(text)

        await asyncio.to_thread(write)
    progress = dict(v.progress or {})
    progress["log_offset"] = next_offset
    v.progress = progress
    return text


async def _samples(v: LoraVersion, agent: AgentClient, run: str) -> None:
    progress = dict(v.progress or {})
    have = {s["file"] for s in progress.get("samples", [])}
    if len(have) >= MAX_SAMPLES:
        return
    folder = f"{run_paths(run)['output_rel']}/samples"
    try:
        listed = await agent.list_files(folder)
    except AgentError:
        return
    # Only the images themselves (<time>__<step>_<index>.jpg directly in samples/), not
    # ai-toolkit's thumbnails (samples/.thumbs) or files still being written (samples/.tmp).
    new = sorted(
        (
            f
            for f in listed
            if Path(f["path"]).parent.as_posix() == folder
            and _SAMPLE_STEP.search(Path(f["path"]).name)
            and Path(f["path"]).name not in have
        ),
        key=lambda f: f.get("mtime") or 0,
    )
    for f in new[: MAX_SAMPLES - len(have)]:
        name = Path(f["path"]).name
        match = _SAMPLE_STEP.search(name)
        dest = storage.absolute(version_dir(v)) / "samples" / name
        try:
            await agent.download_file(f["path"], dest)
        except AgentError:
            continue
        progress.setdefault("samples", []).append(
            {
                "file": name,
                "step": int(match.group(1)) if match else None,
                "index": int(match.group(2)) if match else None,
            }
        )
    v.progress = progress


async def _monitor(db, v: LoraVersion, lora: Lora) -> None:
    agent = _agent(v)
    run = run_name(str(lora.id), v.number)
    job_id = _job_id(v, run)
    if agent is None:
        await _fail(db, v, "agent_missing", "The GPU agent address is missing.")
        return
    try:
        job = await agent.job(job_id)
        health = await agent.health()
    except AgentError as exc:
        alive = await _workload_alive(db, v)
        if alive is False:
            await _fail(
                db, v, "interrupted", "The GPU workload stopped during training.", "interrupted"
            )
        elif exc.code == "not_found":
            await _fail(
                db,
                v,
                "interrupted",
                "The GPU container restarted and the training process was lost.",
                "interrupted",
            )
        return
    progress = dict(v.progress or {})
    if job.get("step") is not None:
        progress["step"] = job["step"]
        progress["total"] = job.get("total")
    if job.get("loss"):
        progress["loss"] = job["loss"][-500:]
    progress["gpu"] = (health.get("gpus") or [None])[0]
    progress["ram"] = {"total": health.get("ram_total"), "available": health.get("ram_available")}
    progress["load_1m"] = health.get("load_1m")
    v.progress = progress
    try:
        await _append_log(v, agent, job_id)
        await _samples(v, agent, run)
    except AgentError:
        pass
    await db.commit()
    publish(v)
    status = job.get("status")
    if status == "completed":
        await _set(db, v, status="downloading", stage_detail="Saving the LoRA to this Mac")
    elif status == "cancelled":
        await _fail(db, v, "cancelled", "Training was cancelled.", "cancelled")
    elif status == "failed":
        tail = ""
        path = storage.absolute(version_dir(v)) / "train.log"
        if await asyncio.to_thread(path.exists):
            tail = (await asyncio.to_thread(path.read_text, "utf-8", "replace"))[-1500:]
        # Only what the latest attempt wrote (the log keeps every attempt).
        if _OOM.search(tail.rsplit("--- Out of GPU memory", 1)[-1]):
            if await _retry_with_less_memory(db, v, agent, run):
                return
            await _fail(db, v, "out_of_memory", f"{OOM_MESSAGE}\n{tail}".strip())
            return
        await _fail(
            db,
            v,
            "training_failed",
            f"ai-toolkit stopped with exit code {job.get('exit_code')}.\n{tail}".strip(),
        )


_CHECKPOINT = re.compile(r"_0*(\d+)\.safetensors$")


async def _collect(db, v: LoraVersion, lora: Lora) -> None:
    agent = _agent(v)
    run = run_name(str(lora.id), v.number)
    if agent is None:
        await _fail(db, v, "agent_missing", "The GPU agent address is missing.")
        return
    out = run_paths(run)["output_rel"]
    try:
        files = [f for f in await agent.list_files(out) if f["path"].endswith(".safetensors")]
    except AgentError as exc:
        if exc.code == "unreachable":
            return
        await _fail(db, v, "download_failed", exc.message)
        return
    final = next((f for f in files if Path(f["path"]).name == f"{run}.safetensors"), None)
    if final is None:
        await _fail(db, v, "no_output", "Training ended without a LoRA file.")
        return
    base = storage.absolute(version_dir(v))
    try:
        sha = await agent.download_file(final["path"], base / f"{run}.safetensors", final["size"])
        checkpoints: list[dict[str, Any]] = []
        for f in files:
            match = _CHECKPOINT.search(Path(f["path"]).name)
            if f is final or not match:
                continue
            name = Path(f["path"]).name
            await agent.download_file(f["path"], base / "checkpoints" / name, f["size"])
            checkpoints.append(
                {
                    "step": int(match.group(1)),
                    "file_path": f"{version_dir(v)}/checkpoints/{name}",
                    "size": f["size"],
                }
            )
        await _append_log(v, agent, _job_id(v, run))
    except AgentError as exc:
        if exc.code == "unreachable":
            return  # resumes from the .part file next tick
        await _fail(db, v, "download_failed", exc.message)
        return
    v.file_path = f"{version_dir(v)}/{run}.safetensors"
    v.file_size = final["size"]
    v.file_sha256 = sha
    v.checkpoints = sorted(checkpoints, key=lambda c: c["step"])
    v.status = "completed"
    v.stage_detail = ""
    v.completed_at = now()
    progress = dict(v.progress or {})
    progress["step"] = progress.get("total") or progress.get("step")
    v.progress = progress
    if lora.preferred_version_id is None:
        lora.preferred_version_id = v.id
    await db.commit()
    if v.auto_cleanup:
        await _cleanup(db, v)
    publish(v)


async def advance(version_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        v = await db.get(LoraVersion, version_id)
        if v is None or v.status not in ACTIVE:
            return
        lora = await db.get(Lora, v.lora_id)
        if lora is None:
            return
        if v.status == "queued":
            await _launch(db, v, lora)
        elif v.status == "preparing":
            await _wait_agent(db, v)
        elif v.status == "uploading":
            await _upload_and_start(db, v, lora)
        elif v.status == "running":
            await _monitor(db, v, lora)
        elif v.status == "downloading":
            await _collect(db, v, lora)


async def cancel(version_id: uuid.UUID, stop_gpu: bool = True) -> None:
    async with SessionLocal() as db:
        v = await db.get(LoraVersion, version_id)
        if v is None or v.status not in ACTIVE:
            return
        lora = await db.get(Lora, v.lora_id)
        agent = _agent(v)
        if agent is not None and lora is not None:
            with contextlib.suppress(AgentError):
                await agent.cancel(_job_id(v, run_name(str(lora.id), v.number)))
        v.status = "cancelled"
        v.error_code = "cancelled"
        v.error_message = "Cancelled."
        v.completed_at = now()
        await db.commit()
        if stop_gpu:
            await _cleanup(db, v)
        publish(v)


async def active_ids() -> list[uuid.UUID]:
    async with SessionLocal() as db:
        return list(
            (
                await db.scalars(
                    select(LoraVersion.id)
                    .where(LoraVersion.status.in_(ACTIVE))
                    .order_by(LoraVersion.created_at)
                )
            ).all()
        )
