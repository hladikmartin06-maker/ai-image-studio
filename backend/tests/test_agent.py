"""The studio agent (gpu/agent/studio_agent.py) as a real process: auth, files with
offsets and ranges, path safety, training job with progress/loss parsing, cancel, logs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from app.gpu.agent import BOOT, agent_env
from app.gpu.agent_client import MISSING, AgentClient, AgentError
from tests.gpu_fakes import free_port, make_toolkit

TOKEN = "agent-test-token-0123456789"


@pytest.fixture
async def agent(tmp_path: Path):
    port = free_port()
    toolkit = make_toolkit(tmp_path)
    env = {
        "PATH": os.environ["PATH"],
        **agent_env(TOKEN),
        "STUDIO_AGENT_PORT": str(port),
        "STUDIO_WORKSPACE": str(tmp_path / "ws"),
        "STUDIO_AITK_DIR": str(toolkit),
        "FAKE_TRAIN_SLEEP": "0.05",
    }
    proc = subprocess.Popen(  # noqa: ASYNC220 - starting a test server is fine here
        ["bash", "-c", BOOT], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    client = AgentClient(f"http://127.0.0.1:{port}", TOKEN)
    for _ in range(100):
        try:
            await client.health()
            break
        except AgentError:
            await asyncio.sleep(0.05)
    yield client, tmp_path / "ws"
    proc.terminate()
    proc.wait(timeout=10)


async def test_auth_and_health(agent):
    client, _ = agent
    health = await client.health()
    assert health["ok"] and health["aitoolkit"] is True
    assert "gpus" in health and "ram_total" in health
    assert health["agent_version"] == "3"
    # RunPod's proxy sits behind Cloudflare: the studio's own User-Agent, not httpx's.
    assert client.headers["User-Agent"].startswith("ai-image-studio/")
    # Model downloads only from huggingface.co.
    with pytest.raises(AgentError) as bad:
        await client.start_job(
            "dl", "download", {"url": "https://evil.com/a", "folder": "vae", "filename": "a"}
        )
    assert bad.value.code == "rejected" and "huggingface.co" in bad.value.message
    # No container start log in this test (the ComfyUI image writes one).
    assert await client.boot_log() == ("", 0)
    wrong = AgentClient(client.base_url, "nope")
    with pytest.raises(AgentError) as err:
        await wrong.health()
    assert err.value.code == "unauthorized"


async def test_files_offsets_ranges_and_paths(agent, tmp_path: Path, monkeypatch):
    client, workspace = agent
    monkeypatch.setattr("app.gpu.agent_client.UPLOAD_CHUNK", 1000)
    monkeypatch.setattr("app.gpu.agent_client.DOWNLOAD_CHUNK", 700)
    data = os.urandom(4321)
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    await client.upload_file("a/b/file.bin", src)
    assert (workspace / "a/b/file.bin").read_bytes() == data
    listed = await client.list_files("a")
    assert listed[0]["path"] == "a/b/file.bin" and listed[0]["size"] == 4321
    dest = tmp_path / "out.bin"
    sha = await client.download_file("a/b/file.bin", dest, 4321)
    assert dest.read_bytes() == data and sha == hashlib.sha256(data).hexdigest()
    # A wrong offset is refused instead of corrupting the file.
    async with httpx.AsyncClient(headers=client.headers) as raw:
        bad = await raw.put(f"{client.base_url}/files/a/b/file.bin?offset=5", content=b"x")
        assert bad.status_code == 409
        escape = await raw.get(f"{client.base_url}/files/..%2F..%2Fetc%2Fpasswd")
        assert escape.status_code in (400, 404)
    await client.delete("a")
    assert not (workspace / "a").exists()


class LosingProxy(httpx.AsyncBaseTransport):
    """Like a proxy in front of a rented GPU that times out: chosen uploads reach the agent
    (all of the chunk, or only its first half) but their answer never comes back."""

    def __init__(self, lose: dict[int, str]):
        self.inner = httpx.AsyncHTTPTransport()
        self.lose = lose  # PUT number -> "full" | "half"
        self.puts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            self.puts += 1
            how = self.lose.get(self.puts)
            if how:
                body = await request.aread()
                sent = body if how == "full" else body[: len(body) // 2]
                forwarded = httpx.Request(
                    "PUT",
                    request.url,
                    headers={**request.headers, "content-length": str(len(sent))},
                    content=sent,
                )
                response = await self.inner.handle_async_request(forwarded)
                await response.aread()
                raise httpx.ReadTimeout("the answer was lost", request=request)
        return await self.inner.handle_async_request(request)


async def test_upload_survives_lost_answers(agent, tmp_path: Path, monkeypatch):
    """A chunk that reached the GPU but whose answer was lost is not sent twice: the agent's
    "offset mismatch" (with its real size) moves the upload on. This used to end with
    "Upload to the GPU failed: offset mismatch"."""
    client, workspace = agent
    monkeypatch.setattr("app.gpu.agent_client.UPLOAD_CHUNK", 1000)
    monkeypatch.setattr("app.gpu.agent_client.asyncio.sleep", _no_sleep)
    data = os.urandom(5555)
    src = tmp_path / "lora.safetensors"
    src.write_bytes(data)
    proxy = LosingProxy({2: "full", 5: "half"})
    monkeypatch.setattr(AgentClient, "transport", proxy)
    await client.upload_file("models/loras/lora.safetensors", src)
    assert (workspace / "models/loras/lora.safetensors").read_bytes() == data
    assert await client.remote_size("models/loras/lora.safetensors") == len(data)
    assert await client.remote_size("models/loras/other.safetensors") == MISSING

    # Two uploads of the same file at once (two runs with the same LoRA) do not mix.
    monkeypatch.setattr(AgentClient, "transport", None)
    await asyncio.gather(
        client.upload_file("models/loras/lora.safetensors", src),
        client.upload_file("models/loras/lora.safetensors", src),
    )
    assert (workspace / "models/loras/lora.safetensors").read_bytes() == data


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_training_job_progress_logs_and_cancel(agent, tmp_path: Path):
    client, workspace = agent
    run = workspace / "runs/r1"
    (run / "dataset").mkdir(parents=True)
    (run / "dataset/0001.png").write_bytes(b"png")
    (run / "dataset/0001.txt").write_text("caption")
    config = {
        "config": {
            "name": "r1",
            "process": [
                {
                    "training_folder": str(run / "output"),
                    "datasets": [{"folder_path": str(run / "dataset")}],
                    "train": {"steps": 100},
                    "save": {"save_every": 50},
                }
            ],
        },
        "meta": {"studio_architecture": "krea2"},
    }
    await client.upload_bytes("runs/r1/config.json", json.dumps(config).encode())
    with pytest.raises(AgentError):
        await client.start_job("bad", "train", {"config": "../outside.json"})
    await client.start_job("train-r1", "train", {"config": "runs/r1/config.json"})
    for _ in range(200):
        job = await client.job("train-r1")
        if job["status"] != "running":
            break
        await asyncio.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["step"] == 100 and job["total"] == 100
    assert job["loss"] and job["loss"][-1][0] == 100
    text, offset = await client.logs("train-r1")
    assert "dataset: 1 images, 1 captions" in text and offset == len(text.encode())
    files = {Path(f["path"]).name for f in await client.list_files("runs/r1/output/r1")}
    assert "r1.safetensors" in files and "r1_000000050.safetensors" in files

    # A second run is cancelled while it runs.
    await client.start_job("train-r2", "train", {"config": "runs/r1/config.json"})
    await client.cancel("train-r2")
    for _ in range(100):
        job = await client.job("train-r2")
        if job["status"] != "running":
            break
        await asyncio.sleep(0.1)
    assert job["status"] == "cancelled"
