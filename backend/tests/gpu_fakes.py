"""Test doubles for the GPU side: a fake RunPod API that really starts the studio agent
(the real gpu/agent code) as a local process for every pod, a fake ai-toolkit `run.py`, and
a fake ComfyUI folder the agent can start. Only the external tools are fakes; the agent,
the runner and the protocol between them are the real ones."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from app.gpu.runpod import RunPodProvider
from tests.test_gpu import RUNPOD_KEY, FakeRunPod

TESTS = Path(__file__).parent

FAKE_RUN_PY = r"""
import json, os, struct, sys, time
# In a real container the workspace is /workspace; the test agent's workspace is a temp dir.
text = open(sys.argv[1]).read().replace("/workspace", os.environ["STUDIO_WORKSPACE"])
cfg = json.loads(text)
proc = cfg["config"]["process"][0]
name = cfg["config"]["name"]
out = os.path.join(proc["training_folder"], name)
os.makedirs(os.path.join(out, "samples"), exist_ok=True)
images = [f for f in os.listdir(proc["datasets"][0]["folder_path"]) if not f.endswith(".txt")]
captions = [f for f in os.listdir(proc["datasets"][0]["folder_path"]) if f.endswith(".txt")]
print(f"dataset: {len(images)} images, {len(captions)} captions", flush=True)
if os.environ.get("FAKE_TRAIN_FAIL"):
    print("CUDA out of memory", flush=True)
    sys.exit(3)
total = int(proc["train"]["steps"])
meta = {"format": "pt", "name": name, "studio_architecture": "krea2",
        "training_info": json.dumps(cfg.get("meta", {}))}
def save(path):
    header = json.dumps({"__metadata__": meta,
        "diffusion_model.blocks.0.attn.lora_A.weight": {"dtype": "F16", "shape": [4, 4],
        "data_offsets": [0, 32]}}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + b"\0" * 32)
slow = float(os.environ.get("FAKE_TRAIN_SLEEP", "0.002"))
for step in range(0, total + 1, max(1, total // 10)):
    loss = 0.5 - step / (total * 4)
    sys.stdout.write(f"\r{name}:  {step * 100 // total}%|##| {step}/{total} "
                     f"[00:01<00:01, 1.0s/it, lr: 1.0e-04 loss: {loss:.3e}]")
    sys.stdout.flush()
    if step and step % proc["save"]["save_every"] == 0 and step < total:
        save(os.path.join(out, f"{name}_{step:09d}.safetensors"))
        sample = os.path.join(out, "samples", f"1700000000{step}__{step:09d}_0.jpg")
        try:
            from PIL import Image
            Image.new("RGB", (256, 256), (40 + step % 200, 70, 140)).save(sample, "JPEG")
        except ImportError:
            with open(sample, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0fake-jpeg")
    time.sleep(slow)
save(os.path.join(out, name + ".safetensors"))
print("\nSaved", flush=True)
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_toolkit(root: Path) -> Path:
    folder = root / "ai-toolkit"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(FAKE_RUN_PY)
    return folder


def make_comfy_dir(root: Path) -> Path:
    """A ComfyUI folder that the agent's "comfyui" job accepts as installed: main.py is the
    fake ComfyUI server, the packages count as installed (STUDIO_COMFY_DEPS_MARK, see
    comfy_env) and the Krea 2 files exist, so nothing is downloaded."""
    folder = root / "ComfyUI"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(TESTS / "fake_comfy_server.py", folder / "main.py")
    (root / "deps-installed").write_text("ok")
    for sub, name in (
        ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
        ("vae", "qwen_image_vae.safetensors"),
        ("diffusion_models", "krea2_turbo_fp8_scaled.safetensors"),
    ):
        (folder / "models" / sub).mkdir(parents=True, exist_ok=True)
        (folder / "models" / sub / name).write_bytes(b"weights")
    return folder


def comfy_env(comfy_dir: Path) -> dict[str, str]:
    """Agent environment for a fake ComfyUI made by make_comfy_dir."""
    return {
        "STUDIO_COMFY_DIR": str(comfy_dir),
        "STUDIO_COMFY_PORT": str(free_port()),
        "STUDIO_COMFY_DEPS_MARK": str(comfy_dir.parent / "deps-installed"),
    }


class SpawningRunPod(FakeRunPod):
    """POST /v1/pods with the studio agent in its env starts that agent locally."""

    def __init__(self, root: Path, extra_env: dict[str, str] | None = None):
        super().__init__()
        self.root = root
        self.extra_env = extra_env or {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.bodies: dict[str, dict[str, Any]] = {}

    def _spawn(self, pod_id: str) -> None:
        body = self.bodies[pod_id]
        env_in = body.get("env") or {}
        if "STUDIO_AGENT_CODE" not in env_in:
            return
        port = free_port()
        workspace = self.root / pod_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.root),
            **env_in,
            "STUDIO_AGENT_PORT": str(port),
            "STUDIO_WORKSPACE": str(workspace),
            "STUDIO_PYTHON": sys.executable,
            **self.extra_env,
        }
        command = [*body["dockerEntrypoint"], *body["dockerStartCmd"]]
        with (self.root / pod_id / "container.log").open("ab") as log:
            self.procs[pod_id] = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pods[pod_id]["agent_port"] = port

    def _kill(self, pod_id: str) -> None:
        proc = self.procs.pop(pod_id, None)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path.startswith("/v1/pods/"):
            if request.headers.get("authorization") != f"Bearer {RUNPOD_KEY}":
                return httpx.Response(401, json={"error": "Unauthorized"})
            pod = self.pods.get(path.split("/")[3])
            if pod is None:
                return httpx.Response(404, json={"error": "pod not found"})
            return httpx.Response(200, json={k: v for k, v in pod.items() if k != "agent_port"})
        if request.method == "POST" and path == "/v1/pods":
            response = super().handle(request)
            if response.status_code == 200:
                pod_id = response.json()["id"]
                self.bodies[pod_id] = json.loads(request.content)
                self._spawn(pod_id)
            return response
        pod_id = path.split("/")[3] if path.startswith("/v1/pods/") else None
        response = super().handle(request)
        if pod_id and response.status_code == 200:
            if request.method == "DELETE" or path.endswith("/stop"):
                self._kill(pod_id)
            elif path.endswith("/start"):
                self._spawn(pod_id)
        return response

    def close(self) -> None:
        for pod_id in list(self.procs):
            self._kill(pod_id)


class LocalRunPodProvider(RunPodProvider):
    """RunPod adapter whose proxy URLs point at the locally started agents."""

    def __init__(self, fake: SpawningRunPod):
        super().__init__(transport=fake.transport())
        self.fake = fake

    async def port_url(self, creds: dict[str, str], workload_id: str, port: int) -> str | None:
        pod = self.fake.pods.get(workload_id)
        if not pod or "agent_port" not in pod:
            return None
        return f"http://127.0.0.1:{pod['agent_port']}"
