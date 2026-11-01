"""A local stand-in for WaveSpeed and Atlas Cloud, for trying the full app without a
provider account (development and the cloud sandbox only; never used in production).

Run:  uv run uvicorn tests.fake_server:app --host 127.0.0.1 --port 8900
Then set in backend/.env:
  WAVESPEED_BASE_URL=http://127.0.0.1:8900/ws
  ATLAS_BASE_URL=http://127.0.0.1:8900/atlas
Any API key that starts with "test-" is accepted. Prompts containing "fail" fail.
The images it returns are generated test patterns, clearly not AI output.

RunPod: also set RUNPOD_REST_URL=http://127.0.0.1:8900/runpod/rest/v1,
RUNPOD_GRAPHQL_URL=http://127.0.0.1:8900/runpod/graphql and
RUNPOD_PROXY_URL=http://127.0.0.1:8900/runpod/proxy/{pod_id}/{port}. A pod created with the
studio agent in its environment runs that (real) agent as a local process, with a fake
ai-toolkit and a fake ComfyUI (tests/gpu_fakes.py), and its port is reached through
/runpod/proxy/… like RunPod's HTTP proxy.

Hugging Face: set HUGGINGFACE_URL=http://127.0.0.1:8900/hf. Tokens that start with "hf_test"
are valid; "hf_test_nolicense…" still has to accept the Krea 2 RAW license
(GET /hf/accept?token=… stands in for the button on huggingface.co).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw

app = FastAPI()
_ids = itertools.count(1)
_jobs: dict[str, dict[str, Any]] = {}
BASE = "http://127.0.0.1:8900"

SIZES = {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (4096, 4096)}
RATIO = {
    "1:1": (1, 1),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "4:5": (4, 5),
    "5:4": (5, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}


def _authorized(request: Request) -> bool:
    return request.headers.get("authorization", "").startswith("Bearer test-")


def _unauthorized():
    return JSONResponse({"code": 401, "message": "Invalid API key"}, status_code=401)


def _size(body: dict[str, Any]) -> tuple[int, int]:
    size = body.get("size")
    if isinstance(size, str) and "*" in size:
        w, h = size.split("*")
        return int(w), int(h)
    if isinstance(size, str) and size in SIZES:
        return SIZES[size]
    ratio = RATIO.get(body.get("aspect_ratio") or "1:1", (1, 1))
    edge = {"1k": 1024, "2k": 2048, "4k": 4096}.get(body.get("resolution") or "1k", 1024)
    scale = edge / max(ratio)
    return int(ratio[0] * scale), int(ratio[1] * scale)


def _pattern(seed: str, width: int, height: int, label: str) -> bytes:
    # Keep test files small: render at most 1024 px on the long side.
    scale = min(1.0, 1024 / max(width, height))
    w, h = max(64, int(width * scale)), max(64, int(height * scale))
    digest = hashlib.sha256(seed.encode()).digest()
    a = (digest[0] // 3 + 20, digest[1] // 5 + 10, digest[2] // 2 + 60)
    b = (digest[3] // 2 + 90, digest[4] // 3 + 50, digest[5] // 2 + 120)
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line([(0, y), (w, y)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)))
    r = min(w, h) // 3
    cx, cy = w // 2 + (digest[6] - 128) * w // 1024, h // 2 + (digest[7] - 128) * h // 1024
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(235, 230, 255), width=max(2, r // 40))
    draw.text((16, h - 28), f"TEST PATTERN · {label}", fill=(240, 240, 250))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _create(provider: str, model: str, body: dict[str, Any]) -> dict[str, Any]:
    pid = f"{provider}-{next(_ids):05d}"
    _jobs[pid] = {
        "model": model,
        "body": body,
        "count": int(body.get("n") or 1),
        "polls": 0,
        "fail": "fail" in str(body.get("prompt", "")).lower(),
    }
    return {"id": pid, "status": "created", "outputs": []}


def _state(pid: str) -> dict[str, Any] | None:
    job = _jobs.get(pid)
    if job is None:
        return None
    job["polls"] += 1
    if job["polls"] < 3:
        return {"id": pid, "status": "processing", "outputs": []}
    if job["fail"]:
        return {"id": pid, "status": "failed", "outputs": [], "error": "Test failure requested"}
    w, h = _size(job["body"])
    outputs = [f"{BASE}/cdn/{pid}-{i}-{w}x{h}.png" for i in range(job["count"])]
    return {
        "id": pid,
        "status": "completed",
        "outputs": outputs,
        "has_nsfw_contents": [False] * len(outputs),
    }


# --- WaveSpeed ---------------------------------------------------------------------------


@app.get("/ws/api/v3/balance")
async def ws_balance(request: Request):
    if not _authorized(request):
        return _unauthorized()
    return {"code": 200, "message": "success", "data": {"balance": 25.0}}


@app.post("/ws/api/v3/media/upload/binary")
async def ws_upload(request: Request, file: UploadFile = File(...)):
    if not _authorized(request):
        return _unauthorized()
    await file.read()
    return {"code": 200, "data": {"download_url": f"{BASE}/cdn/upload-{next(_ids)}-512x512.png"}}


@app.get("/ws/api/v3/predictions/{pid}/result")
async def ws_result(pid: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    state = _state(pid)
    if state is None:
        return JSONResponse({"code": 404, "message": "not found"}, status_code=404)
    return {"code": 200, "message": "success", "data": state}


@app.post("/ws/api/v3/{model:path}")
async def ws_submit(model: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    body = await request.json()
    return {"code": 200, "message": "success", "data": _create("wavespeed", model, body)}


# --- Atlas -------------------------------------------------------------------------------


@app.post("/atlas/api/v1/model/generateImage")
async def atlas_submit(request: Request):
    if not _authorized(request):
        return _unauthorized()
    body = await request.json()
    return {"code": 200, "data": _create("atlas", body.get("model", ""), body)}


@app.get("/atlas/api/v1/model/prediction/{pid}")
async def atlas_result(pid: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    state = _state(pid)
    if state is None:
        return JSONResponse({"code": 404, "message": "prediction not found"}, status_code=404)
    return {"code": 200, "data": state}


@app.post("/atlas/api/v1/model/uploadMedia")
async def atlas_upload(request: Request, file: UploadFile = File(...)):
    if not _authorized(request):
        return _unauthorized()
    await file.read()
    return {"code": 200, "data": {"download_url": f"{BASE}/cdn/upload-{next(_ids)}-512x512.png"}}


# --- CDN ---------------------------------------------------------------------------------


@app.get("/cdn/{name}")
async def cdn(name: str):
    stem = name.rsplit(".", 1)[0]
    size = stem.rsplit("-", 1)[-1]
    try:
        w, h = (int(x) for x in size.split("x"))
    except ValueError:
        w, h = 1024, 1024
    data = await asyncio.to_thread(_pattern, stem, w, h, f"{w}×{h}")
    return Response(data, media_type="image/png")


# --- OpenAI-compatible chat (for the assistant) -------------------------------------------
# Connect "OpenAI-compatible API" with base URL http://127.0.0.1:8900/openai/v1 and any key
# that starts with "test-".

_CHAT_MODELS = ["test-mini", "test-medium", "test-large"]


@app.get("/openai/v1/models")
async def chat_models(request: Request):
    if not _authorized(request):
        return _unauthorized()
    return {"data": [{"id": m} for m in _CHAT_MODELS]}


def _chat_answer(text: str) -> str:
    # Answer the question itself, not the studio context block in front of it.
    lowered = text.split("</studio_context>")[-1].lower()
    if "prompt" in lowered:
        return (
            "Here is a sharper version that keeps your idea:\n\n"
            "```prompt\nA lone lighthouse on a basalt cliff at blue hour, warm lamp glow, "
            "long-exposure sea mist, 35mm, cinematic contrast\n```\n\n"
            "- **Light:** blue hour gives soft contrast.\n- **Lens:** 35mm keeps the scale."
        )
    if "setting" in lowered or "nastaven" in lowered:
        return (
            "For a cinematic landscape I suggest:\n\n"
            '```studio-settings\n{"aspect_ratio": "16:9", "resolution": "2K", "count": 2}\n```'
        )
    return (
        "A **seed** fixes the random start of a generation, so the same settings repeat a result."
    )


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    if not _authorized(request):
        return _unauthorized()
    body = await request.json()
    question = body["messages"][-1]["content"]
    if isinstance(question, list):
        # A vision request (dataset caption): describe what varies, as instructed.
        answer = "standing by a window in a cream knit sweater, soft daylight, half-body shot"
    else:
        answer = _chat_answer(question)

    async def stream():
        for i in range(0, len(answer), 12):
            chunk = {
                "model": body["model"],
                "choices": [{"delta": {"content": answer[i : i + 12]}}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.02)
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
        usage = {"prompt_tokens": 1180, "completion_tokens": len(answer) // 4}
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    from fastapi.responses import StreamingResponse

    return StreamingResponse(stream(), media_type="text/event-stream")


# --- RunPod stand-in (set RUNPOD_REST_URL / RUNPOD_GRAPHQL_URL / RUNPOD_SERVERLESS_URL) -----

_pods: dict[str, dict[str, Any]] = {}
_endpoints: dict[str, dict[str, Any]] = {
    "ep-sdxl": {
        "id": "ep-sdxl",
        "name": "sdxl-worker",
        "gpuTypeIds": ["NVIDIA L4"],
        "workersMin": 0,
        "workersMax": 2,
    }
}


@app.post("/runpod/graphql")
async def runpod_graphql(request: Request):
    # Like Cloudflare in front of api.runpod.io: library User-Agents are refused (error 1010).
    agent = request.headers.get("user-agent", "").lower()
    if agent.startswith(("python-httpx", "python-urllib", "python-requests")):
        return Response(
            "error code: 1010",
            status_code=403,
            headers={"server": "cloudflare"},
            media_type="text/plain",
        )
    if not _authorized(request):
        return _unauthorized()
    query = (await request.json())["query"]
    if "gpuTypes" in query:
        return {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA GeForce RTX 3070",
                        "displayName": "RTX 3070",
                        "memoryInGb": 8,
                        "securePrice": 0.13,
                        "manufacturer": "Nvidia",
                        "communityPrice": 0.07,
                    },
                    {
                        "id": "NVIDIA RTX A4000",
                        "displayName": "RTX A4000",
                        "memoryInGb": 16,
                        "securePrice": 0.25,
                        "manufacturer": "Nvidia",
                        "communityPrice": 0.17,
                    },
                    {
                        "id": "NVIDIA L4",
                        "displayName": "L4",
                        "memoryInGb": 24,
                        "securePrice": 0.43,
                        "manufacturer": "Nvidia",
                        "communityPrice": 0.39,
                    },
                    {
                        "id": "NVIDIA GeForce RTX 4090",
                        "displayName": "RTX 4090",
                        "memoryInGb": 24,
                        "securePrice": 0.69,
                        "manufacturer": "Nvidia",
                        "communityPrice": 0.34,
                    },
                    {
                        "id": "NVIDIA H100 80GB HBM3",
                        "displayName": "H100 SXM",
                        "memoryInGb": 80,
                        "securePrice": 2.99,
                        "manufacturer": "Nvidia",
                        "communityPrice": None,
                    },
                    {
                        "id": "AMD Instinct MI300X OAM",
                        "displayName": "MI300X",
                        "memoryInGb": 192,
                        "securePrice": 2.49,
                        "manufacturer": "AMD",
                        "communityPrice": None,
                    },
                    {
                        # Listed by GraphQL but not accepted by the pod API: never offered.
                        "id": "NVIDIA A100-SXM4-40GB",
                        "displayName": "A100 SXM 40GB",
                        "memoryInGb": 40,
                        "securePrice": 1.19,
                        "manufacturer": "Nvidia",
                        "communityPrice": None,
                    },
                    {
                        "id": "NVIDIA A40",
                        "displayName": "A40",
                        "memoryInGb": 48,
                        "securePrice": None,
                        "manufacturer": "Nvidia",
                        "communityPrice": 0.35,
                    },
                ]
            }
        }
    spend = sum(p["costPerHr"] for p in _pods.values() if p["desiredStatus"] == "RUNNING")
    return {"data": {"myself": {"clientBalance": 18.4, "currentSpendPerHr": spend}}}


@app.get("/runpod/rest/v1/pods")
async def runpod_pods(request: Request):
    if not _authorized(request):
        return _unauthorized()
    return list(_pods.values())


@app.post("/runpod/rest/v1/pods")
async def runpod_create(request: Request):
    if not _authorized(request):
        return _unauthorized()
    body = await request.json()
    # Like RunPod's API: values outside its OpenAPI enums are refused.
    from app.gpu.runpod import REST_CUDA, REST_GPU_IDS

    if any(v not in REST_CUDA for v in body.get("allowedCudaVersions") or []) or any(
        g not in REST_GPU_IDS for g in body.get("gpuTypeIds") or []
    ):
        return JSONResponse(
            {
                "error": "At #/paths/pods/post for POST https://rest.runpod.io/v1/pods, The "
                "request body is defined as an object. However, it does not meet the schema "
                "requirements of the specification."
            },
            status_code=400,
        )
    pod_id = f"pod{uuid.uuid4().hex[:10]}"  # unique across restarts of this fake
    _pods[pod_id] = {
        "id": pod_id,
        "name": body["name"],
        "desiredStatus": "RUNNING",
        "imageName": body["imageName"],
        "gpuCount": body["gpuCount"],
        "costPerHr": 0.43 * body["gpuCount"],
        "ports": body.get("ports", []),
        "gpu": {"id": body["gpuTypeIds"][0], "displayName": body["gpuTypeIds"][0]},
    }
    _bodies[pod_id] = body
    _spawn(pod_id)
    return _pods[pod_id]


@app.get("/runpod/rest/v1/pods/{pod_id}")
async def runpod_get(pod_id: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    if pod_id not in _pods:
        return JSONResponse({"error": "pod not found"}, status_code=404)
    return _pods[pod_id]


# --- pods that run the studio agent locally ----------------------------------------------

_bodies: dict[str, dict[str, Any]] = {}
_procs: dict[str, tuple[subprocess.Popen, int]] = {}
_root = Path(tempfile.gettempdir()) / "ais-fake-gpu"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn(pod_id: str) -> None:
    from tests.gpu_fakes import comfy_env, make_comfy_dir, make_toolkit

    body = _bodies[pod_id]
    env_in = body.get("env") or {}
    if "STUDIO_AGENT_CODE" not in env_in:
        return
    base = _root / pod_id
    port = _free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(base),
        **env_in,
        "STUDIO_AGENT_PORT": str(port),
        "STUDIO_WORKSPACE": str(base / "workspace"),
        "STUDIO_AITK_DIR": str(make_toolkit(_root / "image")),
        "STUDIO_PYTHON": sys.executable,
        "TMPDIR": str(base),
        **comfy_env(make_comfy_dir(base / "gpu")),
        "FAKE_TRAIN_SLEEP": "0.4",
        # The first start copies ComfyUI onto the volume (seconds), like the real image.
        "FAKE_COPY_SLEEP": "8",
    }
    (base / "workspace").mkdir(parents=True, exist_ok=True)
    log = (base / "container.log").open("ab")
    proc = subprocess.Popen(
        [*body["dockerEntrypoint"], *body["dockerStartCmd"]],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _procs[pod_id] = (proc, port)


def _kill(pod_id: str) -> None:
    entry = _procs.pop(pod_id, None)
    if entry and entry[0].poll() is None:
        os.killpg(entry[0].pid, signal.SIGTERM)


@app.api_route(
    "/runpod/proxy/{pod_id}/{port}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"]
)
async def runpod_proxy(pod_id: str, port: int, path: str, request: Request):
    entry = _procs.get(pod_id)
    if entry is None or _pods.get(pod_id, {}).get("desiredStatus") != "RUNNING":
        return Response("Bad Gateway", status_code=502)
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "content-type", "range")
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            upstream = await client.request(
                request.method,
                f"http://127.0.0.1:{entry[1]}/{path}",
                params=request.query_params,
                content=await request.body(),
                headers=headers,
            )
    except httpx.HTTPError:
        return Response("Bad Gateway", status_code=502)
    keep = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in ("content-type", "content-range", "x-next-offset", "accept-ranges")
    }
    return Response(upstream.content, status_code=upstream.status_code, headers=keep)


@app.post("/runpod/rest/v1/pods/{pod_id}/{action}")
async def runpod_action(pod_id: str, action: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    if pod_id not in _pods:
        return JSONResponse({"error": "pod not found"}, status_code=404)
    _pods[pod_id]["desiredStatus"] = "RUNNING" if action == "start" else "EXITED"
    if action == "start":
        _spawn(pod_id)
    else:
        _kill(pod_id)
    return {}


@app.delete("/runpod/rest/v1/pods/{pod_id}")
async def runpod_delete(pod_id: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    _pods.pop(pod_id, None)
    _kill(pod_id)
    return {}


@app.get("/runpod/rest/v1/endpoints")
async def runpod_endpoints(request: Request):
    if not _authorized(request):
        return _unauthorized()
    return list(_endpoints.values())


@app.patch("/runpod/rest/v1/endpoints/{endpoint_id}")
async def runpod_scale(endpoint_id: str, request: Request):
    if not _authorized(request):
        return _unauthorized()
    _endpoints[endpoint_id].update(await request.json())
    return {}


@app.get("/runpod/serverless/{endpoint_id}/health")
async def runpod_health(endpoint_id: str):
    return {"jobs": {"completed": 12, "inQueue": 0}, "workers": {"idle": 0, "running": 0}}


# --- Hugging Face Hub (token and license checks) ---------------------------------------------
# Tokens that start with "hf_test" are valid; "hf_test_nolicense…" has not accepted the
# Krea 2 RAW license yet. Each token can "accept" it at /hf/accept?token=… (the real site
# has a button on the repository page instead).

_hf_accepted: set[str] = set()


def _hf_token(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


@app.get("/hf/api/whoami-v2")
async def hf_whoami(request: Request):
    if not _hf_token(request).startswith("hf_test"):
        return JSONResponse({"error": "Invalid credentials in Authorization header"}, 401)
    return {"type": "user", "name": "studio-tester"}


@app.get("/hf/api/models/{repo:path}/auth-check")
async def hf_auth_check(repo: str, request: Request):
    token = _hf_token(request)
    if not token.startswith("hf_test"):
        return JSONResponse({"error": "Invalid credentials in Authorization header"}, 401)
    if (
        token.startswith("hf_test_nolicense")
        and token not in _hf_accepted
        and repo == "krea/Krea-2-Raw"
    ):
        return JSONResponse(
            {"error": "Access to model krea/Krea-2-Raw is restricted."},
            403,
            headers={"X-Error-Code": "GatedRepo"},
        )
    return {}


@app.get("/hf/accept")
async def hf_accept(token: str):
    _hf_accepted.add(token)
    return {"ok": True}
