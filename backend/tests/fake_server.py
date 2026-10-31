"""A local stand-in for WaveSpeed and Atlas Cloud, for trying the full app without a
provider account (development and the cloud sandbox only; never used in production).

Run:  uv run uvicorn tests.fake_server:app --host 127.0.0.1 --port 8900
Then set in backend/.env:
  WAVESPEED_BASE_URL=http://127.0.0.1:8900/ws
  ATLAS_BASE_URL=http://127.0.0.1:8900/atlas
Any API key that starts with "test-" is accepted. Prompts containing "fail" fail.
The images it returns are generated test patterns, clearly not AI output.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import itertools
from typing import Any

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
