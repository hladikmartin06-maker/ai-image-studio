"""Large file uploads from the browser in chunks (LoRA files, model files).

The Next.js proxy limits one request body, so files are sent in 8 MB pieces and appended
here: storage/<user>/uploads/<id>.part plus <id>.json (name, size). An upload is used once
by the endpoint that imports it, and removed after a day if it never is.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from app.services import storage

CHUNK_SIZE = 8 * 1024 * 1024
MAX_SIZE = 40 * 1024 * 1024 * 1024
MAX_AGE_SECONDS = 24 * 3600


class UploadError(ValueError):
    pass


def _dir(user_id: uuid.UUID) -> Path:
    path = storage.root() / str(user_id) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths(user_id: uuid.UUID, upload_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise UploadError("Unknown upload.")
    base = _dir(user_id)
    return base / f"{upload_id}.part", base / f"{upload_id}.json"


def _create(user_id: uuid.UUID, filename: str, size: int) -> str:
    if size <= 0 or size > MAX_SIZE:
        raise UploadError("File size not allowed.")
    upload_id = uuid.uuid4().hex
    part, meta = _paths(user_id, upload_id)
    part.write_bytes(b"")
    name = Path(filename).name[:200] or "file"
    meta.write_text(json.dumps({"filename": name, "size": size, "created": time.time()}))
    return upload_id


def _append(user_id: uuid.UUID, upload_id: str, offset: int, data: bytes) -> int:
    part, meta = _paths(user_id, upload_id)
    if not meta.exists():
        raise UploadError("Unknown upload.")
    info = json.loads(meta.read_text())
    current = part.stat().st_size
    if offset != current:
        raise UploadError(f"Expected offset {current}.")
    if current + len(data) > info["size"]:
        raise UploadError("More data than announced.")
    with part.open("ab") as f:
        f.write(data)
    return current + len(data)


def _take(user_id: uuid.UUID, upload_id: str) -> tuple[Path, str]:
    part, meta = _paths(user_id, upload_id)
    if not meta.exists():
        raise UploadError("Unknown upload.")
    info = json.loads(meta.read_text())
    if part.stat().st_size != info["size"]:
        raise UploadError("The upload is not complete.")
    return part, info["filename"]


def _discard(user_id: uuid.UUID, upload_id: str) -> None:
    for path in _paths(user_id, upload_id):
        path.unlink(missing_ok=True)


def _cleanup() -> int:
    removed = 0
    root = storage.root()
    for meta in root.glob("*/uploads/*.json"):
        try:
            created = json.loads(meta.read_text()).get("created", 0)
        except (OSError, ValueError):
            created = 0
        if time.time() - created > MAX_AGE_SECONDS:
            meta.with_suffix(".part").unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            removed += 1
    return removed


async def create(user_id: uuid.UUID, filename: str, size: int) -> str:
    return await asyncio.to_thread(_create, user_id, filename, size)


async def append(user_id: uuid.UUID, upload_id: str, offset: int, data: bytes) -> int:
    return await asyncio.to_thread(_append, user_id, upload_id, offset, data)


async def take(user_id: uuid.UUID, upload_id: str) -> tuple[Path, str]:
    return await asyncio.to_thread(_take, user_id, upload_id)


async def discard(user_id: uuid.UUID, upload_id: str) -> None:
    await asyncio.to_thread(_discard, user_id, upload_id)


async def cleanup() -> int:
    return await asyncio.to_thread(_cleanup)
