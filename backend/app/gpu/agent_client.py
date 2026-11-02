"""HTTP client for the studio agent running in a GPU container (see gpu/agent)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import posixpath
from pathlib import Path
from typing import Any

import httpx

from app.config import USER_AGENT

# 8 MB per request: each chunk stays far below the time limits of the proxies in front of a
# rented GPU (RunPod's is ~100 s) even on a slow home upload.
UPLOAD_CHUNK = 8 * 1024 * 1024
DOWNLOAD_CHUNK = 32 * 1024 * 1024


class AgentError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status: int | None = None,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code  # unreachable | unauthorized | not_found | rejected | agent_error
        self.message = message
        self.status = status
        # The agent's JSON answer (e.g. "offset mismatch" carries the file's real size).
        self.detail = detail or {}


# One upload per file on an agent at a time: two uploads of the same file (e.g. two runs
# with the same LoRA) would otherwise write over each other.
_UPLOAD_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


class AgentClient:
    # Tests point this at an in-process transport.
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}

    def _client(self, wait: float = 60) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=httpx.Timeout(wait, connect=15),
            transport=self.transport,
        )

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail: dict[str, Any] = {}
        try:
            body = response.json()
            detail = body if isinstance(body, dict) else {}
            message = str(detail.get("error") or "")
        except ValueError:
            message = response.text[:200]
        code = {401: "unauthorized", 403: "unauthorized", 404: "not_found"}.get(
            response.status_code, "rejected" if response.status_code < 500 else "agent_error"
        )
        raise AgentError(
            code, message or f"HTTP {response.status_code}", response.status_code, detail
        )

    async def _request(self, method: str, path: str, *, wait: float = 60, **kw: Any):
        try:
            async with self._client(wait) as client:
                response = await client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise AgentError("unreachable", type(exc).__name__) from exc
        self._check(response)
        return response

    async def health(self) -> dict[str, Any]:
        return (await self._request("GET", "/health", wait=20)).json()

    async def list_files(self, folder: str) -> list[dict[str, Any]]:
        response = await self._request("GET", "/files", params={"dir": folder})
        return list(response.json().get("files", []))

    async def upload_bytes(self, remote: str, data: bytes) -> None:
        await self._request("PUT", f"/files/{remote}", params={"offset": 0}, content=data)

    async def upload_file(self, remote: str, local: Path) -> None:
        """Chunked upload with resume. A chunk whose answer got lost on the way (a proxy
        timeout, a dropped connection) may still have reached the agent, fully or in part:
        the agent then refuses the repeated chunk with "offset mismatch" and its real size,
        and the upload continues from there. At the end the file's size on the GPU is
        checked; a wrong one is uploaded again from the start."""
        size = await asyncio.to_thread(lambda: local.stat().st_size)
        lock = _UPLOAD_LOCKS.setdefault((self.base_url, remote), asyncio.Lock())
        async with lock:
            for _ in range(3):
                await self._upload_once(remote, local, size)
                if await self._remote_size(remote) in (size, None):
                    return
        raise AgentError("agent_error", "The file on the GPU did not match after the upload")

    async def _upload_once(self, remote: str, local: Path, size: int) -> None:
        offset = 0
        attempts = 0
        while True:
            chunk = await asyncio.to_thread(_read_chunk, local, offset, UPLOAD_CHUNK)
            try:
                response = await self._request(
                    "PUT",
                    f"/files/{remote}",
                    params={"offset": offset},
                    content=chunk,
                    wait=300,
                )
                offset = int(response.json().get("size", offset + len(chunk)))
                attempts = 0
            except AgentError as exc:
                attempts += 1
                have = exc.detail.get("size")
                if exc.status == 409 and isinstance(have, int) and attempts <= 8:
                    # Only bytes of the chunk just sent can be this upload's own (the lock
                    # keeps other uploads of the file out); anything else: start again.
                    offset = have if offset < have <= offset + len(chunk) else 0
                    continue
                if attempts > 4 or exc.code in ("unauthorized", "rejected"):
                    raise
                await asyncio.sleep(2 * attempts)
                continue
            if offset >= size:
                return

    async def _remote_size(self, remote: str) -> int | None:
        """The size of an uploaded file on the agent (None when it cannot be listed)."""
        target = posixpath.normpath(remote)
        try:
            files = await self.list_files(posixpath.dirname(target))
        except AgentError:
            return None
        paths = {posixpath.normpath(str(f.get("path", ""))): int(f.get("size") or 0) for f in files}
        if target in paths:
            return paths[target]
        # The agent lists real paths: a folder reached through a link shows under another
        # name, so the file is found by its name when that is unambiguous.
        name = posixpath.basename(target)
        same = [size for path, size in paths.items() if posixpath.basename(path) == name]
        return same[0] if len(same) == 1 else None

    async def download_file(self, remote: str, local: Path, size: int | None = None) -> str:
        """Ranged download (resumable) into a .part file; returns the SHA-256."""
        part = local.with_suffix(local.suffix + ".part")
        offset = await asyncio.to_thread(_prepare_part, part)
        attempts = 0
        while True:
            end = offset + DOWNLOAD_CHUNK - 1
            try:
                response = await self._request(
                    "GET", f"/files/{remote}", headers={"Range": f"bytes={offset}-{end}"}, wait=300
                )
            except AgentError as exc:
                if exc.status == 416:  # nothing left
                    break
                attempts += 1
                if attempts > 5 or exc.code in ("unauthorized", "not_found"):
                    raise
                await asyncio.sleep(2 * attempts)
                continue
            attempts = 0
            data = response.content
            await asyncio.to_thread(_append, part, data)
            offset += len(data)
            total = None
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                total = int(content_range.rsplit("/", 1)[1])
            elif response.status_code == 200:
                total = offset
            if (total is not None and offset >= total) or not data:
                break
        if size is not None and offset != size:
            raise AgentError("agent_error", f"Download incomplete ({offset} of {size} bytes)")
        return await asyncio.to_thread(_finish_part, part, local)

    async def delete(self, remote: str) -> None:
        await self._request("DELETE", f"/files/{remote}")

    async def start_job(self, job_id: str, kind: str, args: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "POST", "/jobs", json={"id": job_id, "kind": kind, "args": args}
        )
        return response.json()

    async def job(self, job_id: str) -> dict[str, Any]:
        return (await self._request("GET", f"/jobs/{job_id}")).json()

    async def logs(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        response = await self._request("GET", f"/jobs/{job_id}/logs", params={"offset": offset})
        next_offset = int(response.headers.get("X-Next-Offset", offset + len(response.content)))
        return response.content.decode("utf-8", "replace"), next_offset

    async def boot_log(self, offset: int = 0) -> tuple[str, int]:
        """The container's start log (ComfyUI's output) from a byte offset."""
        response = await self._request("GET", "/boot-log", params={"offset": offset}, wait=20)
        next_offset = int(response.headers.get("X-Next-Offset", offset + len(response.content)))
        return response.content.decode("utf-8", "replace"), next_offset

    async def cancel(self, job_id: str) -> None:
        await self._request("POST", f"/jobs/{job_id}/cancel")


def _read_chunk(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(size)


def _prepare_part(part: Path) -> int:
    part.parent.mkdir(parents=True, exist_ok=True)
    return part.stat().st_size if part.exists() else 0


def _append(part: Path, data: bytes) -> None:
    with part.open("ab") as f:
        f.write(data)


def _finish_part(part: Path, local: Path) -> str:
    digest = hashlib.sha256()
    with part.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    os.replace(part, local)
    return digest.hexdigest()
