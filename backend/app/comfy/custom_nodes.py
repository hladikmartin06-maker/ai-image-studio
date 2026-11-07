"""Custom ComfyUI nodes the studio puts on a rented GPU when it starts.

Advanced (comfy/k2edit.py, the owner's request of 2026-11-07) edits photos with Krea 2's
in-context edit LoRAs, which need two nodes ComfyUI v0.35.0 does not have:
lbouaraba/comfyui-krea2edit v2.0.0 (Apache-2.0), one Python file plus its pyproject.toml (the
file reads its version from it), no package to install (torch and einops come with ComfyUI).
The Mac downloads the files of the pinned commit from GitHub, checks each by SHA-256 and sends
them with the GPU's start command, which unpacks them into ComfyUI's custom_nodes before
ComfyUI starts (gpu/agent COMFY_BOOT). A file that does not match is never sent; nothing is
installed with pip.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import tarfile
from dataclasses import dataclass

import httpx

from app.services import storage

log = logging.getLogger("studio.comfy")


@dataclass(frozen=True)
class CustomNode:
    folder: str  # under ComfyUI/custom_nodes
    repo: str  # github.com/<repo>
    commit: str
    files: tuple[tuple[str, str], ...]  # (path, SHA-256)

    def url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{path}"


# Tag v2.0.0 of 2026-10-16 ("reference placement"), Apache-2.0, checked 2026-11-07.
KREA2EDIT = CustomNode(
    folder="comfyui-krea2edit",
    repo="lbouaraba/comfyui-krea2edit",
    commit="13ef60f20828949becada815d3cbc371eb5ccaab",
    files=(
        ("__init__.py", "20c89b7ac7a8c911c05a983b0c1b2b519b7feb7387d2f9d2cb6134ef1731bf77"),
        ("pyproject.toml", "19abb3d580947be79181c05f427c33c67776cb2d8b6f5f63249dd4e5559a5ba0"),
    ),
)
NODES = (KREA2EDIT,)
_lock = asyncio.Lock()
transport: httpx.AsyncBaseTransport | None = None  # tests answer GitHub themselves


def _cached(node: CustomNode, path: str):
    return storage.root() / "custom_nodes" / node.folder / node.commit / path


async def _file(client: httpx.AsyncClient, node: CustomNode, path: str, sha: str) -> bytes:
    cached = _cached(node, path)
    if cached.is_file():
        data = await asyncio.to_thread(cached.read_bytes)
        if hashlib.sha256(data).hexdigest() == sha:
            return data
    response = await client.get(node.url(path))
    response.raise_for_status()
    data = response.content
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{node.repo}/{path} does not match its pinned SHA-256")
    cached.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(cached.write_bytes, data)
    return data


async def packed() -> str | None:
    """The pinned nodes as a gzipped tar (base64) for the start command, or None when they
    cannot be fetched or checked (the GPU then starts without them; Advanced says which node
    is missing)."""
    async with _lock:
        archive = io.BytesIO()
        try:
            async with httpx.AsyncClient(
                timeout=20, follow_redirects=True, transport=transport
            ) as client:
                with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                    for node in NODES:
                        for path, sha in node.files:
                            data = await _file(client, node, path, sha)
                            info = tarfile.TarInfo(f"{node.folder}/{path}")
                            info.size = len(data)
                            info.mode = 0o644
                            tar.addfile(info, io.BytesIO(data))
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.warning("Custom ComfyUI nodes not added to the GPU: %s", exc)
            return None
        return base64.b64encode(archive.getvalue()).decode()
