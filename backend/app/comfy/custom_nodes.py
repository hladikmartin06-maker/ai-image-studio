"""Custom ComfyUI nodes the studio puts on a rented GPU when it starts.

The owner's request (2026-11-06, Advanced phase 1, approved): Krea 2 AnyPaint redraws regions
(comfy/anypaint.py). ComfyUI v0.35.0 has no AnyPaint nodes; they come from
alexw5702-afk/krea2-anypaint (MIT) — four Python files, no package to install (einops, numpy
and Pillow are ComfyUI's own requirements). The Mac downloads the files of the pinned commit
from GitHub, checks each by SHA-256 and sends them with the GPU's start command, which unpacks
them into ComfyUI's custom_nodes before ComfyUI starts (gpu/agent COMFY_BOOT). A file that
does not match is never sent; nothing is installed with pip. An exception to "built-in nodes
only", approved by the owner.
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


# v0.1.0 of 2026-08-04 ("Prepare v0.1.0 Comfy Registry package"), MIT, checked 2026-11-06.
ANYPAINT = CustomNode(
    folder="krea2-anypaint",
    repo="alexw5702-afk/krea2-anypaint",
    commit="675be5a91eadbf8b8997b21c0e8e1848310b9571",
    files=(
        ("__init__.py", "4c0f96b0f0006df5a559d09d44b8d521a32d7992383223310b5ba2a3f88e3e42"),
        ("nodes.py", "08559583183fdded003f7a9b7769fc746c63245e29fc87994800820570c9183e"),
        ("anypaint_helpers.py", "b1ed902f82e09eb5dcc86caff44072e709ab2b7fe8ee7c3c9319e996a83d1fd4"),
        (
            "krea2_registered_core.py",
            "0da66ac85a7b8b0f69b13440577d2e27f014e6a24ccd87a2e012d48958e823c0",
        ),
    ),
)
# v1.0.1 of 2026-07-17 ("Version bump"), MIT, checked 2026-11-06 (phase 2: the pose LoRA).
OSTRIS_EDIT = CustomNode(
    folder="ComfyUI-Krea2-Ostris-Edit",
    repo="ostris/ComfyUI-Krea2-Ostris-Edit",
    commit="7756566160c4a1b24bb1bd9f0ff3ced1a83d7547",
    files=(
        ("__init__.py", "4c0f96b0f0006df5a559d09d44b8d521a32d7992383223310b5ba2a3f88e3e42"),
        ("nodes.py", "63952781029bcd455c8855cbba3f3ff0696f9ed2fa9220c7033a215b04cc7d99"),
    ),
)
NODES = (ANYPAINT, OSTRIS_EDIT)
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
