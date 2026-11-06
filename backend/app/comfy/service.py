"""ComfyUI connections: credentials, URL rules, inspection and LoRA file placement."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.comfy import anypaint, enhance, krea2, pose_control
from app.comfy.client import ComfyClient, ComfyError
from app.gpu.agent_client import AgentClient, AgentError
from app.models import ComfyConnection, Lora, LoraVersion
from app.security import decrypt_secret
from app.services import storage


def comfy_subdir(conn: ComfyConnection) -> str:
    """ComfyUI's folder inside the agent's workspace: runpod-slim/ComfyUI in RunPod's
    ComfyUI image (older studio deployments used ComfyUI)."""
    return str((conn.deploy or {}).get("comfy_dir") or "ComfyUI")


def assoc(conn: ComfyConnection) -> str:
    return f"{conn.user_id}:comfy:{conn.id}"


def check_url(url: str) -> str:
    """HTTPS anywhere; plain HTTP only to this Mac or the local network."""
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Enter a URL such as http://127.0.0.1:8188 or https://…")
    if parts.username or parts.password:
        raise ValueError("Put credentials in the authentication fields, not in the URL.")
    if parts.scheme == "http":
        host = parts.hostname
        local = host in ("localhost",) or host.endswith(".local")
        try:
            ip = ipaddress.ip_address(host)
            local = local or ip.is_loopback or ip.is_private
        except ValueError:
            pass
        if not local:
            raise ValueError("Remote ComfyUI servers must use HTTPS.")
    return url


def secret_of(conn: ComfyConnection) -> str | None:
    if not conn.encrypted_secret:
        return None
    return decrypt_secret(conn.encrypted_secret, associated=assoc(conn))


def agent_for(conn: ComfyConnection) -> AgentClient | None:
    url = (conn.deploy or {}).get("agent_url")
    token = secret_of(conn)
    if conn.auth_type != "agent" or not url or not token:
        return None
    return AgentClient(url, token)


def client_for(conn: ComfyConnection) -> ComfyClient:
    secret = secret_of(conn)
    headers: dict[str, str] = {}
    if conn.auth_type == "agent":
        url = (conn.deploy or {}).get("agent_url")
        if not url or not secret:
            raise ComfyError("unreachable", "This ComfyUI is not deployed yet.")
        return ComfyClient(f"{url.rstrip('/')}/comfy", {"Authorization": f"Bearer {secret}"})
    if conn.auth_type == "bearer" and secret:
        headers["Authorization"] = f"Bearer {secret}"
    elif conn.auth_type == "basic" and secret:
        headers["Authorization"] = "Basic " + base64.b64encode(secret.encode()).decode()
    if not conn.base_url:
        raise ComfyError("unreachable", "No URL configured.")
    return ComfyClient(conn.base_url, headers)


def file_capability(conn: ComfyConnection) -> str:
    """How the studio can put files (LoRAs, models) on this ComfyUI."""
    if conn.auth_type == "agent":
        return "agent"
    if conn.kind == "local" and conn.models_dir:
        return "folder"
    return "none"


async def inspect(conn: ComfyConnection) -> dict[str, Any]:
    client = client_for(conn)
    stats = await client.system_stats()
    info = await client.object_info()
    system = stats.get("system") or {}
    packs = sorted(
        {
            str(n.get("python_module", "")).split(".", 1)[1].split(".")[0]
            for n in info.values()
            if isinstance(n, dict) and str(n.get("python_module", "")).startswith("custom_nodes.")
        }
    )
    files = krea2.installed_files(info)
    result: dict[str, Any] = {
        "version": system.get("comfyui_version"),
        "python": str(system.get("python_version") or "").split(" ")[0] or None,
        "pytorch": system.get("pytorch_version"),
        "os": system.get("os"),
        "ram_total": system.get("ram_total"),
        "ram_free": system.get("ram_free"),
        "devices": [
            {
                "name": d.get("name"),
                "type": d.get("type"),
                "vram_total": d.get("vram_total"),
                "vram_free": d.get("vram_free"),
            }
            for d in stats.get("devices") or []
            if isinstance(d, dict)
        ],
        "node_count": len(info),
        "node_names": sorted(info.keys()),
        "custom_node_packs": packs,
        "files": {k: v[:500] for k, v in files.items()},
        "krea2": krea2.detect(info),
        "enhance": enhance.detect(info),
        "anypaint": anypaint.detect(info),
        "pose": pose_control.detect(info),
        "file_capability": file_capability(conn),
        "checked_at": datetime.now(UTC).isoformat(),
    }
    agent = agent_for(conn)
    if agent is not None:
        try:
            result["agent"] = await agent.health()
        except AgentError:
            result["agent"] = None
    return result


async def object_info(conn: ComfyConnection) -> dict[str, Any]:
    return await client_for(conn).object_info()


# --- LoRA files on a connection ------------------------------------------------------------


def lora_filename(lora: Lora, version: LoraVersion) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", lora.name.lower()).strip("-")[:40] or "lora"
    return f"ai-image-studio/{slug}-{str(lora.id)[:8]}-v{version.number}.safetensors"


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _is_comfy_dir(root: Path) -> bool:
    return (root / "models").is_dir()


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)


async def place_lora(
    db: AsyncSession, conn: ComfyConnection, lora: Lora, version: LoraVersion, force: bool = False
) -> str:
    """Puts the LoRA file into ComfyUI's models/loras (once) and returns the name ComfyUI
    lists it under. Raises ComfyError("unsupported") when this ComfyUI takes no files."""
    if not version.file_path:
        raise ComfyError("missing_models", "This LoRA version has no file yet.")
    name = lora_filename(lora, version)
    done = (version.deployments or {}).get(str(conn.id))
    if (
        not force
        and done
        and done.get("filename") == name
        and done.get("sha256") == version.file_sha256
    ):
        return name
    source = storage.absolute(version.file_path)
    capability = file_capability(conn)
    if capability == "folder":
        root = await asyncio.to_thread(_expand, conn.models_dir or "")
        if not await asyncio.to_thread(_is_comfy_dir, root):
            raise ComfyError(
                "unsupported", f"{root} is not a ComfyUI folder (no models/ folder inside)."
            )
        await asyncio.to_thread(_copy, source, root / "models" / "loras" / name)
    elif capability == "agent":
        agent = agent_for(conn)
        assert agent is not None
        try:
            await agent.upload_file(f"{comfy_subdir(conn)}/models/loras/{name}", source)
        except AgentError as exc:
            raise ComfyError("unreachable", f"Upload to the GPU failed: {exc.message}") from exc
    else:
        raise ComfyError(
            "unsupported",
            "This ComfyUI accepts no file uploads through its API. Copy the LoRA file into "
            f"ComfyUI/models/loras/{name} yourself (download it in My LoRAs), or use a local "
            "folder or a ComfyUI deployed by the studio.",
        )
    version.deployments = {
        **(version.deployments or {}),
        str(conn.id): {
            "filename": name,
            "sha256": version.file_sha256,
            "at": datetime.now(UTC).isoformat(),
        },
    }
    await db.commit()
    return name


async def owned(db: AsyncSession, user_id: uuid.UUID, conn_id: uuid.UUID) -> ComfyConnection | None:
    return await db.scalar(
        select(ComfyConnection).where(
            ComfyConnection.id == conn_id, ComfyConnection.user_id == user_id
        )
    )
