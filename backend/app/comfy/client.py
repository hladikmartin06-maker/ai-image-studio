"""Client for ComfyUI's own HTTP API (server.py in comfyanonymous/ComfyUI):

GET  /system_stats            version, Python/PyTorch, devices with VRAM total/free
GET  /object_info[/{node}]    node definitions; loader nodes list the installed files
GET  /models/{folder}         files in a model folder (newer versions)
POST /prompt                  {prompt: <API-format workflow>, client_id, prompt_id}
                              -> {prompt_id, number, node_errors}; 400 {error, node_errors}
GET  /queue                   {queue_running, queue_pending}  items: [number, prompt_id, ...]
GET  /history/{prompt_id}     {prompt_id: {outputs, status{status_str, completed, messages}}}
GET  /view?filename&subfolder&type
POST /interrupt {prompt_id}   stops the running prompt; POST /queue {delete: [ids]}
WS   /ws?clientId=            "progress" {value, max, prompt_id}
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import USER_AGENT


class ComfyError(Exception):
    """code: unreachable | unauthorized | invalid_workflow | not_found | server_error |
    unsupported | missing_models"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _node_errors(data: dict[str, Any]) -> str:
    parts: list[str] = []
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        parts.append(str(error["message"]))
    elif isinstance(error, str):
        parts.append(error)
    for node_id, info in (data.get("node_errors") or {}).items():
        if not isinstance(info, dict):
            continue
        for err in info.get("errors") or []:
            detail = err.get("details") or err.get("message") or ""
            parts.append(f"{info.get('class_type', node_id)}: {detail}")
    return "; ".join(p for p in parts if p)[:600] or "ComfyUI rejected the workflow."


class ComfyClient:
    # Tests replace this with a mock transport.
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    def _client(self, wait: float = 30) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"User-Agent": USER_AGENT, **self.headers},
            timeout=httpx.Timeout(wait, connect=10),
            transport=self.transport,
        )

    async def _request(self, method: str, path: str, *, wait: float = 30, **kw: Any):
        try:
            async with self._client(wait) as client:
                response = await client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ComfyError(
                "unreachable", f"ComfyUI is not reachable ({type(exc).__name__})."
            ) from exc
        if response.status_code in (401, 403):
            raise ComfyError("unauthorized", "ComfyUI refused the credentials.")
        if response.status_code == 404:
            raise ComfyError("not_found", "Not found on this ComfyUI.")
        if response.status_code == 502:
            raise ComfyError("unreachable", "ComfyUI is not running on the GPU yet.")
        if response.status_code >= 500:
            raise ComfyError("server_error", f"ComfyUI error (HTTP {response.status_code}).")
        return response

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ComfyError(
                "server_error", "The server did not answer like ComfyUI (no JSON)."
            ) from exc

    async def system_stats(self) -> dict[str, Any]:
        data = self._json(await self._request("GET", "/system_stats"))
        if not isinstance(data, dict) or "system" not in data:
            raise ComfyError("server_error", "The server did not answer like ComfyUI.")
        return data

    async def object_info(self, node: str | None = None) -> dict[str, Any]:
        path = f"/object_info/{node}" if node else "/object_info"
        data = self._json(await self._request("GET", path, wait=60))
        return data if isinstance(data, dict) else {}

    async def models(self, folder: str) -> list[str] | None:
        try:
            data = self._json(await self._request("GET", f"/models/{folder}"))
        except ComfyError as exc:
            if exc.code == "not_found":
                return None
            raise
        return [str(x) for x in data] if isinstance(data, list) else None

    async def queue(self) -> dict[str, Any]:
        data = self._json(await self._request("GET", "/queue"))
        return data if isinstance(data, dict) else {}

    async def submit(self, workflow: dict[str, Any], prompt_id: str, client_id: str) -> str:
        response = await self._request(
            "POST",
            "/prompt",
            json={"prompt": workflow, "client_id": client_id, "prompt_id": prompt_id},
            wait=60,
        )
        data = self._json(response)
        if response.status_code >= 400:
            raise ComfyError("invalid_workflow", _node_errors(data))
        if data.get("node_errors"):
            raise ComfyError("invalid_workflow", _node_errors(data))
        # Older ComfyUI versions ignore a client prompt_id and return their own.
        return str(data.get("prompt_id") or prompt_id)

    async def upload_image(self, data: bytes, filename: str, mime: str) -> str:
        """POST /upload/image (ComfyUI's input folder); returns the name ComfyUI stored."""
        response = await self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, data, mime)},
            data={"type": "input", "overwrite": "true"},
            wait=120,
        )
        payload = self._json(response)
        if response.status_code >= 400 or not isinstance(payload, dict) or not payload.get("name"):
            raise ComfyError("server_error", "ComfyUI did not accept the photo.")
        sub = str(payload.get("subfolder") or "")
        return f"{sub}/{payload['name']}" if sub else str(payload["name"])

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self._json(await self._request("GET", f"/history/{prompt_id}"))
        if isinstance(data, dict) and isinstance(data.get(prompt_id), dict):
            return data[prompt_id]
        return None

    async def view(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        response = await self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": folder_type},
            wait=120,
        )
        return response.content

    async def interrupt(self, prompt_id: str) -> None:
        await self._request("POST", "/interrupt", json={"prompt_id": prompt_id})

    async def delete_queued(self, prompt_id: str) -> None:
        await self._request("POST", "/queue", json={"delete": [prompt_id]})


def queue_position(queue: dict[str, Any], prompt_id: str) -> tuple[str, int | None]:
    """("running", None) | ("pending", 1-based position) | ("absent", None)."""
    for item in queue.get("queue_running") or []:
        if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
            return "running", None
    pending = [i for i in queue.get("queue_pending") or [] if isinstance(i, list) and len(i) > 1]
    pending.sort(key=lambda i: i[0] if isinstance(i[0], int | float) else 0)
    for index, item in enumerate(pending):
        if item[1] == prompt_id:
            return "pending", index + 1
    return "absent", None


def history_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
    """Saved images ({filename, subfolder, type}) of a finished prompt, temp previews last."""
    images: list[dict[str, str]] = []
    for output in (entry.get("outputs") or {}).values():
        for image in (output or {}).get("images") or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(
                    {
                        "filename": str(image["filename"]),
                        "subfolder": str(image.get("subfolder") or ""),
                        "type": str(image.get("type") or "output"),
                    }
                )
    saved = [i for i in images if i["type"] == "output"]
    return saved or images


def history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status") or {}
    if status.get("status_str") != "error":
        return None
    for message in status.get("messages") or []:
        if (
            isinstance(message, list)
            and len(message) == 2
            and message[0]
            in (
                "execution_error",
                "execution_interrupted",
            )
        ):
            data = message[1] or {}
            if message[0] == "execution_interrupted":
                return "Interrupted."
            node = data.get("node_type") or ""
            text = data.get("exception_message") or "Execution failed"
            return f"{node}: {text}".strip(": ")[:600]
    return "ComfyUI reported an error."
