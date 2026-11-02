"""Atlas Cloud adapter.

API reference (https://www.atlascloud.ai/docs):
  submit   POST {base}/api/v1/model/generateImage  body {model, prompt, ...} -> data.id
  result   GET  {base}/api/v1/model/prediction/{id} -> data: {status, outputs[], error}
           status: processing | completed | failed
  upload   POST {base}/api/v1/model/uploadMedia (multipart "file") -> data.download_url
Authentication: "Authorization: Bearer <API key>".
Atlas documents no balance endpoint and no price API (prices are on its pricing page), so
the app shows neither for it.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.config import get_settings
from app.providers.base import (
    ConnectionCheck,
    ProviderError,
    TaskState,
    error_from_response,
    normalize_status,
    nsfw_of,
    outputs_of,
    request,
    unwrap,
)

KEY = "atlas"


class AtlasAdapter:
    key = KEY

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport

    @property
    def base(self) -> str:
        return get_settings().atlas_base_url.rstrip("/")

    def _client(self, api_key: str, timeout: float = 60) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15),
            transport=self._transport,
        )

    async def submit(self, api_key: str, model_id: str, payload: dict[str, Any]) -> str:
        body = {"model": model_id, **payload}
        async with self._client(api_key) as client:
            response = await request(
                KEY, client, "POST", f"{self.base}/api/v1/model/generateImage", json=body
            )
        data = unwrap(KEY, response)
        task_id = data.get("id") or data.get("prediction_id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError("provider_error", KEY, "No prediction id in the response")
        return task_id

    async def get_task(self, api_key: str, task_id: str) -> TaskState:
        async with self._client(api_key, timeout=30) as client:
            response = await request(
                KEY, client, "GET", f"{self.base}/api/v1/model/prediction/{task_id}"
            )
        data = unwrap(KEY, response)
        status = normalize_status(data.get("status"))
        state = TaskState(status=status, outputs=outputs_of(data), nsfw_flags=nsfw_of(data))
        if status == "failed":
            state.error = str(data.get("error") or "")
        elif status == "completed" and not state.outputs:
            state.status = "failed"
            state.error = str(data.get("error") or "The provider returned no image")
        return state

    async def upload(self, api_key: str, data: bytes, filename: str, mime: str) -> str:
        async with self._client(api_key, timeout=120) as client:
            response = await request(
                KEY,
                client,
                "POST",
                f"{self.base}/api/v1/model/uploadMedia",
                files={"file": (filename, data, mime)},
            )
        body = unwrap(KEY, response)
        url = body.get("download_url")
        if not isinstance(url, str) or not url:
            raise ProviderError("provider_error", KEY, "The upload returned no URL")
        return url

    async def balance(self, api_key: str) -> float | None:
        return None

    async def list_prices(self, api_key: str) -> dict[str, float]:
        return {}

    async def estimate(self, api_key: str, model_id: str, payload: dict[str, Any]) -> float | None:
        return None

    async def check(self, api_key: str) -> ConnectionCheck:
        """Atlas has no documented "who am I" call. Looking up a prediction id that cannot
        exist costs nothing: 401/403 means the key is rejected, anything else means the key
        was accepted (the lookup itself then fails with "not found")."""
        probe = f"connection-test-{uuid.uuid4().hex}"
        async with self._client(api_key, timeout=20) as client:
            response = await request(
                KEY, client, "GET", f"{self.base}/api/v1/model/prediction/{probe}"
            )
        if response.status_code in (401, 403):
            return ConnectionCheck(False, error_from_response(KEY, response).user_message)
        if response.status_code >= 500:
            raise error_from_response(KEY, response)
        return ConnectionCheck(True, "Connected to Atlas Cloud.")
