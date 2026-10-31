"""WaveSpeedAI adapter.

API reference (https://wavespeed.ai/docs):
  submit   POST {base}/api/v3/{model_id}                 -> {code, message, data: {id, status, ...}}
  result   GET  {base}/api/v3/predictions/{id}/result    -> data: {status, outputs[], error,
                                                              has_nsfw_contents[]}
  upload   POST {base}/api/v3/media/upload/binary (multipart "file") -> data.download_url
  balance  GET  {base}/api/v3/balance                     -> data.balance (USD)
Authentication: "Authorization: Bearer <API key>".
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.providers.base import (
    ConnectionCheck,
    ProviderError,
    TaskState,
    failure_from_message,
    normalize_status,
    nsfw_of,
    outputs_of,
    request,
    unwrap,
)

KEY = "wavespeed"


class WaveSpeedAdapter:
    key = KEY

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport

    @property
    def base(self) -> str:
        return get_settings().wavespeed_base_url.rstrip("/")

    def _client(self, api_key: str, timeout: float = 60) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15),
            transport=self._transport,
        )

    async def submit(self, api_key: str, model_id: str, payload: dict[str, Any]) -> str:
        async with self._client(api_key) as client:
            response = await request(
                KEY, client, "POST", f"{self.base}/api/v3/{model_id}", json=payload
            )
        data = unwrap(KEY, response)
        task_id = data.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError("provider_error", KEY, "No prediction id in the response")
        return task_id

    async def get_task(self, api_key: str, task_id: str) -> TaskState:
        async with self._client(api_key, timeout=30) as client:
            response = await request(
                KEY, client, "GET", f"{self.base}/api/v3/predictions/{task_id}/result"
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
                f"{self.base}/api/v3/media/upload/binary",
                files={"file": (filename, data, mime)},
            )
        body = unwrap(KEY, response)
        url = body.get("download_url")
        if not isinstance(url, str) or not url:
            raise ProviderError("provider_error", KEY, "The upload returned no URL")
        return url

    async def balance(self, api_key: str) -> float | None:
        async with self._client(api_key, timeout=20) as client:
            response = await request(KEY, client, "GET", f"{self.base}/api/v3/balance")
        data = unwrap(KEY, response)
        value = data.get("balance")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    async def check(self, api_key: str) -> ConnectionCheck:
        try:
            balance = await self.balance(api_key)
        except ProviderError as exc:
            if exc.code == "invalid_credentials":
                return ConnectionCheck(False, exc.user_message)
            raise
        return ConnectionCheck(True, "Connected to WaveSpeed.", balance=balance)


def failure(message: str | None) -> ProviderError:
    return failure_from_message(KEY, message)
