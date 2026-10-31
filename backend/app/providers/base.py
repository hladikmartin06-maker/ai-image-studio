"""The provider adapter contract and one normalized error type for all providers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

ErrorCode = Literal[
    "invalid_credentials",
    "insufficient_credits",
    "rate_limited",
    "invalid_parameters",
    "unsupported_model",
    "content_blocked",
    "provider_error",
    "network_error",
    "generation_failed",
    "timeout",
    "interrupted",
]

USER_MESSAGES: dict[str, str] = {
    "invalid_credentials": "The API key was rejected by {provider}. Update it in Integrations.",
    "insufficient_credits": "Your {provider} balance is too low for this request. Top up and "
    "try again.",
    "rate_limited": "{provider} is limiting requests right now. Wait a moment and try again.",
    "invalid_parameters": "{provider} rejected the request parameters.",
    "unsupported_model": "This model is not available on your {provider} account.",
    "content_blocked": "{provider} blocked this request or its result by its content policy.",
    "provider_error": "{provider} had an internal problem. Try again later.",
    "network_error": "Could not reach {provider}. Check your internet connection.",
    "generation_failed": "{provider} could not generate the image.",
    "timeout": "{provider} did not finish in time.",
    "interrupted": "The app was stopped while this request was being sent. It was not retried "
    "so that it cannot be charged twice.",
}


@dataclass(eq=False)
class ProviderError(Exception):
    code: ErrorCode
    provider: str
    detail: str = ""
    status_code: int | None = None

    def __str__(self) -> str:
        return self.user_message

    @property
    def user_message(self) -> str:
        from app.catalog import PROVIDERS

        name = PROVIDERS.get(self.provider, {}).get("name", self.provider)
        base = USER_MESSAGES.get(self.code, "{provider} request failed.").format(provider=name)
        if self.detail and self.code not in ("invalid_credentials", "network_error"):
            return f"{base} ({self.detail[:300]})"
        return base


@dataclass
class TaskState:
    status: Literal["processing", "completed", "failed"]
    outputs: list[str] = field(default_factory=list)
    nsfw_flags: list[bool] = field(default_factory=list)
    error: str | None = None


@dataclass
class ConnectionCheck:
    ok: bool
    message: str
    balance: float | None = None


class ProviderAdapter(Protocol):
    key: str

    async def submit(self, api_key: str, model_id: str, payload: dict[str, Any]) -> str: ...

    async def get_task(self, api_key: str, task_id: str) -> TaskState: ...

    async def upload(self, api_key: str, data: bytes, filename: str, mime: str) -> str: ...

    async def check(self, api_key: str) -> ConnectionCheck: ...

    async def balance(self, api_key: str) -> float | None: ...


# --- helpers shared by the adapters ----------------------------------------------------

_CREDIT_WORDS = ("insufficient", "balance", "credit", "quota", "top up", "recharge", "payment")
_CONTENT_WORDS = ("nsfw", "sensitive", "content policy", "safety", "moderation", "inappropriate")
_PARAM_WORDS = ("invalid", "parameter", "must be", "required", "not supported", "out of range")


def body_message(response: httpx.Response) -> str:
    """The provider's own error text, without echoing anything we sent (never credentials)."""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:300].strip()
    if isinstance(data, dict):
        for key in ("message", "error", "msg", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"]
        inner = data.get("data")
        if isinstance(inner, dict) and isinstance(inner.get("error"), str):
            return inner["error"]
    return ""


def classify_message(message: str) -> ErrorCode | None:
    text = message.lower()
    if any(word in text for word in _CREDIT_WORDS):
        return "insufficient_credits"
    if any(word in text for word in _CONTENT_WORDS):
        return "content_blocked"
    return None


def error_from_response(provider: str, response: httpx.Response) -> ProviderError:
    status = response.status_code
    message = body_message(response)
    code: ErrorCode
    if status in (401, 403):
        code = "invalid_credentials"
    elif status == 402:
        code = "insufficient_credits"
    elif status == 429:
        code = "rate_limited"
    elif status == 404:
        code = "unsupported_model"
    elif status in (400, 422):
        code = classify_message(message) or "invalid_parameters"
    elif status >= 500:
        code = "provider_error"
    else:
        code = classify_message(message) or "provider_error"
    return ProviderError(code=code, provider=provider, detail=message, status_code=status)


def failure_from_message(provider: str, message: str | None) -> ProviderError:
    text = message or ""
    code = classify_message(text) or "generation_failed"
    if any(word in text.lower() for word in _PARAM_WORDS) and code == "generation_failed":
        code = "invalid_parameters"
    return ProviderError(code=code, provider=provider, detail=text)


def unwrap(provider: str, response: httpx.Response) -> dict[str, Any]:
    """Both providers answer {code, message, data}; a non-200 `code` is an error as well."""
    if response.status_code >= 400:
        raise error_from_response(provider, response)
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError("provider_error", provider, "Unexpected response") from exc
    if not isinstance(body, dict):
        raise ProviderError("provider_error", provider, "Unexpected response")
    code = body.get("code")
    if isinstance(code, int) and code not in (0, 200):
        fake = httpx.Response(code if 400 <= code < 600 else 400, json=body)
        raise error_from_response(provider, fake)
    data = body.get("data", body)
    return data if isinstance(data, dict) else {"value": data}


def normalize_status(value: Any) -> Literal["processing", "completed", "failed"]:
    text = str(value or "").lower()
    if text in ("completed", "succeeded", "success", "succeed", "done", "finished"):
        return "completed"
    if text in ("failed", "error", "cancelled", "canceled", "timeout", "expired", "deleted"):
        return "failed"
    return "processing"  # created, queued, starting, processing, running, pending


def outputs_of(data: dict[str, Any]) -> list[str]:
    outputs = data.get("outputs") or data.get("output") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    return [o for o in outputs if isinstance(o, str) and o]


def nsfw_of(data: dict[str, Any]) -> list[bool]:
    flags = data.get("has_nsfw_contents")
    if isinstance(flags, list):
        return [bool(f) for f in flags]
    return []


async def request(
    provider: str, client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    try:
        return await client.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        raise ProviderError("network_error", provider, "Request timed out") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("network_error", provider, type(exc).__name__) from exc
