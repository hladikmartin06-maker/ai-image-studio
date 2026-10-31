"""The contract every chat AI provider adapter implements, and one normalized error."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

AIErrorCode = Literal[
    "invalid_credentials",
    "insufficient_credits",
    "rate_limited",
    "model_not_found",
    "invalid_request",
    "refused",
    "provider_error",
    "network_error",
    "not_connected",
]

MESSAGES: dict[str, str] = {
    "invalid_credentials": "The API key was rejected. Update it in Integrations › AI assistant.",
    "insufficient_credits": "The provider account has no credit left.",
    "rate_limited": "The provider is limiting requests. Try again in a moment.",
    "model_not_found": "This model is not available for your key. Pick another one in "
    "Integrations › AI assistant.",
    "invalid_request": "The provider rejected the request.",
    "refused": "The model declined to answer this request.",
    "provider_error": "The provider had an internal problem. Try again later.",
    "network_error": "Could not reach the provider. Check your internet connection.",
    "not_connected": "Connect an AI provider in Integrations › AI assistant first.",
}


@dataclass(eq=False)
class AIError(Exception):
    code: AIErrorCode
    detail: str = ""

    def __str__(self) -> str:
        return self.message

    @property
    def message(self) -> str:
        base = MESSAGES.get(self.code, "The AI request failed.")
        if self.detail and self.code not in ("invalid_credentials", "network_error"):
            return f"{base} ({self.detail[:240]})"
        return base


@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ChatRequest:
    model: str
    system: str
    messages: list[ChatTurn]
    max_tokens: int
    # low | medium | high — mapped to each provider's own effort/reasoning control.
    effort: Literal["low", "medium", "high"] = "low"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated: bool = False


@dataclass
class StreamEvent:
    """A text delta while streaming, then exactly one final event (`done=True`)."""

    text: str = ""
    done: bool = False
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    model: str | None = None


class ChatProvider(Protocol):
    async def list_models(self, api_key: str, base_url: str | None) -> list[str]: ...

    def stream(
        self, api_key: str, base_url: str | None, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]: ...


def estimate_tokens(text: str) -> int:
    """Rough count for budgeting and for providers that report no usage (~3.5 chars/token)."""
    return max(1, round(len(text) / 3.5)) if text else 0


def classify_status(status: int, message: str) -> AIErrorCode:
    text = message.lower()
    if status in (401, 403):
        return "invalid_credentials"
    if status == 402 or any(
        w in text for w in ("credit balance", "insufficient", "quota", "billing")
    ):
        return "insufficient_credits"
    if status == 429:
        return "rate_limited"
    if status == 404:
        return "model_not_found"
    if status >= 500:
        return "provider_error"
    return "invalid_request"
