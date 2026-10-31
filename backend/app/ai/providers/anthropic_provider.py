"""Anthropic Claude, through the official `anthropic` Python SDK (Messages API, streaming).

Cost controls used here:
- `output_config.effort` (low for chat, medium for the high-quality tier) on models that
  support it; the newest models think adaptively and effort bounds that spend.
- Top-level automatic prompt caching: the stable system prompt and the replayed history
  are read from cache on follow-up turns (~0.1x input price).
- Server-side refusal fallbacks (`fallbacks: "default"`) on the models that support them,
  so a classifier decline is answered by Anthropic's recommended fallback model instead of
  failing (billing and routing: platform.claude.com refusals-and-fallback docs).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable

import anthropic

from app.ai.providers.base import (
    AIError,
    ChatRequest,
    ChatTurn,
    StreamEvent,
    Usage,
    classify_status,
)

# Models that accept output_config.effort (Opus 4.5+, Sonnet 4.6+, Haiku 5.5, Fable).
_EFFORT_MODELS = re.compile(
    r"^claude-(fable-5|mythos-5|opus-5|opus-4-[5-9]|sonnet-5|sonnet-4-6|haiku-5)"
)
# Models that support the server-side "default" refusal fallback (Claude API).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _error(exc: Exception) -> AIError:
    if isinstance(exc, anthropic.APIConnectionError):
        return AIError("network_error", type(exc).__name__)
    if isinstance(exc, anthropic.APIStatusError):
        message = getattr(exc, "message", "") or str(exc)
        return AIError(classify_status(exc.status_code, message), message)
    return AIError("provider_error", type(exc).__name__)


class AnthropicProvider:
    def __init__(self, http_client_factory: Callable[[], object] | None = None):
        # Tests inject an httpx2 mock transport through this factory.
        self._http_client_factory = http_client_factory

    def _client(self, api_key: str) -> anthropic.AsyncAnthropic:
        kwargs: dict = {"api_key": api_key, "max_retries": 1, "timeout": 120.0}
        if self._http_client_factory is not None:
            kwargs["http_client"] = self._http_client_factory()
        return anthropic.AsyncAnthropic(**kwargs)

    async def list_models(self, api_key: str, base_url: str | None) -> list[str]:
        client = self._client(api_key)
        try:
            ids = [m.id async for m in client.models.list(limit=100)]
        except Exception as exc:  # noqa: BLE001 - normalized below
            raise _error(exc) from exc
        finally:
            await client.close()
        return sorted(set(ids))

    async def stream(
        self, api_key: str, base_url: str | None, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]:
        client = self._client(api_key)
        params: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [_message(t) for t in request.messages],
            "cache_control": {"type": "ephemeral"},
        }
        if _EFFORT_MODELS.match(request.model):
            params["output_config"] = {"effort": request.effort}
        use_fallback = request.model in _FALLBACK_MODELS
        try:
            if use_fallback:
                manager = client.beta.messages.stream(
                    **params, betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}
                )
            else:
                manager = client.messages.stream(**params)
            async with manager as stream:
                async for text in stream.text_stream:
                    if text:
                        yield StreamEvent(text=text)
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise _error(exc) from exc
        finally:
            await client.close()

        usage = final.usage
        yield StreamEvent(
            done=True,
            stop_reason=final.stop_reason,
            model=final.model,
            usage=Usage(
                input_tokens=(usage.input_tokens or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0),
                output_tokens=usage.output_tokens or 0,
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
        )


def _message(turn: ChatTurn) -> dict:
    if not turn.images:
        return {"role": turn.role, "content": turn.content}
    blocks: list[dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
        for mime, data in turn.images
    ]
    blocks.append({"type": "text", "text": turn.content})
    return {"role": turn.role, "content": blocks}
