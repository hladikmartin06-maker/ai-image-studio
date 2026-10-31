"""OpenAI Chat Completions API and the many APIs that follow it (xAI, OpenRouter, Groq,
DeepSeek, Mistral, Together, Ollama, …).

  models   GET  {base}/models                      -> {data: [{id, …}]}
  chat     POST {base}/chat/completions (stream)   -> SSE "data: {choices:[{delta:{content}}]}"
                                                       last chunk carries `usage` when
                                                       stream_options.include_usage is set
Auth: "Authorization: Bearer <key>".

Optional parameters that not every compatible server accepts (usage in the stream,
reasoning effort) are sent first and dropped on a 400 that names them; such a request was
rejected before any generation, so retrying it cannot be billed twice.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.ai.providers.base import (
    AIError,
    ChatRequest,
    StreamEvent,
    Usage,
    classify_status,
    estimate_tokens,
)

# Not chat models: filtered out of the model list shown to the user.
_NON_CHAT = (
    "embed",
    "tts",
    "whisper",
    "dall-e",
    "image",
    "audio",
    "realtime",
    "transcribe",
    "moderation",
    "search",
    "davinci",
    "babbage",
    "vision-preview",
    "video",
    "sora",
    "rerank",
)


def is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(word in lowered for word in _NON_CHAT)


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:300]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
        if isinstance(data.get("message"), str):
            return data["message"]
    return ""


class OpenAICompatibleProvider:
    def __init__(
        self,
        default_base_url: str | None,
        *,
        use_max_completion_tokens: bool = False,
        reasoning_effort_prefixes: tuple[str, ...] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.default_base_url = default_base_url
        # OpenAI's reasoning models take max_completion_tokens instead of max_tokens.
        self.use_max_completion_tokens = use_max_completion_tokens
        self.reasoning_effort_prefixes = reasoning_effort_prefixes
        self.transport = transport

    def _base(self, base_url: str | None) -> str:
        base = (base_url or self.default_base_url or "").rstrip("/")
        if not base:
            raise AIError("invalid_request", "No API base URL configured")
        return base

    def _client(self, api_key: str, timeout: float = 120) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return httpx.AsyncClient(
            headers=headers, timeout=httpx.Timeout(timeout, connect=15), transport=self.transport
        )

    async def list_models(self, api_key: str, base_url: str | None) -> list[str]:
        try:
            async with self._client(api_key, timeout=30) as client:
                response = await client.get(f"{self._base(base_url)}/models")
        except httpx.HTTPError as exc:
            raise AIError("network_error", type(exc).__name__) from exc
        if response.status_code >= 400:
            message = _message(response)
            raise AIError(classify_status(response.status_code, message), message)
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else data
        ids = [m.get("id") for m in items or [] if isinstance(m, dict)]
        return sorted({i for i in ids if isinstance(i, str) and is_chat_model(i)})

    def _body(self, request: ChatRequest, *, usage: bool, effort: bool) -> dict[str, Any]:
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system}]
        messages += [{"role": t.role, "content": t.content} for t in request.messages]
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": True}
        if self.use_max_completion_tokens:
            body["max_completion_tokens"] = request.max_tokens
        else:
            body["max_tokens"] = request.max_tokens
        if usage:
            body["stream_options"] = {"include_usage": True}
        if effort and request.model.startswith(self.reasoning_effort_prefixes):
            body["reasoning_effort"] = "low" if request.effort == "low" else request.effort
        return body

    async def stream(
        self, api_key: str, base_url: str | None, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]:
        url = f"{self._base(base_url)}/chat/completions"
        options = {"usage": True, "effort": bool(self.reasoning_effort_prefixes)}
        async with self._client(api_key) as client:
            for _attempt in range(3):
                body = self._body(request, **options)
                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            message = _message(response)
                            lowered = message.lower()
                            if (
                                response.status_code == 400
                                and options["usage"]
                                and ("stream_options" in lowered or "include_usage" in lowered)
                            ):
                                options["usage"] = False
                                continue
                            if (
                                response.status_code == 400
                                and options["effort"]
                                and ("reasoning" in lowered)
                            ):
                                options["effort"] = False
                                continue
                            raise AIError(classify_status(response.status_code, message), message)
                        async for event in self._parse(response, request):
                            yield event
                        return
                except httpx.HTTPError as exc:
                    raise AIError("network_error", type(exc).__name__) from exc
        raise AIError("invalid_request", "The provider rejected the request parameters")

    async def _parse(
        self, response: httpx.Response, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]:
        usage: Usage | None = None
        stop_reason: str | None = None
        model: str | None = None
        produced: list[str] = []
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("error"), dict):
                message = chunk["error"].get("message", "")
                raise AIError(classify_status(500, message), message)
            model = chunk.get("model") or model
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, str) and text:
                    produced.append(text)
                    yield StreamEvent(text=text)
                if choice.get("finish_reason"):
                    stop_reason = choice["finish_reason"]
            raw = chunk.get("usage")
            if isinstance(raw, dict):
                details = raw.get("prompt_tokens_details") or {}
                usage = Usage(
                    input_tokens=int(raw.get("prompt_tokens") or 0),
                    output_tokens=int(raw.get("completion_tokens") or 0),
                    cached_tokens=int(details.get("cached_tokens") or 0),
                )
        if usage is None:
            prompt = request.system + "".join(t.content for t in request.messages)
            usage = Usage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens("".join(produced)),
                estimated=True,
            )
        if stop_reason == "length":
            stop_reason = "max_tokens"
        elif stop_reason == "content_filter":
            stop_reason = "refusal"
        yield StreamEvent(done=True, usage=usage, stop_reason=stop_reason, model=model)
