"""Google Gemini API (Generative Language API, v1beta).

  models   GET  {base}/models?pageSize=1000
           -> {models: [{name, supportedGenerationMethods}]}
  chat     POST {base}/models/{model}:streamGenerateContent?alt=sse
           body {systemInstruction, contents[{role: user|model, parts[{text}]}], generationConfig}
           SSE "data: {candidates:[{content:{parts:[{text}]}, finishReason}], usageMetadata}"
Auth: "x-goog-api-key: <key>".
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

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"
_NON_CHAT = ("embedding", "aqa", "imagen", "image", "tts", "live", "veo", "audio", "robotics")


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:300]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return str(data["error"].get("message", ""))
    return ""


class GeminiProvider:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    def _client(self, api_key: str, timeout: float = 120) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout, connect=15),
            transport=self.transport,
        )

    async def list_models(self, api_key: str, base_url: str | None) -> list[str]:
        base = (base_url or DEFAULT_BASE).rstrip("/")
        try:
            async with self._client(api_key, timeout=30) as client:
                response = await client.get(f"{base}/models", params={"pageSize": 1000})
        except httpx.HTTPError as exc:
            raise AIError("network_error", type(exc).__name__) from exc
        if response.status_code >= 400:
            message = _message(response)
            # Gemini answers an invalid key with 400 API_KEY_INVALID.
            code = "invalid_credentials" if "api key" in message.lower() else None
            raise AIError(code or classify_status(response.status_code, message), message)
        ids: set[str] = set()
        for model in response.json().get("models", []):
            name = str(model.get("name", "")).removeprefix("models/")
            methods = model.get("supportedGenerationMethods") or []
            if name and "generateContent" in methods and not any(w in name for w in _NON_CHAT):
                ids.add(name)
        return sorted(ids)

    def _body(self, request: ChatRequest, *, thinking: bool) -> dict[str, Any]:
        contents = [
            {
                "role": "model" if t.role == "assistant" else "user",
                "parts": [
                    *(
                        {"inline_data": {"mime_type": mime, "data": data}}
                        for mime, data in t.images
                    ),
                    {"text": t.content},
                ],
            }
            for t in request.messages
        ]
        config: dict[str, Any] = {"maxOutputTokens": request.max_tokens}
        if thinking and request.model.startswith("gemini-3"):
            # Gemini 3 models take a thinking level; low keeps reasoning spend small.
            config["thinkingConfig"] = {
                "thinkingLevel": "low" if request.effort != "high" else "high"
            }
        return {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": contents,
            "generationConfig": config,
        }

    async def stream(
        self, api_key: str, base_url: str | None, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]:
        base = (base_url or DEFAULT_BASE).rstrip("/")
        url = f"{base}/models/{request.model}:streamGenerateContent"
        thinking = True
        async with self._client(api_key) as client:
            for _attempt in range(2):
                body = self._body(request, thinking=thinking)
                try:
                    async with client.stream(
                        "POST", url, params={"alt": "sse"}, json=body
                    ) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            message = _message(response)
                            if (
                                response.status_code == 400
                                and thinking
                                and "thinking" in message.lower()
                            ):
                                thinking = False
                                continue
                            code = "invalid_credentials" if "api key" in message.lower() else None
                            raise AIError(
                                code or classify_status(response.status_code, message), message
                            )
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
        produced: list[str] = []
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for candidate in chunk.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    text = part.get("text")
                    # Thought summaries (part.thought == true) are not shown.
                    if isinstance(text, str) and text and not part.get("thought"):
                        produced.append(text)
                        yield StreamEvent(text=text)
                reason = candidate.get("finishReason")
                if reason:
                    stop_reason = reason
            meta = chunk.get("usageMetadata")
            if isinstance(meta, dict):
                usage = Usage(
                    input_tokens=int(meta.get("promptTokenCount") or 0),
                    output_tokens=int(meta.get("candidatesTokenCount") or 0)
                    + int(meta.get("thoughtsTokenCount") or 0),
                    cached_tokens=int(meta.get("cachedContentTokenCount") or 0),
                )
            feedback = chunk.get("promptFeedback") or {}
            if feedback.get("blockReason"):
                stop_reason = "SAFETY"
        if usage is None:
            prompt = request.system + "".join(t.content for t in request.messages)
            usage = Usage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens("".join(produced)),
                estimated=True,
            )
        mapped = {"MAX_TOKENS": "max_tokens", "SAFETY": "refusal", "PROHIBITED_CONTENT": "refusal"}
        yield StreamEvent(
            done=True,
            usage=usage,
            stop_reason=mapped.get(stop_reason or "", (stop_reason or "").lower() or None),
            model=request.model,
        )
