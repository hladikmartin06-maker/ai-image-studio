"""Fake chat AI providers with the documented wire formats (OpenAI, Gemini, Anthropic)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2

AI_KEY = "sk-test-ai-key-123456"


def _sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    out = []
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append((f"event: {name}\n" if name else "") + f"data: {payload}\n\n")
    return "".join(out).encode()


@dataclass
class FakeAI:
    reply: str = "Use Wan 2.7 Pro for this.\n\n```prompt\na red fox in fresh snow, golden hour\n```"
    openai_models: list[str] = field(
        default_factory=lambda: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "text-embedding-3"]
    )
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_status: int | None = None
    reject_stream_options: bool = False
    refuse: bool = False

    # --- OpenAI-compatible (httpx) --------------------------------------------------------

    def openai_transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._openai)

    def _openai(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {AI_KEY}":
            return httpx.Response(401, json={"error": {"message": "Incorrect API key"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.openai_models]})
        body = json.loads(request.content)
        self.calls.append({"provider": "openai", "body": body, "url": str(request.url)})
        if self.fail_status:
            return httpx.Response(self.fail_status, json={"error": {"message": "quota exceeded"}})
        if self.reject_stream_options and "stream_options" in body:
            return httpx.Response(400, json={"error": {"message": "Unknown field stream_options"}})
        events: list[tuple[str | None, Any]] = []
        for word in self.reply.split(" "):
            events.append(
                (None, {"model": body["model"], "choices": [{"delta": {"content": word + " "}}]})
            )
        events.append((None, {"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        if "stream_options" in body:
            events.append(
                (
                    None,
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 1200,
                            "completion_tokens": 40,
                            "prompt_tokens_details": {"cached_tokens": 1024},
                        },
                    },
                )
            )
        events.append((None, "[DONE]"))
        return httpx.Response(
            200, content=_sse(events), headers={"content-type": "text/event-stream"}
        )

    # --- Gemini (httpx) -------------------------------------------------------------------

    def gemini_transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._gemini)

    def _gemini(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-goog-api-key") != AI_KEY:
            return httpx.Response(
                400, json={"error": {"message": "API key not valid. Please pass a valid API key."}}
            )
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3.5-flash-lite",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-3.8-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-3.1-pro-preview",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/text-embedding-004",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ]
                },
            )
        body = json.loads(request.content)
        self.calls.append({"provider": "gemini", "body": body, "url": str(request.url)})
        events = [
            (None, {"candidates": [{"content": {"parts": [{"text": "Gemini "}]}}]}),
            (
                None,
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "answer."}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 12},
                },
            ),
        ]
        return httpx.Response(
            200, content=_sse(events), headers={"content-type": "text/event-stream"}
        )

    # --- Anthropic (httpx2, the SDK's HTTP library) ----------------------------------------

    def anthropic_client(self) -> Any:
        import anthropic

        return anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self._anthropic))

    def _anthropic(self, request: httpx2.Request) -> httpx2.Response:
        sent_key = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        if sent_key != AI_KEY:
            return httpx2.Response(
                401,
                json={
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                },
            )
        if request.url.path.startswith("/v1/models"):
            ids = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
            return httpx2.Response(
                200,
                json={
                    "data": [
                        {
                            "type": "model",
                            "id": i,
                            "display_name": i,
                            "created_at": "2026-01-01T00:00:00Z",
                        }
                        for i in ids
                    ],
                    "has_more": False,
                    "first_id": ids[0],
                    "last_id": ids[-1],
                },
            )
        body = json.loads(request.content)
        self.calls.append(
            {
                "provider": "anthropic",
                "body": body,
                "beta": request.headers.get("anthropic-beta"),
            }
        )
        model = body["model"]
        events: list[tuple[str | None, Any]] = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 50,
                            "output_tokens": 1,
                            "cache_read_input_tokens": 950,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]
        for word in ["Claude ", "says ", "hello."]:
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": word},
                    },
                )
            )
        events += [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal" if self.refuse else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 30},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        return httpx2.Response(
            200, content=_sse(events), headers={"content-type": "text/event-stream"}
        )
