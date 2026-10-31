"""AI assistant: providers, routing, context, streaming, cost controls, isolation."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.ai.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.openai_compat import OpenAICompatibleProvider
from app.ai.registry import AI_PROVIDERS, set_adapter
from app.ai.router import route
from app.ai.service import cache_clear, drain_background
from app.db import SessionLocal
from app.models import AIIntegration, ChatConversation
from tests.ai_fakes import AI_KEY, FakeAI
from tests.conftest import Api, make_user


@pytest.fixture
def ai():
    fake = FakeAI()
    set_adapter(
        "openai",
        OpenAICompatibleProvider(
            "https://api.openai.fake/v1",
            use_max_completion_tokens=True,
            reasoning_effort_prefixes=("gpt-5",),
            transport=fake.openai_transport(),
        ),
    )
    set_adapter(
        "xai", OpenAICompatibleProvider("https://api.x.fake/v1", transport=fake.openai_transport())
    )
    set_adapter(
        "openai_compatible", OpenAICompatibleProvider(None, transport=fake.openai_transport())
    )
    set_adapter("gemini", GeminiProvider(transport=fake.gemini_transport()))
    set_adapter("anthropic", AnthropicProvider(http_client_factory=fake.anthropic_client))
    cache_clear()
    yield fake
    for key in AI_PROVIDERS:
        set_adapter(key, None)


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            out.append((name, data))
    return out


async def connect(api: Api, provider: str, **extra) -> dict:
    r = await api.put(f"/api/ai/providers/{provider}", json={"api_key": AI_KEY, **extra})
    assert r.status_code == 200, r.text
    return r.json()


async def chat(api: Api, conv_id: str, content: str, **extra) -> list[tuple[str, dict]]:
    r = await api.post(
        f"/api/ai/conversations/{conv_id}/messages", json={"content": content, **extra}
    )
    assert r.status_code == 200, r.text
    await drain_background()
    return events(r.text)


async def new_conversation(api: Api) -> str:
    r = await api.post("/api/ai/conversations", json={})
    assert r.status_code == 201
    return r.json()["id"]


async def test_routing_rules():
    assert route("What is a seed?", "low", True).tier == "low"
    assert route("Write a detailed cinematic prompt", "low", True).tier == "low"
    assert route("What is a seed?", "balanced", True).tier == "low"
    assert route("Write a detailed cinematic prompt", "balanced", True).tier == "balanced"
    assert route("What is a seed?", "high", True).tier == "balanced"
    assert route("Prečo zlyhalo generovanie?", "high", True).tier == "high"
    assert route("hi", "balanced", False).tier == "balanced"
    assert route("x", "balanced", True, "improve_prompt").complexity == "complex"


async def test_connect_validates_and_encrypts(user: Api, ai: FakeAI):
    bad = await user.put("/api/ai/providers/openai", json={"api_key": "sk-wrong-key-000000"})
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_credentials"
    body = await connect(user, "openai")
    assert AI_KEY not in json.dumps(body)
    assert body["models"] == ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"]  # no embeddings
    assert body["tiers"]["low"] == {"model": "gpt-5.6-luna", "source": "preferred"}
    assert body["tiers"]["high"]["model"] == "gpt-5.6-sol"
    listing = (await user.get("/api/ai/providers")).json()
    assert listing["active_provider"] == "openai"
    async with SessionLocal() as db:
        rows = (await db.scalars(select(AIIntegration))).all()
        assert all(AI_KEY.encode() not in r.encrypted_key for r in rows)


async def test_chat_streams_and_records_usage(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    ev = await chat(user, conv, "Write a detailed prompt for a fox in snow")
    names = [e[0] for e in ev]
    assert names[0] == "start" and names[-1] == "done" and "delta" in names
    sent = ai.calls[-1]["body"]
    assert sent["model"] == "gpt-5.6-luna"  # low-cost mode is the default
    assert sent["max_completion_tokens"] == 900
    assert sent["reasoning_effort"] == "low"
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][0]["role"] == "system"
    done = ev[-1][1]["message"]
    assert done["status"] == "complete"
    assert "```prompt" in done["content"]
    assert (done["input_tokens"], done["output_tokens"], done["cached_tokens"]) == (1200, 40, 1024)
    usage = (await user.get("/api/ai/usage")).json()
    assert usage["requests"] == 1 and usage["input_tokens"] == 1200
    detail = (await user.get(f"/api/ai/conversations/{conv}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"].startswith("Write a detailed prompt")


async def test_modes_and_auto_routing(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    await user.patch("/api/settings", json={"ai_mode": "balanced"})
    await chat(user, conv, "What is a seed?")
    assert ai.calls[-1]["body"]["model"] == "gpt-5.6-luna"
    await chat(user, conv, "Compare Seedream and Wan for product photos")
    assert ai.calls[-1]["body"]["model"] == "gpt-5.6-terra"
    await user.patch("/api/settings", json={"ai_mode": "high"})
    await chat(user, conv, "Explain why my generation failed")
    assert ai.calls[-1]["body"]["model"] == "gpt-5.6-sol"
    await user.patch("/api/settings", json={"ai_mode": "balanced", "ai_auto_route": False})
    await chat(user, conv, "What is a seed now?")
    assert ai.calls[-1]["body"]["model"] == "gpt-5.6-terra"


async def test_tier_override(user: Api, ai: FakeAI):
    await connect(user, "openai")
    r = await user.patch("/api/ai/providers/openai/models", json={"low": "gpt-5.6-terra"})
    assert r.json()["tiers"]["low"] == {"model": "gpt-5.6-terra", "source": "user"}
    bad = await user.patch("/api/ai/providers/openai/models", json={"low": "not-a-model"})
    assert bad.status_code == 400
    conv = await new_conversation(user)
    await chat(user, conv, "What is a seed?")
    assert ai.calls[-1]["body"]["model"] == "gpt-5.6-terra"


async def test_local_answers_and_cache_cost_nothing(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    ev = await chat(user, conv, "Ahoj!")
    assert ai.calls == []
    assert ev[-1][1]["message"]["source"] == "local"
    # The same first question in a fresh conversation is answered from the cache.
    first = await new_conversation(user)
    await chat(user, first, "What does thinking mode do?")
    second = await new_conversation(user)
    ev = await chat(user, second, "What does thinking mode do?")
    assert len(ai.calls) == 1
    assert ev[-1][1]["message"]["source"] == "cache"
    usage = (await user.get("/api/ai/usage")).json()
    assert usage["free_answers"] == 2


async def test_studio_context_is_sent_once(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    ctx = {
        "page": "generate",
        "model": "wan-2.7-pro",
        "provider": "wavespeed",
        "settings": {"aspect_ratio": "16:9", "resolution": "2K", "seed": 7},
        "prompt": "a lighthouse at night",
        "count": 2,
    }
    await chat(user, conv, "Improve my prompt", context=ctx, quick_action="improve_prompt")
    last_user = ai.calls[-1]["body"]["messages"][-1]["content"]
    assert "<studio_context>" in last_user
    assert "Wan 2.7 Pro" in last_user and "aspect_ratio=16:9" in last_user
    assert "a lighthouse at night" in last_user
    await chat(user, conv, "And a shorter version?", context=ctx)
    msgs = ai.calls[-1]["body"]["messages"]
    assert "<studio_context>" not in msgs[-1]["content"]  # unchanged: not repeated
    assert "<studio_context>" in msgs[1]["content"]  # still in the replayed history
    await user.patch("/api/settings", json={"ai_share_context": False})
    await chat(user, conv, "Thanks, one more variation please", context={**ctx, "prompt": "new"})
    assert "new" not in ai.calls[-1]["body"]["messages"][-1]["content"]


async def test_troubleshooting_includes_recent_failures(connected: Api, ai: FakeAI, fake):
    fake.submit_status = 402
    fake.submit_body = {"code": 402, "message": "Insufficient balance"}
    from tests.conftest import wait_done

    r = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "a cat"}
    )
    await wait_done(connected, r.json()["id"])
    await connect(connected, "openai")
    conv = await new_conversation(connected)
    await chat(connected, conv, "Why did it fail?", quick_action="troubleshoot")
    sent = ai.calls[-1]["body"]["messages"][-1]["content"]
    assert "recent failed generation" in sent and "insufficient_credits" in sent


async def test_anthropic_cost_controls(user: Api, ai: FakeAI):
    body = await connect(user, "anthropic")
    assert body["tiers"]["low"]["model"] == "claude-haiku-5-5"
    conv = await new_conversation(user)
    ev = await chat(user, conv, "What is a seed?")
    call = ai.calls[-1]
    assert call["body"]["model"] == "claude-haiku-5-5"
    assert call["body"]["cache_control"] == {"type": "ephemeral"}
    assert call["body"]["output_config"] == {"effort": "low"}
    assert "temperature" not in call["body"]
    assert not call["beta"]
    message = ev[-1][1]["message"]
    assert message["content"] == "Claude says hello."
    assert (message["input_tokens"], message["cached_tokens"], message["output_tokens"]) == (
        1000,
        950,
        30,
    )
    await user.patch("/api/settings", json={"ai_mode": "balanced"})
    await chat(user, conv, "Write a detailed prompt for a castle")
    call = ai.calls[-1]
    assert call["body"]["model"] == "claude-sonnet-5-5"
    assert FALLBACK_BETA in (call["beta"] or "")
    assert call["body"]["fallbacks"] == "default"


async def test_anthropic_refusal_is_reported(user: Api, ai: FakeAI):
    await connect(user, "anthropic")
    ai.refuse = True
    conv = await new_conversation(user)
    ev = await chat(user, conv, "What is a seed?")
    assert ("error", {"code": "refused", "message": ev[-2][1]["message"]}) in ev
    assert ev[-1][1]["message"]["status"] == "error"
    assert ev[-1][1]["message"]["content"] == ""


async def test_gemini(user: Api, ai: FakeAI):
    body = await connect(user, "gemini")
    assert "text-embedding-004" not in body["models"]
    assert body["tiers"]["balanced"]["model"] == "gemini-3.8-flash"
    conv = await new_conversation(user)
    ev = await chat(user, conv, "What is a seed?")
    call = ai.calls[-1]
    assert "gemini-3.5-flash-lite:streamGenerateContent" in call["url"]
    assert call["body"]["systemInstruction"]["parts"][0]["text"].startswith("You are the built-in")
    assert call["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert ev[-1][1]["message"]["content"] == "Gemini answer."


async def test_openai_compatible_base_url_and_fallbacks(user: Api, ai: FakeAI):
    missing = await user.put("/api/ai/providers/openai_compatible", json={"api_key": AI_KEY})
    assert missing.status_code == 400
    insecure = await user.put(
        "/api/ai/providers/openai_compatible",
        json={"api_key": AI_KEY, "base_url": "http://example.com/v1"},
    )
    assert insecure.status_code == 400
    body = await connect(user, "openai_compatible", base_url="https://router.example.com/v1")
    assert body["base_url"] == "https://router.example.com/v1"
    assert all(body["tiers"][t]["model"] for t in ("low", "balanced", "high"))
    assert body["tiers"]["low"]["source"] == "matched"
    ai.reject_stream_options = True
    conv = await new_conversation(user)
    ev = await chat(user, conv, "What is a seed?")
    assert ai.calls[-1]["url"].startswith("https://router.example.com/v1/chat/completions")
    assert "stream_options" not in ai.calls[-1]["body"]
    assert ev[-1][1]["message"]["usage_estimated"] is True


async def test_provider_errors_are_normalized(user: Api, ai: FakeAI):
    await connect(user, "openai")
    ai.fail_status = 429
    conv = await new_conversation(user)
    ev = await chat(user, conv, "What is a seed?")
    assert "error" in [e[0] for e in ev]
    assert ev[-1][1]["message"]["error_code"] == "insufficient_credits"


async def test_no_provider(user: Api, ai: FakeAI):
    conv = await new_conversation(user)
    r = await user.post(
        f"/api/ai/conversations/{conv}/messages", json={"content": "What is a seed?"}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "not_connected"


async def test_long_conversations_are_summarized(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    ai.reply = "word " * 400  # ~570 tokens per answer: the low-mode window (1200) overflows
    for i in range(4):
        await chat(user, conv, f"Question number {i} about lighting and composition?")
    summaries = [
        c for c in ai.calls if c["body"]["messages"][0]["content"].startswith("You maintain")
    ]
    assert summaries, "older turns should be folded into a summary"
    async with SessionLocal() as db:
        row = await db.get(ChatConversation, __import__("uuid").UUID(conv))
        assert row is not None and row.summary
    last = ai.calls[-1]["body"]["messages"][0]["content"]
    assert "Summary of the earlier conversation" in last or summaries[-1] is ai.calls[-1]
    usage = (await user.get("/api/ai/usage")).json()
    assert usage["summaries"] >= 1


async def test_conversations_are_private(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    await chat(user, conv, "What is a seed?")
    other = await make_user("Other")
    assert (await other.get(f"/api/ai/conversations/{conv}")).status_code == 404
    r = await other.post(f"/api/ai/conversations/{conv}/messages", json={"content": "hi"})
    assert r.status_code == 404
    assert (await other.get("/api/ai/conversations")).json()["items"] == []
    providers = (await other.get("/api/ai/providers")).json()["providers"]
    assert all(p["status"] == "not_connected" for p in providers)
    await other.client.aclose()


async def test_conversation_search_rename_delete(user: Api, ai: FakeAI):
    await connect(user, "openai")
    conv = await new_conversation(user)
    await chat(user, conv, "Lighthouse ideas please")
    found = (await user.get("/api/ai/conversations", params={"q": "lighthouse"})).json()["items"]
    assert [c["id"] for c in found] == [conv]
    await user.patch(f"/api/ai/conversations/{conv}", json={"title": "Coast"})
    assert (await user.get(f"/api/ai/conversations/{conv}")).json()["title"] == "Coast"
    await user.delete(f"/api/ai/conversations/{conv}")
    assert (await user.get(f"/api/ai/conversations/{conv}")).status_code == 404
