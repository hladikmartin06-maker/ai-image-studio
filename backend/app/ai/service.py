"""One assistant turn: route, build context, answer (locally, from cache or by streaming
from the provider), store the result and keep the conversation small."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.context import (
    SUMMARY_SYSTEM,
    StudioContext,
    format_context,
    history_window,
    summary_request_text,
    system_prompt,
)
from app.ai.local import local_answer
from app.ai.providers.base import AIError, ChatRequest, ChatTurn
from app.ai.registry import AI_PROVIDERS, get_adapter, resolve_tier_models
from app.ai.router import HISTORY_BUDGET, Route, route
from app.db import SessionLocal
from app.lora.captions import small_jpeg
from app.models import (
    AIIntegration,
    AIUsageEvent,
    ChatConversation,
    ChatMessage,
    Generation,
    ReferenceImage,
    User,
)
from app.security import decrypt_secret
from app.services import preferences, storage

log = logging.getLogger("studio.ai")

QuickAction = Literal["improve_prompt", "suggest_settings", "troubleshoot", "compare_models"]
_TROUBLE_WORDS = ("error", "fail", "failed", "chyba", "chybu", "zlyhal", "nefunguje", "nejde")
# Photos per message (uploaded like reference images; sent downscaled, only with that message).
MAX_CHAT_IMAGES = 4


class SendBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    context: StudioContext | None = None
    quick_action: QuickAction | None = None
    provider: str | None = Field(default=None, max_length=32)
    mode: Literal["low", "balanced", "high"] | None = None
    # Uploaded reference images (POST /api/reference-images) the user attached.
    image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_CHAT_IMAGES)


class TurnRejected(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# --- response cache (identical request -> same answer, no new API call) ---------------------

_CACHE: OrderedDict[str, tuple[float, str]] = OrderedDict()
_CACHE_TTL = 24 * 3600
_CACHE_MAX = 300


def _cache_key(user_id: uuid.UUID, provider: str, request: ChatRequest) -> str:
    raw = json.dumps(
        [
            str(user_id),
            provider,
            request.model,
            request.system,
            [
                (t.role, t.content, [hashlib.sha256(d.encode()).hexdigest() for _, d in t.images])
                for t in request.messages
            ],
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(key: str) -> str | None:
    item = _CACHE.get(key)
    if item is None:
        return None
    stored_at, text = item
    if time.monotonic() - stored_at > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return text


def cache_put(key: str, text: str) -> None:
    _CACHE[key] = (time.monotonic(), text)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def cache_clear() -> None:
    _CACHE.clear()


# --- serialization --------------------------------------------------------------------------


def message_json(m: ChatMessage) -> dict[str, Any]:
    return {
        "id": str(m.id),
        "conversation_id": str(m.conversation_id),
        "role": m.role,
        "content": m.content,
        "status": m.status,
        "provider": m.provider,
        "model": m.model,
        "tier": m.tier,
        "source": m.source,
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "cached_tokens": m.cached_tokens,
        "usage_estimated": m.usage_estimated,
        "error_code": m.error_code,
        "error_message": m.error_message,
        "meta": m.meta or {},
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def conversation_json(c: ChatConversation) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "title": c.title,
        "has_summary": bool(c.summary),
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def local_title(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:8])
    return (title[:80] + "…") if len(title) > 80 or len(words) > 8 else title


# --- preparing a turn -----------------------------------------------------------------------


@dataclass
class Prepared:
    user_message: ChatMessage
    assistant_message: ChatMessage
    provider: str
    api_key: str
    base_url: str | None
    request: ChatRequest | None
    route: Route | None
    local_text: str | None
    cache_key: str | None
    summarize: bool
    summary_model: str | None


async def _integration(
    db: AsyncSession, user: User, wanted: str | None
) -> tuple[AIIntegration | None, str | None]:
    rows = {
        r.provider: r
        for r in (await db.scalars(select(AIIntegration).where(AIIntegration.user_id == user.id)))
    }
    if wanted and wanted in rows:
        return rows[wanted], wanted
    for key in AI_PROVIDERS:
        if key in rows:
            return rows[key], key
    return None, None


async def _recent_failures(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=2)
    rows = (
        await db.scalars(
            select(Generation)
            .where(
                Generation.user_id == user.id,
                Generation.status == "failed",
                Generation.created_at >= since,
            )
            .order_by(Generation.created_at.desc())
            .limit(3)
        )
    ).all()
    from app.catalog import MODELS, PROVIDERS

    return [
        {
            "model": MODELS[g.model_key].name if g.model_key in MODELS else g.model_key,
            "provider": PROVIDERS.get(g.provider, {}).get("name", g.provider),
            "code": g.error_code or "error",
            "message": (g.error_message or "")[:300],
        }
        for g in rows
    ]


async def prepare_turn(
    db: AsyncSession, user: User, conversation: ChatConversation, body: SendBody
) -> Prepared:
    prefs = await preferences.load(db, user.id)
    content = body.content.strip()
    if not content:
        raise TurnRejected("invalid_request", "Write a message.", 400)

    history = (
        await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.created_at)
        )
    ).all()

    # Attached photos: the user's own uploads only.
    image_ids = list(dict.fromkeys(body.image_ids))
    refs: list[ReferenceImage] = []
    if image_ids:
        rows = (
            await db.scalars(
                select(ReferenceImage).where(
                    ReferenceImage.user_id == user.id, ReferenceImage.id.in_(image_ids)
                )
            )
        ).all()
        by_id = {r.id: r for r in rows}
        if len(by_id) != len(image_ids):
            raise TurnRejected("not_found", "The attached photo was not found.", 404)
        refs = [by_id[i] for i in image_ids]

    user_message = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=content,
        api_content=content,
        meta={
            **({"quick_action": body.quick_action} if body.quick_action else {}),
            **(
                {
                    "images": [
                        {
                            "id": str(r.id),
                            "url": f"/api/reference-images/{r.id}/file",
                            "thumb_url": f"/api/reference-images/{r.id}/thumb",
                        }
                        for r in refs
                    ]
                }
                if refs
                else {}
            ),
        },
    )
    if not conversation.title:
        conversation.title = local_title(content)

    # Zero-cost local answer (greetings, thanks): no provider needed at all.
    local = None if body.quick_action or refs else local_answer(content, prefs.language)
    if local is not None:
        assistant = ChatMessage(
            conversation_id=conversation.id,
            user_id=user.id,
            role="assistant",
            content="",
            status="streaming",
            provider="local",
            model="local",
            source="local",
        )
        db.add_all([user_message, assistant])
        await db.commit()
        return Prepared(
            user_message, assistant, "local", "", None, None, None, local, None, False, None
        )

    integration, provider = await _integration(db, user, body.provider or prefs.ai_provider)
    if integration is None or provider is None:
        raise TurnRejected(
            "not_connected", "Connect an AI provider in Integrations › AI assistant first."
        )
    spec = AI_PROVIDERS[provider]
    mode = body.mode or prefs.ai_mode
    chosen = route(content, mode, prefs.ai_auto_route, body.quick_action)
    tiers = resolve_tier_models(spec, integration.models or [], integration.tier_models or {})
    model = tiers[chosen.tier]["model"]
    if not model:
        raise TurnRejected(
            "model_not_found",
            f"Choose a {chosen.tier} model for {spec.name} in Integrations › AI assistant.",
        )

    # Studio context: only sent when it changed or is no longer in the replayed window.
    window = history_window(list(history), HISTORY_BUDGET[mode], conversation.summary_until)
    failures: list[dict[str, Any]] = []
    wants_trouble = body.quick_action == "troubleshoot" or any(
        w in content.lower() for w in _TROUBLE_WORDS
    )
    if wants_trouble:
        failures = await _recent_failures(db, user)
    ctx_text = format_context(body.context if prefs.ai_share_context else None, failures)
    in_window = any((m.meta or {}).get("context") for m in window.messages)
    if ctx_text and (ctx_text != conversation.last_context or not in_window):
        user_message.api_content = f"{ctx_text}\n\n{content}"
        user_message.meta = {**(user_message.meta or {}), "context": True}
        conversation.last_context = ctx_text

    photos: list[tuple[str, str]] = []
    for ref in refs:
        try:
            data = await asyncio.to_thread(storage.absolute(ref.file_path).read_bytes)
            photos.append(("image/jpeg", await asyncio.to_thread(small_jpeg, data)))
        except (OSError, ValueError) as exc:
            raise TurnRejected("not_found", "The attached photo could not be read.", 404) from exc

    request = ChatRequest(
        model=str(model),
        system=system_prompt(conversation.summary),
        messages=[*window.turns, ChatTurn("user", user_message.api_content, images=photos)],
        max_tokens=chosen.max_tokens,
        effort=chosen.effort,
    )
    assistant = ChatMessage(
        conversation_id=conversation.id,
        user_id=user.id,
        role="assistant",
        content="",
        status="streaming",
        provider=provider,
        model=str(model),
        tier=chosen.tier,
        meta={"complexity": chosen.complexity, "route": chosen.reason, "mode": mode},
    )
    db.add_all([user_message, assistant])
    conversation.updated_at = datetime.now(UTC)
    await db.commit()

    api_key = (
        decrypt_secret(integration.encrypted_key, associated=f"{user.id}:ai:{provider}")
        if integration.encrypted_key
        else ""
    )
    summary_model = tiers["low"]["model"]
    return Prepared(
        user_message=user_message,
        assistant_message=assistant,
        provider=provider,
        api_key=api_key,
        base_url=integration.base_url,
        request=request,
        route=chosen,
        local_text=None,
        cache_key=_cache_key(user.id, provider, request),
        # Fold older turns into the summary once the conversation outgrows the window.
        summarize=len(window.unsummarized) >= 2 and bool(summary_model),
        summary_model=str(summary_model) if summary_model else None,
    )


# --- streaming a turn -----------------------------------------------------------------------


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_background: set[asyncio.Task[Any]] = set()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _finish(message_id: uuid.UUID, **values: Any) -> ChatMessage | None:
    async with SessionLocal() as db:
        await db.execute(update(ChatMessage).where(ChatMessage.id == message_id).values(**values))
        await db.commit()
        return await db.get(ChatMessage, message_id)


async def _record_usage(user_id: uuid.UUID, **values: Any) -> None:
    async with SessionLocal() as db:
        db.add(AIUsageEvent(user_id=user_id, **values))
        await db.commit()


async def stream_turn(prepared: Prepared, user_id: uuid.UUID) -> AsyncIterator[str]:
    assistant = prepared.assistant_message
    yield sse(
        "start",
        {
            "user_message": message_json(prepared.user_message),
            "assistant_message": message_json(assistant),
        },
    )

    # Local answer or cached answer: no provider call.
    cached = cache_get(prepared.cache_key) if prepared.cache_key else None
    if prepared.local_text is not None or cached is not None:
        text = prepared.local_text if prepared.local_text is not None else cached
        source = "local" if prepared.local_text is not None else "cache"
        yield sse("delta", {"text": text})
        done = await _finish(assistant.id, content=text, status="complete", source=source)
        await _record_usage(
            user_id,
            provider=prepared.provider,
            model=assistant.model or "local",
            tier=assistant.tier,
            purpose=source,
        )
        yield sse("done", {"message": message_json(done) if done else {}})
        return

    assert prepared.request is not None
    adapter = get_adapter(AI_PROVIDERS[prepared.provider])
    parts: list[str] = []
    finished = False
    try:
        async for event in adapter.stream(prepared.api_key, prepared.base_url, prepared.request):
            if event.done:
                text = "".join(parts)
                status, error_code, error_message = "complete", None, None
                if event.stop_reason == "refusal":
                    # A mid-stream decline: the partial answer is not reliable.
                    status, error_code = "error", "refused"
                    error_message = AIError("refused").message
                    text = ""
                meta = {**(assistant.meta or {})}
                if event.stop_reason == "max_tokens":
                    meta["truncated"] = True
                if event.model and event.model != assistant.model:
                    meta["served_by"] = event.model
                done = await _finish(
                    assistant.id,
                    content=text,
                    status=status,
                    error_code=error_code,
                    error_message=error_message,
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    cached_tokens=event.usage.cached_tokens,
                    usage_estimated=event.usage.estimated,
                    meta=meta,
                )
                finished = True
                await _record_usage(
                    user_id,
                    provider=prepared.provider,
                    model=event.model or assistant.model or "",
                    tier=assistant.tier,
                    purpose="chat",
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    cached_tokens=event.usage.cached_tokens,
                    estimated=event.usage.estimated,
                )
                if status == "complete" and text and prepared.cache_key:
                    cache_put(prepared.cache_key, text)
                if status == "error":
                    yield sse("error", {"code": error_code, "message": error_message})
                yield sse("done", {"message": message_json(done) if done else {}})
                if prepared.summarize and prepared.summary_model:
                    _spawn(
                        summarize(
                            assistant.conversation_id,
                            user_id,
                            prepared.provider,
                            prepared.summary_model,
                            prepared.api_key,
                            prepared.base_url,
                        )
                    )
                return
            parts.append(event.text)
            yield sse("delta", {"text": event.text})
    except AIError as exc:
        finished = True
        done = await _finish(
            assistant.id,
            content="".join(parts),
            status="error",
            error_code=exc.code,
            error_message=exc.message,
        )
        yield sse("error", {"code": exc.code, "message": exc.message})
        yield sse("done", {"message": message_json(done) if done else {}})
    except (asyncio.CancelledError, GeneratorExit):
        # The browser stopped the answer (Stop button or closed tab): keep what arrived.
        if not finished:
            _spawn(
                _finish(
                    assistant.id,
                    content="".join(parts),
                    status="stopped",
                    usage_estimated=True,
                )
            )
        raise
    except Exception:  # noqa: BLE001
        log.exception("Assistant turn failed")
        finished = True
        await _finish(
            assistant.id,
            content="".join(parts),
            status="error",
            error_code="provider_error",
            error_message="Unexpected error.",
        )
        yield sse("error", {"code": "provider_error", "message": "Unexpected error."})


async def summarize(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
) -> None:
    """Folds turns that left the context window into the conversation summary (cheap tier)."""
    async with SessionLocal() as db:
        conversation = await db.get(ChatConversation, conversation_id)
        if conversation is None:
            return
        prefs = await preferences.load(db, user_id)
        messages = (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation_id)
                .order_by(ChatMessage.created_at)
            )
        ).all()
        window = history_window(
            list(messages), HISTORY_BUDGET[prefs.ai_mode], conversation.summary_until
        )
        if len(window.unsummarized) < 2:
            return
        request = ChatRequest(
            model=model,
            system=SUMMARY_SYSTEM,
            messages=[
                ChatTurn("user", summary_request_text(conversation.summary, window.unsummarized))
            ],
            max_tokens=400,
            effort="low",
        )
        adapter = get_adapter(AI_PROVIDERS[provider])
        parts: list[str] = []
        usage = None
        try:
            async for event in adapter.stream(api_key, base_url, request):
                if event.done:
                    usage = event.usage
                else:
                    parts.append(event.text)
        except AIError as exc:
            log.warning("Summary failed: %s", exc.code)
            return
        text = "".join(parts).strip()
        if not text:
            return
        conversation.summary = text[:2000]
        conversation.summary_until = window.unsummarized[-1].created_at
        await db.commit()
        if usage is not None:
            db.add(
                AIUsageEvent(
                    user_id=user_id,
                    provider=provider,
                    model=model,
                    tier="low",
                    purpose="summary",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=usage.cached_tokens,
                    estimated=usage.estimated,
                )
            )
            await db.commit()


async def drain_background() -> None:
    """Waits for pending summaries (used by tests)."""
    while _background:
        await asyncio.gather(*list(_background), return_exceptions=True)
