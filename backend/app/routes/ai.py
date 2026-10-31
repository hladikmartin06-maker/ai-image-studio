"""AI assistant: provider keys and models, conversations, streaming answers, token usage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.base import AIError
from app.ai.registry import (
    AI_PROVIDERS,
    COMPATIBLE_PRESETS,
    TIERS,
    get_adapter,
    resolve_tier_models,
)
from app.ai.service import (
    SendBody,
    TurnRejected,
    conversation_json,
    message_json,
    prepare_turn,
    stream_turn,
)
from app.db import get_db
from app.deps import api_error, current_user
from app.models import AIIntegration, AIUsageEvent, ChatConversation, ChatMessage, User
from app.security import decrypt_secret, encrypt_secret, mask_key
from app.services import preferences
from app.services.preferences import PreferencesPatch
from app.services.serializers import iso

router = APIRouter(prefix="/api/ai", tags=["assistant"])


class KeyBody(BaseModel):
    api_key: str = Field(default="", max_length=500)
    base_url: str | None = Field(default=None, max_length=300)


class TierBody(BaseModel):
    low: str | None = Field(default=None, max_length=160)
    balanced: str | None = Field(default=None, max_length=160)
    high: str | None = Field(default=None, max_length=160)


class ConversationBody(BaseModel):
    title: str | None = Field(default=None, max_length=200)


def _assoc(user: User, provider: str) -> str:
    return f"{user.id}:ai:{provider}"


def _spec(provider: str):
    spec = AI_PROVIDERS.get(provider)
    if spec is None:
        raise api_error(404, "unknown_provider", "Unknown AI provider.")
    return spec


def _check_base_url(url: str) -> str:
    parsed = urlparse(url.strip())
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise api_error(400, "invalid_base_url", "Enter the API base URL, e.g. https://…/v1.")
    if parsed.scheme == "http" and not local:
        raise api_error(400, "invalid_base_url", "Use https:// (http:// only for localhost).")
    return url.strip().rstrip("/")


async def _row(db: AsyncSession, user: User, provider: str) -> AIIntegration | None:
    return await db.scalar(
        select(AIIntegration).where(
            AIIntegration.user_id == user.id, AIIntegration.provider == provider
        )
    )


def _provider_json(key: str, row: AIIntegration | None, active: str | None) -> dict:
    spec = AI_PROVIDERS[key]
    models = row.models if row else []
    return {
        "key": spec.key,
        "name": spec.name,
        "description": spec.description,
        "keys_url": spec.keys_url,
        "docs_url": spec.docs_url,
        "custom_base_url": spec.custom_base_url,
        "key_optional": spec.key_optional,
        "default_base_url": spec.default_base_url,
        "status": "not_connected" if row is None else row.status,
        "masked_key": (mask_key(row.key_last4) if row.key_last4 else None) if row else None,
        "base_url": row.base_url if row else None,
        "last_error": row.last_error if row else None,
        "last_checked_at": iso(row.last_checked_at) if row else None,
        "models": models,
        "tiers": resolve_tier_models(spec, models, row.tier_models if row else {})
        if row
        else {t: {"model": None, "source": "none"} for t in TIERS},
        "active": row is not None and active == key,
    }


@router.get("/providers")
async def list_providers(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    prefs = await preferences.load(db, user.id)
    rows = {
        r.provider: r
        for r in (await db.scalars(select(AIIntegration).where(AIIntegration.user_id == user.id)))
    }
    ordered = [k for k in AI_PROVIDERS if k in rows]
    active = prefs.ai_provider if prefs.ai_provider in rows else (ordered[0] if ordered else None)
    return {
        "active_provider": active,
        "mode": prefs.ai_mode,
        "auto_route": prefs.ai_auto_route,
        "share_context": prefs.ai_share_context,
        "presets": COMPATIBLE_PRESETS,
        "providers": [_provider_json(k, rows.get(k), active) for k in AI_PROVIDERS],
    }


@router.put("/providers/{provider}")
async def save_key(
    provider: str,
    body: KeyBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Checks the key by listing the provider's models, then stores it encrypted."""
    spec = _spec(provider)
    row = await _row(db, user, provider)
    api_key = body.api_key.strip()
    if any(c.isspace() for c in api_key):
        raise api_error(400, "invalid_key", "An API key cannot contain spaces.")
    if not api_key and row is not None and body.base_url is not None:
        # Only the base URL changes: keep the stored key.
        api_key = decrypt_secret(row.encrypted_key, associated=_assoc(user, provider))
    if not api_key and not spec.key_optional:
        raise api_error(400, "invalid_key", "Enter the API key.")
    base_url = None
    if spec.custom_base_url:
        if not body.base_url:
            raise api_error(400, "invalid_base_url", "Enter the API base URL.")
        base_url = _check_base_url(body.base_url)
    try:
        models = await get_adapter(spec).list_models(api_key, base_url)
    except AIError as exc:
        raise api_error(400, exc.code, exc.message) from exc

    if row is None:
        row = AIIntegration(user_id=user.id, provider=provider, tier_models={})
        db.add(row)
    row.encrypted_key = encrypt_secret(api_key, associated=_assoc(user, provider))
    row.key_last4 = api_key[-4:] if len(api_key) >= 8 else ""
    row.base_url = base_url
    row.models = models
    row.status = "connected"
    row.last_error = None
    row.last_checked_at = datetime.now(UTC)
    await db.commit()

    prefs = await preferences.load(db, user.id)
    if prefs.ai_provider is None or await _row(db, user, prefs.ai_provider) is None:
        prefs = await preferences.save(db, user.id, PreferencesPatch(ai_provider=provider))  # type: ignore[arg-type]
    return _provider_json(provider, row, prefs.ai_provider)


@router.post("/providers/{provider}/test")
async def test_provider(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """Refreshes the model list; that call also proves the key works. It costs nothing."""
    spec = _spec(provider)
    row = await _row(db, user, provider)
    if row is None:
        raise api_error(404, "not_connected", "Add an API key first.")
    api_key = decrypt_secret(row.encrypted_key, associated=_assoc(user, provider))
    ok, message = True, None
    try:
        row.models = await get_adapter(spec).list_models(api_key, row.base_url)
    except AIError as exc:
        ok, message = False, exc.message
    row.status = "connected" if ok else "error"
    row.last_error = message
    row.last_checked_at = datetime.now(UTC)
    await db.commit()
    prefs = await preferences.load(db, user.id)
    return {
        "ok": ok,
        "message": message,
        "provider": _provider_json(provider, row, prefs.ai_provider),
    }


@router.patch("/providers/{provider}/models")
async def set_tier_models(
    provider: str,
    body: TierBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    _spec(provider)
    row = await _row(db, user, provider)
    if row is None:
        raise api_error(404, "not_connected", "Add an API key first.")
    tiers = dict(row.tier_models or {})
    for tier, value in body.model_dump(exclude_unset=True).items():
        if value:
            if row.models and value not in row.models:
                raise api_error(400, "model_not_found", f"{value} is not offered by this key.")
            tiers[tier] = value
        else:
            tiers.pop(tier, None)  # back to automatic
    row.tier_models = tiers
    await db.commit()
    prefs = await preferences.load(db, user.id)
    return _provider_json(provider, row, prefs.ai_provider)


@router.delete("/providers/{provider}")
async def disconnect(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _spec(provider)
    row = await _row(db, user, provider)
    if row is not None:
        await db.delete(row)
        await db.commit()
    prefs = await preferences.load(db, user.id)
    if prefs.ai_provider == provider:
        remaining = (
            await db.scalars(select(AIIntegration.provider).where(AIIntegration.user_id == user.id))
        ).all()
        replacement = next((k for k in AI_PROVIDERS if k in remaining), None)
        prefs = await preferences.save(db, user.id, PreferencesPatch(ai_provider=replacement))  # type: ignore[arg-type]
    return _provider_json(provider, None, prefs.ai_provider)


# --- conversations -------------------------------------------------------------------------


async def _conversation(db: AsyncSession, user: User, conv_id: uuid.UUID) -> ChatConversation:
    conv = await db.scalar(
        select(ChatConversation).where(
            ChatConversation.id == conv_id, ChatConversation.user_id == user.id
        )
    )
    if conv is None:
        raise api_error(404, "not_found", "Conversation not found.")
    return conv


@router.get("/conversations")
async def list_conversations(
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ChatConversation).where(ChatConversation.user_id == user.id)
    if q and q.strip():
        like = f"%{q.strip()}%"
        matching = select(ChatMessage.conversation_id).where(
            ChatMessage.user_id == user.id, ChatMessage.content.ilike(like)
        )
        stmt = stmt.where(
            (ChatConversation.title.ilike(like)) | (ChatConversation.id.in_(matching))
        )
    rows = (await db.scalars(stmt.order_by(ChatConversation.updated_at.desc()).limit(limit))).all()
    return {"items": [conversation_json(c) for c in rows]}


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationBody | None = None,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = ChatConversation(user_id=user.id, title=(body.title if body else None) or "")
    db.add(conv)
    await db.commit()
    return conversation_json(conv)


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    conv = await _conversation(db, user, conv_id)
    messages = (
        await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conv.id)
            .order_by(ChatMessage.created_at)
        )
    ).all()
    totals = {
        "input_tokens": sum(m.input_tokens for m in messages),
        "output_tokens": sum(m.output_tokens for m in messages),
        "cached_tokens": sum(m.cached_tokens for m in messages),
    }
    return {
        **conversation_json(conv),
        "summary": conv.summary,
        "messages": [message_json(m) for m in messages],
        "usage": totals,
    }


@router.patch("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: uuid.UUID,
    body: ConversationBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    conv = await _conversation(db, user, conv_id)
    conv.title = (body.title or "").strip()[:200]
    await db.commit()
    return conversation_json(conv)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    conv = await _conversation(db, user, conv_id)
    await db.execute(delete(ChatMessage).where(ChatMessage.conversation_id == conv.id))
    await db.delete(conv)
    await db.commit()
    return {"ok": True}


@router.post("/conversations/{conv_id}/messages")
async def send_message(
    conv_id: uuid.UUID,
    body: SendBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Answers as server-sent events: start, delta…, done (or error)."""
    conv = await _conversation(db, user, conv_id)
    try:
        prepared = await prepare_turn(db, user, conv, body)
    except TurnRejected as exc:
        raise api_error(exc.status, exc.code, exc.message) from exc
    return StreamingResponse(
        stream_turn(prepared, user.id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/usage")
async def usage(
    days: int = Query(default=30, ge=1, le=365),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.now(UTC) - timedelta(days=days)
    base = (AIUsageEvent.user_id == user.id, AIUsageEvent.created_at >= since)
    total = (
        await db.execute(
            select(
                func.count(AIUsageEvent.id),
                func.coalesce(func.sum(AIUsageEvent.input_tokens), 0),
                func.coalesce(func.sum(AIUsageEvent.output_tokens), 0),
                func.coalesce(func.sum(AIUsageEvent.cached_tokens), 0),
                func.count(AIUsageEvent.id).filter(AIUsageEvent.purpose.in_(("local", "cache"))),
                func.count(AIUsageEvent.id).filter(AIUsageEvent.purpose == "summary"),
            ).where(*base)
        )
    ).one()
    by_model = (
        await db.execute(
            select(
                AIUsageEvent.provider,
                AIUsageEvent.model,
                func.count(AIUsageEvent.id),
                func.coalesce(func.sum(AIUsageEvent.input_tokens), 0),
                func.coalesce(func.sum(AIUsageEvent.output_tokens), 0),
                func.coalesce(func.sum(AIUsageEvent.cached_tokens), 0),
            )
            .where(*base, AIUsageEvent.purpose.in_(("chat", "summary")))
            .group_by(AIUsageEvent.provider, AIUsageEvent.model)
            .order_by(func.count(AIUsageEvent.id).desc())
        )
    ).all()
    by_tier = (
        await db.execute(
            select(AIUsageEvent.tier, func.count(AIUsageEvent.id))
            .where(*base, AIUsageEvent.purpose == "chat")
            .group_by(AIUsageEvent.tier)
        )
    ).all()
    return {
        "days": days,
        "requests": total[0],
        "input_tokens": int(total[1]),
        "output_tokens": int(total[2]),
        "cached_tokens": int(total[3]),
        "free_answers": total[4],
        "summaries": total[5],
        "by_model": [
            {
                "provider": p,
                "model": m,
                "requests": n,
                "input_tokens": int(i),
                "output_tokens": int(o),
                "cached_tokens": int(c),
            }
            for p, m, n, i, o, c in by_model
        ],
        "by_tier": {t or "none": n for t, n in by_tier},
    }
