"""Provider connections. API keys are accepted here, encrypted, and never returned."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import PROVIDERS
from app.db import get_db
from app.deps import api_error, current_user
from app.models import Integration, User
from app.providers import get_adapter
from app.providers.base import ProviderError
from app.security import decrypt_secret, encrypt_secret, mask_key
from app.services import preferences, pricing
from app.services.preferences import PreferencesPatch
from app.services.serializers import iso

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

# Balance per (user, provider) for 60 s, so the dashboard does not call the provider on
# every page view.
_balance_cache: dict[tuple[str, str], tuple[float, float | None]] = {}


class KeyBody(BaseModel):
    api_key: str = Field(min_length=8, max_length=500)


def _assoc(user: User, provider: str) -> str:
    return f"{user.id}:{provider}"


def _check_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise api_error(404, "unknown_provider", "Unknown provider.")


async def _row(db: AsyncSession, user: User, provider: str) -> Integration | None:
    return await db.scalar(
        select(Integration).where(Integration.user_id == user.id, Integration.provider == provider)
    )


def _json(provider: str, row: Integration | None, active: str | None) -> dict:
    info = PROVIDERS[provider]
    return {
        **info,
        "status": "not_connected" if row is None else row.status,
        "masked_key": mask_key(row.key_last4) if row else None,
        "last_error": row.last_error if row else None,
        "last_checked_at": iso(row.last_checked_at) if row else None,
        "connected_at": iso(row.created_at) if row else None,
        "active": row is not None and active == provider,
    }


@router.get("")
async def list_integrations(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    prefs = await preferences.load(db, user.id)
    rows = {
        r.provider: r
        for r in (await db.scalars(select(Integration).where(Integration.user_id == user.id)))
    }
    return {
        "active_provider": prefs.active_provider if prefs.active_provider in rows else None,
        "integrations": [_json(p, rows.get(p), prefs.active_provider) for p in PROVIDERS],
    }


@router.put("/{provider}")
async def save_key(
    provider: str,
    body: KeyBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Validates the key with the provider, then stores it encrypted."""
    _check_provider(provider)
    api_key = body.api_key.strip()
    if any(c.isspace() for c in api_key):
        raise api_error(400, "invalid_key", "An API key cannot contain spaces.")
    status_value, last_error = "connected", None
    try:
        result = await get_adapter(provider).check(api_key)
    except ProviderError as exc:
        # The provider could not be reached: keep the key, show the problem.
        status_value, last_error = "error", exc.user_message
    else:
        if not result.ok:
            raise api_error(400, "invalid_credentials", result.message)

    row = await _row(db, user, provider)
    encrypted = encrypt_secret(api_key, associated=_assoc(user, provider))
    now = datetime.now(UTC)
    if row is None:
        row = Integration(user_id=user.id, provider=provider)
        db.add(row)
    row.encrypted_key = encrypted
    row.key_last4 = api_key[-4:]
    row.status = status_value
    row.last_error = last_error
    row.last_checked_at = now
    await db.commit()
    _balance_cache.pop((str(user.id), provider), None)
    pricing.forget(str(user.id), provider)

    prefs = await preferences.load(db, user.id)
    if prefs.active_provider is None or await _row(db, user, prefs.active_provider) is None:
        prefs = await preferences.save(db, user.id, PreferencesPatch(active_provider=provider))
    return _json(provider, row, prefs.active_provider)


@router.delete("/{provider}")
async def disconnect(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _check_provider(provider)
    row = await _row(db, user, provider)
    if row is not None:
        await db.delete(row)
        await db.commit()
    _balance_cache.pop((str(user.id), provider), None)
    pricing.forget(str(user.id), provider)
    prefs = await preferences.load(db, user.id)
    if prefs.active_provider == provider:
        other = next((p for p in PROVIDERS if p != provider), None)
        replacement = other if other and await _row(db, user, other) else None
        prefs = await preferences.save(db, user.id, PreferencesPatch(active_provider=replacement))
    return _json(provider, None, prefs.active_provider)


@router.post("/{provider}/test")
async def test_connection(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _check_provider(provider)
    row = await _row(db, user, provider)
    if row is None:
        raise api_error(404, "not_connected", "Add an API key first.")
    api_key = decrypt_secret(row.encrypted_key, associated=_assoc(user, provider))
    balance = None
    try:
        result = await get_adapter(provider).check(api_key)
        ok, message, balance = result.ok, result.message, result.balance
    except ProviderError as exc:
        ok, message = False, exc.user_message
    row.status = "connected" if ok else "error"
    row.last_error = None if ok else message
    row.last_checked_at = datetime.now(UTC)
    await db.commit()
    if balance is not None:
        _balance_cache[(str(user.id), provider)] = (time.monotonic(), balance)
    prefs = await preferences.load(db, user.id)
    return {
        "ok": ok,
        "message": message,
        "balance": balance,
        "integration": _json(provider, row, prefs.active_provider),
    }


@router.post("/{provider}/activate")
async def activate(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _check_provider(provider)
    if await _row(db, user, provider) is None:
        raise api_error(409, "not_connected", "Connect this provider before making it active.")
    await preferences.save(db, user.id, PreferencesPatch(active_provider=provider))
    return {"active_provider": provider}


@router.get("/{provider}/balance")
async def balance(
    provider: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _check_provider(provider)
    if not PROVIDERS[provider]["supports_balance"]:
        return {"supported": False, "balance": None, "currency": None}
    row = await _row(db, user, provider)
    if row is None:
        return {"supported": True, "balance": None, "currency": "USD", "connected": False}
    cached = _balance_cache.get((str(user.id), provider))
    if cached and time.monotonic() - cached[0] < 60:
        return {"supported": True, "balance": cached[1], "currency": "USD", "connected": True}
    api_key = decrypt_secret(row.encrypted_key, associated=_assoc(user, provider))
    try:
        value = await get_adapter(provider).balance(api_key)
    except ProviderError as exc:
        return {
            "supported": True,
            "balance": None,
            "currency": "USD",
            "connected": True,
            "error": exc.user_message,
        }
    _balance_cache[(str(user.id), provider)] = (time.monotonic(), value)
    return {"supported": True, "balance": value, "currency": "USD", "connected": True}
