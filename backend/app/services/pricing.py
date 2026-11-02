"""Prices of the hosted models, shown before anything is sent (never invented).

WaveSpeed has a price API: the list price of each model (`base_price`) and an estimate for
one request with its inputs (`unit_price`); both are USD and both are shown as estimates.
Atlas Cloud has none: the price is "unknown" with a link to its pricing page.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import MODELS, PROVIDERS, build_payload
from app.models import Integration, User
from app.providers import get_adapter
from app.providers.base import ProviderError
from app.security import decrypt_secret
from app.services.generation import Prepared

# List prices per (user, provider) for 10 minutes: they change rarely, and the Generate
# page shows them on every model card.
LIST_TTL = 600
_list_cache: dict[tuple[str, str], tuple[float, dict[str, float]]] = {}


def _key(user: User, row: Integration) -> str:
    return decrypt_secret(row.encrypted_key, associated=f"{user.id}:{row.provider}")


def forget(user_id: str, provider: str) -> None:
    _list_cache.pop((user_id, provider), None)


async def _list(user: User, row: Integration) -> dict[str, float]:
    cache_key = (str(user.id), row.provider)
    cached = _list_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < LIST_TTL:
        return cached[1]
    prices = await get_adapter(row.provider).list_prices(_key(user, row))
    _list_cache[cache_key] = (time.monotonic(), prices)
    return prices


async def model_prices(db: AsyncSession, user: User) -> dict[str, Any]:
    """The list price of every catalog endpoint on each connected provider."""
    rows = (await db.scalars(select(Integration).where(Integration.user_id == user.id))).all()
    providers: dict[str, Any] = {}
    for row in rows:
        info = PROVIDERS.get(row.provider)
        if info is None:
            continue
        entry: dict[str, Any] = {
            "supported": info["supports_prices"],
            "pricing_url": info["pricing_url"],
            "prices": [],
            "error": None,
        }
        providers[row.provider] = entry
        if not info["supports_prices"]:
            continue
        try:
            listed = await _list(user, row)
        except ProviderError as exc:
            entry["error"] = exc.user_message
            continue
        for model in MODELS.values():
            for variant in model.variants:
                if variant.provider != row.provider:
                    continue
                entry["prices"].append(
                    {
                        "model": model.key,
                        "mode": variant.mode,
                        "model_id": variant.model_id,
                        "price": listed.get(variant.model_id),
                    }
                )
    return {"currency": "USD", "providers": providers}


async def estimate(user: User, prepared: Prepared, count: int) -> dict[str, Any]:
    """The price of a checked request: the provider's estimate for each request it will
    make, or its list price per request when the estimate is not available."""
    p = prepared
    info = PROVIDERS[p.provider]
    out: dict[str, Any] = {
        "provider": p.provider,
        "provider_name": info["name"],
        "model": p.model.key,
        "model_name": p.model.name,
        "mode": p.variant.mode,
        "images": count,
        "requests": len(p.plan),
        "currency": "USD",
        "total": None,
        "per_image": None,
        "source": None,
        "supported": info["supports_prices"],
        "pricing_url": info["pricing_url"],
        "error": None,
    }
    if not info["supports_prices"]:
        return out
    adapter = get_adapter(p.provider)
    api_key = _key(user, p.integration)
    total = 0.0
    source = "estimate"
    by_size: dict[int, float | None] = {}
    try:
        for size in p.plan:
            if size not in by_size:
                payload = build_payload(
                    p.variant,
                    prompt=p.final_prompt or p.prompt,
                    settings=p.settings,
                    image_urls=[f"<reference {i + 1}>" for i in range(len(p.ref_ids))],
                    count=size,
                )
                try:
                    by_size[size] = await adapter.estimate(api_key, p.variant.model_id, payload)
                except ProviderError as exc:
                    if exc.code in ("invalid_credentials", "network_error"):
                        raise
                    by_size[size] = None
            price = by_size[size]
            if price is None and p.variant.video:
                # A video's price depends on its length and resolution: without the
                # provider's estimate it stays unknown (a list price would mislead).
                return out
            if price is None:
                listed = (await _list(user, p.integration)).get(p.variant.model_id)
                if listed is None:
                    return out
                # The list price is per request; a request for several images is counted
                # as that many requests (an upper estimate).
                price = listed * max(1, size)
                source = "list_price"
            total += price
    except ProviderError as exc:
        out["error"] = exc.user_message
        return out
    out.update(
        total=round(total, 4),
        per_image=round(total / max(1, count), 4),
        source=source,
    )
    return out
