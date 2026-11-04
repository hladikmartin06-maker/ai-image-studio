"""Questions to the user's own AI provider (vision): the plan of a request and the check of
a result, as JSON. The same providers, keys, tier models and usage records as the assistant
and the LoRA captions (`ai/`); tokens are recorded with purpose "advanced"."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.base import AIError, ChatRequest, ChatTurn, Usage
from app.ai.registry import AI_PROVIDERS, get_adapter, resolve_tier_models
from app.lora.captions import small_jpeg
from app.models import AIUsageEvent, User
from app.security import decrypt_secret
from app.services import preferences

# Output budgets: newer models count their reasoning in it (see lora/captions.py).
BUDGETS = (2048, 6144)


class IntelligenceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def parse_json(text: str) -> dict[str, Any]:
    """The first JSON object in the answer (models sometimes wrap it in ```json fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise IntelligenceError("bad_answer", "The AI did not answer with a plan.")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise IntelligenceError("bad_answer", "The AI's answer was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise IntelligenceError("bad_answer", "The AI's answer was not a JSON object.")
    return data


async def ask_json(
    db: AsyncSession,
    user: User,
    *,
    system: str,
    text: str,
    images: list[bytes],
    tier: Literal["low", "balanced", "high"] = "balanced",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(answer, {provider, model, input_tokens, output_tokens})."""
    from app.ai.service import _integration

    prefs = await preferences.load(db, user.id)
    integration, provider = await _integration(db, user, prefs.ai_provider)
    if integration is None or provider is None:
        raise IntelligenceError(
            "not_connected", "Connect an AI provider in Integrations › AI assistant first."
        )
    spec = AI_PROVIDERS[provider]
    tiers = resolve_tier_models(spec, integration.models or [], integration.tier_models or {})
    model = tiers[tier]["model"]
    if not model:
        raise IntelligenceError("model_not_found", f"Choose a {tier} model for {spec.name}.")
    api_key = (
        decrypt_secret(integration.encrypted_key, associated=f"{user.id}:ai:{provider}")
        if integration.encrypted_key
        else ""
    )
    pictures = [("image/jpeg", small_jpeg(data)) for data in images]
    total = Usage()
    answer = ""
    for budget in BUDGETS:
        request = ChatRequest(
            model=str(model),
            system=system,
            messages=[ChatTurn("user", text, images=pictures)],
            max_tokens=budget,
            effort="low",
        )
        answer, stop, usage = "", None, Usage()
        try:
            async for event in get_adapter(spec).stream(api_key, integration.base_url, request):
                if event.done:
                    usage = event.usage
                    stop = event.stop_reason
                else:
                    answer += event.text
        except AIError as exc:
            raise IntelligenceError(exc.code, exc.message) from exc
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        db.add(
            AIUsageEvent(
                user_id=user.id,
                provider=provider,
                model=str(model),
                tier=tier,
                purpose="advanced",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                estimated=usage.estimated,
            )
        )
        await db.commit()
        if stop == "refusal":
            raise IntelligenceError("refused", "The AI provider declined to look at this.")
        if stop != "max_tokens":
            break
    meta = {
        "provider": provider,
        "model": str(model),
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
    }
    return parse_json(answer), meta
