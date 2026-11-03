"""Automatic dataset captions through the user's connected AI provider (vision models:
Claude, GPT, Gemini, Grok or a compatible API). One request per image; usage is recorded
like any assistant call. The caption describes what varies between photos and leaves the
character's fixed identity to the trigger word."""

from __future__ import annotations

import base64
import io
from typing import Any, Literal

from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.base import AIError, ChatRequest, ChatTurn, Usage
from app.ai.registry import AI_PROVIDERS, get_adapter, resolve_tier_models
from app.models import AIUsageEvent, User
from app.security import decrypt_secret
from app.services import preferences
from app.services.safety import mentions_minors

CAPTION_SYSTEM = (
    "You write training captions for a LoRA of one fictional adult character. Describe only "
    "what changes from photo to photo: clothing, pose, expression, framing, setting, lighting, "
    "camera angle and style. Do not describe the person's permanent identity (face shape, eye "
    "color, hair color or hairstyle, body type, skin) and never guess who it is. Answer with "
    "one plain sentence under 40 words, no quotes. If the image may show a minor, answer only "
    "with the word REFUSED."
)
MAX_EDGE = 1024


class CaptionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def small_jpeg(data: bytes) -> str:
    """Downscaled copy for the request (fewer image tokens); the dataset keeps the original."""
    with PILImage.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_EDGE, MAX_EDGE), PILImage.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
    return base64.b64encode(out.getvalue()).decode()


def apply_template(template: str, caption: str, trigger: str, class_word: str) -> str:
    text = template or "{trigger}, {caption}"
    if "{caption}" not in text:
        text = text + ", {caption}"
    result = (
        text.replace("{trigger}", trigger)
        .replace("{class}", class_word)
        .replace("{caption}", caption)
    )
    return " ".join(result.replace(" ,", ",").split()).strip(" ,")


# Output token budgets: the first try, and one more when the answer was cut off.
CAPTION_BUDGETS = (1024, 4096)


def complete_part(text: str) -> str:
    """The caption up to its last finished phrase (comma, semicolon or full stop)."""
    text = text.strip()
    cut = max(text.rfind(","), text.rfind(";"), text.rfind("."))
    return text[:cut].strip(" ,;.") if cut > 0 else ""


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return complete_part(text[:limit]) or text[:limit].rsplit(" ", 1)[0]


async def _ask(
    spec: Any, api_key: str, base_url: str | None, model: str, data: bytes, budget: int
) -> tuple[str, str | None, Usage]:
    request = ChatRequest(
        model=model,
        system=CAPTION_SYSTEM,
        messages=[
            ChatTurn(
                "user", "Caption this training image.", images=[("image/jpeg", small_jpeg(data))]
            )
        ],
        max_tokens=budget,
        effort="low",
    )
    text = ""
    usage = Usage()
    stop: str | None = None
    try:
        async for event in get_adapter(spec).stream(api_key, base_url, request):
            if event.done:
                usage = event.usage
                stop = event.stop_reason
            else:
                text += event.text
    except AIError as exc:
        raise CaptionError(exc.code, exc.message) from exc
    return text, stop, usage


async def describe(
    db: AsyncSession,
    user: User,
    data: bytes,
    tier: Literal["low", "balanced", "high"] = "low",
) -> tuple[str, str, str]:
    """(caption, provider, model)."""
    from app.ai.service import _integration

    prefs = await preferences.load(db, user.id)
    integration, provider = await _integration(db, user, prefs.ai_provider)
    if integration is None or provider is None:
        raise CaptionError(
            "not_connected", "Connect an AI provider in Integrations › AI assistant first."
        )
    spec = AI_PROVIDERS[provider]
    tiers = resolve_tier_models(spec, integration.models or [], integration.tier_models or {})
    model = tiers[tier]["model"]
    if not model:
        raise CaptionError("model_not_found", f"Choose a {tier} model for {spec.name}.")
    api_key = (
        decrypt_secret(integration.encrypted_key, associated=f"{user.id}:ai:{provider}")
        if integration.encrypted_key
        else ""
    )
    text, stop = "", None
    for budget in CAPTION_BUDGETS:
        # The budget also covers the model's own reasoning (newer models think before they
        # answer): too small, and the caption stopped mid-sentence.
        text, stop, usage = await _ask(
            spec, api_key, integration.base_url, str(model), data, budget
        )
        db.add(
            AIUsageEvent(
                user_id=user.id,
                provider=provider,
                model=str(model),
                tier=tier,
                purpose="caption",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                estimated=usage.estimated,
            )
        )
        await db.commit()
        if stop != "max_tokens":
            break
    caption = " ".join(text.strip().strip('"').split())
    if stop == "max_tokens":
        # Still cut off: keep only the complete parts, never half a word or phrase.
        caption = complete_part(caption)
        if not caption:
            raise CaptionError(
                "incomplete", "The model's answer was cut off. Try again or use another model."
            )
    if not caption or caption.upper().startswith("REFUSED") or mentions_minors(caption):
        raise CaptionError(
            "refused", "The model did not caption this image. Write the caption yourself."
        )
    return shorten(caption, 600), provider, str(model)
