"""What the assistant knows: a stable system prompt (cacheable), the studio context (sent
only when it changed), a token-budgeted window of recent turns and a rolling summary of
everything older."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.ai.providers.base import ChatTurn, estimate_tokens
from app.catalog import MATCH_INPUT, MODELS, PROVIDERS
from app.models import ChatMessage


class StudioContext(BaseModel):
    """What the user is looking at in the studio, sent by the browser with each message."""

    page: str | None = Field(default=None, max_length=40)
    model: str | None = Field(default=None, max_length=40)
    provider: str | None = Field(default=None, max_length=32)
    settings: dict[str, Any] = Field(default_factory=dict)
    prompt: str | None = Field(default=None, max_length=4000)
    reference_count: int = Field(default=0, ge=0, le=20)
    count: int | None = Field(default=None, ge=1, le=4)


def _catalog_facts() -> str:
    lines: list[str] = []
    for model in MODELS.values():
        content = "SFW only" if model.content == "sfw" else "SFW and adult (NSFW) content"
        lines.append(f"- {model.name} (key `{model.key}`, {model.family}): {content}.")
        for variant in model.variants:
            parts: list[str] = []
            if variant.aspect_ratios:
                ratios = ", ".join(variant.aspect_ratios)
                parts.append(
                    f"aspect {ratios}" + (" or match input" if variant.allow_match_input else "")
                )
            if variant.resolutions:
                parts.append("resolution " + "/".join(variant.resolutions))
            if variant.output_formats:
                parts.append("format " + "/".join(variant.output_formats))
            if variant.seed:
                parts.append("seed")
            if variant.thinking_mode:
                parts.append("thinking mode")
            if variant.max_reference_images:
                parts.append(f"up to {variant.max_reference_images} reference images")
            controls = "; ".join(parts) if parts else "prompt only (size follows input)"
            provider = PROVIDERS[variant.provider]["name"]
            lines.append(f"  - {provider} {variant.mode}: {controls}.")
    return "\n".join(lines)


_BASE_PROMPT = """You are the built-in assistant of AI Image Studio, a private Mac app for AI \
image generation. You help with writing and improving image prompts, choosing a model and \
settings, troubleshooting failed generations and questions about the app. Be concise and \
practical: short paragraphs or bullets, no filler. Never invent models, settings, prices or \
features.

The app has exactly four image models, run through the user's own WaveSpeed or Atlas Cloud account:
{catalog}

How the app works:
- Generate page: pick a model and provider, write a prompt, optional reference images (PNG, JPEG \
or WebP up to 10 MB) switch to image editing, settings shown are only those the endpoint supports, \
1–4 images per run (each image is a separate paid request unless the endpoint batches them).
- A fixed seed with the same settings reproduces a result; empty seed = random.
- Image Library (with History of every request and its exact parameters), Favorites, Integrations \
(image providers, AI assistant providers, GPU compute: Modal and RunPod), Settings.
- Common errors: invalid_credentials (fix the key in Integrations), insufficient_credits (top up \
the provider), rate_limited (wait), invalid_parameters (setting not accepted), content_blocked \
(provider moderation or the app's rules), timeout, interrupted (app stopped while sending; never \
resent to avoid double charges).
- Nano Banana Pro and Seedream are SFW only; for adult content the user must choose Wan 2.7 or \
Wan 2.7 Pro. Never write or improve sexual content involving minors or anyone who could be a minor.

Output conventions (the app turns these into buttons):
- When you give a complete, ready-to-use image prompt, put only the prompt inside a fenced block \
with the language `prompt`.
- When you recommend concrete settings, add one fenced block with the language `studio-settings` \
containing JSON with any of: "model" (one of the model keys), "aspect_ratio", "resolution", \
"count" (1-4), "seed" (integer), "thinking_mode" (true/false). Use only values the chosen model \
supports.

Answer in the language of the user's message."""


def system_prompt(summary: str) -> str:
    text = _BASE_PROMPT.format(catalog=_catalog_facts())
    if summary:
        text += f"\n\nSummary of the earlier conversation:\n{summary}"
    return text


def format_context(ctx: StudioContext | None, failures: list[dict[str, Any]]) -> str:
    if ctx is None and not failures:
        return ""
    lines: list[str] = []
    if ctx is not None:
        if ctx.page:
            lines.append(f"page: {ctx.page}")
        model = MODELS.get(ctx.model or "")
        if model:
            provider = PROVIDERS.get(ctx.provider or "", {}).get("name", ctx.provider or "?")
            mode = "image edit" if ctx.reference_count else "text-to-image"
            lines.append(f"selected model: {model.name} (`{model.key}`) via {provider}, {mode}")
        allowed = ("aspect_ratio", "resolution", "output_format", "seed", "thinking_mode")
        settings = []
        for key in allowed:
            value = ctx.settings.get(key)
            if value in (None, ""):
                continue
            if key == "aspect_ratio" and value == MATCH_INPUT:
                value = "match input"
            settings.append(f"{key}={value}")
        if ctx.count:
            settings.append(f"images={ctx.count}")
        if ctx.reference_count:
            settings.append(f"reference_images={ctx.reference_count}")
        if settings:
            lines.append("settings: " + ", ".join(str(s) for s in settings))
        if ctx.prompt and ctx.prompt.strip():
            prompt = ctx.prompt.strip()
            if len(prompt) > 1500:
                prompt = prompt[:1500] + "…"
            lines.append(f'current prompt: "{prompt}"')
    for failure in failures:
        lines.append(
            "recent failed generation: {model} via {provider}: {code} — {message}".format(**failure)
        )
    return "<studio_context>\n" + "\n".join(lines) + "\n</studio_context>" if lines else ""


@dataclass
class Window:
    turns: list[ChatTurn]
    messages: list[ChatMessage]
    # Older messages that are not in the window and not yet in the summary.
    unsummarized: list[ChatMessage]


def history_window(messages: list[ChatMessage], budget: int, summary_until) -> Window:
    """Newest turns that fit the token budget; the first replayed turn is always the user's."""
    usable = [
        m
        for m in messages
        if m.status in ("complete", "stopped") and (m.api_content or m.content).strip()
    ]
    picked: list[ChatMessage] = []
    used = 0
    for message in reversed(usable):
        cost = estimate_tokens(message.api_content or message.content)
        if picked and used + cost > budget:
            break
        picked.append(message)
        used += cost
    picked.reverse()
    while picked and picked[0].role != "user":
        picked.pop(0)
    first = picked[0].created_at if picked else None
    older = [
        m
        for m in usable
        if (first is None or m.created_at < first)
        and (summary_until is None or m.created_at > summary_until)
    ]
    turns = [ChatTurn(role=m.role, content=m.api_content or m.content) for m in picked]  # type: ignore[arg-type]
    return Window(turns=turns, messages=picked, unsummarized=older)


SUMMARY_SYSTEM = (
    "You maintain a short running summary of a conversation between a user and the assistant of "
    "an AI image generation app. Merge the previous summary with the new messages. Keep: the "
    "user's goals, decided prompts (verbatim if short), chosen models and settings, open "
    "questions. "
    "At most 120 words, plain text, same language as the conversation."
)


def summary_request_text(previous: str, messages: list[ChatMessage]) -> str:
    parts = [f"Previous summary:\n{previous or '(none)'}", "New messages:"]
    for m in messages:
        text = (m.content or "").strip()
        if len(text) > 1200:
            text = text[:1200] + "…"
        parts.append(f"{m.role}: {text}")
    return "\n\n".join(parts)
