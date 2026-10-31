"""Chat AI providers and how their models map to the three cost tiers.

The model lists come from each provider's own models endpoint (fetched when a key is
connected or tested and cached). The tier preferences below are documented model ids used
to pick sensible defaults from that live list; anything not offered by the user's key is
skipped, and the user can choose any listed model per tier. To add a provider: add an entry
here and, if its API differs from the three adapters, an adapter in `providers/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.ai.providers.anthropic_provider import AnthropicProvider
from app.ai.providers.base import ChatProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.openai_compat import OpenAICompatibleProvider

Tier = Literal["low", "balanced", "high"]
TIERS: tuple[Tier, ...] = ("low", "balanced", "high")


@dataclass(frozen=True)
class AIProviderSpec:
    key: str
    name: str
    api: Literal["anthropic", "openai", "gemini"]
    description: str
    default_base_url: str | None
    keys_url: str
    docs_url: str
    tier_preferences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Only the generic provider takes a user-supplied base URL.
    custom_base_url: bool = False
    key_optional: bool = False


AI_PROVIDERS: dict[str, AIProviderSpec] = {
    "anthropic": AIProviderSpec(
        key="anthropic",
        name="Anthropic Claude",
        api="anthropic",
        description="Claude models through the Anthropic API.",
        default_base_url=None,
        keys_url="https://platform.claude.com/settings/keys",
        docs_url="https://platform.claude.com/docs",
        tier_preferences={
            "low": ("claude-haiku-5-5",),
            "balanced": ("claude-sonnet-5-5",),
            "high": ("claude-opus-5-5",),
        },
    ),
    "openai": AIProviderSpec(
        key="openai",
        name="OpenAI",
        api="openai",
        description="GPT models through the OpenAI API.",
        default_base_url="https://api.openai.com/v1",
        keys_url="https://platform.openai.com/api-keys",
        docs_url="https://platform.openai.com/docs/models",
        tier_preferences={
            "low": ("gpt-5.6-luna", "gpt-5-nano", "gpt-5-mini"),
            "balanced": ("gpt-5.6-terra", "gpt-5-mini", "gpt-5"),
            "high": ("gpt-5.6-sol", "gpt-5.6", "gpt-5"),
        },
    ),
    "xai": AIProviderSpec(
        key="xai",
        name="xAI Grok",
        api="openai",
        description="Grok models through the xAI API (OpenAI-compatible).",
        default_base_url="https://api.x.ai/v1",
        keys_url="https://console.x.ai",
        docs_url="https://docs.x.ai/models",
        tier_preferences={
            "low": ("grok-4.20-non-reasoning", "grok-4.20-0309-non-reasoning"),
            "balanced": ("grok-4.20-non-reasoning", "grok-4.20-0309-non-reasoning"),
            "high": ("grok-4.20-reasoning", "grok-4.20-0309-reasoning"),
        },
    ),
    "gemini": AIProviderSpec(
        key="gemini",
        name="Google Gemini",
        api="gemini",
        description="Gemini models through the Gemini API (Google AI Studio key).",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        keys_url="https://aistudio.google.com/apikey",
        docs_url="https://ai.google.dev/gemini-api/docs/models",
        tier_preferences={
            "low": ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"),
            "balanced": ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"),
            "high": ("gemini-3.1-pro-preview", "gemini-3.1-pro", "gemini-2.5-pro"),
        },
    ),
    "openai_compatible": AIProviderSpec(
        key="openai_compatible",
        name="OpenAI-compatible API",
        api="openai",
        description="Any API that follows OpenAI Chat Completions: OpenRouter, Groq, DeepSeek, "
        "Mistral, Together or a local Ollama / LM Studio server.",
        default_base_url=None,
        keys_url="",
        docs_url="https://platform.openai.com/docs/api-reference/chat",
        custom_base_url=True,
        key_optional=True,
    ),
}

# Well-known base URLs offered as presets for the OpenAI-compatible provider.
COMPATIBLE_PRESETS: list[dict[str, str]] = [
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1"},
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1"},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1"},
    {"name": "Together", "base_url": "https://api.together.xyz/v1"},
    {"name": "Ollama (local)", "base_url": "http://localhost:11434/v1"},
    {"name": "LM Studio (local)", "base_url": "http://localhost:1234/v1"},
]

# Name hints used when none of the preferred ids is in the live list (e.g. custom APIs).
_TIER_HINTS: dict[str, tuple[str, ...]] = {
    "low": ("luna", "nano", "flash-lite", "haiku", "lite", "mini", "small", "8b", "fast", "flash"),
    "balanced": ("terra", "sonnet", "medium", "chat", "70b", "flash", "mini"),
    "high": ("sol", "opus", "pro", "large", "reasoning", "max", "405b"),
}


def resolve_tier_models(
    spec: AIProviderSpec, live_models: list[str], overrides: dict[str, str]
) -> dict[str, dict[str, object]]:
    """For each tier: {model, source: user|preferred|matched|fallback|none}."""
    available = set(live_models)
    result: dict[str, dict[str, object]] = {}
    for tier in TIERS:
        chosen = overrides.get(tier)
        if chosen and (not available or chosen in available):
            result[tier] = {"model": chosen, "source": "user"}
            continue
        preferred = spec.tier_preferences.get(tier, ())
        match = next((m for m in preferred if m in available), None)
        if match:
            result[tier] = {"model": match, "source": "preferred"}
            continue
        if available:
            candidates = sorted(available, reverse=True)
            hinted = next(
                (m for hint in _TIER_HINTS[tier] for m in candidates if hint in m.lower()), None
            )
            result[tier] = {"model": hinted or candidates[0], "source": "matched"}
            continue
        if preferred:
            result[tier] = {"model": preferred[0], "source": "fallback"}
        else:
            result[tier] = {"model": None, "source": "none"}
    return result


_ADAPTERS: dict[str, ChatProvider] = {}


def get_adapter(spec: AIProviderSpec) -> ChatProvider:
    adapter = _ADAPTERS.get(spec.key)
    if adapter is not None:
        return adapter
    if spec.api == "anthropic":
        adapter = AnthropicProvider()
    elif spec.api == "gemini":
        adapter = GeminiProvider()
    elif spec.key == "openai":
        adapter = OpenAICompatibleProvider(
            spec.default_base_url,
            use_max_completion_tokens=True,
            reasoning_effort_prefixes=("gpt-5", "o1", "o3", "o4"),
        )
    else:
        adapter = OpenAICompatibleProvider(spec.default_base_url)
    _ADAPTERS[spec.key] = adapter
    return adapter


def set_adapter(key: str, adapter: ChatProvider | None) -> None:
    """Tests replace adapters with fakes."""
    if adapter is None:
        _ADAPTERS.pop(key, None)
    else:
        _ADAPTERS[key] = adapter
