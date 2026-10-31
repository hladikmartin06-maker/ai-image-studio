"""Cost-aware routing: which tier answers a message, with which output limit and effort.

No API call is spent on routing: a local heuristic decides whether a message is simple
(short question, small talk) or complex (writing/optimizing prompts, comparisons,
troubleshooting, long input). Simple messages go to a cheaper tier.

  mode       simple   complex          (auto routing off: always the mode's own tier)
  low        low      low
  balanced   low      balanced
  high       balanced high
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Mode = Literal["low", "balanced", "high"]
Tier = Literal["low", "balanced", "high"]

_COMPLEX_WORDS = re.compile(
    r"\b("
    r"write|create|improve|optimi[sz]e|rewrite|enhance|compare|explain|why|analy[sz]e|"
    r"detailed|step by step|troubleshoot|debug|error|failed|fail|variations?|"
    r"napíš|napis|vytvor|vylepši|vylepsi|zlepši|zlepsi|optimalizuj|prepíš|prepis|porovnaj|"
    r"vysvetli|prečo|preco|analyzuj|podrobne|chyba|chybu|zlyhal[oa]?|nefunguje|variácie|"
    r"varianty"
    r")\b",
    re.IGNORECASE,
)

COMPLEX_ACTIONS = {"improve_prompt", "suggest_settings", "troubleshoot", "compare_models"}

_ROUTES: dict[str, tuple[Tier, Tier]] = {
    "low": ("low", "low"),
    "balanced": ("low", "balanced"),
    "high": ("balanced", "high"),
}
MAX_TOKENS: dict[str, int] = {"low": 900, "balanced": 1400, "high": 2400}
EFFORT: dict[str, Literal["low", "medium", "high"]] = {
    "low": "low",
    "balanced": "low",
    "high": "medium",
}
# How much recent conversation (estimated tokens) is replayed verbatim per mode.
HISTORY_BUDGET: dict[str, int] = {"low": 1200, "balanced": 2500, "high": 5000}


@dataclass
class Route:
    tier: Tier
    complexity: Literal["simple", "complex"]
    max_tokens: int
    effort: Literal["low", "medium", "high"]
    reason: str


def complexity(text: str, quick_action: str | None) -> tuple[Literal["simple", "complex"], str]:
    if quick_action in COMPLEX_ACTIONS:
        return "complex", f"action:{quick_action}"
    if len(text) > 400:
        return "complex", "long message"
    if "```" in text:
        return "complex", "code or prompt block"
    if text.count("?") >= 2:
        return "complex", "several questions"
    if _COMPLEX_WORDS.search(text):
        return "complex", "task keywords"
    return "simple", "short question"


def route(text: str, mode: Mode, auto: bool, quick_action: str | None = None) -> Route:
    kind, reason = complexity(text, quick_action)
    if auto:
        tier = _ROUTES[mode][0 if kind == "simple" else 1]
    else:
        tier = mode
        reason = "fixed by mode"
    return Route(
        tier=tier,
        complexity=kind,
        max_tokens=MAX_TOKENS[tier],
        effort=EFFORT[tier],
        reason=reason,
    )
