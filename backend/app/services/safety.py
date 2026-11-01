"""Prompt checks that run before anything is sent to a provider.

- Sexual content involving minors is always refused, for every model.
- Explicit prompts are refused for the SFW-only models (Nano Banana Pro, Seedream), with a
  hint to use Wan 2.7 / Wan 2.7 Pro, which allow mature content for adults, and for Krea 2
  unless its "18+ content" switch is on.
The providers apply their own moderation on top of this.
"""

from __future__ import annotations

import re

_MINOR_TERMS = (
    r"child",
    r"children",
    r"kid",
    r"kids",
    r"minor",
    r"minors",
    r"underage",
    r"under-age",
    r"teen",
    r"teens",
    r"teenager",
    r"preteen",
    r"pre-teen",
    r"schoolgirl",
    r"schoolboy",
    r"toddler",
    r"infant",
    r"baby",
    r"loli",
    r"lolita",
    r"shota",
    r"young girl",
    r"young boy",
    r"little girl",
    r"little boy",
    r"(?:1[0-7]|[1-9])[ -]?(?:yo|y/o|year[ -]?old|years[ -]?old)",
)
_EXPLICIT_TERMS = (
    r"nude",
    r"nudes",
    r"naked",
    r"nsfw",
    r"sex",
    r"sexual",
    r"sexy",
    r"porn",
    r"pornographic",
    r"erotic",
    r"explicit",
    r"topless",
    r"bottomless",
    r"lingerie",
    r"genitals?",
    r"nipples?",
    r"breasts?",
    r"boobs?",
    r"penis",
    r"vagina",
    r"orgasm",
    r"fetish",
    r"bdsm",
    r"hentai",
    r"undressed",
    r"seductive",
)

_MINOR_RE = re.compile(r"\b(?:" + "|".join(_MINOR_TERMS) + r")\b", re.IGNORECASE)
_EXPLICIT_RE = re.compile(r"\b(?:" + "|".join(_EXPLICIT_TERMS) + r")\b", re.IGNORECASE)


# Common phrases that contain a listed word but say nothing about age.
_HARMLESS_RE = re.compile(
    r"\bbaby[ -]?(?:blue|pink|hairs?|bangs)\b|\bkid[ -]?gloves?\b", re.IGNORECASE
)


def mentions_minors(text: str) -> bool:
    """True when the text refers to children or teenagers (used for AI model profiles,
    which must describe adults)."""
    return bool(_MINOR_RE.search(_HARMLESS_RE.sub(" ", text)))


class PromptRefused(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def is_explicit(text: str) -> bool:
    """True when the text asks for sexual content or nudity."""
    return bool(_EXPLICIT_RE.search(text))


def check_prompt(prompt: str, *, allows_mature: bool) -> None:
    explicit = bool(_EXPLICIT_RE.search(prompt))
    if explicit and _MINOR_RE.search(prompt):
        raise PromptRefused(
            "content_blocked",
            "This prompt combines sexual content with terms that refer to minors. "
            "It is never allowed.",
        )
    if explicit and not allows_mature:
        raise PromptRefused(
            "content_blocked",
            "This model is for SFW images only. Use Wan 2.7 or Wan 2.7 Pro for mature content.",
        )
