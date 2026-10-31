"""Prompt checks that run before anything is sent to a provider.

- Sexual content involving minors is always refused, for every model.
- Explicit prompts are refused for the SFW-only models (Nano Banana Pro, Seedream), with a
  hint to use Wan 2.7 / Wan 2.7 Pro, which allow mature content for adults.
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


class PromptRefused(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


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
