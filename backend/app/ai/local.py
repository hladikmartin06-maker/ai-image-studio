"""Answers that need no AI call at all (greetings and thanks): zero cost, instant."""

from __future__ import annotations

import re

_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "good morning",
    "good evening",
    "ahoj",
    "čau",
    "cau",
    "čaute",
    "dobrý deň",
    "dobry den",
    "dobré ráno",
    "dobre rano",
    "dobrý večer",
    "dobry vecer",
    "zdravím",
    "zdravim",
}
_THANKS = {
    "thanks",
    "thank you",
    "thx",
    "ty",
    "great thanks",
    "ďakujem",
    "dakujem",
    "vďaka",
    "vdaka",
    "ďakujem pekne",
    "dakujem pekne",
    "super ďakujem",
    "super dakujem",
}
_SK_HINT = re.compile(r"[áäčďéíĺľňóôŕšťúýž]|ahoj|cau|dakujem|vdaka|dobry|zdravim", re.IGNORECASE)

REPLIES = {
    ("greeting", "en"): "Hi! I can improve a prompt, recommend a model and settings, or explain "
    "why a generation failed. What are you working on?",
    ("greeting", "sk"): "Ahoj! Viem vylepšiť prompt, odporučiť model a nastavenia alebo vysvetliť, "
    "prečo generovanie zlyhalo. Na čom pracuješ?",
    ("thanks", "en"): "You're welcome! Ask any time.",
    ("thanks", "sk"): "Rado sa stalo! Kedykoľvek sa pýtaj.",
}


def local_answer(text: str, ui_language: str) -> str | None:
    cleaned = re.sub(r"[!.?,🙂😊👍\s]+", " ", text.lower()).strip()
    if not cleaned or len(cleaned) > 30:
        return None
    kind = "greeting" if cleaned in _GREETINGS else "thanks" if cleaned in _THANKS else None
    if kind is None:
        return None
    language = "sk" if _SK_HINT.search(cleaned) or ui_language == "sk" else "en"
    return REPLIES[(kind, language)]
