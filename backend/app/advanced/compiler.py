"""The plan of an Advanced request: what to keep, what to change, the Krea 2 prompts, the
regions to redraw and the yes/no checks of the result.

With "analysis" on, the user's AI provider writes it (looking at the photo for an edit);
without it the plan is the user's own text and the regions they chose. Either way the
studio validates it: known targets only, prompts checked like every prompt (minors never,
explicit only with the 18+ switch), and contradictions about the mouth removed."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.advanced import graphs
from app.advanced.intelligence import IntelligenceError, ask_json
from app.models import User
from app.services.safety import ADULT_OFF, PromptRefused, check_prompt, is_explicit

Target = Literal["hair", "mouth", "eyes", "face", "expression", "clothing", "background", "whole"]
TARGETS: tuple[str, ...] = (
    "hair",
    "mouth",
    "eyes",
    "face",
    "expression",
    "clothing",
    "background",
    "whole",
)


class Edit(BaseModel):
    target: Target
    # SAM 3.1's text prompt for the region (English, short); empty for "whole".
    region: str = Field(default="", max_length=80)
    # The whole photo as it should look after this change (Krea 2 sees the whole image).
    prompt: str = Field(min_length=1, max_length=2000)
    denoise: float | None = Field(default=None, ge=graphs.DENOISE_MIN, le=graphs.DENOISE_MAX)


class Check(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    # A yes/no question in English; "yes" means the requirement is met.
    question: str = Field(min_length=3, max_length=300)
    kind: Literal["modify", "preserve"] = "modify"
    # The region a failed check can be fixed in, when there is one.
    target: Target | None = None


class Spec(BaseModel):
    mode: Literal["text-to-image", "edit"]
    summary: str = Field(default="", max_length=600)
    preserve: list[str] = Field(default_factory=list, max_length=20)
    modify: list[str] = Field(default_factory=list, max_length=12)
    # Text to image: the Krea 2 prompt (and RAW's negative prompt).
    prompt: str = Field(default="", max_length=2000)
    negative: str = Field(default="", max_length=600)
    edits: list[Edit] = Field(default_factory=list, max_length=4)
    checks: list[Check] = Field(default_factory=list, max_length=10)
    source: Literal["ai", "rules", "user"] = "rules"
    ai: dict[str, Any] | None = None

    @field_validator("preserve", "modify")
    @classmethod
    def _short(cls, value: list[str]) -> list[str]:
        return [" ".join(v.split())[:160] for v in value if v and v.strip()]


SYSTEM = """You plan requests for an image studio whose generator is Krea 2, a photorealistic \
text-to-image model, for fictional adult characters. Reply with one JSON object only.

Schema:
{"summary": str, "preserve": [str], "modify": [str], "prompt": str, "negative": str,
 "edits": [{"target": T, "region": str, "prompt": str}],
 "checks": [{"id": str, "question": str, "kind": "modify"|"preserve", "target": T|null}]}
T is one of: hair, mouth, eyes, face, expression, clothing, background, whole.

Rules:
- Keep the user's intent exactly. Never invent details the user did not ask for.
- No contradictions: closed lips/mouth means never mention teeth, tongue or an open mouth;
  a colour and a hairstyle are both stated when the user names both.
- Sexual content only if the user's text asks for it; nothing about minors ever.
- "checks": one yes/no question per requirement, in English, where "yes" means met
  (e.g. "Is the mouth closed with the lips together?"). For an edit add preserve checks for
  identity, background and anything the user said to keep.

TEXT TO IMAGE: "prompt" is an optimized English Krea 2 prompt: subject, appearance,
expression and mouth, hair, clothing, environment, lighting, composition, camera, photoreal
details — only what the user's text says or clearly implies. "negative" lists what must not
appear (only when the user excluded something). "edits" is []; a check's "target" is the
region it concerns or null.

EDIT (a photo is attached): "preserve" lists every visible property that must stay
(identity, face geometry, eyes, nose, lips, body proportions, pose, camera angle and
framing, clothing, background, lighting, …) except what the user asked to change; "modify"
lists only the requested changes. One edit per changed region, in the order to apply:
"region" is a short English noun phrase a segmentation model can find (e.g. "hair",
"mouth and lips", "background"); "prompt" describes the WHOLE photo faithfully as it should
look after that change (the subject, pose, clothing, place, light as seen), with the change
stated clearly. Use target "whole" only when the change concerns the entire image."""


def _clean_mouth(text: str, closed: bool) -> str:
    """Closed lips: no words that ask for an open mouth, teeth or tongue."""
    if not closed:
        return text
    cleaned = re.sub(
        r",?\s*(open(ed)? mouth|mouth open|teeth (showing|visible)|showing teeth|visible teeth|"
        r"tongue( out)?|smiling with teeth|parted lips|lips parted)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip(" ,")


CLOSED_MOUTH = re.compile(
    r"\b(clos(e|ed|es)\b.*\b(mouth|lips)|lips (closed|together)|mouth closed|zavret|zatvor)",
    re.IGNORECASE,
)


def _wants_closed_mouth(text: str) -> bool:
    return bool(CLOSED_MOUTH.search(text))


def validate(spec: Spec, *, user_prompt: str, adult: bool) -> Spec:
    """Prompts checked like every prompt; regions filled in; mouth contradictions removed."""
    closed = _wants_closed_mouth(user_prompt) or any(
        _wants_closed_mouth(c.question) for c in spec.checks
    )
    spec.prompt = _clean_mouth(spec.prompt, closed)
    for edit in spec.edits:
        edit.prompt = _clean_mouth(edit.prompt, closed or edit.target == "mouth")
        if edit.target != "whole" and not edit.region.strip():
            edit.region = graphs.REGIONS.get(edit.target, edit.target)
        if edit.target == "whole":
            edit.region = ""
    texts = [
        spec.prompt,
        spec.negative,
        *(e.prompt for e in spec.edits),
        *(e.region for e in spec.edits),
    ]
    for text in texts:
        if not text:
            continue
        check_prompt(text, allows_mature=True)
        if not adult and is_explicit(text):
            raise PromptRefused("adult_off", ADULT_OFF)
    if spec.mode == "text-to-image" and not spec.prompt.strip():
        spec.prompt = user_prompt.strip()
    if spec.mode == "edit" and not spec.edits:
        raise PromptRefused("invalid_parameter", "The plan has no change to make.")
    return spec


def by_rules(*, mode: str, prompt: str, targets: list[str], negative: str = "") -> Spec:
    """Without analysis: the user's text as the prompt; for an edit, the regions they chose
    (each redrawn with that text). No AI checks — only the measured ones."""
    text = " ".join(prompt.split())
    if mode == "text-to-image":
        return Spec(mode="text-to-image", prompt=text, negative=negative, source="rules")
    chosen = [t for t in targets if t in TARGETS] or ["whole"]
    return Spec(
        mode="edit",
        modify=[text],
        edits=[
            Edit(target=t, region=graphs.REGIONS.get(t, ""), prompt=text)  # type: ignore[arg-type]
            for t in chosen[:4]
        ],
        source="rules",
    )


async def by_ai(
    db: AsyncSession,
    user: User,
    *,
    mode: str,
    prompt: str,
    photo: bytes | None,
    lora_note: str | None,
) -> Spec:
    """The user's AI provider writes the plan (it sees the photo for an edit)."""
    lines = [f"MODE: {'EDIT' if mode == 'edit' else 'TEXT TO IMAGE'}", f"USER REQUEST: {prompt}"]
    if lora_note:
        lines.append(f"CHARACTER LoRA: {lora_note} (its trigger word is added by the studio).")
    data, meta = await ask_json(
        db,
        user,
        system=SYSTEM,
        text="\n".join(lines),
        images=[photo] if photo else [],
    )
    try:
        spec = Spec(
            mode="edit" if mode == "edit" else "text-to-image",
            summary=str(data.get("summary") or ""),
            preserve=list(data.get("preserve") or []),
            modify=list(data.get("modify") or []),
            prompt=str(data.get("prompt") or ""),
            negative=str(data.get("negative") or ""),
            edits=[Edit(**e) for e in (data.get("edits") or [])][:4] if mode == "edit" else [],
            checks=[Check(**c) for c in (data.get("checks") or [])][:10],
            source="ai",
            ai=meta,
        )
    except (ValidationError, TypeError) as exc:
        raise IntelligenceError(
            "bad_answer", "The AI's plan did not have the expected form."
        ) from exc
    return spec
