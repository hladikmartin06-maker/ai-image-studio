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

Target = Literal[
    "hair", "mouth", "eyes", "face", "expression", "clothing", "background", "body", "whole"
]
TARGETS: tuple[str, ...] = (
    "hair",
    "mouth",
    "eyes",
    "face",
    "expression",
    "clothing",
    "background",
    "body",
    "whole",
)
# The order edits run in (rules and the LoRA identity steps): the whole photo or the person
# first, the face next, the hair last so its colour has the final say.
ORDER = ("whole", "body", "background", "clothing", "face", "expression", "eyes", "mouth", "hair")
MAX_EDITS = 6


class Edit(BaseModel):
    target: Target
    # SAM 3.1's text prompt for the region (English, short); empty for "whole".
    region: str = Field(default="", max_length=80)
    # The whole photo as it should look after this change (Krea 2 sees the whole image).
    prompt: str = Field(min_length=1, max_length=2000)
    denoise: float | None = Field(default=None, ge=graphs.DENOISE_MIN, le=graphs.DENOISE_MAX)
    # Hair or clothing: the new colour ("#RRGGBB"), the region is tinted with it before the
    # redraw. None = found in the text when it names one; "" = none (the user removed it).
    color: str | None = Field(default=None, pattern=r"^(#[0-9a-fA-F]{6})?$")


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
    edits: list[Edit] = Field(default_factory=list, max_length=MAX_EDITS)
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
 "edits": [{"target": T, "region": str, "prompt": str, "color": "#RRGGBB"|null}],
 "checks": [{"id": str, "question": str, "kind": "modify"|"preserve", "target": T|null}]}
T is one of: hair, mouth, eyes, face, expression, clothing, background, body, whole
("body" = the whole person: figure, proportions, skin; region "person").

Rules:
- Keep the user's intent exactly. Never invent details the user did not ask for.
- No contradictions: closed lips/mouth means never mention teeth, tongue or an open mouth;
  a colour and a hairstyle are both stated when the user names both.
- Sexual content only if the user's text asks for it; nothing about minors ever.
- "checks": one yes/no question per requirement, in English, where "yes" means met
  (e.g. "Is the mouth closed with the lips together?"). For an edit add preserve checks for
  identity, background and anything the user said to keep.

TEXT TO IMAGE: "prompt" is an optimized English Krea 2 prompt: subject, appearance,
head pose and gaze, expression and mouth, hair (colour first, then style), clothing,
environment, lighting, composition, camera, photoreal details — only what the user's text
says or clearly implies. State the head pose, gaze and mouth plainly when the user names
them (e.g. "facing the camera, head straight, looking into the lens, neutral expression,
lips closed"). "negative" lists what must not
appear (only when the user excluded something). "edits" is []; a check's "target" is the
region it concerns or null.

EDIT (a photo is attached): "preserve" lists every visible property that must stay
(identity, face geometry, head pose and angle, gaze, eyes, nose, lips, body proportions,
pose, camera angle and framing, clothing, background, lighting, …) except what the user
asked to change; "modify" lists only the requested changes. One edit per changed region, in
the order to apply: "region" is a short English noun phrase a segmentation model can find
(e.g. "hair", "mouth and lips", "background"); "prompt" describes the WHOLE photo faithfully
as it should look after that change (the subject, head pose, clothing, place, light as
seen), with the change stated first and clearly (e.g. "dark brown hair, a woman with dark
brown hair, …"; "closed mouth, lips gently together, …"). "color": for a colour change of
hair or clothing, the new colour as a typical hex value of how it looks in a photo
(e.g. dark brown hair "#281B14", platinum blonde "#E5DCC8"); otherwise null. Add a preserve
check for the head pose ("Is the head turned and tilted the same way as in the original?").
Use target "whole" only when the change concerns the entire image.

IDENTITY (the request says so): the studio itself redraws the figure (the person without
face, hair, hands and feet) and then the face with the character LoRA, so body, proportions
and face become the character's while pose, hands and feet stay; the face is measured
against the LoRA's photos. Add one "body" edit (region "person") whose prompt describes the
person as seen — pose, clothing, what she holds — without the original woman's body shape
(e.g. "a woman sitting on the floor against a sofa, legs bent, wearing a white ribbed top
and leggings"), and edits for the other requested changes (hair, clothing, background,
mouth …); a background edit only when the user asks for a new background. No preserve check
about the original person's identity, face or body proportions, no check about the
character's likeness."""


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


# Hair colours (the studio's palette: typical tones as they look in photos), most specific
# first; English and Slovak words. Used to tint the hair before the redraw when the plan
# names a colour but gives no hex value (no AI analysis, or the AI left it out).
HAIR_COLOURS: tuple[tuple[str, str, str], ...] = (
    (r"platinum(?: blonde?)?|platinov\w*", "platinum blonde", "#E5DCC8"),
    (r"strawberry blonde?", "strawberry blonde", "#C8915E"),
    (r"(?:honey|golden) blonde?|zlat\w* blond\w*", "golden blonde", "#C9A05A"),
    (r"(?:dark|deep) blonde?|tmavo ?blond\w*", "dark blonde", "#A88654"),
    (r"(?:light|pale) blonde?|svetlo ?blond\w*", "light blonde", "#DCC594"),
    (r"blonde?|blond\w*", "blonde", "#CDAE74"),
    (
        r"(?:dark|deep) brown|dark chocolate|tmavo ?hned[éeýyáaoíuú]\w*",
        "very dark brown, almost black",
        "#281B14",
    ),
    (r"(?:light|pale) brown|svetlo ?hned[éeýyáaoíuú]\w*", "light brown", "#8A6646"),
    (r"chestnut(?: brown)?|gaštan\w*|gastan\w*", "chestnut brown", "#5C3622"),
    (r"chocolate(?: brown)?|čokolád\w*|cokolad\w*", "chocolate brown", "#4A2E20"),
    (r"brunette|brown|hned[éeýyáaoíuú]\w*", "brown", "#5A3E2B"),
    (r"jet black|uhľovo čiern\w*", "jet black", "#0F0D0D"),
    (r"black|čiern\w*|ciern\w*", "black", "#1B1716"),
    (r"auburn|kaštan\w*|kastan\w*", "auburn", "#7B3522"),
    (r"copper|meden\w*", "copper", "#B05A2C"),
    (r"ginger|redhead|red|ryšav\w*|rysav\w*|červen\w*|cerven\w*", "red", "#9C4128"),
    (r"silver|grey|gray|siv\w*|strieborn\w*", "silver grey", "#A9A9AC"),
    (r"white|biel\w*", "white", "#E9E7E3"),
    (r"pink|ružov\w*|ruzov\w*", "pink", "#E29AB9"),
    (r"blue|modr\w*", "blue", "#3B5DA6"),
    (r"purple|violet|fialov\w*", "purple", "#764C9E"),
    (r"green|zelen\w*", "green", "#4E8A58"),
)
_HAIR = r"(?:hair|vlas\w*)"
_GAP = r"[\s,-]+(?:[^\W\d_]+[\s,-]+){0,3}?"
# "from blonde to dark brown" / "z blond na tmavohnedé": the old colour is not the new one.
_FROM_TO = re.compile(r"\b(?:from|z|zo)\b[^.;]*?\b(?:to|into|na|do)\b", re.IGNORECASE)


# "her platinum blonde hair … to deep dark brown" / "vlasy na čierne": in a sentence about
# the hair, a colour a few words after "to"/"into"/"na"/"do" is the new one.
_TO = re.compile(r"\b(?:to|into|na|do)\b(?P<after>(?:[\s,]+[^\W\d_]+){1,6})", re.IGNORECASE)


def _target_colour(text: str) -> tuple[str, str] | None:
    for sentence in re.split(r"[.;!?\n]+", text):
        if not re.search(rf"\b{_HAIR}", sentence, re.IGNORECASE):
            continue
        for m in _TO.finditer(sentence):
            after = m.group("after")
            found = [
                (c.start(), -len(c.group(0)), name, hex_colour)
                for pattern, name, hex_colour in HAIR_COLOURS
                for c in re.finditer(rf"\b(?:{pattern})\b", after, re.IGNORECASE)
            ]
            if found:
                _, _, name, hex_colour = min(found)
                return name, hex_colour
    return None


def hair_colour(text: str) -> tuple[str, str] | None:
    """The hair colour a text asks for: (English name, "#RRGGBB"), or None. The colour it
    changes to wins over the one it changes from; otherwise only a colour next to
    "hair"/"vlasy" counts, so "a blue dress" is not taken for the hair."""
    target = _target_colour(text)
    if target:
        return target
    clean = _FROM_TO.sub(" ", text)
    best: tuple[int, int, str, str] | None = None
    for pattern, name, hex_colour in HAIR_COLOURS:
        for regex in (
            rf"\b(?P<c>{pattern})\b{_GAP}{_HAIR}",
            rf"{_HAIR}{_GAP}\b(?P<c>{pattern})\b",
        ):
            for m in re.finditer(regex, clean, re.IGNORECASE):
                key = (m.start("c"), -len(m.group("c")), name, hex_colour)
                if best is None or key[:2] < best[:2]:
                    best = key
    return (best[2], best[3]) if best else None


# Said after the colour: without it Krea 2 drew the old blonde strands as lighter highlights
# (the owner's car photo, 2026-11-05).
EVEN = "one even natural colour from roots to ends without highlights"


def _colour_first(edit: Edit, name: str) -> None:
    """The new colour stated first in the edit's prompt, so Krea 2 weighs it most."""
    if not edit.prompt.lower().startswith(f"{name} hair"):
        edit.prompt = f"{name} hair, {EVEN}, {edit.prompt}"


# Without AI analysis the regions a request clearly asks to change are found in its text, one
# sentence at a time: a region word with a change word (or a hair colour), in a sentence
# that does not keep something ("Preserve the body position and the hair" changes nothing).
_KEEP = re.compile(
    r"\b(preserve|maintain|retain|keep (?:the|her|his|their|its?|them|everything)|unchanged|"
    r"(?:stays?|remains?) the same|as in the original|ponechaj\w*|zachovaj\w*|nemeň\w*|"
    r"nemen\w*|bez zmeny)\b",
    re.IGNORECASE,
)
_CHANGE = re.compile(
    r"\b(change|make|turn|transform|match|replace|give|switch|swap|new|different|dye|"
    r"recolou?r|bigger|larger|smaller|longer|shorter|slimmer|thinner|thicker|curvier|fuller|"
    r"put|place|move|presuň\w*|presun\w*|prenes\w*|"
    r"zmeň\w*|zmen\w*|sprav\w*|daj|uprav\w*|prefarb\w*|nov\w*|väčš\w*|vacs\w*|menš\w*|"
    r"mens\w*|dlhš\w*|dlhs\w*|kratš\w*|krats\w*)\b",
    re.IGNORECASE,
)
_REGION_WORDS = {
    "body": r"body|figure|proportions?|curves|buttocks?|butt|hips|waist|breasts?|"
    r"telo|tela|tele|telom|postav\w*|proporci\w*|zadok\w*|boky|bokov",
    "face": r"face|facial|tvár\w*|tvari|tvarou",
    "hair": r"hair\w*|ponytail|bangs|braids?|vlas\w*|účes\w*|uces\w*|vrkoč\w*|vrkoc\w*",
    "background": r"background|backdrop|scene|setting|surroundings|environment|room|interior|"
    r"location|pozad\w*|prostred\w*|miestnos\w*|izb\w*|interiér\w*|interier\w*",
}


# Without analysis each region is redrawn with the sentences of the request about it, so a
# long text about the hair does not paint hair over the face (the owner's selfie test,
# 2026-11-05: with the whole text the face became bangs; with its own sentences a face).
_FACE_WORDS = (
    r"face|facial|expression|eyes?|eyelids?|eyebrows?|gaze|mouth|lips|tongue|teeth|smile|"
    r"head|tvár\w*|tvar\w*|oči|oci|ústa|usta|pery|jazyk\w*|výraz\w*|vyraz\w*"
)
# The figure's prompt says what she wears and how she sits or lies (the LoRA gives the body),
# never the hair or the face (they are not in its mask and would be drawn into it).
_CLOTHING_WORDS = (
    r"outfit|clothes|clothing|dress|top|shirt|t-shirt|blouse|tank|bra|bikini|lingerie|"
    r"underwear|panties|shorts|leggings|jeans|pants|trousers|skirt|socks|jacket|hoodie|"
    r"sweater|choker|necklace|oblečen\w*|šat\w*|tričk\w*|nohavic\w*|sukn\w*|podprsenk\w*"
)
_POSE_WORDS = (
    r"pose|posture|position|sitting|seated|lying|standing|kneeling|leaning|arms?|legs?|"
    r"hands?|phone|póz\w*|sed\w*|leží\w*|lezi\w*|stoj\w*|ruk\w*|noh\w*"
)
_NOT_BODY = (
    rf"{_REGION_WORDS['hair']}|facial|expression|mouth|lips?|tongue|teeth|smile|eyes?|eyelids?|"
    r"eyebrows?|brows?|gaze|tvár\w*|oči|oci|ústa|usta|pery|jazyk\w*|výraz\w*|vyraz\w*"
)
_SENTENCE_WORDS = {
    "face": (_FACE_WORDS, _REGION_WORDS["hair"]),
    "expression": (_FACE_WORDS, _REGION_WORDS["hair"]),
    "eyes": (_FACE_WORDS, _REGION_WORDS["hair"]),
    "mouth": (_FACE_WORDS, _REGION_WORDS["hair"]),
    "hair": (_REGION_WORDS["hair"], r"facial|identity|lora|body|figure|proportions?|buttocks?"),
    "body": (rf"{_REGION_WORDS['body']}|{_CLOTHING_WORDS}|{_POSE_WORDS}", _NOT_BODY),
    "background": (_REGION_WORDS["background"], rf"{_NOT_BODY}|{_FACE_WORDS}"),
}


def _changes(text: str) -> str:
    """The request without its "keep/preserve" sentences: a diffusion model does not read
    "preserve the exact mouth shape, lip position" as an instruction but draws mouth and lips
    (an open mouth, then a tongue, on the owner's car photo, 2026-11-05)."""
    kept = [s.strip() for s in re.split(r"[.;!?\n]+", text) if s.strip() and not _KEEP.search(s)]
    return ". ".join(kept) or text


def _region_text(text: str, target: str) -> str:
    """A region's prompt without analysis: the face regions without the "keep" sentences
    (redrawn gently, the photo keeps their shape); the hair with them, since it is redrawn
    strongly and "keep the wavy hairstyle, bangs" says what it should look like (without it
    the owner's selfie lost its waves, 2026-11-05)."""
    if target not in _SENTENCE_WORDS:
        return text
    face = target in ("face", "expression", "eyes", "mouth")
    return region_prompt(_changes(text) if face else text, target)


def region_prompt(text: str, target: str, *, fallback: str | None = None) -> str:
    """The request's sentences about this region (its first words — the trigger and "woman" —
    kept in front, except for the background, where they would draw a person); `fallback`,
    else the whole request, when none speaks of it."""
    words = _SENTENCE_WORDS.get(target)
    if not words:
        return text
    about, other = words
    sentences = [s.strip() for s in re.split(r"[.;!?\n]+", text) if s.strip()]
    said = []
    for sentence in sentences:
        if target == "body":
            # The figure keeps a sentence's clauses about clothing and pose and drops those
            # about the face or hair ("…the pose, the finger touching her lips, the black
            # dress…" lost the dress with the whole sentence, 2026-11-06).
            sentence = ", ".join(
                c.strip()
                for c in sentence.split(",")
                if not re.search(rf"\b({other})\b", c, re.IGNORECASE)
            )
        if re.search(rf"\b({about})\b", sentence, re.IGNORECASE) and not re.search(
            rf"\b({other})\b", sentence, re.IGNORECASE
        ):
            said.append(sentence)
    if not said:
        return text if fallback is None else fallback
    lead = sentences[0].split(",")[0].strip() if sentences and target != "background" else ""
    body = ". ".join(said)
    return body if not lead or body.startswith(lead) else f"{lead}, {body}"


def detected_targets(text: str) -> list[str]:
    """The regions (body, face, hair) the text asks to change."""
    found: list[str] = []
    for sentence in re.split(r"[.;!?\n]+", text):
        if not sentence.strip() or _KEEP.search(sentence):
            continue
        change = bool(_CHANGE.search(sentence))
        for target, words in _REGION_WORDS.items():
            if target in found or not re.search(rf"\b({words})\b", sentence, re.IGNORECASE):
                continue
            if change or (target == "hair" and hair_colour(sentence)):
                found.append(target)
    return found


def _ordered(edits: list[Edit]) -> list[Edit]:
    return sorted(edits, key=lambda e: ORDER.index(e.target))


def _with_identity(spec: Spec, text: str) -> None:
    """The person becomes the LoRA's character: first the figure (body, proportions, skin)
    with the LoRA, then the other changes (background, hair …), the face last, so the face
    that is measured is the face delivered (on the owner's photos, 2026-11-06, a hair step
    after the face pulled its likeness from 47 % to 16–28 %; with the face last it stayed at
    58–66 %). The plan's own figure and face edits are kept (the user may have edited them)."""
    edits = [e for e in spec.edits if e.target != "whole"]
    body = next((e for e in edits if e.target == "body"), None) or Edit(
        target="body", prompt=region_prompt(text, "body")
    )
    face = next((e for e in edits if e.target == "face"), None) or face_edit(text)
    rest = _ordered([e for e in edits if e.target not in ("body", "face")])
    spec.edits = [body, *rest[: MAX_EDITS - 2], face]


def face_edit(text: str) -> Edit:
    """The LoRA character's face: the request's sentences about the face without the "keep"
    ones (`_changes`), or the whole request when none speaks of it."""
    return Edit(
        target="face", region=graphs.REGIONS["face"], prompt=region_prompt(_changes(text), "face")
    )


# The background redrawn sharper (the switch "improve the background"): the request's
# sentences about it, or the same place in a sharp photo. Never the LoRA (its studio
# backdrops would turn the place grey) nor its trigger.
SHARP = "sharp, detailed, natural light, photorealistic"


def background_edit(text: str) -> Edit:
    said = region_prompt(_changes(text), "background", fallback="")
    prompt = f"{said}, {SHARP}" if said else f"the same place, {SHARP}"
    return Edit(
        target="background",
        region=graphs.REGIONS["background"],
        prompt=prompt,
        denoise=graphs.DENOISE_BACKGROUND_SHARPER,
    )


def validate(
    spec: Spec,
    *,
    user_prompt: str,
    adult: bool,
    identity: bool = False,
    improve_background: bool = False,
) -> Spec:
    """Prompts checked like every prompt; regions filled in; mouth contradictions removed.
    `identity` (an edit with a character LoRA): the figure and the face are redrawn with it.
    `improve_background`: the background is redrawn sharper unless the plan changes it."""
    text = " ".join(user_prompt.split())
    if (
        improve_background
        and spec.mode == "edit"
        and not any(e.target == "background" for e in spec.edits)
    ):
        first = sum(1 for e in spec.edits if e.target in ("whole", "body"))
        spec.edits.insert(first, background_edit(text))
    if identity and spec.mode == "edit":
        _with_identity(spec, text)
    spec.edits = spec.edits[:MAX_EDITS]
    closed = _wants_closed_mouth(user_prompt) or any(
        _wants_closed_mouth(c.question) for c in spec.checks
    )
    spec.prompt = _clean_mouth(spec.prompt, closed)
    for edit in spec.edits:
        edit.prompt = _clean_mouth(edit.prompt, closed or edit.target == "mouth")
        if edit.target == "mouth" and closed and not edit.prompt.lower().startswith("closed mouth"):
            # Turbo ignores negative prompts: the closed mouth is said first instead.
            edit.prompt = f"closed mouth, lips gently together, {edit.prompt}"
        if edit.target != "whole" and not edit.region.strip():
            edit.region = graphs.REGIONS.get(edit.target, edit.target)
        if edit.target == "whole":
            edit.region = ""
        if edit.target not in graphs.COLOUR_TARGETS:
            edit.color = None
        elif edit.target == "hair" and edit.color is None:
            found = hair_colour(edit.prompt) or hair_colour(user_prompt)
            if found:
                edit.color = found[1]
                _colour_first(edit, found[0])
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


def by_rules(
    *, mode: str, prompt: str, targets: list[str], negative: str = "", whole: bool = True
) -> Spec:
    """Without analysis: the user's text as the prompt; for an edit, the regions they chose
    plus those the text asks to change (body, face, hair, background), each redrawn with that
    text; none = the whole photo (`whole`; off when only the background is improved). No
    AI checks — only the measured ones."""
    text = " ".join(prompt.split())
    if mode == "text-to-image":
        return Spec(mode="text-to-image", prompt=text, negative=negative, source="rules")
    chosen = [t for t in targets if t in TARGETS]
    chosen += [t for t in detected_targets(text) if t not in chosen]
    edits = [
        Edit(
            target=t,  # type: ignore[arg-type]
            region=graphs.REGIONS.get(t, ""),
            prompt=_region_text(text, t),
        )
        for t in chosen or (["whole"] if whole else [])
    ]
    return Spec(mode="edit", modify=[text], edits=_ordered(edits)[:MAX_EDITS], source="rules")


async def by_ai(
    db: AsyncSession,
    user: User,
    *,
    mode: str,
    prompt: str,
    photo: bytes | None,
    lora_note: str | None,
    identity: bool = False,
) -> Spec:
    """The user's AI provider writes the plan (it sees the photo for an edit)."""
    lines = [f"MODE: {'EDIT' if mode == 'edit' else 'TEXT TO IMAGE'}", f"USER REQUEST: {prompt}"]
    if lora_note:
        lines.append(f"CHARACTER LoRA: {lora_note} (its trigger word is added by the studio).")
    if identity:
        lines.append("IDENTITY: yes — the person in the photo becomes the LoRA's character.")
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
            edits=[Edit(**_edit_fields(e)) for e in (data.get("edits") or [])][:MAX_EDITS]
            if mode == "edit"
            else [],
            checks=[Check(**c) for c in (data.get("checks") or [])][:10],
            source="ai",
            ai=meta,
        )
    except (ValidationError, TypeError) as exc:
        raise IntelligenceError(
            "bad_answer", "The AI's plan did not have the expected form."
        ) from exc
    return spec


def _edit_fields(raw: Any) -> dict[str, Any]:
    """An edit from the AI's answer; a colour that is not "#RRGGBB" counts as none given."""
    if not isinstance(raw, dict):
        raise TypeError("edit")
    fields = {k: raw.get(k) for k in ("target", "region", "prompt") if raw.get(k) is not None}
    colour = raw.get("color")
    if isinstance(colour, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", colour.strip()):
        fields["color"] = colour.strip().upper()
    return fields
