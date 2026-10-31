"""AI models: saved fictional adult characters with a visual description and references.

The description is turned into one predictable block of text that is appended to the
user's prompt (only when the user keeps it switched on), and the reference images are
attached only when the selected endpoint accepts reference images, within its limit.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Generation, Persona, PersonaImage, ReferenceImage
from app.services.safety import PromptRefused, check_prompt, mentions_minors
from app.services.serializers import iso

MAX_PERSONA_IMAGES = 200
NAME_MAX = 80
DESCRIPTION_MAX = 1500
NOTES_MAX = 4000
# Upper bound of the text an AI model adds to a prompt (description, appearance, identity).
PROMPT_TEXT_MAX = 5000

# Appearance fields in the order they are written into the prompt, with their labels there.
PROMPT_LABELS: dict[str, str] = {
    "ethnicity": "ethnicity",
    "skin": "skin",
    "eyes": "eyes",
    "hair_color": "hair color",
    "hairstyle": "hairstyle",
    "face": "facial features",
    "body": "body proportions",
    "height": "height",
    "style": "style",
    "distinctive": "distinctive features",
    "extra": "other details",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class Appearance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Characters are adults: an age, when given, is 18 or more.
    age: int | None = Field(default=None, ge=18, le=120)
    gender: str = Field(default="", max_length=60)
    ethnicity: str = Field(default="", max_length=120)
    skin: str = Field(default="", max_length=160)
    eyes: str = Field(default="", max_length=160)
    hair_color: str = Field(default="", max_length=120)
    hairstyle: str = Field(default="", max_length=300)
    face: str = Field(default="", max_length=400)
    body: str = Field(default="", max_length=400)
    height: str = Field(default="", max_length=60)
    style: str = Field(default="", max_length=300)
    distinctive: str = Field(default="", max_length=300)
    extra: str = Field(default="", max_length=400)

    @field_validator(
        "gender",
        "ethnicity",
        "skin",
        "eyes",
        "hair_color",
        "hairstyle",
        "face",
        "body",
        "height",
        "style",
        "distinctive",
        "extra",
    )
    @classmethod
    def _strip(cls, value: str) -> str:
        return _clean(value)


class PersonaCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=NAME_MAX)
    description: str = Field(default="", max_length=DESCRIPTION_MAX)
    notes: str = Field(default="", max_length=NOTES_MAX)
    appearance: Appearance = Field(default_factory=Appearance)
    # The user confirms that the character is fictional and an adult.
    adult_confirmed: bool = False


class PersonaPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=NAME_MAX)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    notes: str | None = Field(default=None, max_length=NOTES_MAX)
    appearance: Appearance | None = None
    # A reference id of this model's collection; null = the first image.
    cover_reference_id: uuid.UUID | None = None
    # Reference ids in the new order (ids not listed keep their place after these).
    image_order: list[uuid.UUID] | None = Field(default=None, max_length=MAX_PERSONA_IMAGES)


class PersonaRejected(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def check_text(*texts: str) -> None:
    """AI models are fictional adults: words that refer to minors are never accepted."""
    if any(mentions_minors(text) for text in texts if text):
        raise PersonaRejected(
            422,
            "content_blocked",
            "AI models must be fictional adults. Remove words that refer to minors.",
        )


def check_persona_fields(name: str, description: str, appearance: Appearance) -> None:
    values = [str(v) for v in appearance.model_dump().values() if isinstance(v, str)]
    check_text(name, description, *values)


# --- prompt text ----------------------------------------------------------------------------


def character_text(persona: Persona) -> str:
    """The description block added to the prompt, e.g.
    "Character: adult woman, 28 years old; freckled, warm smile; hairstyle: long waves."
    Empty when the model has no description at all."""
    look = Appearance.model_validate(persona.appearance or {})
    lead = f"adult {look.gender}" if look.gender else "adult"
    if look.age:
        lead += f", {look.age} years old"
    parts = [lead]
    description = _clean(persona.description or "").rstrip(".")
    if description:
        parts.append(description)
    for key, label in PROMPT_LABELS.items():
        value = getattr(look, key)
        if value:
            parts.append(f"{label}: {value.rstrip('.')}")
    if len(parts) == 1 and not look.gender and not look.age:
        return ""
    return "Character: " + "; ".join(parts) + "."


def identity_text(reference_count: int) -> str:
    if reference_count <= 0:
        return ""
    images = "the reference image" if reference_count == 1 else "the reference images"
    return (
        f"Keep the same person as in {images}: identical face, facial features, hairstyle "
        "and body proportions."
    )


def compose_prompt(prompt: str, persona: Persona, reference_count: int) -> str:
    extra = " ".join(p for p in (character_text(persona), identity_text(reference_count)) if p)
    return f"{prompt}\n\n{extra}" if extra else prompt


def check_composed(composed: str, persona: Persona, allows_mature: bool) -> None:
    """Runs the prompt rules on the prompt with the description added (the user's own
    prompt is checked first), with a message that points at the AI model."""
    try:
        check_prompt(composed, allows_mature=allows_mature)
    except PromptRefused as exc:
        raise PromptRefused(
            exc.code,
            f"The description of the AI model “{persona.name}” is not allowed here: "
            f"{exc.message} Edit the AI model or switch its description off.",
        ) from exc


# --- loading and JSON ----------------------------------------------------------------------


def with_images(stmt: Select) -> Select:
    return stmt.options(selectinload(Persona.images).selectinload(PersonaImage.reference))


async def owned(db: AsyncSession, user_id: uuid.UUID, persona_id: uuid.UUID) -> Persona | None:
    return await db.scalar(
        with_images(
            select(Persona).where(Persona.id == persona_id, Persona.user_id == user_id)
        ).execution_options(populate_existing=True)
    )


def ordered_references(persona: Persona) -> list[ReferenceImage]:
    """The collection with the primary image first, then in the user's order."""
    refs = [link.reference for link in sorted(persona.images, key=lambda i: i.position)]
    primary = next((r for r in refs if r.id == persona.cover_reference_id), None)
    if primary is not None:
        refs.remove(primary)
        refs.insert(0, primary)
    return refs


def primary_reference(persona: Persona) -> ReferenceImage | None:
    refs = ordered_references(persona)
    return refs[0] if refs else None


async def generation_counts(
    db: AsyncSession, user_id: uuid.UUID, persona_ids: list[uuid.UUID]
) -> dict[uuid.UUID, int]:
    if not persona_ids:
        return {}
    rows = await db.execute(
        select(Generation.persona_id, func.count(Generation.id))
        .where(Generation.user_id == user_id, Generation.persona_id.in_(persona_ids))
        .group_by(Generation.persona_id)
    )
    return {pid: n for pid, n in rows.all() if pid is not None}


def image_json(ref: ReferenceImage, primary: bool) -> dict[str, Any]:
    return {
        "id": str(ref.id),
        "url": f"/api/reference-images/{ref.id}/file",
        "thumb_url": f"/api/reference-images/{ref.id}/thumb",
        "original_name": ref.original_name,
        "mime_type": ref.mime_type,
        "width": ref.width,
        "height": ref.height,
        "size_bytes": ref.size_bytes,
        "is_primary": primary,
        "created_at": iso(ref.created_at),
    }


def persona_json(persona: Persona, generation_count: int = 0) -> dict[str, Any]:
    refs = ordered_references(persona)
    images = [image_json(ref, i == 0) for i, ref in enumerate(refs)]
    look = Appearance.model_validate(persona.appearance or {})
    return {
        "id": str(persona.id),
        "name": persona.name,
        "description": persona.description,
        "notes": persona.notes,
        "appearance": look.model_dump(),
        "primary_image": images[0] if images else None,
        "images": images,
        "image_count": len(images),
        "max_images": MAX_PERSONA_IMAGES,
        "prompt_text": character_text(persona),
        # Added after it when reference images are attached (one / several images).
        "identity_text": {"one": identity_text(1), "many": identity_text(2)},
        "generation_count": generation_count,
        "last_used_at": iso(persona.last_used_at),
        "created_at": iso(persona.created_at),
        "updated_at": iso(persona.updated_at),
    }


async def link_reference(db: AsyncSession, persona: Persona, ref: ReferenceImage) -> bool:
    """Adds a reference to the collection; False when it is already there."""
    if any(link.reference_image_id == ref.id for link in persona.images):
        return False
    if len(persona.images) >= MAX_PERSONA_IMAGES:
        raise PersonaRejected(
            409,
            "limit_reached",
            f"An AI model holds up to {MAX_PERSONA_IMAGES} reference images.",
        )
    position = max((link.position for link in persona.images), default=-1) + 1
    persona.images.append(
        PersonaImage(
            persona_id=persona.id,
            user_id=persona.user_id,
            reference_image_id=ref.id,
            position=position,
            reference=ref,
        )
    )
    await db.flush()
    return True
