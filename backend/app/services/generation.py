"""Creating generation jobs: validation against the catalog, then hand-off to the worker."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import (
    MAX_IMAGES_PER_GENERATION,
    MAX_PROMPT_CHARS,
    PROVIDERS,
    CatalogError,
    ModelInfo,
    Variant,
    build_payload,
    get_model,
    normalize_settings,
    plan_tasks,
    resolve_variant,
)
from app.models import Generation, Integration, Persona, ProviderTask, ReferenceImage, User
from app.services import personas, preferences
from app.services.safety import ADULT_OFF, PromptRefused, check_prompt, is_explicit
from app.services.worker import worker


class PersonaUse(BaseModel):
    """The saved AI model to use for this request."""

    id: uuid.UUID
    # Which of its reference images to attach (in the AI model's order). None = all, primary
    # first, as many as the endpoint accepts.
    reference_image_ids: list[uuid.UUID] | None = Field(default=None, max_length=24)
    use_references: bool = True
    # Adds the AI model's description to the prompt.
    use_description: bool = True


class GenerationRequest(BaseModel):
    model: str
    provider: str | None = None
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    count: int = Field(default=1, ge=1, le=MAX_IMAGES_PER_GENERATION)
    settings: dict[str, Any] = Field(default_factory=dict)
    reference_image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=10)
    client_request_id: str | None = Field(default=None, max_length=64)
    source_generation_id: uuid.UUID | None = None
    persona: PersonaUse | None = None
    # The price the user confirmed (USD, the provider's estimate shown before the run), kept
    # for the spending overview; None when no price was known.
    price_estimate: float | None = Field(default=None, ge=0, le=100_000)
    # "18+ content" (adults only) on the models that allow it (Wan 2.7, Wan 2.7 Pro): off =
    # explicit prompts are refused there too; on = the images are marked 18+.
    adult: bool = False


class Replay(BaseModel):
    """What a repeated request copies from the original instead of recomposing it."""

    final_prompt: str | None = None
    persona_id: uuid.UUID | None = None
    persona_meta: dict[str, Any] = Field(default_factory=dict)


class GenerationRejected(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


async def _existing(db: AsyncSession, user: User, client_request_id: str) -> Generation | None:
    return await db.scalar(
        select(Generation)
        .where(Generation.user_id == user.id, Generation.client_request_id == client_request_id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
    )


async def _persona_references(
    db: AsyncSession,
    user: User,
    use: PersonaUse,
    *,
    model_key: str,
    provider: str,
    manual_count: int,
) -> tuple[Persona, list[uuid.UUID], dict[str, Any]]:
    """Which of the AI model's references go with this request. They are attached only
    when the endpoint takes reference images, within its limit (after the user's own)."""
    persona = await personas.owned(db, user.id, use.id)
    if persona is None:
        raise GenerationRejected("not_found", "The selected AI model no longer exists.", 404)
    ordered = [r.id for r in personas.ordered_references(persona)]
    if use.reference_image_ids is not None:
        wanted = set(use.reference_image_ids)
        if not wanted <= set(ordered):
            raise GenerationRejected(
                "invalid_parameter", "A selected image is no longer part of this AI model."
            )
        chosen = [r for r in ordered if r in wanted]
        explicit = True
    else:
        chosen = ordered
        explicit = False
    meta: dict[str, Any] = {
        "name": persona.name,
        "description_used": use.use_description,
        "references_used": use.use_references,
        "reference_ids": [],
        "references_available": len(ordered),
    }
    if not use.use_references or not chosen:
        return persona, [], meta
    model = get_model(model_key)
    edit = model.variant(provider, model.mode_for(True))
    if edit is None:
        meta["references_skipped"] = "unsupported"
        return persona, [], meta
    room = max(0, edit.max_reference_images - manual_count)
    if explicit and len(chosen) > room:
        raise GenerationRejected(
            "invalid_parameter",
            f"{get_model(model_key).name} on {PROVIDERS[provider]['name']} accepts up to "
            f"{edit.max_reference_images} reference images. Choose fewer images of the AI model.",
        )
    if len(chosen) > room:
        meta["references_skipped"] = len(chosen) - room
    chosen = chosen[:room]
    meta["reference_ids"] = [str(r) for r in chosen]
    return persona, chosen, meta


@dataclass
class Prepared:
    """A request checked against the catalog and the prompt rules, ready to be stored."""

    model: ModelInfo
    variant: Variant
    provider: str
    integration: Integration
    prompt: str
    final_prompt: str | None
    settings: dict[str, Any]
    ref_ids: list[uuid.UUID]
    persona: Persona | None
    persona_meta: dict[str, Any]
    plan: list[int]


async def create_generation(
    db: AsyncSession, user: User, req: GenerationRequest, *, replay: Replay | None = None
) -> Generation:
    if req.client_request_id:
        existing = await _existing(db, user, req.client_request_id)
        if existing is not None:
            return existing

    p = await prepare(db, user, req, replay=replay)
    model, variant, provider, prompt = p.model, p.variant, p.provider, p.prompt
    final_prompt, settings, ref_ids, persona = p.final_prompt, p.settings, p.ref_ids, p.persona
    persona_meta, plan = p.persona_meta, p.plan
    preview = build_payload(
        variant,
        prompt=final_prompt or prompt,
        settings=settings,
        image_urls=[f"<reference {i + 1}>" for i in range(len(ref_ids))],
        count=plan[0],
    )
    gen = Generation(
        user_id=user.id,
        client_request_id=req.client_request_id,
        model_key=model.key,
        provider=provider,
        mode=variant.mode,
        provider_model_id=variant.model_id,
        prompt=prompt,
        settings=settings,
        request_payload=preview,
        reference_image_ids=[str(r) for r in ref_ids],
        image_count=req.count,
        status="queued",
        source_generation_id=req.source_generation_id,
        persona_id=persona.id if persona is not None else (replay.persona_id if replay else None),
        persona_meta=persona_meta,
        final_prompt=final_prompt,
        cost_estimate=req.price_estimate,
        tasks=[ProviderTask(index=i, images_requested=n) for i, n in enumerate(plan)],
    )
    db.add(gen)
    if persona is not None:
        persona.last_used_at = datetime.now(UTC)
    try:
        await db.commit()
    except IntegrityError:
        # Two identical submits raced; return the one that won.
        await db.rollback()
        if req.client_request_id:
            existing = await _existing(db, user, req.client_request_id)
            if existing is not None:
                return existing
        raise
    worker.enqueue(gen.id)
    loaded = await db.scalar(
        select(Generation)
        .where(Generation.id == gen.id)
        .options(selectinload(Generation.tasks), selectinload(Generation.images))
        .execution_options(populate_existing=True)
    )
    assert loaded is not None
    return loaded


async def prepare(
    db: AsyncSession, user: User, req: GenerationRequest, *, replay: Replay | None = None
) -> Prepared:
    """Every check of a request, without storing or sending anything (also used for the
    price shown before a run)."""
    prompt = req.prompt.strip()
    if not prompt:
        raise GenerationRejected("invalid_parameter", "Enter a prompt.")

    prefs = await preferences.load(db, user.id)
    provider = req.provider or prefs.active_provider
    if provider is None:
        raise GenerationRejected(
            "not_connected", "Connect WaveSpeed or Atlas Cloud in Integrations first.", 409
        )
    if provider not in PROVIDERS:
        raise GenerationRejected("unsupported_provider", f"Unknown provider “{provider}”.")

    integration = await db.scalar(
        select(Integration).where(Integration.user_id == user.id, Integration.provider == provider)
    )
    if integration is None:
        raise GenerationRejected(
            "not_connected",
            f"{PROVIDERS[provider]['name']} is not connected. Add your API key in Integrations.",
            409,
        )

    ref_ids = list(dict.fromkeys(req.reference_image_ids))
    persona: Persona | None = None
    persona_meta: dict[str, Any] = {}
    if req.persona is not None and replay is None:
        try:
            get_model(req.model)
        except CatalogError as exc:
            raise GenerationRejected(exc.code, exc.message) from exc
        persona, persona_refs, persona_meta = await _persona_references(
            db,
            user,
            req.persona,
            model_key=req.model,
            provider=provider,
            manual_count=len(ref_ids),
        )
        # The AI model's images go first (primary first), then the user's own references.
        ref_ids = list(dict.fromkeys([*persona_refs, *ref_ids]))
    if ref_ids:
        found = (
            await db.scalars(
                select(ReferenceImage.id).where(
                    ReferenceImage.user_id == user.id, ReferenceImage.id.in_(ref_ids)
                )
            )
        ).all()
        if len(found) != len(ref_ids):
            raise GenerationRejected(
                "invalid_parameter", "A reference image no longer exists. Upload it again."
            )

    try:
        model = get_model(req.model)
        variant = resolve_variant(req.model, provider, bool(ref_ids))
        settings = normalize_settings(variant, req.settings)
    except CatalogError as exc:
        raise GenerationRejected(exc.code, exc.message) from exc

    if ref_ids and len(ref_ids) < variant.min_reference_images:
        raise GenerationRejected("invalid_parameter", "Add a reference image.")
    if len(ref_ids) > variant.max_reference_images:
        raise GenerationRejected(
            "invalid_parameter",
            f"{model.name} on {PROVIDERS[provider]['name']} accepts up to "
            f"{variant.max_reference_images} reference images.",
        )

    if req.adult and model.content != "sfw+nsfw":
        raise GenerationRejected(
            "invalid_parameter",
            f"{model.name} is for SFW images only. 18+ content is available on Wan 2.7 and "
            "Wan 2.7 Pro.",
        )
    allows_mature = model.content == "sfw+nsfw"
    try:
        check_prompt(prompt, allows_mature=allows_mature)
        if allows_mature and not req.adult and is_explicit(prompt):
            raise PromptRefused("adult_off", ADULT_OFF)
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc

    final_prompt: str | None = None
    if replay is not None:
        final_prompt = replay.final_prompt
        persona_meta = dict(replay.persona_meta)
    elif persona is not None and req.persona is not None and req.persona.use_description:
        composed = personas.compose_prompt(prompt, persona, len(persona_meta["reference_ids"]))
        final_prompt = composed if composed != prompt else None
    if final_prompt is not None:
        if len(final_prompt) > MAX_PROMPT_CHARS + personas.PROMPT_TEXT_MAX:
            raise GenerationRejected(
                "invalid_parameter", "The prompt with the AI model's description is too long."
            )
        try:
            if persona is not None:
                personas.check_composed(final_prompt, persona, allows_mature)
                if allows_mature and not req.adult and is_explicit(final_prompt):
                    raise PromptRefused(
                        "adult_off",
                        f"The description of the AI model “{persona.name}” asks for 18+ "
                        "content. Turn on “18+ content” or switch its description off.",
                    )
            else:
                check_prompt(final_prompt, allows_mature=allows_mature)
                if allows_mature and not req.adult and is_explicit(final_prompt):
                    raise PromptRefused("adult_off", ADULT_OFF)
        except PromptRefused as exc:
            raise GenerationRejected(exc.code, exc.message, 422) from exc

    if req.adult:
        # Kept with the request (never sent to the provider: build_payload uses only the
        # documented settings); the worker marks these images 18+.
        settings["adult"] = True

    return Prepared(
        model=model,
        variant=variant,
        provider=provider,
        integration=integration,
        prompt=prompt,
        final_prompt=final_prompt,
        settings=settings,
        ref_ids=ref_ids,
        persona=persona,
        persona_meta=persona_meta,
        plan=plan_tasks(variant, req.count),
    )
