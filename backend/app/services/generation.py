"""Creating generation jobs: validation against the catalog, then hand-off to the worker."""

from __future__ import annotations

import uuid
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
    build_payload,
    get_model,
    normalize_settings,
    plan_tasks,
    resolve_variant,
)
from app.models import Generation, Integration, ProviderTask, ReferenceImage, User
from app.services import preferences
from app.services.safety import PromptRefused, check_prompt
from app.services.worker import worker


class GenerationRequest(BaseModel):
    model: str
    provider: str | None = None
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    count: int = Field(default=1, ge=1, le=MAX_IMAGES_PER_GENERATION)
    settings: dict[str, Any] = Field(default_factory=dict)
    reference_image_ids: list[uuid.UUID] = Field(default_factory=list, max_length=10)
    client_request_id: str | None = Field(default=None, max_length=64)
    source_generation_id: uuid.UUID | None = None


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


async def create_generation(db: AsyncSession, user: User, req: GenerationRequest) -> Generation:
    prompt = req.prompt.strip()
    if not prompt:
        raise GenerationRejected("invalid_parameter", "Enter a prompt.")

    if req.client_request_id:
        existing = await _existing(db, user, req.client_request_id)
        if existing is not None:
            return existing

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

    try:
        check_prompt(prompt, allows_mature=model.content == "sfw+nsfw")
    except PromptRefused as exc:
        raise GenerationRejected(exc.code, exc.message, 422) from exc

    plan = plan_tasks(variant, req.count)
    preview = build_payload(
        variant,
        prompt=prompt,
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
        tasks=[ProviderTask(index=i, images_requested=n) for i, n in enumerate(plan)],
    )
    db.add(gen)
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
