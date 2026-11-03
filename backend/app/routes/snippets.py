"""Prompt snippets: saved pieces of prompt text (lighting, outfit, place, …), for every AI model
or only for one, inserted into a prompt with one click."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.models import Persona, PromptSnippet, User
from app.services.safety import mentions_minors
from app.services.serializers import iso

router = APIRouter(prefix="/api/snippets", tags=["snippets"])

MAX_SNIPPETS = 300


class SnippetBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1000)
    # Only for this AI model; None = for all.
    persona_id: uuid.UUID | None = None


class SnippetPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    text: str | None = Field(default=None, min_length=1, max_length=1000)


def snippet_json(s: PromptSnippet) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "persona_id": str(s.persona_id) if s.persona_id else None,
        "name": s.name,
        "text": s.text,
        "created_at": iso(s.created_at),
        "updated_at": iso(s.updated_at),
    }


def _check(*texts: str) -> None:
    # Snippets go into prompts: the same rule as AI model profiles.
    if any(mentions_minors(t) for t in texts):
        raise api_error(
            400, "content_blocked", "Snippets must not mention minors. Remove those words."
        )


async def _owned(db: AsyncSession, user: User, snippet_id: uuid.UUID) -> PromptSnippet:
    row = await db.scalar(
        select(PromptSnippet).where(
            PromptSnippet.id == snippet_id, PromptSnippet.user_id == user.id
        )
    )
    if row is None:
        raise api_error(404, "not_found", "Snippet not found.")
    return row


@router.get("")
async def list_snippets(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = (
        await db.scalars(
            select(PromptSnippet)
            .where(PromptSnippet.user_id == user.id)
            .order_by(func.lower(PromptSnippet.name))
        )
    ).all()
    return {"items": [snippet_json(r) for r in rows]}


@router.post("", status_code=201)
async def create_snippet(
    body: SnippetBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    _check(body.name, body.text)
    if body.persona_id is not None:
        persona = await db.scalar(
            select(Persona).where(Persona.id == body.persona_id, Persona.user_id == user.id)
        )
        if persona is None:
            raise api_error(404, "not_found", "That AI model no longer exists.")
    count = await db.scalar(
        select(func.count()).select_from(PromptSnippet).where(PromptSnippet.user_id == user.id)
    )
    if (count or 0) >= MAX_SNIPPETS:
        raise api_error(409, "too_many", f"Up to {MAX_SNIPPETS} snippets.")
    row = PromptSnippet(
        user_id=user.id,
        persona_id=body.persona_id,
        name=body.name.strip(),
        text=body.text.strip(),
    )
    db.add(row)
    await db.commit()
    return snippet_json(row)


@router.patch("/{snippet_id}")
async def update_snippet(
    snippet_id: uuid.UUID,
    body: SnippetPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await _owned(db, user, snippet_id)
    _check(body.name or "", body.text or "")
    if body.name is not None:
        row.name = body.name.strip()
    if body.text is not None:
        row.text = body.text.strip()
    await db.commit()
    return snippet_json(row)


@router.delete("/{snippet_id}")
async def delete_snippet(
    snippet_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    row = await _owned(db, user, snippet_id)
    await db.delete(row)
    await db.commit()
    return {"ok": True}
