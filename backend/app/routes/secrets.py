"""Other credentials: the Hugging Face token for gated downloads (Krea 2 RAW/Turbo
repositories) on the user's GPU. Encrypted like every key; only the last 4 characters are
ever returned."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.models import User, UserSecret
from app.security import decrypt_secret, encrypt_secret, mask_key
from app.services import huggingface
from app.services.serializers import iso

router = APIRouter(prefix="/api/secrets", tags=["secrets"])

Name = Literal["huggingface"]


class SecretBody(BaseModel):
    value: str = Field(min_length=8, max_length=300)


async def _row(db: AsyncSession, user: User, name: str) -> UserSecret | None:
    return await db.scalar(
        select(UserSecret).where(UserSecret.user_id == user.id, UserSecret.name == name)
    )


async def hf_token(db: AsyncSession, user: User) -> str | None:
    row = await _row(db, user, "huggingface")
    if row is None:
        return None
    return decrypt_secret(row.encrypted_value, associated=f"{user.id}:secret:huggingface")


@router.get("")
async def list_secrets(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    row = await _row(db, user, "huggingface")
    return {
        "items": [
            {
                "name": "huggingface",
                "set": row is not None,
                "masked": mask_key(row.last4) if row else None,
                "updated_at": iso(row.updated_at) if row else None,
            }
        ]
    }


@router.get("/huggingface/status")
async def huggingface_status(
    refresh: bool = False, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """The token's account and whether it can download each gated Krea 2 repository."""
    row = await _row(db, user, "huggingface")
    if row is None:
        return {
            "set": False,
            "masked": None,
            "token": None,
            "token_message": None,
            "account": None,
            "ready": False,
            "repos": [
                {**r, "access": "unknown", "message": None} for r in huggingface.gated_repos()
            ],
        }
    token = decrypt_secret(row.encrypted_value, associated=f"{user.id}:secret:huggingface")
    result = await huggingface.status(token, refresh=refresh)
    return {"set": True, "masked": mask_key(row.last4), **result}


@router.put("/{name}")
async def save_secret(
    name: Name,
    body: SecretBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    value = body.value.strip()
    if name == "huggingface":
        if not value.startswith("hf_"):
            raise api_error(400, "invalid_parameter", "Hugging Face tokens start with hf_.")
        # A token Hugging Face rejects is never saved; when Hugging Face cannot be reached
        # the token is saved and checked again later.
        state, _, reason = await huggingface.HubClient(value).whoami()
        if state == "invalid":
            raise api_error(
                400,
                "invalid_key",
                "Hugging Face rejected this token"
                + (f" ({reason})" if reason else "")
                + ". Copy the whole token (it starts with hf_), or create a new one of type "
                "Read, and paste it.",
            )
    row = await _row(db, user, name)
    if row is None:
        row = UserSecret(user_id=user.id, name=name, encrypted_value=b"", last4="")
        db.add(row)
    row.encrypted_value = encrypt_secret(value, associated=f"{user.id}:secret:{name}")
    row.last4 = value[-4:]
    await db.commit()
    return {"name": name, "set": True, "masked": mask_key(row.last4)}


@router.delete("/{name}")
async def delete_secret(
    name: Name, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    await db.execute(
        delete(UserSecret).where(UserSecret.user_id == user.id, UserSecret.name == name)
    )
    await db.commit()
    return {"ok": True}
