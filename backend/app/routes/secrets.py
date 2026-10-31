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
from app.security import encrypt_secret, mask_key
from app.services.serializers import iso

router = APIRouter(prefix="/api/secrets", tags=["secrets"])

Name = Literal["huggingface"]


class SecretBody(BaseModel):
    value: str = Field(min_length=8, max_length=300)


@router.get("")
async def list_secrets(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    rows = (await db.scalars(select(UserSecret).where(UserSecret.user_id == user.id))).all()
    found = {r.name: r for r in rows}
    return {
        "items": [
            {
                "name": "huggingface",
                "set": "huggingface" in found,
                "masked": mask_key(found["huggingface"].last4) if "huggingface" in found else None,
                "updated_at": iso(found["huggingface"].updated_at)
                if "huggingface" in found
                else None,
            }
        ]
    }


@router.put("/{name}")
async def save_secret(
    name: Name,
    body: SecretBody,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    value = body.value.strip()
    if name == "huggingface" and not value.startswith("hf_"):
        raise api_error(400, "invalid_parameter", "Hugging Face tokens start with hf_.")
    row = await db.scalar(
        select(UserSecret).where(UserSecret.user_id == user.id, UserSecret.name == name)
    )
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
