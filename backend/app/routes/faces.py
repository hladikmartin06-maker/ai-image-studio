"""Face similarity of generated images to an AI model's photos or a LoRA's photos."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import api_error, current_user
from app.models import Image, Lora, LoraDatasetItem, Persona, PersonaImage, ReferenceImage, User
from app.services import faces, storage

router = APIRouter(prefix="/api/faces", tags=["faces"])

MAX_REFERENCES = 24


class CompareBody(BaseModel):
    image_ids: list[uuid.UUID] = Field(min_length=1, max_length=60)
    persona_id: uuid.UUID | None = None
    lora_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _one_reference(self) -> CompareBody:
        if (self.persona_id is None) == (self.lora_id is None):
            raise ValueError("Compare with one AI model or one LoRA")
        return self


@router.post("/compare")
async def compare(
    body: CompareBody, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)
):
    """The face in each image against the AI model's photos (or the LoRA's training photos):
    cosine similarity of the face features; 0.363 or more = the same person."""
    if body.persona_id is not None:
        owner = await db.scalar(
            select(Persona).where(Persona.id == body.persona_id, Persona.user_id == user.id)
        )
        if owner is None:
            raise api_error(404, "not_found", "That AI model no longer exists.")
        ref_rows = (
            await db.scalars(
                select(ReferenceImage)
                .join(PersonaImage, PersonaImage.reference_image_id == ReferenceImage.id)
                .where(PersonaImage.persona_id == owner.id, PersonaImage.user_id == user.id)
                .order_by(PersonaImage.position)
                .limit(MAX_REFERENCES)
            )
        ).all()
        reference = {"kind": "persona", "name": owner.name}
    else:
        lora = await db.scalar(select(Lora).where(Lora.id == body.lora_id, Lora.user_id == user.id))
        if lora is None:
            raise api_error(404, "not_found", "That LoRA no longer exists.")
        ref_rows = (
            await db.scalars(
                select(ReferenceImage)
                .join(LoraDatasetItem, LoraDatasetItem.reference_image_id == ReferenceImage.id)
                .where(
                    LoraDatasetItem.lora_id == lora.id,
                    LoraDatasetItem.user_id == user.id,
                    LoraDatasetItem.excluded.is_(False),
                )
                .order_by(LoraDatasetItem.position)
                .limit(MAX_REFERENCES)
            )
        ).all()
        reference = {"kind": "lora", "name": lora.name}
    if not ref_rows:
        raise api_error(409, "no_photos", f"{reference['name']} has no photos to compare with.")
    images = (
        await db.scalars(
            select(Image).where(Image.user_id == user.id, Image.id.in_(body.image_ids))
        )
    ).all()
    by_id = {img.id: img for img in images if not storage.is_video(img.mime_type)}
    wanted = [i for i in dict.fromkeys(body.image_ids) if i in by_id]
    if not wanted:
        raise api_error(404, "not_found", "No images to compare (videos are not compared).")
    try:
        result = await faces.compare(
            [storage.absolute(by_id[i].file_path) for i in wanted],
            [storage.absolute(r.file_path) for r in ref_rows],
        )
    except faces.FaceError as exc:
        raise api_error(502, "face_model", exc.message) from exc
    return {
        "reference": {**reference, "photos": len(ref_rows), "faces": result["reference_faces"]},
        "threshold": faces.MATCH_THRESHOLD,
        "items": [{"image_id": str(i), **r} for i, r in zip(wanted, result["items"], strict=True)],
    }
