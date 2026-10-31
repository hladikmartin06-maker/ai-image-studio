"""Chunked uploads (see services/uploads.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.deps import api_error, current_user
from app.models import User
from app.services import uploads

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


class NewUpload(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=uploads.MAX_SIZE)


@router.post("", status_code=201)
async def start(body: NewUpload, user: User = Depends(current_user)):
    try:
        upload_id = await uploads.create(user.id, body.filename, body.size)
    except uploads.UploadError as exc:
        raise api_error(400, "invalid_upload", str(exc)) from exc
    return {"id": upload_id, "chunk_size": uploads.CHUNK_SIZE}


@router.put("/{upload_id}")
async def chunk(
    upload_id: str,
    request: Request,
    offset: int = Query(ge=0),
    user: User = Depends(current_user),
):
    data = await request.body()
    if len(data) > uploads.CHUNK_SIZE:
        raise api_error(413, "chunk_too_large", "Send at most 8 MB per request.")
    try:
        received = await uploads.append(user.id, upload_id, offset, data)
    except uploads.UploadError as exc:
        raise api_error(409, "upload_conflict", str(exc)) from exc
    return {"received": received}


@router.delete("/{upload_id}")
async def cancel(upload_id: str, user: User = Depends(current_user)):
    try:
        await uploads.discard(user.id, upload_id)
    except uploads.UploadError as exc:
        raise api_error(404, "not_found", str(exc)) from exc
    return {"ok": True}
