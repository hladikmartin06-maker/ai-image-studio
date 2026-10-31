"""Reference images: validation, de-duplication, provider upload reuse and clean-up.

Every reference is a file under storage/<user>/refs plus a ReferenceImage row. The same
file uploaded again by the same user reuses the existing row (SHA-256), so it is stored
once and uploaded to each provider once. AI models (personas) link reference rows instead
of copying files.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Generation, PersonaImage, ReferenceImage
from app.services import storage

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
MIN_EDGE = 64
MAX_EDGE = 8192
# How long a provider keeps an uploaded file reachable is not documented by WaveSpeed or
# Atlas Cloud. An upload is reused for this long, then the file is uploaded again, so a
# saved AI model never points the provider at an expired URL.
UPLOAD_REUSE = timedelta(hours=24)


class ReferenceRejected(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def max_bytes() -> int:
    return get_settings().max_upload_mb * 1024 * 1024


def validate(data: bytes) -> None:
    if len(data) > max_bytes():
        raise ReferenceRejected(
            413, "file_too_large", f"Images up to {get_settings().max_upload_mb} MB."
        )
    if not data:
        raise ReferenceRejected(400, "empty_file", "The file is empty.")
    try:
        mime, _, width, height = storage.inspect(data)
    except storage.InvalidImage as exc:
        raise ReferenceRejected(415, "invalid_image", str(exc)) from exc
    if mime not in ALLOWED_MIME:
        raise ReferenceRejected(
            415, "invalid_image", "Only PNG, JPEG and WebP images are supported."
        )
    if min(width, height) < MIN_EDGE or max(width, height) > MAX_EDGE:
        raise ReferenceRejected(
            400,
            "invalid_dimensions",
            f"Images must be {MIN_EDGE} to {MAX_EDGE} pixels per side.",
        )


async def store(
    db: AsyncSession, user_id: uuid.UUID, data: bytes, original_name: str = ""
) -> tuple[ReferenceImage, bool]:
    """Validates and stores an image; returns (row, created). Adds the row to the session
    without committing. An identical file of the same user returns the existing row."""
    validate(data)
    digest = hashlib.sha256(data).hexdigest()
    existing = await db.scalar(
        select(ReferenceImage)
        .where(ReferenceImage.user_id == user_id, ReferenceImage.sha256 == digest)
        .limit(1)
    )
    if existing is not None:
        return existing, False
    ref_id = uuid.uuid4()
    stored = await storage.store_reference(user_id, ref_id, data)
    ref = ReferenceImage(
        id=ref_id,
        user_id=user_id,
        file_path=stored.file_path,
        thumb_path=stored.thumb_path,
        original_name=original_name[:255],
        mime_type=stored.mime_type,
        width=stored.width,
        height=stored.height,
        size_bytes=stored.size_bytes,
        provider_urls={},
        provider_uploaded_at={},
        sha256=digest,
    )
    db.add(ref)
    await db.flush()
    return ref, True


async def from_library_file(
    db: AsyncSession, user_id: uuid.UUID, relative_path: str, name: str
) -> tuple[ReferenceImage, bool]:
    """A reference made from a generated image (copied, so deleting either keeps the other)."""
    data = await asyncio.to_thread(storage.absolute(relative_path).read_bytes)
    data = await asyncio.to_thread(storage.fit_for_reference, data, max_bytes(), MAX_EDGE)
    return await store(db, user_id, data, name)


def reusable_url(ref: ReferenceImage, provider: str, now: datetime | None = None) -> str | None:
    url = (ref.provider_urls or {}).get(provider)
    when = (ref.provider_uploaded_at or {}).get(provider)
    if not url or not when:
        return None
    try:
        uploaded = datetime.fromisoformat(when)
    except ValueError:
        return None
    return url if (now or datetime.now(UTC)) - uploaded < UPLOAD_REUSE else None


def remember_upload(ref: ReferenceImage, provider: str, url: str) -> None:
    ref.provider_urls = {**(ref.provider_urls or {}), provider: url}
    ref.provider_uploaded_at = {
        **(ref.provider_uploaded_at or {}),
        provider: datetime.now(UTC).isoformat(),
    }


async def delete_unused(
    db: AsyncSession, user_id: uuid.UUID, ref_ids: list[uuid.UUID]
) -> list[str]:
    """Deletes the given references unless an AI model still links them or a running
    generation needs them. Returns the file paths to remove after the commit."""
    if not ref_ids:
        return []
    active = (
        await db.scalars(
            select(Generation.reference_image_ids).where(
                Generation.user_id == user_id,
                Generation.status.in_(("queued", "processing")),
            )
        )
    ).all()
    busy = {ref for ids in active for ref in ids}
    rows = (
        await db.scalars(
            select(ReferenceImage).where(
                ReferenceImage.user_id == user_id,
                ReferenceImage.id.in_(ref_ids),
                ~exists().where(PersonaImage.reference_image_id == ReferenceImage.id),
            )
        )
    ).all()
    paths: list[str] = []
    for ref in rows:
        if str(ref.id) in busy:
            continue
        paths.extend((ref.file_path, ref.thumb_path))
        await db.delete(ref)
    return paths
