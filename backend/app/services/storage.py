"""Files on disk: generated images, their thumbnails and reference uploads.

Layout: <storage_dir>/<user id>/{images,thumbs,refs}/<file>. The database keeps only
paths relative to storage_dir. Originals are stored byte for byte as the provider
returned them, so downloads keep the full quality and resolution.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from app.config import get_settings

PILImage.MAX_IMAGE_PIXELS = 12_000 * 12_000  # refuse decompression bombs

FORMATS = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
}
THUMB_EDGE = 640


class InvalidImage(ValueError):
    pass


@dataclass
class StoredFile:
    file_path: str
    thumb_path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int


def root() -> Path:
    path = get_settings().storage_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def absolute(relative: str) -> Path:
    """Resolves a stored relative path and refuses anything outside the storage folder."""
    base = root().resolve()
    path = (base / relative).resolve()
    if base not in path.parents:
        raise FileNotFoundError(relative)
    return path


def inspect(data: bytes) -> tuple[str, str, int, int]:
    """(mime, extension, width, height) of a PNG/JPEG/WebP image, or InvalidImage."""
    try:
        with PILImage.open(io.BytesIO(data)) as probe:
            probe.verify()
        with PILImage.open(io.BytesIO(data)) as img:
            fmt = img.format or ""
            width, height = img.size
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise InvalidImage("The file is not a valid image.") from exc
    if fmt not in FORMATS:
        raise InvalidImage("Only PNG, JPEG and WebP images are supported.")
    mime, ext = FORMATS[fmt]
    return mime, ext, width, height


def _thumbnail(data: bytes) -> bytes:
    with PILImage.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        img.thumbnail((THUMB_EDGE, THUMB_EDGE), PILImage.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=82, method=4)
        return out.getvalue()


def _write(user_id: uuid.UUID, folder: str, name: str, data: bytes) -> str:
    relative = f"{user_id}/{folder}/{name}"
    path = root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return relative


def _store(user_id: uuid.UUID, file_id: uuid.UUID, data: bytes, folder: str) -> StoredFile:
    mime, ext, width, height = inspect(data)
    thumb = _thumbnail(data)
    file_path = _write(user_id, folder, f"{file_id}.{ext}", data)
    thumb_path = _write(user_id, "thumbs", f"{file_id}.webp", thumb)
    return StoredFile(file_path, thumb_path, mime, width, height, len(data))


async def store_image(user_id: uuid.UUID, file_id: uuid.UUID, data: bytes) -> StoredFile:
    return await asyncio.to_thread(_store, user_id, file_id, data, "images")


async def store_reference(user_id: uuid.UUID, file_id: uuid.UUID, data: bytes) -> StoredFile:
    return await asyncio.to_thread(_store, user_id, file_id, data, "refs")


def fit_for_reference(data: bytes, max_bytes: int, max_edge: int) -> bytes:
    """Keeps an image as it is when it fits the reference limits; otherwise re-encodes it
    as a high-quality JPEG, scaled down only as far as needed."""
    with PILImage.open(io.BytesIO(data)) as probe:
        size = probe.size
    if len(data) <= max_bytes and max(size) <= max_edge:
        return data
    with PILImage.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        edge = min(max(img.size), max_edge)
        while True:
            copy = img.copy()
            copy.thumbnail((edge, edge), PILImage.Resampling.LANCZOS)
            out = io.BytesIO()
            copy.save(out, format="JPEG", quality=92, optimize=True)
            if out.tell() <= max_bytes or edge <= 1024:
                return out.getvalue()
            edge = int(edge * 0.8)


def delete_files(*relative_paths: str) -> None:
    for relative in relative_paths:
        with contextlib.suppress(FileNotFoundError):
            absolute(relative).unlink(missing_ok=True)


def user_usage(user_id: uuid.UUID) -> dict[str, int]:
    """Bytes on disk per folder for one user."""
    usage = {"images": 0, "thumbs": 0, "refs": 0}
    base = root() / str(user_id)
    for folder in usage:
        directory = base / folder
        if directory.is_dir():
            usage[folder] = sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
    return usage
