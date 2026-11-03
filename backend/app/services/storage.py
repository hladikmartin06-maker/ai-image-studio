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


# --- videos (Wan 3.0) ------------------------------------------------------------------------


def video_kind(data: bytes) -> tuple[str, str] | None:
    """(mime, extension) of an MP4/MOV or WebM file, else None."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4", "mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm", "webm"
    return None


def mp4_size(data: bytes) -> tuple[int, int] | None:
    """Width and height of the first video track of an MP4 (its `tkhd` box), else None."""

    def boxes(start: int, end: int):
        pos = start
        while pos + 8 <= end:
            size = int.from_bytes(data[pos : pos + 4], "big")
            kind = data[pos + 4 : pos + 8]
            header = 8
            if size == 1 and pos + 16 <= end:
                size = int.from_bytes(data[pos + 8 : pos + 16], "big")
                header = 16
            elif size == 0:
                size = end - pos
            if size < header or pos + size > end:
                return
            yield kind, pos + header, pos + size
            pos += size

    def search(start: int, end: int) -> tuple[int, int] | None:
        for kind, body, stop in boxes(start, end):
            if kind == b"tkhd" and stop - body >= 8:
                width = int.from_bytes(data[stop - 8 : stop - 4], "big") >> 16
                height = int.from_bytes(data[stop - 4 : stop], "big") >> 16
                if width and height:
                    return width, height
            elif kind in (b"moov", b"trak"):
                found = search(body, stop)
                if found:
                    return found
        return None

    return search(0, len(data))


def expected_size(settings: dict) -> tuple[int, int]:
    """The size a video was asked for (resolution + aspect ratio), when the file says none."""
    short = {"480p": 480, "720p": 720, "1080p": 1080}.get(str(settings.get("resolution")), 720)
    try:
        w, h = (int(x) for x in str(settings.get("aspect_ratio") or "16:9").split(":"))
    except ValueError:
        w, h = 16, 9
    if w >= h:
        return round(short * w / h / 2) * 2, short
    return short, round(short * h / w / 2) * 2


def _store_video(
    user_id: uuid.UUID, file_id: uuid.UUID, data: bytes, fallback: tuple[int, int]
) -> StoredFile:
    kind = video_kind(data)
    if kind is None:
        raise InvalidImage("The provider returned no video file.")
    mime, ext = kind
    width, height = (mp4_size(data) if ext == "mp4" else None) or fallback
    file_path = _write(user_id, "images", f"{file_id}.{ext}", data)
    # No picture is made from a video: the browser shows its first frame.
    return StoredFile(file_path, file_path, mime, width, height, len(data))


async def store_video(
    user_id: uuid.UUID, file_id: uuid.UUID, data: bytes, fallback: tuple[int, int]
) -> StoredFile:
    return await asyncio.to_thread(_store_video, user_id, file_id, data, fallback)


def is_video(mime_type: str | None) -> bool:
    return bool(mime_type and mime_type.startswith("video/"))


# --- export for social media -----------------------------------------------------------------

# Format -> (width, height) of the exported picture; "original" keeps the shape.
SOCIAL_FORMATS: dict[str, tuple[int, int] | None] = {
    "story": (1080, 1920),  # 9:16 — Stories, Reels, TikTok
    "portrait": (1080, 1350),  # 4:5 — feed portrait
    "square": (1080, 1080),  # 1:1
    "original": None,
}


def social_export(data: bytes, fmt: str, watermark: str = "") -> bytes:
    """A JPEG for posting: cropped from the centre to the format and scaled, without any
    metadata (EXIF, generation data), with an optional text watermark at the bottom right."""
    from PIL import ImageDraw, ImageFont

    with PILImage.open(io.BytesIO(data)) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
    target = SOCIAL_FORMATS[fmt]
    if target is not None:
        img = ImageOps.fit(img, target, method=PILImage.Resampling.LANCZOS)
    elif max(img.size) > 4096:
        img.thumbnail((4096, 4096), PILImage.Resampling.LANCZOS)
    text = watermark.strip()
    if text:
        draw = ImageDraw.Draw(img)
        size = max(14, img.width // 36)
        font = ImageFont.load_default(size=size)
        box = draw.textbbox((0, 0), text, font=font)
        margin = size
        x = img.width - (box[2] - box[0]) - margin
        y = img.height - (box[3] - box[1]) - margin * 1.4
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
    out = io.BytesIO()
    # A new file: no EXIF, no XMP, no PNG text chunks of the original.
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


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
