"""Dataset tools: analysis (resolution, aspect, duplicates, captions) and packaging the
training copy (crop applied, originals untouched)."""

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageOps

from app.services import storage

MIN_IMAGES = 10
GOOD_IMAGES = 15
MAX_IMAGES = 200
MIN_EDGE_OK = 768
MIN_EDGE_HARD = 512
PACK_MAX_EDGE = 2048
NEAR_DUPLICATE_BITS = 6


def dhash(data: bytes) -> str:
    """64-bit difference hash (hex) of an image, for near-duplicate detection."""
    with PILImage.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("L").resize((9, 8), PILImage.Resampling.LANCZOS)
        pixels = list(img.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def aspect_bucket(width: int, height: int) -> str:
    ratio = width / height
    for label, value in (
        ("1:1", 1.0),
        ("4:5", 0.8),
        ("3:4", 0.75),
        ("2:3", 0.667),
        ("9:16", 0.5625),
        ("5:4", 1.25),
        ("4:3", 1.333),
        ("3:2", 1.5),
        ("16:9", 1.778),
    ):
        if abs(ratio - value) / value < 0.06:
            return label
    return "other"


def cropped_size(width: int, height: int, crop: dict[str, float] | None) -> tuple[int, int]:
    if not crop:
        return width, height
    return max(1, round(width * crop["w"])), max(1, round(height * crop["h"]))


@dataclass
class ItemView:
    id: str
    width: int
    height: int
    crop: dict[str, float] | None
    caption: str
    excluded: bool
    split: str
    dhash: str | None


def analyze(items: list[ItemView], trigger_word: str) -> dict[str, Any]:
    used = [i for i in items if not i.excluded]
    train = [i for i in used if i.split == "train"]
    sizes = [cropped_size(i.width, i.height, i.crop) for i in train]
    edges = [min(w, h) for w, h in sizes]
    buckets: dict[str, int] = {}
    for w, h in sizes:
        bucket = aspect_bucket(w, h)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    warnings: list[dict[str, Any]] = []
    if len(train) < MIN_IMAGES:
        warnings.append({"code": "too_few", "level": "error", "n": len(train)})
    elif len(train) < GOOD_IMAGES:
        warnings.append({"code": "few", "level": "warning", "n": len(train)})
    if len(train) > MAX_IMAGES:
        warnings.append({"code": "too_many", "level": "warning", "n": len(train)})
    low = [i.id for i, (w, h) in zip(train, sizes, strict=True) if min(w, h) < MIN_EDGE_OK]
    very_low = [i.id for i, (w, h) in zip(train, sizes, strict=True) if min(w, h) < MIN_EDGE_HARD]
    if very_low:
        warnings.append({"code": "very_low_res", "level": "error", "items": very_low})
    elif low:
        warnings.append({"code": "low_res", "level": "warning", "items": low})
    missing = [i.id for i in train if not i.caption.strip()]
    if missing:
        warnings.append({"code": "missing_captions", "level": "warning", "items": missing})
    if trigger_word:
        pattern = re.compile(rf"(?<![\w]){re.escape(trigger_word)}(?![\w])", re.IGNORECASE)
        no_trigger = [i.id for i in train if i.caption.strip() and not pattern.search(i.caption)]
        if no_trigger:
            warnings.append({"code": "missing_trigger", "level": "warning", "items": no_trigger})
    pairs: list[list[str]] = []
    hashed = [i for i in used if i.dhash]
    for a_index, a in enumerate(hashed):
        for b in hashed[a_index + 1 :]:
            if hamming(a.dhash or "0", b.dhash or "0") <= NEAR_DUPLICATE_BITS:
                pairs.append([a.id, b.id])
    if pairs:
        warnings.append({"code": "near_duplicates", "level": "warning", "pairs": pairs[:50]})
    if len(buckets) == 1 and len(train) >= MIN_IMAGES:
        warnings.append({"code": "single_aspect", "level": "info"})
    return {
        "total": len(items),
        "used": len(used),
        "train": len(train),
        "validation": len([i for i in used if i.split == "validation"]),
        "excluded": len(items) - len(used),
        "captioned": len([i for i in train if i.caption.strip()]),
        "min_edge": min(edges) if edges else None,
        "median_edge": int(statistics.median(edges)) if edges else None,
        "aspect_buckets": buckets,
        "warnings": warnings,
        "ready": len(train) >= MIN_IMAGES and not very_low,
    }


def _encode(img: PILImage.Image, mime: str) -> tuple[bytes, str]:
    out = io.BytesIO()
    if mime == "image/jpeg":
        img.convert("RGB").save(out, format="JPEG", quality=95)
        return out.getvalue(), "jpg"
    img.save(out, format="PNG", optimize=True)
    return out.getvalue(), "png"


def pack_image(relative_path: str, mime: str, crop: dict[str, float] | None) -> tuple[bytes, str]:
    """The training copy: crop applied, at most 2048 px per side (the trainer resizes to
    its buckets anyway), JPEG kept as JPEG, PNG/WebP saved as PNG (ai-toolkit reads
    jpg/png). Returns (bytes, extension)."""
    data = storage.absolute(relative_path).read_bytes()
    with PILImage.open(io.BytesIO(data)) as source:
        img = ImageOps.exif_transpose(source)
        changed = img is not source
        if crop:
            w, h = img.size
            box = (
                round(crop["x"] * w),
                round(crop["y"] * h),
                round((crop["x"] + crop["w"]) * w),
                round((crop["y"] + crop["h"]) * h),
            )
            img = img.crop(box)
            changed = True
        if max(img.size) > PACK_MAX_EDGE:
            img.thumbnail((PACK_MAX_EDGE, PACK_MAX_EDGE), PILImage.Resampling.LANCZOS)
            changed = True
        if not changed and mime in ("image/jpeg", "image/png"):
            return data, "jpg" if mime == "image/jpeg" else "png"
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        return _encode(img, mime)


def write_pack(folder: Path, files: list[tuple[str, bytes, str]]) -> None:
    """files: (stem, image bytes, caption)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        old.unlink()
    for stem, data, caption in files:
        (folder / stem).write_bytes(data)
        (folder / (stem.rsplit(".", 1)[0] + ".txt")).write_text(caption, encoding="utf-8")
