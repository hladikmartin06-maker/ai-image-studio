"""The check of a result: per requirement, not one overall score.

- The user's AI provider looks at the result (and the original photo for an edit) and
  answers each yes/no check of the plan — an estimate; "not evaluated" when it declines.
- Face identity: SFace on this Mac (services/faces.py) — the median similarity to the
  original photo (edit) or to the LoRA's training photos (text to image with a LoRA, and an
  edit that turns the person into the LoRA's character: then it is also a check of its own,
  passed at SFace's match threshold, so a face that does not match is corrected).
- Changed area (edit): the share of pixels that differ from the original photo, measured.
Nothing here is a guarantee; the numbers are what was measured."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageChops, ImageOps
from sqlalchemy.ext.asyncio import AsyncSession

from app.advanced.compiler import Spec
from app.advanced.intelligence import IntelligenceError, ask_json
from app.models import User
from app.services import faces

SYSTEM = """You check an image made by an image studio against a list of requirements. \
Look carefully. Reply with one JSON object only:
{"checks": [{"id": str, "passed": true|false|null, "note": str}], "artifacts": str}
"passed" is true when the requirement is clearly met, false when it is clearly not, null
when you cannot tell. "note" says briefly what you see (e.g. "mouth slightly open").
"artifacts" names visible problems (extra fingers, warped face, seams, blur) or "". When
two images are attached, the first is the ORIGINAL photo and the second the RESULT; judge
the RESULT, using the original for "keep"/"preserve" checks."""

# A pixel counts as changed when a channel differs by more than this (out of 255); JPEG and
# resizing alone move pixels a little.
CHANGE_LEVEL = 24
# The measured face check of a request that turns the person into the LoRA's character. It
# passes at this share of how alike the LoRA's own photos are (their baseline), never below
# SFace's match threshold: at the threshold alone a face 41 % alike passed while the photos
# are 64 % alike, and the owner saw a different woman (2026-11-05).
FACE_MATCH = "face_match"
IDENTITY_SHARE = 0.85


# A mouth or tongue the photo does not have: share (%) of the pixels below the lips that turned
# clearly redder (red minus the mean of green and blue up by more than MOUTH_RISE). The owner's
# car photo (2026-11-05, LoRA Taila 2, whose photos include one with the tongue out): faces
# with a new tongue 1.1–2.1 %, faces with the closed mouth kept 0.0 % on six seeds.
MOUTH_RISE = 25
NEW_MOUTH = 0.5


def _landmarks(path: Path) -> tuple[list[float], int, int] | None:
    """YuNet's largest face: x, y, w, h and five points, in the photo's own pixels."""
    engine = faces._engine
    detector = getattr(engine, "detector", None)
    if detector is None:
        return None
    cv2 = engine.cv2
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    oh, ow = image.shape[:2]
    scale = min(1.0, 1600 / max(oh, ow))
    if scale < 1.0:
        image = cv2.resize(image, (int(ow * scale), int(oh * scale)))
    with engine.lock:
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, found = detector.detect(image)
    if found is None or len(found) == 0:
        return None
    face = max(found, key=lambda f: f[2] * f[3])
    return [float(v) / scale for v in face[:14]], ow, oh


def new_mouth(original: Path, result: Path) -> float | None:
    """% of the area below the original's lips that became redder (a tongue or an opened
    mouth the photo does not have); None without a face."""
    found = _landmarks(original)
    if found is None:
        return None
    points, ow, oh = found
    _, _, fw, fh = points[:4]
    cx = (points[10] + points[12]) / 2
    low = max(points[11], points[13])
    with PILImage.open(result) as b:
        size = b.size
        after = np.asarray(ImageOps.exif_transpose(b).convert("RGB")).astype(float)
    with PILImage.open(original) as a:
        before = np.asarray(
            ImageOps.exif_transpose(a).convert("RGB").resize(size, PILImage.Resampling.LANCZOS)
        ).astype(float)
    sx, sy = size[0] / ow, size[1] / oh
    box = (
        max(0, int((cx - 0.4 * fw) * sx)),
        max(0, int((low + 0.02 * fh) * sy)),
        min(size[0], int((cx + 0.4 * fw) * sx)),
        min(size[1], int((low + 0.45 * fh) * sy)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    def red(pixels: np.ndarray) -> np.ndarray:
        part = pixels[box[1] : box[3], box[0] : box[2]]
        return part[..., 0] - (part[..., 1] + part[..., 2]) / 2

    return round(float(((red(after) - red(before)) > MOUTH_RISE).mean() * 100), 2)


def changed_area(original: Path, result: Path) -> float:
    """Percent of the result's pixels that differ from the original (resized to match)."""
    with PILImage.open(original) as a, PILImage.open(result) as b:
        b = ImageOps.exif_transpose(b).convert("RGB")
        a = ImageOps.exif_transpose(a).convert("RGB").resize(b.size, PILImage.Resampling.LANCZOS)
        diff = ImageChops.difference(a, b).convert("L")
        histogram = diff.histogram()
    changed = sum(histogram[CHANGE_LEVEL + 1 :])
    total = b.size[0] * b.size[1]
    return round(100.0 * changed / total, 2) if total else 0.0


async def identity(result: Path, references: list[Path]) -> tuple[float | None, float | None]:
    """Median face similarity to the reference photos (SFace), or None without faces, and how
    alike the reference photos are to each other (None with fewer than two faces)."""
    if not references:
        return None, None
    report = await faces.likeness([(0, result)], references)
    rows = report.get("steps") or []
    return (rows[0]["score"] if rows else None), report.get("baseline")


async def evaluate(
    db: AsyncSession,
    user: User,
    *,
    spec: Spec,
    result: Path,
    original: Path | None,
    identity_refs: list[Path],
    use_ai: bool,
    lora_identity: bool = False,
) -> dict[str, Any]:
    use_lora = lora_identity or original is None
    report: dict[str, Any] = {
        "checks": [
            {
                "id": c.id,
                "question": c.question,
                "kind": c.kind,
                "target": c.target,
                "passed": None,
                "note": "",
            }
            for c in spec.checks
        ],
        "artifacts": "",
        "identity": None,
        "identity_goal": None,
        "identity_reference": ("lora" if identity_refs else None) if use_lora else "original",
        "threshold": faces.MATCH_THRESHOLD,
        "changed_area": None,
        "ai": None,
        "error": None,
    }
    if original is not None:
        report["changed_area"] = await asyncio.to_thread(changed_area, original, result)
    refs = identity_refs if use_lora else [original]
    baseline = None
    try:
        report["identity"], baseline = await identity(result, refs)
    except faces.FaceError as exc:
        report["error"] = exc.message
    if use_ai and spec.checks:
        lines = [f"- {c.id} [{c.kind}]: {c.question}" for c in spec.checks]
        images = [await asyncio.to_thread(original.read_bytes)] if original is not None else []
        images.append(await asyncio.to_thread(result.read_bytes))
        try:
            data, meta = await ask_json(
                db,
                user,
                system=SYSTEM,
                text="REQUIREMENTS:\n" + "\n".join(lines),
                images=images,
            )
            answers = {str(a.get("id")): a for a in data.get("checks") or [] if isinstance(a, dict)}
            for check in report["checks"]:
                answer = answers.get(check["id"]) or {}
                passed = answer.get("passed")
                check["passed"] = passed if isinstance(passed, bool) else None
                check["note"] = str(answer.get("note") or "")[:200]
            report["artifacts"] = str(data.get("artifacts") or "")[:300]
            report["ai"] = meta
        except IntelligenceError as exc:
            report["error"] = exc.message
    if lora_identity and identity_refs:
        score = report["identity"]
        goal = round(max(faces.MATCH_THRESHOLD, IDENTITY_SHARE * (baseline or 0.0)), 3)
        report["identity_goal"] = goal
        report["checks"].append(
            {
                "id": FACE_MATCH,
                "question": "Does the face match the LoRA's photos? (measured)",
                "kind": "modify",
                "target": "face",
                "passed": None if score is None else score >= goal,
                "note": "" if score is None else f"{round(score * 100)} % / {round(goal * 100)} %",
                "measured": True,
            }
        )
    return report


def failed(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in report.get("checks") or [] if c.get("passed") is False]


def rank(report: dict[str, Any]) -> tuple[int, int, float]:
    """For choosing the best attempt: fewer failed checks, more passed, then identity."""
    checks = report.get("checks") or []
    return (
        -sum(1 for c in checks if c.get("passed") is False),
        sum(1 for c in checks if c.get("passed") is True),
        report.get("identity") or 0.0,
    )
