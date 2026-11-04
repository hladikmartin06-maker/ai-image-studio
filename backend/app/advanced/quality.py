"""The check of a result: per requirement, not one overall score.

- The user's AI provider looks at the result (and the original photo for an edit) and
  answers each yes/no check of the plan — an estimate; "not evaluated" when it declines.
- Face identity: SFace on this Mac (services/faces.py) — the median similarity to the
  original photo (edit) or to the LoRA's training photos (text to image with a LoRA).
- Changed area (edit): the share of pixels that differ from the original photo, measured.
Nothing here is a guarantee; the numbers are what was measured."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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


async def identity(result: Path, references: list[Path]) -> float | None:
    """Median face similarity to the reference photos (SFace), or None without faces."""
    if not references:
        return None
    report = await faces.likeness([(0, result)], references)
    rows = report.get("steps") or []
    return rows[0]["score"] if rows else None


async def evaluate(
    db: AsyncSession,
    user: User,
    *,
    spec: Spec,
    result: Path,
    original: Path | None,
    identity_refs: list[Path],
    use_ai: bool,
) -> dict[str, Any]:
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
        "identity_reference": "original"
        if original is not None
        else ("lora" if identity_refs else None),
        "threshold": faces.MATCH_THRESHOLD,
        "changed_area": None,
        "ai": None,
        "error": None,
    }
    if original is not None:
        report["changed_area"] = await asyncio.to_thread(changed_area, original, result)
    refs = [original] if original is not None else identity_refs
    try:
        report["identity"] = await identity(result, refs)
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
