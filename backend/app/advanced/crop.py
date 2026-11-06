"""Face edits at a usable size ("crop and stitch").

A mouth or the eyes are small in a whole photo: Krea 2 would redraw them on a handful of
latent pixels (1/8 of the image), which is why closed lips, teeth, the tongue and the
expression come out badly. For face regions the engine finds the face on this Mac (OpenCV
YuNet, the face-match detector, never sent anywhere), crops a square around it with some
context, redraws that crop at 1024 px and puts it back — only inside the mask, so every
other pixel, the head's pose and the framing stay the photo's own."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.services import faces

log = logging.getLogger(__name__)

FACE_TARGETS = ("mouth", "eyes", "face", "expression")
CROP_SIZE = 1024  # the crop is redrawn at this size (Krea 2's ~1 MP)
CONTEXT = 1.8  # crop side = the face box's longer side × this (forehead, hair line, chin)
MIN_SIDE = 320  # px at the edit size: at most ~3.2× enlarged, so a tiny face keeps its context
MIN_GAIN = 1.3  # crop only when the face gets at least this much bigger


async def face_box(path: Path) -> tuple[float, float, float, float, int, int] | None:
    """The main face (`faces.main_face`: the largest of the sure ones) as x, y, w, h in the
    photo's own pixels, and the photo's size; None without a face or the face models."""
    try:
        engine = await faces._get_engine()
    except Exception:  # noqa: BLE001 - no crop is fine; the edit runs on the whole photo
        log.info("Face detector unavailable; face edit without crop")
        return None
    detector = getattr(engine, "detector", None)
    if detector is None:
        return None

    def detect() -> tuple[float, float, float, float, int, int] | None:
        import numpy as np

        cv2 = engine.cv2
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        oh, ow = image.shape[:2]
        scale = min(1.0, 1600 / max(oh, ow))
        if scale < 1.0:
            image = cv2.resize(image, (int(ow * scale), int(oh * scale)))
        h, w = image.shape[:2]
        with engine.lock:
            detector.setInputSize((w, h))
            _, found = detector.detect(image)
        if found is None or len(found) == 0:
            return None
        face = faces.main_face(found)
        x, y, fw, fh = (float(v) / scale for v in face[:4])
        return x, y, fw, fh, ow, oh

    return await asyncio.to_thread(detect)


def plan_crop(
    box: tuple[float, float, float, float, int, int] | None, width: int, height: int
) -> tuple[int, int, int] | None:
    """(left, top, side) of the square to redraw, at the edit size (width × height); None =
    the face is already big enough (a close-up) or there is no face."""
    if box is None:
        return None
    x, y, fw, fh, ow, oh = box
    sx, sy = width / ow, height / oh
    cx, cy = (x + fw / 2) * sx, (y + fh / 2) * sy
    side = max(MIN_SIDE, int(max(fw * sx, fh * sy) * CONTEXT))
    side = min(side, width, height)
    if side * MIN_GAIN > CROP_SIZE:
        return None
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, side
