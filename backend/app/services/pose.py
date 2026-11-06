"""Body pose on this Mac (the owner's plan, phase 2, approved 2026-11-06): DWPose — a person
box from YOLOX-L, then 133 whole-body keypoints (COCO-WholeBody: 17 body, 6 feet, 68 face,
2 × 21 hands) — run with OpenCV's DNN (already used for the faces; no new package), and an
OpenPose-style skeleton map like the one the Krea 2 pose LoRA was trained with (black
background, coloured limbs, hands, face points). Measured 0.4 s per photo on the owner's Mac.

Files, pinned like the face models (huggingface.co/yzd-v/DWPose, commit 1a714410, Apache
2.0), downloaded once into storage/models/pose and checked by SHA-256:
  - yolox_l.onnx — 216,746,733 bytes
  - dw-ll_ucoco_384.onnx — 134,399,116 bytes (input 288 × 384, SimCC output ×2)
Pre- and post-processing as in DWPose's own ONNX code (letterbox 640 with 114, the person
class, NMS; an affine crop with 1.25 padding, ImageNet normalisation, SimCC argmax / 2).
Nothing leaves the Mac.
"""

from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.faces import FaceError, FaceModel, _download, _ok

HF = "https://huggingface.co/yzd-v/DWPose/resolve/1a7144101628d69ee7a3768d1ee3a094070dc388"
DETECTOR = FaceModel(
    "yolox_l.onnx",
    216_746_733,
    "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    (f"{HF}/yolox_l.onnx",),
)
ESTIMATOR = FaceModel(
    "dw-ll_ucoco_384.onnx",
    134_399_116,
    "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
    (f"{HF}/dw-ll_ucoco_384.onnx",),
)
THRESHOLD = 0.3  # a keypoint below this confidence is not drawn (DWPose's own value)


class PoseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class Person:
    points: Any  # (133, 2) pixels of the photo
    conf: Any  # (133,)
    box: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return (x1 - x0) * (y1 - y0)


def models_dir() -> Path:
    path = get_settings().storage_dir / "models" / "pose"
    path.mkdir(parents=True, exist_ok=True)
    return path


_lock = asyncio.Lock()


async def ensure_models() -> tuple[Path, Path]:
    async with _lock:
        paths = []
        for model in (DETECTOR, ESTIMATOR):
            path = models_dir() / model.filename
            if not await asyncio.to_thread(_ok, path, model):
                try:
                    await _download(model, path, "The pose model")
                except FaceError as exc:
                    raise PoseError(exc.message) from exc
            paths.append(path)
        return paths[0], paths[1]


class _Engine:
    def __init__(self, detector: Path, estimator: Path):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.dnn.readNetFromONNX(str(detector))
        self.estimator = cv2.dnn.readNetFromONNX(str(estimator))
        self.lock = threading.Lock()

    def boxes(self, img: Any) -> list[Any]:
        import numpy as np

        cv2 = self.cv2
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        padded = np.full((640, 640, 3), 114, np.uint8)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.detector.setInput(padded.transpose(2, 0, 1)[None].astype(np.float32))
        out = self.detector.forward()[0]
        grids, strides = [], []
        for s in (8, 16, 32):
            xv, yv = np.meshgrid(np.arange(640 // s), np.arange(640 // s))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full(((640 // s) ** 2, 1), s))
        grid, stride = np.concatenate(grids), np.concatenate(strides)
        out[:, :2] = (out[:, :2] + grid) * stride
        out[:, 2:4] = np.exp(out[:, 2:4]) * stride
        boxes = (
            np.stack(
                [
                    out[:, 0] - out[:, 2] / 2,
                    out[:, 1] - out[:, 3] / 2,
                    out[:, 0] + out[:, 2] / 2,
                    out[:, 1] + out[:, 3] / 2,
                ],
                1,
            )
            / r
        )
        scores = out[:, 4] * out[:, 5]  # objectness × the person class
        keep = scores > THRESHOLD
        boxes, scores = boxes[keep], scores[keep]
        rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        picked = cv2.dnn.NMSBoxes(rects, [float(x) for x in scores], THRESHOLD, 0.45)
        return [boxes[i] for i in np.array(picked).flatten()]

    def keypoints(self, img: Any, box: Any) -> tuple[Any, Any]:
        import numpy as np

        cv2 = self.cv2
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        if bw > bh * 288 / 384:
            bh = bw * 384 / 288
        else:
            bw = bh * 288 / 384
        src = np.float32(
            [[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx - bw / 2, cy + bh / 2]]
        )
        warp = cv2.getAffineTransform(src, np.float32([[0, 0], [288, 0], [0, 384]]))
        crop = cv2.warpAffine(img, warp, (288, 384), flags=cv2.INTER_LINEAR)
        rgb = (crop[:, :, ::-1].astype(np.float32) - [123.675, 116.28, 103.53]) / [
            58.395,
            57.12,
            57.375,
        ]
        self.estimator.setInput(rgb.transpose(2, 0, 1)[None].astype(np.float32))
        names = self.estimator.getUnconnectedOutLayersNames()
        outs = [o[0] for o in self.estimator.forward(names)]
        sx, sy = sorted(outs, key=lambda o: o.shape[-1])  # simcc_x (…, 576), simcc_y (…, 768)
        pts = np.stack([sx.argmax(1) / 2.0, sy.argmax(1) / 2.0, np.ones(sx.shape[0])], 1)
        conf = np.minimum(sx.max(1), sy.max(1))
        return pts @ cv2.invertAffineTransform(warp).T, conf

    def people(self, data: bytes) -> list[Person]:
        import numpy as np

        img = np.ascontiguousarray(np.asarray(upright(data))[:, :, ::-1])  # RGB → BGR
        with self.lock:
            found = [(b, *self.keypoints(img, b)) for b in self.boxes(img)]
        people = [Person(p, c, tuple(float(v) for v in b)) for b, p, c in found]
        return sorted(people, key=lambda p: p.area, reverse=True)


_engine: _Engine | None = None


async def people(path: Path) -> list[Person]:
    """The people in a photo, the largest first (their 133 keypoints in its pixels)."""
    global _engine
    if _engine is None:
        detector, estimator = await ensure_models()
        _engine = await asyncio.to_thread(_Engine, detector, estimator)
    data = await asyncio.to_thread(path.read_bytes)
    return await asyncio.to_thread(_engine.people, data)


def upright(data: bytes) -> Any:
    """The photo as ComfyUI's LoadImage sees it (turned by its EXIF orientation), in RGB."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PoseError("The photo could not be read.") from exc


def size(path: Path) -> tuple[int, int]:
    return upright(path.read_bytes()).size


async def main_person(path: Path, what: str = "the pose photo") -> Person:
    found = await people(path)
    if not found:
        raise PoseError(f"No person was found in {what}.")
    return found[0]


# --- the skeleton map (OpenPose colours, RGB) -----------------------------------------------

# OpenPose's 18 body points from COCO's 17 (index 1, the neck, is between the shoulders).
COCO_TO_OPENPOSE = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
LIMBS = [
    (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12),
    (12, 13), (13, 14), (2, 1), (1, 15), (15, 17), (1, 16), (16, 18),
]  # fmt: skip
COLOURS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]  # fmt: skip
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
    (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19),
    (19, 20),
]  # fmt: skip
# The head and neck (OpenPose points): nose, neck, eyes, ears — the limbs between them.
HEAD_POINTS = {0, 1, 14, 15, 16, 17}


def _openpose(person: Person) -> list[tuple[Any, float]]:
    body, conf = person.points[:17], person.conf[:17]
    neck = (body[5] + body[6]) / 2
    return [
        (neck, float(min(conf[5], conf[6]))) if i is None else (body[i], float(conf[i]))
        for i in COCO_TO_OPENPOSE
    ]


def render(person: Person, width: int, height: int, part: str = "body") -> bytes:
    """The skeleton as a PNG on black at width × height (person in those pixels): the whole
    body with hands and face, or only the head and neck (part="head")."""
    import cv2
    import numpy as np

    canvas = np.zeros((height, width, 3), np.uint8)
    stick = max(2, round(min(width, height) / 256))
    points = _openpose(person)
    for k, (a, b) in enumerate(LIMBS):
        if part == "head" and not ({a - 1, b - 1} <= HEAD_POINTS):
            continue
        (pa, ca), (pb, cb) = points[a - 1], points[b - 1]
        if ca < THRESHOLD or cb < THRESHOLD:
            continue
        mx, my = (pa + pb) / 2
        length = float(np.hypot(*(pa - pb)))
        angle = float(np.degrees(np.arctan2(pa[1] - pb[1], pa[0] - pb[0])))
        poly = cv2.ellipse2Poly(
            (int(mx), int(my)), (int(length / 2), stick * 2), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLOURS[k]])
    for k, (p, c) in enumerate(points):
        if c >= THRESHOLD and (part != "head" or k in HEAD_POINTS):
            cv2.circle(canvas, (int(p[0]), int(p[1])), stick * 2, COLOURS[k], -1)
    if part != "head":
        for start in (91, 112):
            hand, hc = person.points[start : start + 21], person.conf[start : start + 21]
            for i, j in HAND_EDGES:
                if hc[i] >= THRESHOLD and hc[j] >= THRESHOLD:
                    cv2.line(
                        canvas,
                        (int(hand[i][0]), int(hand[i][1])),
                        (int(hand[j][0]), int(hand[j][1])),
                        (0, 0, 255),
                        max(1, stick // 2),
                    )
    for p, c in zip(person.points[23:91], person.conf[23:91], strict=True):
        if c >= THRESHOLD:
            cv2.circle(canvas, (int(p[0]), int(p[1])), max(1, stick // 2), (255, 255, 255), -1)
    ok, png = cv2.imencode(".png", canvas[:, :, ::-1])  # drawn in RGB, saved as BGR
    if not ok:
        raise PoseError("The skeleton could not be drawn.")
    return png.tobytes()


def scaled(person: Person, factor_x: float, factor_y: float) -> Person:
    """The same pose in a photo resized by these factors."""
    import numpy as np

    pts = person.points * np.array([factor_x, factor_y])
    x0, y0, x1, y1 = person.box
    box = (x0 * factor_x, y0 * factor_y, x1 * factor_x, y1 * factor_y)
    return Person(pts, person.conf, box)


def _anchor(person: Person) -> tuple[Any, float] | None:
    """The neck and the head's size (the larger of the eye–ear span and the shoulder half
    width), or None when they are not visible."""
    import numpy as np

    points = _openpose(person)
    neck, neck_conf = points[1]
    if neck_conf < THRESHOLD:
        return None
    head = [p for k, (p, c) in enumerate(points) if k in (0, 14, 15, 16, 17) and c >= THRESHOLD]
    if not head:
        return None
    spread = float(max(np.ptp(np.array(head)[:, 0]), np.ptp(np.array(head)[:, 1]), 1.0))
    shoulders = float(np.hypot(*(person.points[5] - person.points[6]))) / 2
    return neck, max(spread, shoulders * 0.6)


def head_onto(reference: Person, target: Person) -> Person:
    """The reference's head pose moved and scaled onto the target's neck (the head's size
    from its eyes, ears and shoulders), so it can be drawn on the target's photo."""
    a, b = _anchor(reference), _anchor(target)
    if a is None or b is None:
        raise PoseError("The head and neck are not visible enough in one of the photos.")
    (neck_r, size_r), (neck_t, size_t) = a, b
    k = size_t / size_r
    pts = (reference.points - neck_r) * k + neck_t
    return Person(pts, reference.conf, target.box)


def head_square(person: Person, width: int, height: int, factor: float) -> tuple[int, int, int]:
    """(left, top, side) of a square around the head with its neck, `factor` times the head
    and neck's size, inside the photo."""
    import numpy as np

    points = _openpose(person)
    found = [p for k, (p, c) in enumerate(points) if k in HEAD_POINTS and c >= THRESHOLD]
    face = person.points[23:91][person.conf[23:91] >= THRESHOLD]
    pts = np.concatenate([np.array(found).reshape(-1, 2), face.reshape(-1, 2)])
    if not len(pts):
        raise PoseError("No head was found in the photo.")
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    side = int(min(max(x1 - x0, y1 - y0, 64) * factor, width, height))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, side


def sheet(photo: bytes, skeleton: bytes) -> bytes:
    """The photo and its skeleton side by side (one file for one ComfyUI input)."""
    from PIL import Image

    a = upright(photo)
    b = Image.open(io.BytesIO(skeleton)).convert("RGB").resize(a.size)
    out = Image.new("RGB", (a.width * 2, a.height))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
