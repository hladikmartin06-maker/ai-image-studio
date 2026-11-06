"""Face similarity: how close the face in an image is to an AI model's or a LoRA's photos.

Runs on this Mac (CPU) with OpenCV's built-in face detector (YuNet) and face recognizer
(SFace), the models of the OpenCV Model Zoo:
  - face_detection_yunet_2023mar.onnx (MIT) — huggingface.co/opencv/face_detection_yunet
  - face_recognition_sface_2021dec.onnx (Apache-2.0) — huggingface.co/opencv/face_recognition_sface
Both are downloaded once into storage/models/face (pinned commit, SHA-256 checked; GitHub's
copy of the OpenCV Model Zoo is the second source). The score is the cosine similarity of
SFace's face features; 0.363 or more = the same person (the threshold in the zoo's sface.py).
Nothing leaves the Mac.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import statistics
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

log = logging.getLogger("studio.faces")

MATCH_THRESHOLD = 0.363
# Smaller faces (e.g. a full-body shot) are not compared: SFace aligns every face to 112 px,
# and a face of a few dozen pixels gives an unreliable score.
MIN_FACE_PX = 64
# The face measured is the largest of those YuNet is sure of (at least this share of its best
# score): in the owner's lift photo (2026-11-06) the cut-off profile at the edge (0.76) was
# larger than the frontal face in the mirror (0.93), the photo's subject.
SURE_SHARE = 0.9
HF = "https://huggingface.co/opencv"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class FaceModel:
    filename: str
    size: int
    sha256: str
    urls: tuple[str, ...]


DETECTOR = FaceModel(
    "face_detection_yunet_2023mar.onnx",
    232_589,
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    (
        f"{HF}/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/"
        "face_detection_yunet_2023mar.onnx",
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
)
RECOGNIZER = FaceModel(
    "face_recognition_sface_2021dec.onnx",
    38_696_353,
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    (
        f"{HF}/face_recognition_sface/resolve/3d7082438a6e4551e840c9b2bb60b71e8da4b524/"
        "face_recognition_sface_2021dec.onnx",
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
)

# Tests replace the download transport.
transport: httpx.AsyncBaseTransport | None = None


class FaceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def models_dir() -> Path:
    path = get_settings().storage_dir / "models" / "face"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ok(path: Path, model: FaceModel) -> bool:
    if not path.is_file() or path.stat().st_size != model.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == model.sha256


_download_lock = asyncio.Lock()


async def ensure_models() -> tuple[Path, Path]:
    """Downloads the two model files once (checked against their SHA-256)."""
    async with _download_lock:
        paths = []
        for model in (DETECTOR, RECOGNIZER):
            path = models_dir() / model.filename
            if not await asyncio.to_thread(_ok, path, model):
                await _download(model, path)
            paths.append(path)
        return paths[0], paths[1]


async def _download(model: FaceModel, path: Path, what: str = "The face model") -> None:
    part = path.with_suffix(".part")
    last = "no source answered"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300, connect=20), follow_redirects=True, transport=transport
    ) as client:
        for url in model.urls:
            try:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        last = f"HTTP {response.status_code}"
                        continue
                    with part.open("wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
            except httpx.HTTPError as exc:
                last = type(exc).__name__
                continue
            if await asyncio.to_thread(_ok, part, model):
                part.replace(path)
                return
            last = "the file did not match its checksum"
    part.unlink(missing_ok=True)
    raise FaceError(f"{what} could not be downloaded ({last}). Check the internet.")


class _Engine:
    """OpenCV's detector and recognizer, loaded once; one image at a time (not thread safe)."""

    def __init__(self, detector: Path, recognizer: Path):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.lock = threading.Lock()

    def analyze(self, data: bytes) -> tuple[Any | None, float | None]:
        """The largest face's normalized feature and its width as a share of the image width;
        (None, None) without a face big enough to compare."""
        import numpy as np

        image = self.cv2.imdecode(np.frombuffer(data, np.uint8), self.cv2.IMREAD_COLOR)
        if image is None:
            return None, None
        h, w = image.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        if scale < 1.0:
            image = self.cv2.resize(image, (int(w * scale), int(h * scale)))
            h, w = image.shape[:2]
        with self.lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) == 0:
                return None, None
            face = main_face(faces)
            if float(face[2]) < MIN_FACE_PX:
                return None, None
            vector = self.recognizer.feature(self.recognizer.alignCrop(image, face)).flatten()
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm else None), round(float(face[2]) / w, 3)

    def feature(self, data: bytes) -> Any | None:
        return self.analyze(data)[0]


def main_face(found: Any) -> Any:
    """The largest face among those detected with a score near the best one (YuNet's rows:
    x, y, w, h, landmarks, score)."""
    best = max(float(f[14]) for f in found)
    return max((f for f in found if float(f[14]) >= SURE_SHARE * best), key=lambda f: f[2] * f[3])


_engine: _Engine | None = None
# Features per file (path + size + mtime), so the AI model's photos are read once.
_cache: OrderedDict[tuple[str, int, int], Any] = OrderedDict()


async def _get_engine() -> _Engine:
    global _engine
    if _engine is None:
        detector, recognizer = await ensure_models()
        _engine = await asyncio.to_thread(_Engine, detector, recognizer)
    return _engine


def _analysis_of(engine: Any, path: Path) -> tuple[Any | None, float | None]:
    st = path.stat()
    key = (str(path), st.st_size, int(st.st_mtime))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    data = path.read_bytes()
    analyze = getattr(engine, "analyze", None)
    value = analyze(data) if analyze else (engine.feature(data), None)
    _cache[key] = value
    if len(_cache) > 2000:
        _cache.popitem(last=False)
    return value


def _feature_of(engine: Any, path: Path) -> Any | None:
    return _analysis_of(engine, path)[0]


def _baseline(refs: list[Any]) -> float | None:
    """How alike the photos' faces are to each other, measured the same way as the
    validation images: the median over the photos of each photo's median similarity to the
    others. Shown next to the run's scores as a measured reference, not converted into one."""
    if len(refs) < 2:
        return None
    per_photo = [
        statistics.median(float((a * b).sum()) for j, b in enumerate(refs) if j != i)
        for i, a in enumerate(refs)
    ]
    return statistics.median(per_photo)


def _score(features: list[Any], refs: list[Any]) -> float:
    """The best match against the reference photos (cosine similarity)."""
    return max(float((f * r).sum()) for f in features for r in refs)


async def compare(images: list[Path], references: list[Path]) -> dict[str, Any]:
    engine = await _get_engine()

    def run() -> dict[str, Any]:
        refs = [f for f in (_feature_of(engine, p) for p in references) if f is not None]
        out = []
        for path in images:
            feature = _feature_of(engine, path)
            if feature is None or not refs:
                out.append({"face": feature is not None, "score": None, "match": None})
                continue
            score = round(_score([feature], refs), 3)
            out.append({"face": True, "score": score, "match": score >= MATCH_THRESHOLD})
        return {"reference_faces": len(refs), "items": out}

    return await asyncio.to_thread(run)


async def likeness(samples: list[tuple[int, Path]], references: list[Path]) -> dict[str, Any]:
    """How close a training run's validation images are to its photos, per step: the mean of
    the samples' scores at that step (each sample: its median similarity to the photos' faces,
    so one odd photo cannot decide it), and the step with the highest score."""
    engine = await _get_engine()

    def run() -> dict[str, Any]:
        refs = [f for f in (_feature_of(engine, p) for p in references) if f is not None]
        counts: dict[int, int] = {}
        scores: dict[int, list[float]] = {}
        for step, path in samples:
            counts[step] = counts.get(step, 0) + 1
            feature = _feature_of(engine, path) if path.is_file() else None
            if feature is not None and refs:
                similarity = statistics.median(float((feature * r).sum()) for r in refs)
                scores.setdefault(step, []).append(similarity)
        rows = [
            {
                "step": step,
                "samples": counts[step],
                "faces": len(scores.get(step, [])),
                "score": round(statistics.fmean(scores[step]), 3) if scores.get(step) else None,
            }
            for step in sorted(counts)
        ]
        baseline = _baseline(refs)
        scored = [r for r in rows if r["score"] is not None]
        best = max(scored, key=lambda r: (r["score"], -r["step"]))["step"] if scored else None
        return {
            "reference_faces": len(refs),
            "baseline": round(baseline, 3) if baseline is not None else None,
            "steps": rows,
            "best_step": best,
        }

    return await asyncio.to_thread(run)


async def consistency(paths: list[Path]) -> list[dict[str, Any]]:
    """For each photo of one character: is there a face, and how close it is to the faces
    in the other photos (median similarity). Below the match threshold the photo may show
    someone else — or a face from the side, which SFace compares less well."""
    engine = await _get_engine()

    def run() -> list[dict[str, Any]]:
        analyses = [_analysis_of(engine, p) for p in paths]
        features = [a[0] for a in analyses]
        found = [(i, f) for i, f in enumerate(features) if f is not None]
        out: list[dict[str, Any]] = []
        for index, feature in enumerate(features):
            others = (
                [float((feature * g).sum()) for j, g in found if j != index]
                if (feature is not None)
                else []
            )
            score = round(statistics.median(others), 3) if others else None
            out.append(
                {
                    "face": feature is not None,
                    "score": score,
                    "outlier": score is not None and score < MATCH_THRESHOLD,
                    # The face's width as a share of the photo's width (close-up ≈ 0.3+).
                    "share": analyses[index][1] if feature is not None else None,
                }
            )
        return out

    return await asyncio.to_thread(run)
