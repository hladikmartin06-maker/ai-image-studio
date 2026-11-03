"""Face similarity: the API with a stand-in for OpenCV, and the real YuNet + SFace models when
their files are on this machine (FACE_MODELS_DIR)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from app.services import faces
from tests.conftest import Api, make_user, wait_done
from tests.fakes import png_bytes
from tests.test_personas import LENA


class FakeEngine:
    """The red channel of the photo stands for "the face": same colour = same person."""

    def feature(self, data: bytes):
        import io

        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            r, g, b = img.convert("RGB").getpixel((0, 0))
        if (r, g, b) == (0, 0, 0):
            return None  # "no face"
        vector = np.array([r, g, b], dtype=float)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()

    async def get_engine():
        return engine

    monkeypatch.setattr(faces, "_get_engine", get_engine)
    faces._cache.clear()
    return engine


async def test_compare_with_an_ai_model(connected: Api, fake, fake_engine):
    r = await connected.post("/api/personas", json=LENA)
    persona = r.json()
    files = {"file": ("a.png", png_bytes(200, 200, (200, 40, 40)), "image/png")}
    up = await connected.post(f"/api/personas/{persona['id']}/images", files=files)
    assert up.status_code in (200, 201), up.text
    # A generated image of "another person" (the fake provider's colour).
    g = await connected.post(
        "/api/generations", json={"model": "wan-2.7", "provider": "wavespeed", "prompt": "x"}
    )
    image = (await wait_done(connected, g.json()["id"]))["images"][0]
    res = await connected.post(
        "/api/faces/compare", json={"image_ids": [image["id"]], "persona_id": persona["id"]}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["reference"] == {"kind": "persona", "name": "Lena", "photos": 1, "faces": 1}
    item = body["items"][0]
    assert item["face"] and isinstance(item["score"], float)
    assert item["match"] == (item["score"] >= faces.MATCH_THRESHOLD)
    # One of: an AI model or a LoRA; never another user's images.
    assert (
        await connected.post("/api/faces/compare", json={"image_ids": [image["id"]]})
    ).status_code == 422
    other = await make_user("faces-other")
    assert (
        await other.post(
            "/api/faces/compare", json={"image_ids": [image["id"]], "persona_id": persona["id"]}
        )
    ).status_code == 404
    await other.client.aclose()


async def test_models_are_downloaded_once_and_checked(tmp_path, monkeypatch):
    """The model files come from the pinned sources and must match their SHA-256."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "storage_dir", tmp_path)
    detector = b"d" * 10
    recognizer = b"r" * 20
    calls: list[str] = []
    monkeypatch.setattr(
        faces,
        "DETECTOR",
        faces.FaceModel(
            "det.onnx",
            10,
            __import__("hashlib").sha256(detector).hexdigest(),
            ("https://huggingface.co/x/det.onnx", "https://zoo.example/det.onnx"),
        ),
    )
    monkeypatch.setattr(
        faces,
        "RECOGNIZER",
        faces.FaceModel(
            "rec.onnx",
            20,
            __import__("hashlib").sha256(recognizer).hexdigest(),
            ("https://huggingface.co/x/rec.onnx",),
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "huggingface.co" and request.url.path.endswith("det.onnx"):
            return httpx.Response(503)  # the first source fails: the second one is used
        body = detector if request.url.path.endswith("det.onnx") else recognizer
        return httpx.Response(200, content=body)

    monkeypatch.setattr(faces, "transport", httpx.MockTransport(handler))
    det, rec = await faces.ensure_models()
    assert det.read_bytes() == detector and rec.read_bytes() == recognizer
    assert calls == [
        "https://huggingface.co/x/det.onnx",
        "https://zoo.example/det.onnx",
        "https://huggingface.co/x/rec.onnx",
    ]
    await faces.ensure_models()
    assert len(calls) == 3  # already there
    # A file that does not match its checksum is never used.
    rec.write_bytes(b"x" * 20)
    monkeypatch.setattr(
        faces, "transport", httpx.MockTransport(lambda r: httpx.Response(200, content=b"y" * 20))
    )
    with pytest.raises(faces.FaceError):
        await faces.ensure_models()


MODELS = os.environ.get("FACE_MODELS_DIR")


@pytest.mark.skipif(not MODELS, reason="set FACE_MODELS_DIR to the two ONNX files to run")
async def test_real_models_tell_people_apart():
    folder = Path(MODELS or "")
    engine = faces._Engine(folder / faces.DETECTOR.filename, folder / faces.RECOGNIZER.filename)
    lena = (folder / "lena.jpg").read_bytes()
    messi = (folder / "messi5.jpg").read_bytes()
    a, b = engine.feature(lena), engine.feature(messi)
    assert a is not None and b is not None
    assert float((a * a).sum()) == pytest.approx(1.0, abs=1e-5)
    assert float((a * b).sum()) < faces.MATCH_THRESHOLD
    assert engine.feature(png_bytes(64, 64)) is None  # no face
