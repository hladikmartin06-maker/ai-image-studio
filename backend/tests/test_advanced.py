"""Generate › Advanced: Krea 2 Turbo photo edits (comfy/k2edit.py, comfy/advanced.py), the
custom nodes sent to a rented GPU (comfy/custom_nodes.py) and the Modal volume mount."""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile

import httpx
import pytest
from PIL import Image

from app.comfy import custom_nodes, deploy, k2edit
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.db import SessionLocal
from app.models import ComfyConnection, Generation
from tests.conftest import Api, make_user, wait_done
from tests.fake_comfy_server import KREA_FILES, FakeComfy
from tests.fakes import png_bytes

EDIT_FILE = k2edit.EDIT_LORAS[k2edit.DEFAULT_EDIT].filename


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def test_edit_graph_follows_the_node_pack_example():
    wf = k2edit.build_edit(
        image="me.png",
        prompt="Make her smile.",
        width=880,
        height=1184,
        seed=5,
        edit_lora="general_v2",
        loras=[("ai-image-studio/taila.safetensors", 0.9)],
    )
    kinds = {k: v["class_type"] for k, v in wf.items()}
    assert kinds["13"] == "Krea2EditModelPatch" and kinds["4"] == "Krea2EditGroundedEncode"
    # Edit LoRA first, the owner's LoRA after it, the patch on top of both.
    assert wf["20"]["inputs"]["lora_name"] == EDIT_FILE
    assert wf["21"]["inputs"]["model"] == ["20", 0]
    assert wf["21"]["inputs"]["strength_model"] == 0.9
    assert wf["13"]["inputs"]["model"] == ["21", 0]
    # The photo is scaled to the output size and used by the patch and the encoder alike.
    assert wf["101"]["inputs"]["width"] == 880 and wf["101"]["inputs"]["height"] == 1184
    assert wf["13"]["inputs"]["source_image"] == ["101", 0]
    assert wf["4"]["inputs"]["image"] == ["101", 0]
    assert wf["13"]["inputs"]["target_latent"] == ["6", 0]
    assert wf["6"]["class_type"] == "EmptySD3LatentImage"
    ks = wf["7"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (
        8,
        1.0,
        "euler",
        "simple",
    )
    assert wf["5"]["class_type"] == "ConditioningZeroOut"  # CFG 1: no negative encode
    two = k2edit.build_edit(
        image="scene.png",
        image_b="person.png",
        prompt="x",
        width=64,
        height=64,
        seed=1,
        edit_lora="identity_v2",
        ref_position=k2edit.REF_POSITIONS[1],
    )
    assert two["13"]["inputs"]["source_image_b"] == ["301", 0]
    assert two["4"]["inputs"]["image_b"] == ["301", 0]
    with pytest.raises(ValueError):
        k2edit.build_edit(
            image="a",
            prompt="x",
            width=64,
            height=64,
            seed=1,
            edit_lora="general_v2",
            ref_position="sideways",
        )


def test_downloads_are_pinned_and_detection():
    files = k2edit.downloads()
    assert [f["filename"] for f in files] == [
        "qwen3vl_4b_fp8_scaled.safetensors",
        "qwen_image_vae.safetensors",
        "krea2_turbo_fp8_scaled.safetensors",
        EDIT_FILE,
    ]
    edit = files[-1]
    assert edit["folder"] == "loras" and len(edit["sha256"]) == 64
    assert f"/resolve/{k2edit.HF_REVISION}/" in edit["url"]
    info = FakeComfy(
        files={**KREA_FILES, "loras": [EDIT_FILE]}, custom_nodes=list(k2edit.NODES)
    ).object_info()
    assert k2edit.detect(info)["ready"]
    missing = FakeComfy().object_info()
    found = k2edit.detect(missing)
    assert not found["ready"] and found["missing_nodes"] == list(k2edit.NODES)
    assert k2edit.catalog_json()["total_bytes"] > 18e9


def test_character_is_named_once():
    assert k2edit.with_character("Make her smile.", "lmxx2", "woman") == (
        "Turn the woman into lmxx2 woman. Make her smile."
    )
    assert k2edit.with_character("Smile.", "lmxx2", "woman", "long straight black") == (
        "Turn the woman into lmxx2 woman, with long straight black hair. Smile."
    )
    assert k2edit.with_character("Turn her into LMXX2 woman", "lmxx2", "woman") == (
        "Turn her into LMXX2 woman"
    )


def test_keep_unchanged_puts_the_original_back():
    src = Image.new("RGB", (400, 300), (40, 90, 160))
    for x in range(0, 400, 8):  # some texture
        src.paste((200, 200, 200), (x, 0, x + 2, 300))
    original = _png(src)
    # The model's version: same picture at another size, with one square changed.
    edit = src.resize((320, 240))
    edit.paste((250, 30, 30), (120, 80, 220, 180))
    result, changed = k2edit.keep_unchanged(original, _png(edit))
    out = Image.open(io.BytesIO(result)).convert("RGB")
    assert out.size == (400, 300)  # the original's size
    assert 0.05 < changed < 0.5
    # Far from the change: the original's own pixels, exactly.
    for x, y in ((5, 5), (390, 290), (20, 250), (396, 10)):
        assert out.getpixel((x, y)) == src.getpixel((x, y))
    r, g, b = out.getpixel((212, 162))  # inside the change: the edit
    assert r > 200 and g < 80
    # Nothing changed: the original comes back unchanged.
    same, none = k2edit.keep_unchanged(original, _png(src.resize((320, 240))))
    assert none < 0.01
    assert Image.open(io.BytesIO(same)).convert("RGB").tobytes() == src.tobytes()


def test_keep_unchanged_size_limits():
    big = _png(Image.new("RGB", (4000, 3000), (10, 10, 10)))
    out, _ = k2edit.keep_unchanged(big, _png(Image.new("RGB", (1024, 768), (10, 10, 10))))
    assert Image.open(io.BytesIO(out)).size == (1536, 1152)  # at most 1.5x the edit
    small = _png(Image.new("RGB", (400, 300), (10, 10, 10)))
    out, _ = k2edit.keep_unchanged(small, _png(Image.new("RGB", (1024, 768), (10, 10, 10))))
    assert Image.open(io.BytesIO(out)).size == (1024, 768)  # never smaller than the edit


async def test_custom_nodes_are_pinned(tmp_path, monkeypatch):
    files = {
        "__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
        "pyproject.toml": b'[project]\nversion = "2.0.0"\n',
    }
    pinned = custom_nodes.CustomNode(
        folder="comfyui-krea2edit",
        repo="lbouaraba/comfyui-krea2edit",
        commit="abc",
        files=tuple((n, hashlib.sha256(d).hexdigest()) for n, d in files.items()),
    )
    monkeypatch.setattr(custom_nodes, "NODES", (pinned,))
    monkeypatch.setattr(custom_nodes.storage, "root", lambda: tmp_path)
    served: dict[str, bytes] = dict(files)

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "raw.githubusercontent.com"
        assert "/lbouaraba/comfyui-krea2edit/abc/" in request.url.path
        return httpx.Response(200, content=served[request.url.path.rsplit("/", 1)[1]])

    monkeypatch.setattr(custom_nodes, "transport", httpx.MockTransport(answer))
    packed = await custom_nodes.packed()
    assert packed
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(packed)), mode="r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == ["comfyui-krea2edit/__init__.py", "comfyui-krea2edit/pyproject.toml"]
    # A file that does not match its SHA-256 is never sent (nor cached).
    for f in tmp_path.rglob("*"):
        if f.is_file():
            f.unlink()
    served["__init__.py"] = b"import os; os.system('evil')\n"
    assert await custom_nodes.packed() is None


def test_modal_comfyui_volume_is_not_mounted_over_workspace():
    """The ComfyUI image has files in /workspace; a Modal Volume mounted there ends the
    sandbox at once (exit code 1, no output). The Volume goes to /studio instead."""
    conn = ComfyConnection(
        id=__import__("uuid").uuid4(),
        user_id=__import__("uuid").uuid4(),
        name="m",
        kind="modal",
        deploy={"provider": "modal", "gpu_type": "L40S"},
    )
    spec = deploy.spec_for(conn, "token", None, "image", [], nodes="Tk9ERVM=")
    assert spec.volume_name == deploy.MODAL_VOLUME and spec.volume_path == "/studio"
    assert spec.env["STUDIO_WORKSPACE"] == "/studio"
    assert spec.env["STUDIO_NODES_CODE"] == "Tk9ERVM="
    runpod = ComfyConnection(
        id=conn.id,
        user_id=conn.user_id,
        name="r",
        kind="runpod",
        deploy={"provider": "runpod", "gpu_type": "A40"},
    )
    assert deploy.spec_for(runpod, "t", None, "image", []).env["STUDIO_WORKSPACE"] == "/workspace"


async def _ready_connection(user: Api) -> dict:
    r = await user.post(
        "/api/comfy/connections",
        json={"name": "Mac ComfyUI", "kind": "local", "base_url": "http://127.0.0.1:8188"},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def test_advanced_edit_runs_and_keeps_the_original(user: Api):
    fake = FakeComfy(
        files={**KREA_FILES, "loras": [*KREA_FILES["loras"], EDIT_FILE]},
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    try:
        conn = await _ready_connection(user)
        assert conn["info"]["k2edit"]["ready"], conn["info"]["k2edit"]
        photo = png_bytes(64, 80, (10, 200, 10))
        ref = (
            await user.post("/api/reference-images", files={"file": ("me.png", photo, "image/png")})
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her smile and change the background to a beach.",
        }
        # Explicit instructions need the 18+ switch; minors are always refused.
        off = await user.post("/api/comfy/advanced", json={**body, "prompt": "make her nude"})
        assert off.status_code == 422 and off.json()["error"]["code"] == "adult_off"
        minor = await user.post(
            "/api/comfy/advanced", json={**body, "prompt": "nude teen girl", "adult": True}
        )
        assert minor.status_code == 422
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        gen = await wait_done(user, r.json()["id"])
        assert gen["status"] == "completed" and len(gen["images"]) == 1, gen
        assert gen["comfy"]["family"] == "k2edit" and gen["settings"]["keep_unchanged"] is True
        assert gen["comfy"]["input"]["thumb_url"].endswith("/thumb")
        sent = list(fake.prompts.values())[-1]["prompt"]
        uploaded = sent["10"]["inputs"]["image"]
        assert uploaded in fake.uploads
        assert sent["4"]["inputs"]["prompt"] == body["prompt"]
        assert sent["20"]["inputs"]["lora_name"] == EDIT_FILE
        # The fake's result is 96x128; the photo was smaller, so the result is the edit's size.
        image = gen["images"][0]
        assert (image["width"], image["height"]) == (102, 128)
        assert sent["13"]["inputs"]["ref_boost"] == 4.0
        strong = await user.post("/api/comfy/advanced", json={**body, "strength": "strong"})
        assert (await wait_done(user, strong.json()["id"]))["settings"]["change"] == "strong"
        assert list(fake.prompts.values())[-1]["prompt"]["13"]["inputs"]["ref_boost"] == 0.6
        # Off: the edit as it came from ComfyUI.
        raw = await user.post("/api/comfy/advanced", json={**body, "keep_unchanged": False})
        done = await wait_done(user, raw.json()["id"])
        assert (done["images"][0]["width"], done["images"][0]["height"]) == (96, 128)
        # Never another user's connection or photo.
        other = await make_user("advanced-other@example.com")
        assert (await other.post("/api/comfy/advanced", json=body)).status_code == 404
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


@pytest.fixture
def comfy():
    fake = FakeComfy()
    ComfyClient.transport = fake.transport()
    yield fake
    ComfyClient.transport = None


async def test_advanced_needs_its_models(user: Api, comfy):
    conn = await _ready_connection(user)
    ref = (
        await user.post(
            "/api/reference-images", files={"file": ("me.png", png_bytes(64, 80), "image/png")}
        )
    ).json()
    r = await user.post(
        "/api/comfy/advanced",
        json={
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her smile.",
        },
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "missing_nodes"
    assert comfy.prompts == {}


async def _character(user: Api, conn_id: str) -> tuple[str, str]:
    """A finished LoRA version of the user's that the connection already has."""
    import uuid

    from app.comfy.service import lora_filename
    from app.models import Lora, LoraVersion

    r = await user.post(
        "/api/loras",
        json={
            "name": "Taila",
            "trigger_word": "tla",
            "class_word": "woman",
            "base_variant": "turbo",
            "adult_confirmed": True,
        },
    )
    assert r.status_code == 201, r.text
    async with SessionLocal() as db:
        lora = await db.get(Lora, uuid.UUID(r.json()["id"]))
        version = LoraVersion(
            lora_id=lora.id,
            user_id=lora.user_id,
            number=1,
            status="completed",
            config={},
            trainer_config={},
            file_path="loras/tla.safetensors",
            file_sha256="0" * 64,
            deployments={conn_id: {"filename": "x"}},
        )
        db.add(version)
        await db.commit()
        return str(version.id), lora_filename(lora, version)


async def test_character_face_is_redrawn_close_up(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={**KREA_FILES, "loras": [*KREA_FILES["loras"], EDIT_FILE]},
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    boxes: list[tuple[float, float, float, float] | None] = [(30.0, 40.0, 24.0, 24.0)]

    async def box(_data: bytes):
        return boxes[0]

    monkeypatch.setattr(faces, "face_box", box)
    try:
        conn = await _ready_connection(user)
        version_id, lora_file = await _character(user, conn["id"])
        fake.files["loras"].append(lora_file)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(64, 80, (10, 200, 10)), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her smile.",
            "lora": {"version_id": version_id, "strength": 1.1},
        }
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        gen = await wait_done(user, r.json()["id"])
        assert gen["status"] == "completed", gen
        assert gen["final_prompt"] == "Turn the woman into tla woman. Make her smile."
        assert gen["settings"]["face_refine"] is True
        edit, face = [p["prompt"] for p in list(fake.prompts.values())[-2:]]
        # The edit: the character LoRA after the edit LoRA, ref_boost 2 (room for her face).
        assert edit["21"]["inputs"]["lora_name"] == lora_file
        assert edit["21"]["inputs"]["strength_model"] == 1.1
        assert edit["13"]["inputs"]["ref_boost"] == 2.0
        # The face: Krea 2 Turbo image to image of the crop with the same LoRA.
        assert face["20"]["inputs"]["lora_name"] == lora_file
        assert face["7"]["inputs"]["denoise"] == k2edit.FACE_DENOISE
        assert face["4"]["inputs"]["text"].startswith("close-up photo of tla woman")
        assert face["10"]["inputs"]["image"] in fake.uploads
        # No face found (or too big): one prompt only. Text to image: a batch, no photo.
        boxes[0] = None
        before = len(fake.prompts)
        text = await user.post(
            "/api/comfy/advanced",
            json={
                "connection_id": conn["id"],
                "prompt": "standing on a beach",
                "count": 2,
                "aspect_ratio": "3:4",
                "lora": {"version_id": version_id, "strength": 1.0},
            },
        )
        assert text.status_code == 201, text.text
        done = await wait_done(user, text.json()["id"])
        assert done["status"] == "completed" and len(done["images"]) == 2, done
        assert len(fake.prompts) == before + 1
        sent = list(fake.prompts.values())[-1]["prompt"]
        assert sent["6"]["inputs"]["batch_size"] == 2
        assert sent["4"]["inputs"]["text"] == "tla, standing on a beach"
        assert done["mode"] == "text-to-image" and not done["comfy"].get("input")
        bad = await user.post(
            "/api/comfy/advanced",
            json={"connection_id": conn["id"], "prompt": "x", "aspect_ratio": "7:3"},
        )
        assert bad.status_code == 400
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_face_square_rules():
    assert k2edit.face_square((10, 10, 10, 10), 1000, 1000) is None  # too small to redraw
    assert k2edit.face_square((100, 100, 400, 400), 1000, 1000) is None  # already big
    left, top, side = k2edit.face_square((900, 20, 100, 120), 1000, 1500)
    assert side == 264 and left + side <= 1000 and top >= 0
    img = Image.new("RGB", (400, 400), (0, 0, 255))
    patch = _png(Image.new("RGB", (1024, 1024), (255, 0, 0)))
    out = Image.open(io.BytesIO(k2edit.face_paste(_png(img), patch, (100, 100, 200)))).convert(
        "RGB"
    )
    assert out.getpixel((200, 200))[0] > 240  # the middle of the face: the new one
    assert out.getpixel((105, 105)) == (0, 0, 255)  # the square's corner: untouched
    assert out.getpixel((10, 10)) == (0, 0, 255)


async def test_face_only_edits_just_the_face(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={**KREA_FILES, "loras": [*KREA_FILES["loras"], EDIT_FILE]},
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    found: list = [(100.0, 80.0, 60.0, 70.0)]

    async def box(_data: bytes):
        return found[0]

    monkeypatch.setattr(faces, "face_box", box)
    try:
        conn = await _ready_connection(user)
        photo = png_bytes(400, 500, (10, 200, 10))
        ref = (
            await user.post("/api/reference-images", files={"file": ("me.png", photo, "image/png")})
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her smile widely.",
            "face_only": True,
        }
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        gen = await wait_done(user, r.json()["id"])
        assert gen["status"] == "completed", gen
        assert gen["settings"]["face_only"] is True
        sent = list(fake.prompts.values())[-1]["prompt"]
        assert sent["6"]["inputs"]["width"] == sent["6"]["inputs"]["height"] == 1024
        uploaded = Image.open(io.BytesIO(fake.uploads[sent["10"]["inputs"]["image"]]))
        assert uploaded.size == (1024, 1024)  # the face crop, not the photo
        image = gen["images"][0]
        assert (image["width"], image["height"]) == (400, 500)  # back in the whole photo
        found[0] = None
        none = await user.post("/api/comfy/advanced", json=body)
        assert none.status_code == 422 and none.json()["error"]["code"] == "no_face"
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_documented_workflows_match_the_code():
    """docs/advanced/*.json are the graphs the studio sends (example values)."""
    import json
    from pathlib import Path

    docs = Path(__file__).resolve().parents[2] / "docs" / "advanced"
    edit = k2edit.build_edit(
        image="your_photo.png",
        prompt="Make her smile and change the background to a beach at sunset.",
        width=880,
        height=1184,
        seed=7,
        edit_lora=k2edit.DEFAULT_EDIT,
        loras=[("your_character_lora.safetensors", 1.0)],
        **k2edit.settings_for(character=True),
    )
    face = k2edit.face_graph(
        image="face_crop_1024.png",
        lora="your_character_lora.safetensors",
        strength=1.0,
        who="lmxx2 woman",
        seed=7,
    )
    assert json.loads((docs / "advanced_edit_api.json").read_text()) == edit
    assert json.loads((docs / "advanced_face_api.json").read_text()) == face
