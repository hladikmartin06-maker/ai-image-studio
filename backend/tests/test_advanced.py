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


def _sent(fake: FakeComfy) -> list[dict]:
    """The graphs sent, without the 2K detail pass (the last step of character edits)."""
    return [
        p["prompt"]
        for p in fake.prompts.values()
        if not any(
            n.get("class_type") == "SaveImage"
            and n["inputs"].get("filename_prefix", "").endswith("-detail")
            for n in p["prompt"].values()
        )
    ]


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
        "sam3.1_multiplex_fp16.safetensors",
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
        sent = _sent(fake)[-1]
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
        assert _sent(fake)[-1]["13"]["inputs"]["ref_boost"] == 0.6
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
        edit, face = _sent(fake)[-2:]
        # The edit: the character LoRA after the edit LoRA, ref_boost 2 (room for her face).
        assert edit["21"]["inputs"]["lora_name"] == lora_file
        assert edit["21"]["inputs"]["strength_model"] == 1.1
        assert edit["13"]["inputs"]["ref_boost"] == 2.0
        # The face: the edit LoRA with her LoRA gives the crop her face (keeping what it
        # shows), then Turbo with her LoRA alone redraws it lightly from that latent.
        assert face["21"]["inputs"]["lora_name"] == lora_file
        assert face["21"]["inputs"]["strength_model"] == min(
            k2edit.FACE_LORA_MAX, 1.1 * k2edit.FACE_EDIT_LORA
        )
        assert face["13"]["inputs"]["ref_boost"] == 1.0
        assert face["4"]["inputs"]["prompt"].startswith("Give her the face of tla woman")
        assert face["90"]["inputs"]["model"] == ["1", 0]
        assert face["90"]["inputs"]["strength_model"] == min(
            k2edit.FACE_LORA_MAX, 1.1 * k2edit.FACE_LORA_BOOST
        )
        assert face["93"]["inputs"]["latent_image"] == ["7", 0]
        assert face["93"]["inputs"]["denoise"] == k2edit.FACE_DENOISE
        assert face["91"]["inputs"]["text"] == (
            "close-up photo of tla woman, natural skin texture. Make her smile."
        )
        assert face["8"]["inputs"]["samples"] == ["93", 0]
        assert face["10"]["inputs"]["image"] in fake.uploads
        # Without SAM 3.1 there is no detail pass (it redraws only her, found by SAM).
        assert len(fake.prompts) == 2
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
                "resolution": "1K",
                "lora": {"version_id": version_id, "strength": 1.0},
            },
        )
        assert text.status_code == 201, text.text
        done = await wait_done(user, text.json()["id"])
        assert done["status"] == "completed" and len(done["images"]) == 2, done
        assert len(fake.prompts) == before + 1
        sent = _sent(fake)[-1]
        assert sent["6"]["inputs"]["batch_size"] == 2
        assert (sent["6"]["inputs"]["width"], sent["6"]["inputs"]["height"]) == (880, 1184)
        # No trigger word added (it got written into the photo as a sign).
        assert sent["4"]["inputs"]["text"] == "standing on a beach"
        # The pose from the words: her LoRA only after the first steps, on the same noise.
        first, rest = sent[k2edit.COMPOSE_NODE]["inputs"], sent["7"]["inputs"]
        light, full = sent[first["model"][0]]["inputs"], sent[rest["model"][0]]["inputs"]
        assert light["model"] == ["1", 0] and light["lora_name"] == full["lora_name"]
        assert light["strength_model"] == k2edit.COMPOSE_LORA * full["strength_model"]
        assert first["end_at_step"] == rest["start_at_step"] == k2edit.COMPOSE_STEPS
        assert rest["latent_image"] == [k2edit.COMPOSE_NODE, 0]
        assert first["noise_seed"] == rest["noise_seed"] == done["settings"]["seed"]
        assert first["add_noise"] == "enable" and rest["add_noise"] == "disable"
        # Words without a pose (a plain mirror selfie): her LoRA on every step.
        plain = await user.post(
            "/api/comfy/advanced",
            json={
                "connection_id": conn["id"],
                "prompt": "a mirror selfie in a bathroom",
                "aspect_ratio": "3:4",
                "resolution": "1K",
                "lora": {"version_id": version_id, "strength": 1.0},
            },
        )
        assert (await wait_done(user, plain.json()["id"]))["status"] == "completed"
        sent = _sent(fake)[-1]
        assert sent["7"]["class_type"] == "KSampler" and k2edit.COMPOSE_NODE not in sent
        # 18+ and words asking for her naked: the sentence that keeps clothes off is added
        # (History shows it); without the switch nothing is added.
        naked = await user.post(
            "/api/comfy/advanced",
            json={
                "connection_id": conn["id"],
                "prompt": "completely nude on a bed, her pussy",
                "adult": True,
                "resolution": "1K",
            },
        )
        done = await wait_done(user, naked.json()["id"])
        assert done["final_prompt"] == (
            f"completely nude on a bed, her pussy {k2edit.NUDE_NOTE} {k2edit.VULVA_NOTE}"
        )
        # A pose in a front selfie with her LoRA: her face beside the phone.
        squat = await user.post(
            "/api/comfy/advanced",
            json={
                "connection_id": conn["id"],
                "prompt": "squatting, a mirror selfie",
                "resolution": "1K",
                "lora": {"version_id": version_id, "strength": 1.0},
            },
        )
        done = await wait_done(user, squat.json()["id"])
        assert done["final_prompt"] == f"squatting, a mirror selfie {k2edit.FACE_NOTE}"
        assert _sent(fake)[-1]["4"]["inputs"]["text"] == done["final_prompt"]
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


async def test_moved_finds_tilted_faces_and_a_lost_face(monkeypatch):
    a, b = _png(Image.new("RGB", (100, 100))), _png(Image.new("RGB", (100, 100), (1, 1, 1)))
    found = {a: (a, 0.0, (40.0, 40.0, 20.0, 20.0)), b: None}

    async def level(data: bytes):
        return found[data]

    monkeypatch.setattr(comfy_runner, "_level_face", level)
    assert await comfy_runner._moved(a, b)  # her face lost: moved
    assert not await comfy_runner._moved(b, a)  # no face to begin with: not
    # A face found turned by 90° is put back where it is: the same place, not moved.
    turned = k2edit.turn(a, 90.0)
    x, y = k2edit.unturn_point(50.0, 50.0, -90.0, (100, 100))  # the centre, in the turned photo
    found[b] = (turned, 90.0, (x - 10, y - 10, 20.0, 20.0))
    assert not await comfy_runner._moved(a, b)
    found[b] = (turned, 90.0, (0.0, 0.0, 20.0, 20.0))
    assert await comfy_runner._moved(a, b)


def test_keep_unchanged_takes_the_edit_where_the_old_hair_was():
    wall = Image.new("RGB", (100, 100), (230, 230, 230))
    original = wall.copy()
    original.paste((240, 235, 220), (10, 10, 30, 90))  # pale curls, close to the wall
    edited = wall.copy()
    hair = Image.new("L", (50, 50), 0)
    hair.paste(255, (0, 0, 25, 50))
    plain, _ = k2edit.keep_unchanged(_png(original), _png(edited))
    fixed, _ = k2edit.keep_unchanged(_png(original), _png(edited), [_png(hair)])
    assert Image.open(io.BytesIO(plain)).getpixel((20, 50)) == (240, 235, 220)  # a ghost
    assert Image.open(io.BytesIO(fixed)).getpixel((20, 50)) == (230, 230, 230)


def test_paste_held_puts_back_only_the_held_part():
    before = _png(Image.new("RGB", (100, 100), (0, 0, 255)))
    after = _png(Image.new("RGB", (200, 200), (255, 0, 0)))
    held = Image.new("L", (50, 50), 0)
    held.paste(255, (0, 0, 25, 50))  # the left half: a hand
    out = Image.open(io.BytesIO(k2edit.paste_held(before, after, _png(held))))
    assert out.size == (200, 200)
    assert out.getpixel((10, 100)) == (0, 0, 255)  # held: from before
    assert out.getpixel((190, 100)) == (255, 0, 0)  # the body pass's change


def test_reflection_pass_graph_has_no_masks_left():
    wf = k2edit.build_edit(
        image="a.png",
        prompt="p",
        width=64,
        height=64,
        seed=1,
        edit_lora=k2edit.DEFAULT_EDIT,
        loras=[("l.safetensors", 1.0)],
        hold_body=True,
        blur_hair=True,
    )
    plain = k2edit.without_masks(wf)
    assert "40" not in plain and k2edit.HAIR_NODE not in plain and "184" not in plain
    assert plain["12"]["inputs"]["pixels"] == ["101", 0]
    assert "ref_boost_mask" not in plain["13"]["inputs"]
    for node in plain.values():  # every link points at a node that is still there
        for v in node["inputs"].values():
            if isinstance(v, list) and v and isinstance(v[0], str):
                assert v[0] in plain, v


def test_graphs_have_no_clashing_nodes():
    """Every link points at a node of the right kind: a preview once took the id of a mask
    merge and the body pass failed on the GPU."""
    makers = {
        "MaskToImage": "IMAGE",
        "PreviewImage": None,
        "SAM3_Detect": "MASK",
        "GrowMask": "MASK",
        "MaskComposite": "MASK",
        "InvertMask": "MASK",
        "ImageBlur": "IMAGE",
        "ImageCompositeMasked": "IMAGE",
        "ImageToMask": "MASK",
        "ImageScale": "IMAGE",
    }
    wants = {
        ("GrowMask", "mask"),
        ("MaskComposite", "destination"),
        ("MaskComposite", "source"),
        ("InvertMask", "mask"),
        ("MaskToImage", "mask"),
        ("ImageCompositeMasked", "mask"),
    }
    graphs = [
        k2edit.build_edit(
            image="a",
            prompt="p",
            width=64,
            height=64,
            seed=1,
            edit_lora=k2edit.DEFAULT_EDIT,
            loras=[("l", 1.0)],
            **kw,
        )
        for kw in (
            {"region": "bust+butt+waist", "hold_rest": True, "keep_limbs": True},
            {"region": "butt"},
            {"region": "hair"},
            {"region": "clothes", "hold_rest": True},
            {"hold_body": True, "blur_hair": True},
        )
    ]
    graphs.append(
        k2edit.detail_graph(image="a", width=64, height=64, lora="l", strength=1.0, seed=1)
    )
    for wf in graphs:
        for node in wf.values():
            for name, v in node["inputs"].items():
                if (node["class_type"], name) in wants:
                    assert makers.get(wf[v[0]]["class_type"]) == "MASK", (node, name)


def test_face_step_keeps_an_open_mouth():
    plain = k2edit.face_graph(image="f.png", lora="l.safetensors", strength=1.0, who="w", seed=1)
    assert plain["93"]["inputs"]["latent_image"] == ["7", 0] and "40" not in plain
    wf = k2edit.face_graph(
        image="f.png", lora="l.safetensors", strength=1.0, who="w", seed=1, keep_mouth=True
    )
    words = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert {"teeth", "tongue"} <= set(words)
    # SAM looks at the first part's result; teeth and tongue are left out of the redraw.
    assert wf["146"]["inputs"]["image"] == ["144", 0] and wf["144"]["inputs"]["samples"] == ["7", 0]
    assert wf["93"]["inputs"]["latent_image"] == ["152", 0]
    assert wf["152"]["inputs"] == {"samples": ["7", 0], "mask": ["151", 0]}
    assert wf["151"]["class_type"] == "InvertMask"


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
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
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
        sent = _sent(fake)[-1]
        assert sent["6"]["inputs"]["width"] == sent["6"]["inputs"]["height"] == 1024
        uploaded = Image.open(io.BytesIO(fake.uploads[sent["10"]["inputs"]["image"]]))
        assert uploaded.size == (1024, 1024)  # the face crop, not the photo
        # Hard expressions: the crop is held loosely; hands are found to be kept as they were.
        assert sent["13"]["inputs"]["ref_boost"] == 1.0
        assert sent["61"]["class_type"] == "SAM3_Detect" and sent["60"]["inputs"]["text"] == "hand"
        assert sent["4"]["inputs"]["prompt"].endswith(k2edit.FACE_ONLY_NOTE)
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


async def test_body_part_frees_only_that_part(user: Api):
    fake = FakeComfy(
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    try:
        conn = await _ready_connection(user)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(64, 80), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her bust much larger.",
            "body_part": "bust",
        }
        bad = await user.post("/api/comfy/advanced", json={**body, "body_part": "nose"})
        assert bad.status_code == 400
        both = await user.post("/api/comfy/advanced", json={**body, "face_only": True})
        assert both.status_code == 400
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        gen = await wait_done(user, r.json()["id"])
        assert gen["status"] == "completed" and gen["settings"]["body_part"] == "bust"
        sent = _sent(fake)[-1]
        words = sorted(
            n["inputs"]["text"] for n in sent.values() if n["class_type"] == "CLIPTextEncode"
        )
        assert words == ["breasts", "chest", "face"]
        # The bust let go (a visible change); the fallback holds everything but the bust,
        # for when the person moves.
        assert sent["13"]["inputs"]["ref_boost"] == k2edit.REGION_BOOST
        async with SessionLocal() as db:
            row = await db.get(Generation, __import__("uuid").UUID(gen["id"]))
            fallback = row.comfy_meta["pose_fallback"]
        assert fallback["13"]["inputs"]["ref_boost"] == k2edit.HOLD_BOOST
        assert fallback["13"]["inputs"]["ref_boost_mask"] == ["46", 0]
        assert fallback["46"]["class_type"] == "InvertMask"
        assert sent["43"]["inputs"]["mask"] == ["70", 0]  # both words added together
        assert sent["45"]["inputs"]["operation"] == "subtract"  # minus the face
        assert sent["4"]["inputs"]["prompt"].endswith(f"{k2edit.KEEP_CLOTHES} {k2edit.KEEP_POSE}")
        # Strong: no fallback, no words keeping the pose.
        strong = await user.post("/api/comfy/advanced", json={**body, "strength": "strong"})
        done = await wait_done(user, strong.json()["id"])
        async with SessionLocal() as db:
            row = await db.get(Generation, __import__("uuid").UUID(done["id"]))
            assert row.comfy_meta["pose_fallback"] is None
            assert k2edit.KEEP_POSE not in row.final_prompt
        # Clothes: SAM's clothes edited, everything else held, the pose kept in words.
        cloth = await user.post(
            "/api/comfy/advanced",
            json={
                "connection_id": conn["id"],
                "image": {"kind": "reference", "id": ref["id"]},
                "prompt": "Change her top to a red satin camisole.",
                "area": "clothes",
            },
        )
        assert cloth.status_code == 201, cloth.text
        assert (await wait_done(user, cloth.json()["id"]))["status"] == "completed"
        sent = _sent(fake)[-1]
        words = sorted(
            n["inputs"]["text"] for n in sent.values() if n["class_type"] == "CLIPTextEncode"
        )
        assert words == ["clothes", "face", "swimsuit"]
        assert sent["13"]["inputs"]["ref_boost"] == k2edit.HOLD_BOOST
        assert sent["46"]["class_type"] == "InvertMask"
        assert sent["4"]["inputs"]["prompt"].endswith(k2edit.KEEP_POSE)
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_face_paste_keeps_the_hands():
    img = Image.new("RGB", (400, 400), (0, 0, 255))
    patch = _png(Image.new("RGB", (1024, 1024), (255, 0, 0)))
    hands = Image.new("L", (1024, 1024), 0)
    hands.paste(255, (480, 480, 600, 600))  # a finger in the middle of the square
    out = Image.open(
        io.BytesIO(k2edit.face_paste(_png(img), patch, (100, 100, 200), _png(hands)))
    ).convert("RGB")
    assert out.getpixel((200, 200)) == (0, 0, 255)  # the finger: the photo's own pixels
    assert out.getpixel((160, 200))[0] > 240  # the rest of the face: the new one


def test_rewrite_answer_is_parsed():
    from app.comfy.advanced import parse_rewrite

    assert parse_rewrite(
        '```json\n{"instruction": "Make her bust  larger.", "target": "body", "part": "bust"}\n```'
    ) == ("Make her bust larger.", "body", "bust")
    assert parse_rewrite('{"instruction": "Smile.", "target": "face", "part": "bust"}') == (
        "Smile.",
        "face",
        None,
    )
    assert parse_rewrite("Make her smile.") == ("Make her smile.", None, None)
    assert parse_rewrite('{"instruction": 5}')[1] is None


def test_keep_unchanged_fills_holes_in_a_change():
    """A new beanie the colour of the old hair: strands of the same colour inside the new
    beanie must not come back from the original."""
    src = Image.new("RGB", (400, 400), (30, 30, 30))
    src.paste((200, 180, 140), (195, 120, 205, 260))  # a blond strand
    edit = src.copy()
    edit.paste((200, 180, 140), (120, 100, 280, 280))  # a beige beanie over it
    for x in range(120, 280, 6):  # knit texture
        edit.paste((150, 130, 100), (x, 100, x + 2, 280))
    result, _ = k2edit.keep_unchanged(_png(src), _png(edit))
    out = Image.open(io.BytesIO(result)).convert("RGB")
    assert out.tobytes() != src.tobytes()
    # Inside the beanie, where the strand was, the edit's own (textured) pixels.
    assert out.getpixel((198, 200)) == edit.getpixel((198, 200))
    assert out.getpixel((10, 10)) == src.getpixel((10, 10))


async def test_head_and_hair_areas(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02

    async def box(_data: bytes):
        return (180.0, 60.0, 40.0, 48.0)  # a small face

    monkeypatch.setattr(faces, "face_box", box)
    try:
        conn = await _ready_connection(user)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(400, 600, (10, 200, 10)), "image/png")},
            )
        ).json()
        base = {"connection_id": conn["id"], "image": {"kind": "reference", "id": ref["id"]}}
        head = await user.post(
            "/api/comfy/advanced",
            json={**base, "prompt": "Add black eyeglasses.", "area": "head"},
        )
        assert head.status_code == 201, head.text
        gen = await wait_done(user, head.json()["id"])
        assert gen["status"] == "completed" and gen["settings"]["area"] == "head", gen
        sent = _sent(fake)[-1]
        crop = Image.open(io.BytesIO(fake.uploads[sent["10"]["inputs"]["image"]]))
        assert crop.size == (1024, 1024)  # the head square, edited close up
        assert sent["4"]["inputs"]["prompt"].endswith(k2edit.HEAD_NOTE)
        assert sent["13"]["inputs"]["ref_boost"] == 4.0
        image = gen["images"][0]
        assert (image["width"], image["height"]) == (400, 600)  # back in the whole photo
        hair = await user.post(
            "/api/comfy/advanced",
            json={**base, "prompt": "Dye her hair copper red.", "area": "hair"},
        )
        assert hair.status_code == 201, hair.text
        assert (await wait_done(user, hair.json()["id"]))["status"] == "completed"
        sent = _sent(fake)[-1]
        texts = [n["inputs"]["text"] for n in sent.values() if n["class_type"] == "CLIPTextEncode"]
        assert (
            texts == ["hair", "face"] and sent["13"]["inputs"]["ref_boost"] == k2edit.REGION_BOOST
        )
        no_part = await user.post(
            "/api/comfy/advanced", json={**base, "prompt": "Bigger.", "area": "body"}
        )
        assert no_part.status_code == 400
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_head_square_and_crop_paste():
    left, top, side = k2edit.head_square((180, 60, 40, 48), 400, 600)
    assert side == 168 and top > 0  # 3.5x the face, reaching down to the shoulders
    # A big face: no close-up (the whole photo is edited instead).
    assert k2edit.head_square((100, 100, 200, 240), 400, 600) is None
    img = Image.new("RGB", (400, 600), (0, 0, 255))
    edit = Image.new("RGB", (1024, 1024), (0, 0, 255))
    edit.paste((255, 0, 0), (400, 300, 620, 420))  # glasses in the middle of the square
    out = Image.open(
        io.BytesIO(k2edit.crop_paste(_png(img), _png(edit), (left, top, side)))
    ).convert("RGB")
    cx, cy = left + side * 510 // 1024, top + side * 360 // 1024
    assert out.getpixel((cx, cy))[0] > 200  # the change came back
    assert out.getpixel((10, 10)) == (0, 0, 255)
    assert out.getpixel((left + 3, top + 3)) == (0, 0, 255)


def test_keep_unchanged_uses_the_whole_edit_for_a_new_background():
    src = Image.new("RGB", (400, 300), (240, 240, 240))  # a white sky…
    src.paste((40, 90, 40), (0, 0, 60, 300))  # …with a tree at the edge
    edit = Image.new("RGB", (400, 300), (30, 30, 200))  # a whole new scene…
    edit.paste((242, 242, 242), (200, 0, 400, 300))  # …partly as white as the old sky
    result, changed = k2edit.keep_unchanged(_png(src), _png(edit))
    assert changed > k2edit.KEEP_MAX_CHANGE
    assert Image.open(io.BytesIO(result)).convert("RGB").tobytes() == edit.tobytes()


async def test_auto_resolution_follows_the_photo(user: Api):
    from app.comfy import krea2

    fake = FakeComfy(
        files={**KREA_FILES, "loras": [*KREA_FILES["loras"], EDIT_FILE]},
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    try:
        conn = await _ready_connection(user)
        for size, tier in (((2000, 1500), "1K"), ((800, 600), "1K")):
            ref = (
                await user.post(
                    "/api/reference-images",
                    files={"file": ("me.png", png_bytes(*size), "image/png")},
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
            assert r.status_code == 201, r.text
            gen = await wait_done(user, r.json()["id"])
            assert gen["settings"]["resolution"] == tier
            sent = _sent(fake)[-1]
            w, h = krea2.size_for_source(*size, tier)
            assert (sent["6"]["inputs"]["width"], sent["6"]["inputs"]["height"]) == (w, h)
            # Sent at the edit's size, as a JPEG (big photos timed out on the way).
            sent_photo = Image.open(io.BytesIO(fake.uploads[sent["10"]["inputs"]["image"]]))
            assert sent_photo.format == "JPEG" and max(sent_photo.size) <= max(w, h)
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_keep_unchanged_keeps_thin_new_things():
    """Wire-rimmed glasses: a thin new line must come through whole."""
    from PIL import ImageDraw

    src = Image.new("RGB", (400, 400), (120, 100, 90))
    edit = src.copy()
    ImageDraw.Draw(edit).ellipse((100, 150, 220, 250), outline=(230, 190, 60), width=2)
    result, _ = k2edit.keep_unchanged(_png(src), _png(edit))
    out = Image.open(io.BytesIO(result)).convert("RGB")
    for x, y in ((160, 150), (160, 249), (100, 200), (219, 200)):  # top, bottom, left, right
        assert out.getpixel((x, y))[2] < 100, (x, y)  # gold, not the old skin


def test_hold_face_puts_the_face_back():
    src = Image.new("RGB", (400, 400), (200, 150, 120))
    edit = Image.new("RGB", (800, 800), (10, 10, 10))  # the edit's size may differ
    out = Image.open(
        io.BytesIO(k2edit.hold_face(_png(src), _png(edit), (150.0, 120.0, 100.0, 120.0)))
    ).convert("RGB")
    assert out.size == (800, 800)
    assert out.getpixel((400, 380)) == (200, 150, 120)  # the face's middle: the original
    assert out.getpixel((40, 40)) == (10, 10, 10)  # elsewhere: the edit


async def test_character_edit_holds_her_body(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02

    async def no_face(_data: bytes):
        return None

    monkeypatch.setattr(faces, "face_box", no_face)
    try:
        conn = await _ready_connection(user)
        version_id, lora_file = await _character(user, conn["id"])
        fake.files["loras"].append(lora_file)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(64, 80), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Give her long straight black hair.",
            "lora": {"version_id": version_id, "strength": 1.0},
        }
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        sent = _sent(fake)[-1]
        texts = [n["inputs"]["text"] for n in sent.values() if n["class_type"] == "CLIPTextEncode"]
        assert sorted(texts) == ["accessory", "face", "hair", "hand", "jewelry", "woman"]
        # Her body (woman minus hair and face, plus her hands) held at HOLD_BOOST: the pose
        # stays; where her old hair was is shown for keep_unchanged.
        assert sent["13"]["inputs"]["ref_boost"] == k2edit.HOLD_BOOST
        assert sent["13"]["inputs"]["ref_boost_mask"] == ["131", 0]
        assert sent["131"]["inputs"]["source"] == ["138", 0]
        assert sent[k2edit.HAIR_NODE]["class_type"] == "PreviewImage"
        # New hair: the old hair (minus hands and face) blurred in the reference.
        assert sent["12"]["inputs"]["pixels"] == ["187", 0]
        assert sent["13"]["inputs"]["source_image"] == ["187", 0]
        assert sent["4"]["inputs"]["image"] == ["187", 0]
        # Once over her, three times over the background; not her accessories, not her hands
        # (but the hair in them).
        assert sent["184"]["inputs"]["mask"] == ["212", 0]  # soft edges
        assert sent["187"]["inputs"]["mask"] == ["216", 0]
        assert sent["210"]["inputs"]["mask"] == ["194", 0]
        assert sent["194"]["inputs"]["destination"] == ["193", 0]
        assert sent["131"]["inputs"]["source"] == ["138", 0]
        assert sent["138"]["inputs"]["operation"] == "subtract"
        assert sent["190"]["inputs"]["text"] == "accessory"
        assert k2edit.asks_for_hair("Give her long black hair.")
        assert not k2edit.asks_for_hair("Make her smile. Keep her hair as it is.")
        r = await user.post("/api/comfy/advanced", json={**body, "prompt": "Make her smile."})
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        smile = _sent(fake)[-1]
        assert "184" not in smile and smile["12"]["inputs"]["pixels"] == ["101", 0]
        assert sent["127"]["inputs"]["operation"] == "subtract"
        # Normal keeps the pose in words too, unless a new pose is asked for (or Strong).
        assert sent["4"]["inputs"]["prompt"].endswith(k2edit.KEEP_POSE)
        for prompt, strength in (
            ("Make her sit with her legs crossed.", "normal"),
            ("Give her long straight black hair.", "strong"),
        ):
            r = await user.post(
                "/api/comfy/advanced", json={**body, "prompt": prompt, "strength": strength}
            )
            assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
            sent = _sent(fake)[-1]
            assert k2edit.KEEP_POSE not in sent["4"]["inputs"]["prompt"]
            assert "128" not in sent  # her body not held: the new pose can come
        # Without a character LoRA: no hold (the photo at ref_boost 4 everywhere).
        plain = {k: v for k, v in body.items() if k != "lora"}
        r = await user.post("/api/comfy/advanced", json={**plain, "prompt": "Make her smile."})
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        sent = _sent(fake)[-1]
        assert "40" not in sent and sent["13"]["inputs"]["ref_boost"] == 4.0
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_pose_words_and_body_words():
    # What is to stay is not a new pose (the studio's rewrite said "Keep ... her pose").
    assert not k2edit.asks_for_pose(
        "Give her long black hair. Keep her current facial expression, her pose, and the "
        "background exactly as they are."
    )
    assert not k2edit.asks_for_pose("Turn her into a blonde. Nechaj ju v tej istej póze.")
    assert k2edit.asks_for_pose("Make her sit and keep the background.")
    assert k2edit.asks_for_pose("Make her turn around.")
    assert k2edit.asks_for_pose("Nech si sadne.")
    # Body words: only the body clauses, the parts named; the edit gets the rest.
    assert k2edit.body_change(
        "Change her facial features, give her long straight black hair, make her figure "
        "slimmer with a much larger bust and a larger, rounder buttocks. Keep everything else "
        "unchanged."
    ) == (
        "Make her figure slimmer with a much larger bust, a larger, rounder buttocks.",
        "bust+butt+waist",
        "Change her facial features, give her long straight black hair. Keep everything else "
        "unchanged.",
    )
    assert k2edit.body_change("Daj jej väčšie prsia a väčší zadok.")[1] == "bust+butt"
    assert k2edit.body_change("Make her slimmer.") == ("Make her slimmer.", "figure", "")
    assert k2edit.body_change("Make her a bit curvier and keep the background.") == (
        "Make her a bit curvier.",
        "figure",
        "Keep the background.",
    )
    assert k2edit.body_change("Make her smile, and make her butt bigger.") == (
        "Make her butt bigger.",
        "butt",
        "Make her smile.",
    )
    for none in (
        "Give her long black hair.",
        "Make her cross her legs.",
        "Put her hands on her hips.",
        "Give her a slimmer face.",
        "Make her smile and give her bigger lips.",
    ):
        assert k2edit.body_change(none) is None, none
    # Many parts: the whole figure (SAM words in one region are limited).
    many = "Make her waist slimmer, her arms thinner, her legs longer and her bust bigger."
    assert k2edit.body_change(many)[1] == "figure"


def test_parts_together_are_one_region():
    wf = k2edit.build_edit(
        image="a.png",
        prompt="p",
        width=64,
        height=64,
        seed=1,
        edit_lora=k2edit.DEFAULT_EDIT,
        region="bust+butt",
    )
    words = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert words == ["breasts", "chest", "butt", "bottom", "face"]
    assert wf["13"]["inputs"]["ref_boost"] == k2edit.REGION_BOOST


async def test_whole_photo_body_change_gets_a_body_pass(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02

    async def no_face(_data: bytes):
        return None

    monkeypatch.setattr(faces, "face_box", no_face)
    try:
        conn = await _ready_connection(user)
        version_id, lora_file = await _character(user, conn["id"])
        fake.files["loras"].append(lora_file)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(64, 80), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Give her long straight black hair, a much larger bust and a rounder "
            "butt. Keep her pose and the background.",
            "lora": {"version_id": version_id, "strength": 1.0},
        }
        before = len(fake.prompts)
        r = await user.post("/api/comfy/advanced", json=body)
        assert r.status_code == 201, r.text
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        edit, shape, detail = [p["prompt"] for p in list(fake.prompts.values())[before:]]
        # Last, her at 2K once more with her LoRA (one quality for face and rest), no trigger
        # word in the text (it got written into the photo), only her redrawn.
        assert detail["7"]["inputs"]["denoise"] == k2edit.DETAIL_DENOISE
        assert detail["21"]["inputs"]["lora_name"] == lora_file
        assert "tla" not in detail["4"]["inputs"]["text"]
        assert detail["7"]["inputs"]["latent_image"] == ["233", 0]
        assert detail["9"]["inputs"]["images"] == ["237", 0]
        assert detail["231"]["inputs"]["operation"] == "subtract"  # her hair, not accessories
        words = {
            n["inputs"]["text"] for n in detail.values() if n["class_type"] == "CLIPTextEncode"
        }
        assert {"hair", "accessory", "jewelry", "hand", "face"} <= words and "skin" not in words
        assert detail["242"]["inputs"]["operation"] == "subtract"  # not her hands and face
        assert detail["241"]["inputs"]["source"] == ["244", 0]
        assert max(detail["101"]["inputs"]["width"], detail["101"]["inputs"]["height"]) > 1500
        # The edit holds her body (the pose); the second pass changes the bust and butt only
        # (everything else held, her face, arms and hands too), told only the body words,
        # kept natural, with her LoRA at full strength.
        assert edit["13"]["inputs"]["ref_boost_mask"] == ["131", 0]
        # The edit is not told about the body (a lying woman stood up to show a new bust).
        assert edit["4"]["inputs"]["prompt"].startswith(
            "Turn the woman into tla woman. Give her long straight black hair. Keep her pose"
        )
        assert "bust" not in edit["4"]["inputs"]["prompt"]
        words = [n["inputs"]["text"] for n in shape.values() if n["class_type"] == "CLIPTextEncode"]
        # (stored as JSON, the graph comes back in another key order)
        assert sorted(words) == sorted(
            ["breasts", "chest", "butt", "bottom", "face", "arm", "hand", "feet", "hair"]
        )
        assert shape["13"]["inputs"]["ref_boost"] == k2edit.HOLD_BOOST
        assert shape["13"]["inputs"]["ref_boost_mask"] == ["46", 0]  # the face held too
        assert shape["46"]["inputs"]["mask"] == ["165", 0]
        assert shape["165"]["inputs"]["source"] == ["166", 0]  # arms and hands, grown
        assert shape["166"]["inputs"]["expand"] == k2edit.LIMB_GROW
        assert "128" not in shape and "47" not in shape
        # Her arms, hands, face and hair are shown for putting back (paste_held).
        assert shape[k2edit.HELD_NODE]["class_type"] == "PreviewImage"
        assert shape["176"]["inputs"]["destination"] == ["175", 0]
        assert shape["175"]["inputs"]["destination"] == ["171", 0]  # limbs put back tight
        assert shape["171"]["inputs"]["expand"] == k2edit.PASTE_GROW
        # The part it changed is shown, so keep_unchanged never brings the old shape back.
        assert shape[k2edit.PART_NODE]["class_type"] == "PreviewImage"
        assert shape["188"]["inputs"]["mask"] == ["165", 0]
        assert shape["166"]["inputs"]["mask"] == ["179", 0]  # arms + hands + feet
        text = shape["4"]["inputs"]["prompt"]
        assert "Give her a much larger bust, a rounder butt." in text and "background" not in text
        assert text.endswith(f"{k2edit.KEEP_CLOTHES} {k2edit.KEEP_POSE}")
        assert shape["21"]["inputs"]["strength_model"] == 1.0  # her own figure
        assert k2edit.BODY_NATURAL in text
        assert shape["10"]["inputs"]["image"] != edit["10"]["inputs"]["image"]  # the edit's result

        # She moved in the body pass anyway: the edit is kept as it was.
        async def moved(_a: bytes, _b: bytes) -> bool:
            return True

        monkeypatch.setattr(comfy_runner, "_moved", moved)
        before = len(fake.prompts)
        r = await user.post("/api/comfy/advanced", json=body)
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        sent = [p["prompt"] for p in list(fake.prompts.values())[before:]]
        assert len(sent) == 3  # edit, body pass (kept as it was), detail
        # No body words: no second pass.
        before = len(fake.prompts)
        r = await user.post(
            "/api/comfy/advanced", json={**body, "prompt": "Give her long straight black hair."}
        )
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        assert len(fake.prompts) == before + 2  # edit, detail
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


async def test_reflection_gets_a_second_pass(user: Api, monkeypatch):
    from app.services import faces

    fake = FakeComfy(
        files={**KREA_FILES, "loras": [*KREA_FILES["loras"], EDIT_FILE]},
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02

    async def main_face(_data: bytes):
        return (30.0, 30.0, 40.0, 40.0)

    async def others(_data: bytes, min_score: float = 0.6):
        return [(0.9, 0.4)]  # her reflection at the right edge

    monkeypatch.setattr(faces, "face_box", main_face)
    monkeypatch.setattr(faces, "other_faces", others)
    try:
        conn = await _ready_connection(user)
        version_id, lora_file = await _character(user, conn["id"])
        fake.files["loras"].append(lora_file)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(100, 100), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her smile.",
            "lora": {"version_id": version_id, "strength": 1.0},
        }
        before = len(fake.prompts)
        r = await user.post("/api/comfy/advanced", json=body)
        gen = await wait_done(user, r.json()["id"])
        assert gen["status"] == "completed", gen
        assert len(fake.prompts) == before + 3  # edit, reflection, face step (no SAM: no detail)
        second = _sent(fake)[-2]
        assert second["4"]["inputs"]["prompt"].startswith("The person at the right edge")
        assert "Make her smile." in second["4"]["inputs"]["prompt"]
        assert second["21"]["inputs"]["lora_name"] == lora_file
        assert second["10"]["inputs"]["image"] in fake.uploads  # the first result
        # Her own face comes back from the first pass (the second redraws the whole photo).
        held: list[bytes] = []

        async def face_at(data: bytes):
            held.append(data)
            return (10.0, 10.0, 20.0, 20.0), (64, 80)

        monkeypatch.setattr(comfy_runner, "_face_at", face_at)
        r = await user.post("/api/comfy/advanced", json=body)
        assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
        assert held  # the face was looked up on the first pass's result
        # Switched off: one pass.
        before = len(fake.prompts)
        r = await user.post("/api/comfy/advanced", json={**body, "all_faces": False})
        await wait_done(user, r.json()["id"])
        assert len(fake.prompts) == before + 2
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


def test_places_in_words():
    assert k2edit.place(0.9, 0.5) == "right edge" and k2edit.place(0.1, 0.5) == "left edge"
    assert k2edit.place(0.5, 0.5) == "middle" and k2edit.place(0.5, 0.1) == "top"


def test_face_paste_with_a_box_takes_only_the_face():
    img = Image.new("RGB", (400, 400), (0, 0, 255))
    patch = _png(Image.new("RGB", (1024, 1024), (255, 0, 0)))
    out = Image.open(
        io.BytesIO(k2edit.face_paste(_png(img), patch, (100, 100, 220), None, (170, 160, 80, 100)))
    ).convert("RGB")
    assert out.getpixel((210, 210))[0] > 240  # the face
    assert out.getpixel((120, 300)) == (0, 0, 255)  # inside the square, beside the face
    # Face only: a face at the square's edge (the square moved inside the photo) is taken
    # whole; the square's own centred ellipse left its far side as it was (a seam).
    edge = (100, 100, 220, 10)  # square 100..320; face 250..310 wide
    face = (250, 170, 60, 80)
    old_way = Image.open(io.BytesIO(k2edit.face_paste(_png(img), patch, edge[:3]))).convert("RGB")
    new_way = Image.open(
        io.BytesIO(k2edit.face_paste(_png(img), patch, edge[:3], None, face, k2edit.FACE_ONLY_GROW))
    ).convert("RGB")
    assert old_way.getpixel((305, 210))[0] < 128 and new_way.getpixel((305, 210))[0] > 200
    assert new_way.getpixel((130, 130)) == (0, 0, 255)  # hair beside the face stays


def test_figure_is_let_go_gently():
    wf = k2edit.build_edit(
        image="a", prompt="x", width=64, height=64, seed=1, edit_lora="general_v2", region="figure"
    )
    assert wf["13"]["inputs"]["ref_boost"] == k2edit.REGION_BOOSTS["figure"]
    words = sorted(n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode")
    assert words == ["breasts", "butt", "face", "hips", "thighs", "waist"]


def test_moved_tells_a_new_pose():
    size = (1000, 1000)
    assert not k2edit.moved((400, 200, 100, 120), (410, 205, 105, 118), size, size)
    assert k2edit.moved((200, 200, 100, 120), (450, 150, 100, 120), size, size)  # shifted
    assert k2edit.moved((400, 200, 100, 120), (400, 200, 200, 240), size, size)  # bigger
    # Another output size, the same place: not moved.
    assert not k2edit.moved((400, 200, 100, 120), (800, 400, 200, 240), size, (2000, 2000))


def test_turn_and_unturn_put_a_level_face_back():
    from PIL import ImageDraw

    data = _png(Image.new("RGB", (300, 400), (0, 0, 255)))
    turned = Image.open(io.BytesIO(k2edit.turn(data, 45))).convert("RGB")
    assert turned.size != (300, 400)  # grown: no corner is cut off
    ImageDraw.Draw(turned).rectangle((150, 180, 250, 280), fill=(255, 0, 0))  # the new face
    out = Image.open(
        io.BytesIO(k2edit.unturn(data, _png(turned), 45, (150, 180, 100, 100)))
    ).convert("RGB")
    assert out.size == (300, 400)
    reds = [(x, y) for x in range(300) for y in range(400) if out.getpixel((x, y))[0] > 200]
    assert reds and out.getpixel((2, 2)) == (0, 0, 255) == out.getpixel((297, 397))


async def test_a_face_found_only_turned_is_levelled(monkeypatch):
    from app.services import faces

    data = _png(Image.new("RGB", (300, 400), (0, 0, 255)))
    level = k2edit.turn(data, 45)

    async def not_as_it_is(image: bytes):
        return None

    async def found(image: bytes):  # surest of the level turn
        return (10.0, 10.0, 50.0, 50.0), 0.9 if image == level else 0.7

    monkeypatch.setattr(faces, "face_box", not_as_it_is)
    monkeypatch.setattr(faces, "face_box_score", found)
    image, angle, box = await comfy_runner._level_face(data)
    assert angle == 45 and image == level

    # A back view has no face: what YuNet finds only turned and unsure of (soles, a bottom,
    # 0.60-0.70) is no face, and no face is drawn there.
    async def unsure(image: bytes):
        return (10.0, 10.0, 50.0, 50.0), 0.7

    monkeypatch.setattr(faces, "face_box_score", unsure)
    assert await comfy_runner._level_face(data) is None

    # Found as it is but surer upside down (lying on her back): turned 180; a level face stays.
    photo = Image.new("RGB", (300, 400), (0, 0, 255))
    photo.paste((255, 0, 0), (0, 0, 100, 100))
    data = _png(photo)
    flipped = k2edit.turn(data, 180)

    async def anywhere(image: bytes):
        return (10.0, 10.0, 50.0, 50.0)

    async def upside_down(image: bytes):
        return (10.0, 10.0, 50.0, 50.0), 0.92 if image == flipped else 0.85

    async def level_face(image: bytes):
        return (10.0, 10.0, 50.0, 50.0), 0.82 if image == flipped else 0.94

    monkeypatch.setattr(faces, "face_box", anywhere)
    monkeypatch.setattr(faces, "face_box_score", upside_down)
    image, angle, _ = await comfy_runner._level_face(data)
    assert angle == 180 and image == flipped
    monkeypatch.setattr(faces, "face_box_score", level_face)
    image, angle, _ = await comfy_runner._level_face(data)
    assert angle == 0 and image == data


def test_nude_words_and_colours():
    for text in ("completely nude", "naked on a bed", "nahá žena", "nude body", "nude topless"):
        assert k2edit.asks_for_nude(text), text
    for text in ("nude lipstick", "wearing nude heels", "a nude bodysuit", "nude bra", "nahor"):
        assert not k2edit.asks_for_nude(text), text
    assert k2edit.names_vulva("showing her pussy") and k2edit.names_vulva("ukáž pičku")
    assert not k2edit.names_vulva("pick up a picture of a pickup")


def test_front_selfies():
    assert k2edit.front_selfie("squatting, holding the phone for a front selfie")
    assert k2edit.front_selfie("mirror selfie v kúpeľni")
    for text in (
        "rear selfie from behind",
        "looking back over her shoulder at the phone",
        "a beach",
    ):
        assert not k2edit.front_selfie(text), text


def test_clothes_leave_her_face_free_with_a_character():
    args = dict(image="a", prompt="x", width=64, height=64, seed=1, edit_lora="general_v2")
    plain = k2edit.build_edit(**args, region="clothes", hold_rest=True)
    assert plain["13"]["inputs"]["ref_boost_mask"] == ["46", 0]  # the face held with the rest
    hers = k2edit.build_edit(
        **args, region="clothes", hold_rest=True, loras=[("her.safetensors", 1.0)]
    )
    assert hers["13"]["inputs"]["ref_boost_mask"] == ["47", 0]
    assert hers["47"]["inputs"]["destination"] == ["46", 0]
    assert hers["47"]["inputs"]["source"] == ["81", 0]  # SAM's face
    assert hers["47"]["inputs"]["operation"] == "subtract"


async def test_body_edit_uses_half_the_character_lora(user: Api, monkeypatch):
    from app.services import faces

    async def no_face(_data: bytes):
        return None

    monkeypatch.setattr(faces, "face_box", no_face)
    monkeypatch.setattr(faces, "face_box_score", no_face)
    fake = FakeComfy(
        files={
            **KREA_FILES,
            "loras": [*KREA_FILES["loras"], EDIT_FILE],
            "checkpoints": [k2edit.SAM3.filename],
        },
        custom_nodes=list(k2edit.NODES),
    )
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    try:
        conn = await _ready_connection(user)
        lora, lora_file = await _character(user, conn["id"])
        fake.files["loras"].append(lora_file)
        ref = (
            await user.post(
                "/api/reference-images",
                files={"file": ("me.png", png_bytes(64, 80), "image/png")},
            )
        ).json()
        body = {
            "connection_id": conn["id"],
            "image": {"kind": "reference", "id": ref["id"]},
            "prompt": "Make her bust much smaller.",
            "lora": {"version_id": lora, "strength": 0.8},
        }
        strengths = {}
        for area, part in (("body", "bust"), ("clothes", None)):
            r = await user.post(
                "/api/comfy/advanced", json={**body, "area": area, "body_part": part}
            )
            assert (await wait_done(user, r.json()["id"]))["status"] == "completed"
            sent = _sent(fake)[-1]
            strengths[area] = [
                n["inputs"]["strength_model"]
                for n in sent.values()
                if n["class_type"] == "LoraLoaderModelOnly"
                and n["inputs"]["lora_name"] != EDIT_FILE
            ]
        assert strengths == {"body": [0.8 * k2edit.BODY_LORA], "clothes": [0.8]}
    finally:
        ComfyClient.transport = None
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()
