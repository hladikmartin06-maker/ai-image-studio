"""Generate › Advanced: plan → Krea 2 generation or masked edits → checks → corrections, on a
fake ComfyUI and a fake AI provider. The built-in Krea 2 workflow, Enhance photo and LoRA
training are only read; a snapshot proves their graphs did not change."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import select

from app.advanced import compiler, crop, engine, graphs
from app.ai.providers.openai_compat import OpenAICompatibleProvider
from app.ai.registry import AI_PROVIDERS, set_adapter
from app.comfy import anypaint, enhance, finish, krea2, pose_control
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.db import SessionLocal
from app.models import AdvancedJob, Generation, Image
from app.services import faces, pose
from tests.ai_fakes import AI_KEY, FakeAI
from tests.conftest import Api, make_user
from tests.fake_comfy_server import ENHANCE_FILES, KREA_FILES, FakeComfy
from tests.fakes import png_bytes
from tests.test_faces import FakeEngine


class PlanningAI(FakeAI):
    """Answers the planner and the checker with what each test sets."""

    def __init__(self) -> None:
        super().__init__()
        self.plans: list[dict] = []
        self.checks: list[dict] = []

    def _openai(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/models"):
            body = json.loads(request.content)
            system = json.dumps(body.get("messages", [])[:1])
            queue = self.plans if "You plan requests" in system else self.checks
            self.reply = json.dumps(queue.pop(0) if len(queue) > 1 else queue[0])
        return super()._openai(request)


@pytest.fixture
def comfy(tmp_path: Path):
    (tmp_path / "ComfyUI" / "models" / "loras").mkdir(parents=True)
    files = {k: list(KREA_FILES.get(k, [])) + ENHANCE_FILES.get(k, []) for k in KREA_FILES}
    files["loras"].append(anypaint.LORA.filename)  # installed with one click
    files["loras"].append(pose_control.LORA.filename)
    files["loras"].append(finish.SKIN_LORA.filename)
    fake = FakeComfy(files=files, models_dir=str(tmp_path / "ComfyUI"))
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    engine.POLL = 0.01
    yield fake
    ComfyClient.transport = None


@pytest.fixture
def ai():
    fake = PlanningAI()
    set_adapter(
        "openai",
        OpenAICompatibleProvider(
            "https://api.openai.fake/v1",
            use_max_completion_tokens=True,
            reasoning_effort_prefixes=("gpt-5",),
            transport=fake.openai_transport(),
        ),
    )
    yield fake
    for key in AI_PROVIDERS:
        set_adapter(key, None)


@pytest.fixture
def face_engine(monkeypatch):
    fake = FakeEngine()

    async def get_engine():
        return fake

    monkeypatch.setattr(faces, "_get_engine", get_engine)
    faces._cache.clear()
    return fake


async def connect(api: Api, tmp_path: Path) -> dict:
    r = await api.post(
        "/api/comfy/connections",
        json={
            "name": "This Mac",
            "kind": "local",
            "base_url": "http://127.0.0.1:8188",
            "models_dir": str(tmp_path / "ComfyUI"),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


async def run(api: Api, body: dict) -> dict:
    r = await api.post("/api/advanced/jobs", json=body)
    assert r.status_code == 201, r.text
    await engine.drain()
    job = (await api.get(f"/api/advanced/jobs/{r.json()['id']}")).json()
    return job


# Input limits ComfyUI v0.35.0 enforces (read from the owner's GPU's /object_info): the fake
# ComfyUI does not, so every graph sent is checked here (a sigma of 15.5 was refused).
LIMITS = {
    ("ImageBlur", "blur_radius"): (1, 31),
    ("ImageBlur", "sigma"): (0.1, 10.0),
    ("SolidMask", "value"): (0.0, 1.0),
    ("ImageBlend", "blend_factor"): (0.0, 1.0),
    ("KSampler", "denoise"): (0.0, 1.0),
    ("SAM3_Detect", "threshold"): (0.0, 1.0),
}


def sent_graphs(fake: FakeComfy) -> list[dict]:
    graphs_sent = [p["prompt"] for p in fake.prompts.values()]
    for graph in graphs_sent:
        for node in graph.values():
            for name, value in node["inputs"].items():
                low, high = LIMITS.get((node["class_type"], name), (value, value))
                assert low <= value <= high, (node["class_type"], name, value)
    return graphs_sent


async def photo(api: Api) -> dict:
    data = png_bytes(64, 80, (200, 40, 40))
    r = await api.post("/api/reference-images", files={"file": ("me.png", data, "image/png")})
    assert r.status_code in (200, 201), r.text
    return r.json()


# --- TEST 10: the protected workflows are unchanged ----------------------------------------


def _hash(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()


def test_built_in_krea2_and_enhance_graphs_are_unchanged():
    """Snapshots of the original Krea 2 (text and photo) and Enhance photo graphs, taken
    before the Advanced engine was added (2026-11-04): the engine builds its own graphs and
    never changes these."""
    base = {
        "unet": "krea2_turbo_fp8_scaled.safetensors",
        "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "prompt": "a woman in a park",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "seed": 7,
        "steps": 8,
        "cfg": 1.0,
        "batch": 1,
        "loras": [("x.safetensors", 1.0)],
    }
    turbo = krea2.build_workflow(variant="turbo", **base)
    raw = krea2.build_workflow(
        variant="raw", **{**base, "steps": 52, "cfg": 3.5, "negative_prompt": "blur"}
    )
    photo_graph = krea2.build_workflow(
        variant="turbo", **base, image=krea2.INPUT_PLACEHOLDER, denoise=0.55
    )
    upscale = enhance.build_workflow(
        image=enhance.INPUT_PLACEHOLDER, strength=0.3, face_only=True, upscale=2.0, seed=7
    )
    assert {
        "turbo": _hash(turbo),
        "raw": _hash(raw),
        "photo": _hash(photo_graph),
        "enhance": _hash(upscale),
    } == SNAPSHOTS


SNAPSHOTS = {
    "turbo": "7967e6f04484c836549a7d8d388d6c1e521f6f7fb080fed202b9bb7a5b7a028b",
    "raw": "205ac46c577714496f34251f4ebc256d935f41b7ca117b80c91e82c989e94270",
    "photo": "df40dc4d86a7894ad5e176257f06d2f62647e9123f29abb23f459f73d7e57e01",
    "enhance": "4c165f4b8b6ae91bc42112e01061d23b7b1bb066c6341e7e5315781878e5e396",
}


# --- TEST 1: text to image on Krea 2 Turbo, the minimum of stages -------------------------


async def test_text_to_image_on_turbo_with_only_the_needed_stages(user: Api, comfy, tmp_path):
    conn = await connect(user, tmp_path)
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "text-to-image",
            "variant": "turbo",
            "prompt": "photorealistic portrait of a woman in a cafe, soft window light",
            "analysis": False,
            "quality": False,
            "corrections": 0,
        },
    )
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "generate"]
    assert all(s["status"] == "completed" for s in job["stages"])
    assert job["result"]["thumb_url"].endswith("/thumb") and job["spec"]["source"] == "rules"
    graph = sent_graphs(comfy)[-1]
    kinds = {n["class_type"] for n in graph.values()}
    # The engine's own text-to-image graph: Krea 2 Turbo (8 steps, CFG 1, no negative).
    assert graph["9"]["inputs"]["filename_prefix"] == "advanced"
    assert graph["7"]["inputs"]["steps"] == 8 and graph["7"]["inputs"]["cfg"] == 1.0
    # No seed given: one is chosen when the request is made and kept, so it can be repeated.
    assert graph["7"]["inputs"]["seed"] == job["options"]["first_seed"]
    assert job["options"]["seed"] is None
    # The result is an image like everywhere else (download, viewer, delete).
    assert job["result"]["download_url"] == f"/api/images/{job['result']['id']}/download"
    assert job["result"]["in_library"] is True and job["result"]["comfy"]["advanced"]
    assert [i["id"] for i in job["images"]] == [job["result"]["id"]]
    assert job["steps_running"] == 0
    assert "ConditioningZeroOut" in kinds and "SAM3_Detect" not in kinds
    assert (
        graph["4"]["inputs"]["text"]
        == "photorealistic portrait of a woman in a cafe, soft window light"
    )
    # The run is an ordinary ComfyUI generation, linked to the job.
    async with SessionLocal() as db:
        gen = await db.scalar(select(Generation).where(Generation.provider == "comfyui"))
    assert gen.comfy_meta["advanced"]["job_id"] == job["id"]
    assert gen.comfy_meta["workflow"]["name"] == "Advanced: Krea 2"
    # The browser sees the link too (the step does not raise its own "image ready" toast).
    shown = (await user.get(f"/api/generations/{gen.id}")).json()
    assert shown["comfy"]["advanced"] == {"job_id": job["id"], "stage": "generate"}
    listed = (await user.get("/api/advanced/jobs")).json()["items"]
    assert listed[0]["id"] == job["id"] and "unet" not in listed[0]["options"]


# --- TEST 2 + 3: Krea 2 RAW with a trained (imported) RAW LoRA, plan and check by the AI ---


async def test_raw_with_a_lora_planned_and_checked(user: Api, comfy, ai, tmp_path, face_engine):
    from tests.test_loras import chunked, safetensors

    upload = await chunked(
        user,
        "lena.safetensors",
        safetensors({"training_info": json.dumps({"studio_architecture": "krea2"})}),
    )
    lora = (
        await user.post(
            "/api/loras/import",
            json={
                "upload_id": upload,
                "name": "Lena",
                "trigger_word": "lnx",
                "class_word": "woman",
                "declared_variant": "raw",
                "adult_confirmed": True,
            },
        )
    ).json()
    await user.put("/api/ai/providers/openai", json={"api_key": AI_KEY})
    conn = await connect(user, tmp_path)
    ai.plans = [
        {
            "summary": "A portrait of the character with closed lips.",
            "preserve": [],
            "modify": [],
            "prompt": (
                "close-up portrait of a woman, lips closed, teeth visible, dark brown high ponytail"
            ),
            "negative": "",
            "edits": [],
            "checks": [
                {
                    "id": "mouth",
                    "question": "Are the lips closed?",
                    "kind": "modify",
                    "target": "mouth",
                },
                {
                    "id": "hair",
                    "question": "Is the hair a dark brown high ponytail?",
                    "kind": "modify",
                    "target": "hair",
                },
            ],
        }
    ]
    ai.checks = [
        {
            "checks": [
                {"id": "mouth", "passed": True, "note": "closed"},
                {"id": "hair", "passed": True, "note": "ok"},
            ],
            "artifacts": "",
        }
    ]
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "text-to-image",
            "variant": "raw",
            "prompt": "portrait of her with closed lips and a dark brown high ponytail",
            "lora": {"version_id": lora["preferred_version_id"], "strength": 0.9},
            "analysis": True,
            "quality": True,
            "corrections": 2,
        },
    )
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "generate", "check"]
    # Closed lips: the contradiction "teeth visible" never reaches Krea 2.
    graph = sent_graphs(comfy)[-1]
    text = graph["4"]["inputs"]["text"]
    assert text.startswith("lnx, close-up portrait") and "teeth" not in text
    assert graph["7"]["inputs"]["steps"] == 52 and graph["7"]["inputs"]["cfg"] == 3.5
    assert graph["5"]["class_type"] == "CLIPTextEncode"  # RAW has a negative prompt
    lora_node = next(n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora_node["inputs"]["strength_model"] == 0.9
    assert (tmp_path / "ComfyUI" / "models" / "loras" / lora_node["inputs"]["lora_name"]).is_file()
    report = job["reports"][0]
    assert [c["passed"] for c in report["checks"]] == [True, True]
    assert report["ai"]["provider"] == "openai" and report["changed_area"] is None
    # A Turbo-only LoRA is never loaded into RAW.
    upload = await chunked(
        user,
        "t.safetensors",
        safetensors({"training_info": json.dumps({"studio_architecture": "krea2"})}),
    )
    turbo_lora = (
        await user.post(
            "/api/loras/import",
            json={
                "upload_id": upload,
                "name": "T",
                "trigger_word": "tbx",
                "declared_variant": "turbo",
                "adult_confirmed": True,
            },
        )
    ).json()
    r = await user.post(
        "/api/advanced/jobs",
        json={
            "connection_id": conn["id"],
            "mode": "text-to-image",
            "variant": "raw",
            "prompt": "portrait",
            "lora": {"version_id": turbo_lora["preferred_version_id"]},
            "analysis": False,
            "quality": False,
            "corrections": 0,
        },
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "lora_incompatible"


# --- TEST 4, 5, 7, 8: change only the hair and the mouth, check, correct what failed --------


async def test_edit_hair_and_mouth_corrects_only_what_failed(
    user: Api, comfy, ai, tmp_path, face_engine, monkeypatch
):
    # The face found on the Mac: 16 × 20 px at (24, 20) of the 64 × 80 photo.
    seen: list[Path] = []

    async def face_box(path: Path):
        seen.append(path)
        return (24.0, 20.0, 16.0, 20.0, 64, 80)

    monkeypatch.setattr(crop, "face_box", face_box)
    await user.put("/api/ai/providers/openai", json={"api_key": AI_KEY})
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    ai.plans = [
        {
            "summary": "Dark brown hair and a closed mouth; everything else kept.",
            "preserve": ["facial identity", "pose", "outfit", "background", "lighting"],
            "modify": ["hair colour dark brown", "mouth closed"],
            "prompt": "",
            "negative": "",
            "edits": [
                {
                    "target": "hair",
                    "region": "hair",
                    "prompt": "photo of a woman with dark brown hair in a park",
                },
                {
                    "target": "mouth",
                    "region": "mouth and lips",
                    "prompt": "photo of a woman in a park, lips closed, tongue out",
                },
            ],
            "checks": [
                {
                    "id": "hair",
                    "question": "Is the hair dark brown?",
                    "kind": "modify",
                    "target": "hair",
                },
                {
                    "id": "mouth",
                    "question": "Is the mouth closed?",
                    "kind": "modify",
                    "target": "mouth",
                },
                {
                    "id": "bg",
                    "question": "Is the background the same as in the original?",
                    "kind": "preserve",
                    "target": "background",
                },
            ],
        }
    ]
    ai.checks = [
        {
            "checks": [
                {"id": "hair", "passed": True},
                {"id": "mouth", "passed": False, "note": "slightly open"},
                {"id": "bg", "passed": True},
            ],
            "artifacts": "",
        },
        {
            "checks": [
                {"id": "hair", "passed": True},
                {"id": "mouth", "passed": True},
                {"id": "bg", "passed": True},
            ],
            "artifacts": "",
        },
    ]
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "edit",
            "variant": "turbo",
            "prompt": (
                "Change her hair colour to dark brown and close her mouth; keep everything else."
            ),
            "image": {"kind": "reference", "id": ref["id"]},
            "analysis": True,
            "quality": True,
            "corrections": 2,
        },
    )
    assert job["status"] == "completed", job
    keys = [s["key"] for s in job["stages"]]
    # No enlarging first (no upscaler in Advanced since 2026-11-06), even for this 64 × 80.
    assert keys == ["plan", "edit1", "edit2", "check", "fix1-mouth", "check2"]
    graphs_sent = sent_graphs(comfy)
    hair, mouth, fix = graphs_sent[-3:]
    assert len(graphs_sent) == 3
    # Only the region is redrawn, with Krea 2 AnyPaint, and pasted back into the photo.
    for graph, region in ((hair, "hair:2"), (mouth, "mouth and lips"), (fix, "mouth and lips")):
        kinds = {n["class_type"] for n in graph.values()}
        assert {
            "SAM3_Detect",
            "Krea2AnyPaintPrepare",
            "Krea2AnyPaintEncode",
            "Krea2AnyPaintModelPatch",
            "ImageCompositeMasked",
            "LoadImage",
        } <= kinds
        assert "DifferentialDiffusion" not in kinds and "SetLatentNoiseMask" not in kinds
        assert graph["31"]["inputs"]["text"] == region
        # AnyPaint's LoRA first at its authors' 1.0, then the model patch (K/V cache on).
        assert graph["20"]["inputs"]["lora_name"] == anypaint.LORA.filename
        assert graph["20"]["inputs"]["strength_model"] == 1.0
        assert graph["97"]["class_type"] == "Krea2AnyPaintModelPatch"
        assert graph["97"]["inputs"]["kv_cache"] is True
        assert graph["95"]["inputs"]["reference_max_edge"] == 384
        assert graph["95"]["inputs"]["boundary_redraw_px"] == 32
        assert graph["96"]["inputs"]["vlm_reference"] is True
        assert graph["96"]["inputs"]["semantic_reference"] == ["95", 0]
        assert graph["96"]["inputs"]["known_image"] == ["95", 1]
        assert graph["96"]["inputs"]["keep_mask"] == ["95", 3]
        sampler = graph["7"]["inputs"]
        assert sampler["model"] == ["97", 0] and sampler["positive"] == ["96", 0]
        assert sampler["latent_image"] == ["96", 1] and sampler["steps"] == 8
        # Pasted back 40 px past the mask (AnyPaint's 32 px blend band and 8 more).
        paste = graph[graph["38"]["inputs"]["mask"][0]]
        image = graph[paste["inputs"]["image"][0]]
        grown = graph[graph[image["inputs"]["image"][0]]["inputs"]["mask"][0]]
        assert grown["class_type"] == "GrowMask" and grown["inputs"]["expand"] == 40
        assert grown["inputs"]["mask"] == graph["95"]["inputs"]["generated_mask"]
    # Hair colour: the plan named "dark brown", so the hair is tinted with it first (only
    # what SAM 3.1 found, multiplied so the strands keep their light); the colour leads the
    # prompt. No extra blend towards the flat colour: on platinum hair it left a flat patch
    # without strands (owner's test, 2026-11-05).
    assert hair["40"]["class_type"] == "EmptyImage"
    assert hair["40"]["inputs"]["color"] == 0x281B14
    assert hair["41"]["inputs"] == {
        "image1": ["11", 0],
        "image2": ["40", 0],
        "blend_factor": 1.0,
        "blend_mode": "multiply",
    }
    assert "43" not in hair
    assert hair["42"]["inputs"]["source"] == ["41", 0]
    assert hair["95"]["inputs"]["source"] == ["42", 0]
    # The eyes and brows (SAM 3.1, grown 8 px) are taken out of the hair: out of the grown
    # mask that is redrawn and out of what is tinted, so the hair redraw leaves them alone.
    assert hair["70"]["inputs"]["text"] == "eyes and eyebrows:2"
    assert hair["72"]["inputs"] == {"mask": ["71", 0], "expand": 8, "tapered_corners": True}
    assert hair["47"]["inputs"]["destination"] == ["33", 0]
    assert hair["47"]["inputs"]["source"] == ["65", 0]
    assert hair["47"]["inputs"]["operation"] == "subtract"
    # The face is left alone too, except where SAM 3.1 found hair over it (bangs).
    assert hair["61"]["inputs"]["text"] == "face:2"
    assert hair["64"]["inputs"]["destination"] == ["63", 0]
    assert hair["64"]["inputs"]["source"] == ["56", 0]
    assert hair["65"]["inputs"]["destination"] == ["72", 0]
    assert hair["65"]["inputs"]["source"] == ["64", 0]
    assert hair["95"]["inputs"]["generated_mask"] == ["47", 0]
    # Every head of hair of the photo's people (a woman and her reflection: two), only on
    # what "person" and "reflection of a person" found, grown 8 px.
    assert hair["53"]["inputs"]["text"] == "person"
    assert hair["45"]["inputs"]["text"] == "reflection of a person"
    assert hair["48"]["inputs"]["destination"] == ["54", 0]
    assert hair["56"]["inputs"] == {
        "destination": ["32", 0],
        "source": ["55", 0],
        "x": 0,
        "y": 0,
        "operation": "and",
    }
    assert hair["33"]["inputs"]["mask"] == ["56", 0]
    assert hair["49"]["inputs"]["destination"] == ["56", 0]
    # The tint goes a little past what SAM 3.1 found (8 px) and fades out over 12 px.
    tint_mask = hair[hair["42"]["inputs"]["mask"][0]]
    assert tint_mask["class_type"] == "ImageToMask"
    grown = next(
        n
        for n in hair.values()
        if n["class_type"] == "GrowMask" and n["inputs"]["mask"] == ["49", 0]
    )
    assert grown["inputs"]["expand"] == 8
    assert "71" not in mouth and mouth["95"]["inputs"]["generated_mask"] == ["33", 0]
    assert hair["33"]["inputs"]["expand"] == 32 and mouth["33"]["inputs"]["expand"] == 16
    assert hair["96"]["inputs"]["prompt"].startswith(
        "very dark brown, almost black hair, one even natural colour from roots to ends"
    )
    assert hair["7"]["inputs"]["denoise"] == 0.85 and mouth["7"]["inputs"]["denoise"] == 0.7
    assert "50" not in hair  # hair is not a face region: no crop
    # The mouth is redrawn in a square around the face, enlarged to 1024 px, and pasted back
    # at the same place inside the mask; no tint (no colour change).
    left, top, side = (
        mouth["50"]["inputs"]["x"],
        mouth["50"]["inputs"]["y"],
        mouth["50"]["inputs"]["width"],
    )
    assert mouth["50"]["class_type"] == "ImageCrop" and mouth["50"]["inputs"]["height"] == side
    assert mouth["51"]["inputs"]["width"] == 1024 and mouth["51"]["inputs"]["height"] == 1024
    assert mouth["32"]["inputs"]["image"] == ["51", 0]
    assert mouth["95"]["inputs"]["source"] == ["51", 0]
    assert mouth["52"]["inputs"]["width"] == side
    assert mouth["38"]["inputs"]["source"] == ["52", 0]
    assert (mouth["38"]["inputs"]["x"], mouth["38"]["inputs"]["y"]) == (left, top)
    assert mouth["38"]["inputs"]["destination"] == ["11", 0] and "40" not in mouth
    width = mouth["11"]["inputs"]["width"]
    height = mouth["11"]["inputs"]["height"]
    assert left >= 0 and left + side <= width and top >= 0 and top + side <= height
    # The face's centre stays the crop's centre (within rounding).
    assert abs(left + side / 2 - 32 * width / 64) <= 1
    assert abs(top + side / 2 - 30 * height / 80) <= 1
    assert len(seen) == 2  # the mouth edit and its correction
    # The correction: only the mouth again, a bit stronger, another seed.
    assert fix["7"]["inputs"]["denoise"] == pytest.approx(0.8)
    assert fix["7"]["inputs"]["seed"] != mouth["7"]["inputs"]["seed"]
    assert fix["50"]["class_type"] == "ImageCrop"
    # Closed mouth: "tongue out" was removed from the plan before Krea 2 saw it, and the
    # closed mouth comes first (Turbo has no negative prompt).
    assert "tongue" not in mouth["96"]["inputs"]["prompt"]
    assert mouth["96"]["inputs"]["prompt"].startswith("closed mouth, lips gently together, ")
    assert len(job["reports"]) == 2 and job["reports"][0]["checks"][1]["passed"] is False
    first = job["reports"][0]
    assert first["identity_reference"] == "original" and first["changed_area"] is not None
    # The best attempt is the result; the steps' other images left the library.
    assert job["result"]["image_id"] == job["reports"][1]["image_id"]
    async with SessionLocal() as db:
        images = (await db.scalars(select(Image))).all()
    in_library = {str(i.id): i.in_library for i in images}
    assert in_library[job["result"]["image_id"]] is True
    # The two first edits left the library.
    assert sum(1 for v in in_library.values() if v is False) == 2
    listed = [i["id"] for i in job["images"]]
    assert len(listed) == 3 and listed[-1] == job["result"]["id"]
    assert [i["in_library"] for i in job["images"]] == [False, False, True]
    files = [Path(p) for i in images if str(i.id) in listed for p in (i.file_path, i.thumb_path)]
    assert len(files) == 6
    from app.services import storage

    assert all(storage.absolute(str(f)).exists() for f in files)

    # Deleting the request removes every image it made and its steps' history entries.
    other = await make_user("advanced-delete-other")
    try:
        assert (await other.delete(f"/api/advanced/jobs/{job['id']}")).status_code == 404
    finally:
        await other.client.aclose()
    assert (await user.delete(f"/api/advanced/jobs/{job['id']}")).json() == {"ok": True}
    assert (await user.get(f"/api/advanced/jobs/{job['id']}")).status_code == 404
    async with SessionLocal() as db:
        ids = [uuid.UUID(i) for i in listed]
        assert (await db.scalars(select(Image).where(Image.id.in_(ids)))).all() == []
        steps = (
            await db.scalars(
                select(Generation).where(
                    Generation.comfy_meta["advanced"]["job_id"].astext == job["id"]
                )
            )
        ).all()
    assert steps == []
    assert not any(storage.absolute(str(f)).exists() for f in files)
    # The photo it started from is the user's own and stays.
    assert (await user.get(f"/api/reference-images/{ref['id']}/file")).status_code == 200


# --- without AI: the regions the user chose, measured checks only ------------------------


async def test_edit_without_ai_uses_the_chosen_region(user: Api, comfy, tmp_path, face_engine):
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    body = {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": "photo of a woman with dark brown hair",
        "image": {"kind": "reference", "id": ref["id"]},
        "analysis": False,
        "quality": True,
        "corrections": 0,
    }
    # Nothing to change without analysis, a chosen region or a text that names one.
    vague = {**body, "prompt": "a nicer photo"}
    assert (await user.post("/api/advanced/jobs", json=vague)).status_code == 400
    job = await run(user, {**body, "targets": ["hair"], "denoise": 0.75})
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "edit1", "check"]
    # The region chosen and the one the text asks for are the same: one edit.
    assert [e["target"] for e in job["spec"]["edits"]] == ["hair"]
    graph = sent_graphs(comfy)[-1]
    assert graph["31"]["inputs"]["text"] == "hair:2" and graph["7"]["inputs"]["denoise"] == 0.75
    report = job["reports"][0]
    assert report["checks"] == [] and report["ai"] is None
    assert isinstance(report["changed_area"], float) and report["identity"] is not None


# --- a photo and a character LoRA: the person becomes the LoRA's character ---------------


async def imported_lora(api: Api, trigger: str = "lmx") -> dict:
    from tests.test_loras import chunked, safetensors

    upload = await chunked(
        api,
        f"{trigger}.safetensors",
        safetensors({"training_info": json.dumps({"studio_architecture": "krea2"})}),
    )
    r = await api.post(
        "/api/loras/import",
        json={
            "upload_id": upload,
            "name": "Taila",
            "trigger_word": trigger,
            "class_word": "woman",
            "declared_variant": "turbo",
            "adult_confirmed": True,
        },
    )
    assert r.status_code in (200, 201), r.text
    return r.json()


async def test_identity_redraws_the_figure_then_tries_the_face_last(
    user: Api, comfy, tmp_path, face_engine, monkeypatch
):
    from app.advanced import quality

    async def face_box(path: Path):
        return (28.0, 30.0, 6.0, 8.0, 64, 80)  # a small face in a body shot

    monkeypatch.setattr(crop, "face_box", face_box)
    # Three tries of the face (the LoRA's photos are 60 % alike, so the goal is 51 %): no
    # match, then short of the goal, then a match. The final check measures it once more.
    measured: list[list[Path]] = []
    scores = [0.21, 0.45, 0.55, 0.55]

    async def identity(result: Path, references: list[Path]):
        measured.append(references)
        return scores.pop(0), 0.6

    monkeypatch.setattr(quality, "identity", identity)
    lora = await imported_lora(user)
    for i, colour in enumerate([(200, 40, 40), (201, 41, 40)]):
        files = {"file": (f"t{i}.png", png_bytes(64, 64, colour), "image/png")}
        r = await user.post(f"/api/loras/{lora['id']}/dataset/upload", files=files)
        assert r.status_code == 201, r.text
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    body = {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": OWNER_PROMPT,
        "image": {"kind": "reference", "id": ref["id"]},
        "lora": {"version_id": lora["preferred_version_id"], "strength": 1.0},
        "analysis": False,
        "quality": False,  # the owner's setting: checks off, no corrections
        "corrections": 0,
        "resolution": "2K",
    }
    # The plan before anything runs shows the same steps: the figure, the buttocks the text
    # names, the hair, the face last.
    shown = (await user.post("/api/advanced/plan", json={**body, "targets": []})).json()
    assert [e["target"] for e in shown["edits"]] == ["body", "hips", "hair", "face"]
    # No region chosen: with a LoRA the person becomes its character anyway.
    job = await run(user, body)
    assert job["status"] == "completed", job
    assert job["options"]["identity"] is True and "prepare" not in job["options"]
    keys = [s["key"] for s in job["stages"]]
    assert keys == ["plan", "edit1", "edit2", "edit3", "edit4", "edit4-try2", "edit4-try3", "check"]
    labels = [s["label"] for s in job["stages"] if s["key"].startswith("edit")]
    # Every step says it ran on AnyPaint (never another method in its place).
    assert labels[:3] == ["body · AnyPaint", "hips · AnyPaint", "hair · AnyPaint"]
    figure, hips, hair, face, try2, try3 = sent_graphs(comfy)[-6:]
    # The figure: SAM 3.1 "body", grown 48 px so a new shape can pass the old outline, the
    # face, head, hair, hands and feet left alone (they pin the pose), redrawn by AnyPaint at
    # 0.85 with the LoRA at 0.65 of its strength.
    assert figure["31"]["inputs"]["text"] == "body"
    assert figure["33"]["inputs"]["expand"] == 48
    assert [figure[str(70 + 4 * i)]["inputs"]["text"] for i in range(5)] == [
        "face:2",
        "head:2",
        "hair:2",
        "hands:4",
        "feet:4",
    ]
    assert figure["95"]["inputs"]["generated_mask"] == ["47", 0]
    assert figure["7"]["inputs"]["denoise"] == 0.85
    loras = [
        (n["inputs"]["lora_name"], n["inputs"]["strength_model"])
        for k, n in sorted(figure.items(), key=lambda kv: int(kv[0]))
        if n["class_type"] == "LoraLoaderModelOnly"
    ]
    assert loras[0] == (anypaint.LORA.filename, 1.0) and loras[1][1] == 0.65
    text = figure["96"]["inputs"]["prompt"]
    assert text.startswith("lmx, lmxx2 woman, Preserve the exact original body position")
    assert "buttock size" in text and "facial" not in text
    # The buttocks: their own region and sentence, the hands left alone.
    assert hips["31"]["inputs"]["text"] == "hips and buttocks"
    assert hips["70"]["inputs"]["text"] == "hands:4" and hips["33"]["inputs"]["expand"] == 48
    assert "buttock size" in hips["96"]["inputs"]["prompt"]
    # The hair with its colour.
    assert hair["31"]["inputs"]["text"] == "hair:2"
    assert hair["41"]["inputs"]["blend_mode"] == "multiply"
    # Then the face last, in its enlarged crop, gently, with the LoRA and its own sentences.
    assert face["31"]["inputs"]["text"] == "face" and face["50"]["class_type"] == "ImageCrop"
    assert face["7"]["inputs"]["denoise"] == try3["7"]["inputs"]["denoise"] == 0.65
    assert len({face["7"]["inputs"]["seed"], try2["7"]["inputs"]["seed"]}) == 2
    face_loras = [
        n["inputs"]["strength_model"]
        for k, n in sorted(face.items(), key=lambda kv: int(kv[0]))
        if n["class_type"] == "LoraLoaderModelOnly"
    ]
    assert face_loras == [1.0, 1.0]  # AnyPaint, then the character at its full strength
    assert face["96"]["inputs"]["prompt"].startswith("lmx, lmxx2 woman, fully transform")
    assert "buttock" not in face["96"]["inputs"]["prompt"]
    # The mouth is redrawn with the face (kept at half strength it left the photo's lips on
    # the new face, 2026-11-06).
    assert all("70" not in g for g in (face, try2, try3))
    # Every try starts from the same image (the one with the new hair); the result is the
    # third try, the first that reached the goal.
    async with SessionLocal() as db:
        inputs = {
            g.comfy_meta["advanced"]["stage"]: (g.comfy_meta.get("input") or {}).get("id")
            for g in await db.scalars(
                select(Generation).where(
                    Generation.comfy_meta["advanced"]["job_id"].astext == job["id"]
                )
            )
        }
    stage = {s["key"]: s for s in job["stages"]}
    assert inputs["edit4"] == inputs["edit4-try2"] == inputs["edit4-try3"]
    assert inputs["edit4"] == stage["edit3"]["image_id"]
    assert job["result"]["image_id"] == stage["edit4-try3"]["image_id"]
    assert all(len(refs) == 2 for refs in measured)  # the LoRA's photos, not the original
    # The final check measures it; nothing is redrawn after it.
    (report,) = job["reports"]
    match = next(c for c in report["checks"] if c["id"] == "face_match")
    assert match["passed"] is True and report["identity_goal"] == 0.51
    assert report["ai"] is None  # no AI checks with the checks off
    assert not scores
    # Identity off: the LoRA only helps; the regions the text names are redrawn one by one.
    r = await user.post("/api/advanced/jobs", json={**body, "identity": False})
    assert r.status_code == 201


def test_region_prompts_leave_out_what_to_keep():
    from app.advanced.compiler import _changes, _region_text, region_prompt

    selfie = (
        "lmxx2 woman, transform the woman's facial identity to match the trained LoRA identity. "
        "Completely change her platinum blonde hair color to deep natural dark brown. "
        "Preserve the exact original facial expression, eyes looking upward, open mouth and "
        "tongue sticking out. Keep the original grey top and background."
    )
    face = region_prompt(_changes(selfie), "face")
    # "Preserve … open mouth and tongue" made Krea 2 open a closed mouth and draw a tongue
    # (the owner's car, 2026-11-05); the expression is kept by the photo itself.
    assert face == (
        "lmxx2 woman, transform the woman's facial identity to match the trained LoRA identity"
    )
    # The hair is redrawn strongly: its "keep" sentence says what it should look like.
    wavy = selfie + " Preserve the exact original wavy hairstyle and bangs."
    hair = _region_text(wavy, "hair")
    assert hair.startswith("lmxx2 woman, Completely change her platinum blonde hair color")
    assert "wavy hairstyle and bangs" in hair
    assert "tongue" not in hair and "background" not in hair
    assert _region_text(wavy, "face") == face
    # Nothing but "keep" sentences: the request as written.
    assert _changes("Keep her pose.") == "Keep her pose."


# --- the checks before anything runs ------------------------------------------------------


async def test_refusals_before_anything_runs(user: Api, comfy, tmp_path):
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    base = {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": "dark brown hair",
        "image": {"kind": "reference", "id": ref["id"]},
        "analysis": True,
        "quality": True,
        "corrections": 1,
    }
    r = await user.post("/api/advanced/jobs", json=base)
    assert r.status_code == 409 and r.json()["error"]["code"] == "ai_not_connected"
    r = await user.post(
        "/api/advanced/jobs",
        json={**base, "analysis": False, "targets": ["hair"], "quality": False},
    )
    assert r.status_code == 400  # corrections need the check
    r = await user.post(
        "/api/advanced/jobs",
        json={
            **base,
            "mode": "text-to-image",
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "prompt": "nude woman",
        },
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "adult_off"
    r = await user.post(
        "/api/advanced/jobs",
        json={
            **base,
            "mode": "text-to-image",
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "resolution": "4K",
        },
    )
    assert r.status_code == 400
    caps = (
        await user.get("/api/advanced/capabilities", params={"connection_id": conn["id"]})
    ).json()
    assert (
        caps["sam3"] is True
        and caps["variants"] == {"turbo": True, "raw": True}
        and caps["ai"] is False
    )
    assert caps["sam3_file"]["url"].startswith("https://huggingface.co/")
    assert comfy.prompts == {}
    other = await make_user("advanced-other")
    try:
        r = await other.get("/api/advanced/capabilities", params={"connection_id": conn["id"]})
        assert r.status_code == 404
    finally:
        await other.client.aclose()


async def test_edit_needs_sam3(user: Api, tmp_path):
    files = {k: list(v) for k, v in KREA_FILES.items()}
    fake = FakeComfy(files=files)
    ComfyClient.transport = fake.transport()
    try:
        conn = await connect(user, tmp_path)
        ref = await photo(user)
        r = await user.post(
            "/api/advanced/jobs",
            json={
                "connection_id": conn["id"],
                "mode": "edit",
                "prompt": "dark brown hair",
                "image": {"kind": "reference", "id": ref["id"]},
                "analysis": False,
                "targets": ["hair"],
                "quality": False,
                "corrections": 0,
            },
        )
        assert r.status_code == 409 and "SAM 3.1" in r.json()["error"]["message"]
    finally:
        ComfyClient.transport = None


# --- cancel, restart --------------------------------------------------------------------


async def test_cancel_and_restart(user: Api, comfy, tmp_path, monkeypatch):
    conn = await connect(user, tmp_path)
    monkeypatch.setattr(engine, "start", lambda _job_id: None)
    r = await user.post(
        "/api/advanced/jobs",
        json={
            "connection_id": conn["id"],
            "mode": "text-to-image",
            "prompt": "a woman in a park",
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "client_request_id": "adv-1",
        },
    )
    job_id = r.json()["id"]
    # The same request twice starts one job.
    again = await user.post(
        "/api/advanced/jobs",
        json={
            "connection_id": conn["id"],
            "mode": "text-to-image",
            "prompt": "a woman in a park",
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "client_request_id": "adv-1",
        },
    )
    assert again.json()["id"] == job_id
    r = await user.delete(f"/api/advanced/jobs/{job_id}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "in_progress"
    assert (await user.post(f"/api/advanced/jobs/{job_id}/cancel")).status_code == 200
    await engine._run(uuid.UUID(job_id))
    job = (await user.get(f"/api/advanced/jobs/{job_id}")).json()
    assert job["status"] == "cancelled"
    assert next(s for s in job["stages"] if s["key"] == "generate")["status"] == "cancelled"
    # A job running when the app stops ends as "interrupted" (not continued, not resent).
    async with SessionLocal() as db:
        row = await db.get(AdvancedJob, uuid.UUID(job_id))
        row.status = "running"
        await db.commit()
    await engine.recover()
    job = (await user.get(f"/api/advanced/jobs/{job_id}")).json()
    assert job["status"] == "interrupted"


# --- hair colour and face crops (pure functions) -------------------------------------------


def test_hair_colour_from_the_request():
    from app.advanced.compiler import hair_colour

    assert hair_colour("dark brown hair, lips closed") == (
        "very dark brown, almost black",
        "#281B14",
    )
    # The new colour, not the old one.
    assert hair_colour("change her hair from blonde to dark brown")[1] == "#281B14"
    assert hair_colour("Zmeň jej farbu vlasov na tmavohnedú a zatvor jej ústa.")[1] == "#281B14"
    assert hair_colour("zmeň jej vlasy z blond na čierne")[0] == "black"
    # Only a colour next to the hair counts.
    assert hair_colour("a blue dress and long wavy blonde hair")[0] == "blonde"
    assert hair_colour("a blue dress, close her mouth") is None
    # The colour it changes to, not the one it has (owner's test, 2026-11-05).
    assert hair_colour(
        "Completely change her platinum blonde hair color to deep natural dark brown, "
        "with no remaining blonde sections."
    ) == ("very dark brown, almost black", "#281B14")
    assert hair_colour("Change her platinum blonde hair colour to jet black")[0] == "jet black"
    assert hair_colour("prefarbi jej vlasy do čierna")[0] == "black"
    assert hair_colour("Walk to the red car. Her hair is blonde.")[0] == "blonde"
    assert hair_colour("make the hair platinum blonde")[0] == "platinum blonde"


def test_colour_in_the_plan():
    from app.advanced.compiler import Edit, Spec, _edit_fields, validate

    def plan(**edit) -> Spec:
        return Spec(mode="edit", edits=[Edit(**edit)])

    spec = validate(
        plan(target="hair", prompt="a woman in a park"),
        user_prompt="dark brown hair please",
        adult=False,
    )
    assert spec.edits[0].color == "#281B14"
    assert spec.edits[0].prompt == (
        "very dark brown, almost black hair, one even natural colour from roots to ends "
        "without highlights, a woman in a park"
    )
    # The user removed the colour in the plan: it stays removed.
    spec = validate(
        plan(target="hair", prompt="dark brown hair", color=""), user_prompt="x", adult=False
    )
    assert spec.edits[0].color == ""
    # The AI's own value wins; a colour on a region without colours is dropped.
    spec = validate(
        plan(target="hair", prompt="auburn hair", color="#7B3522"), user_prompt="x", adult=False
    )
    assert spec.edits[0].color == "#7B3522"
    spec = validate(
        plan(target="mouth", prompt="closed mouth", color="#FF0000"), user_prompt="x", adult=False
    )
    assert spec.edits[0].color is None
    # A colour from the AI that is not "#RRGGBB" counts as none given.
    assert "color" not in _edit_fields({"target": "hair", "prompt": "x", "color": "brown"})
    assert _edit_fields({"target": "hair", "prompt": "x", "color": "#3a281d"})["color"] == (
        "#3A281D"
    )


def test_face_crop_geometry():
    # A small face in a full-body photo: a square around it, inside the image.
    left, top, side = crop.plan_crop((420.0, 100.0, 160.0, 240.0, 1000, 1500), 896, 1344)
    assert side == int(240 * 1344 / 1500 * crop.CONTEXT)
    assert 0 <= left <= 896 - side and 0 <= top <= 1344 - side
    # A tiny face still gets a square of at least MIN_SIDE px.
    assert crop.plan_crop((480.0, 100.0, 20.0, 25.0, 1000, 1500), 896, 1344)[2] == (crop.MIN_SIDE)
    # A face at the edge: the square is moved inside, not cut.
    left, top, side = crop.plan_crop((0.0, 0.0, 60.0, 80.0, 1000, 1500), 896, 1344)
    assert (left, top) == (0, 0)
    # A close-up (the face fills the photo) or no face: the whole photo as before.
    assert crop.plan_crop((100.0, 100.0, 700.0, 900.0, 1000, 1500), 896, 1344) is None
    assert crop.plan_crop(None, 896, 1344) is None


def test_hair_tint_keeps_the_strands_and_grows_with_each_correction():
    from app.advanced import graphs

    args = {
        "variant": "turbo",
        "unet": "u",
        "text_encoder": "t",
        "vae": "v",
        "prompt": "dark brown hair",
        "negative": "",
        "width": 896,
        "height": 1152,
        "seed": 1,
        "steps": 8,
        "cfg": 1.0,
        "loras": [],
        "denoise": 0.75,
        "region": "hair",
        "colour": "#3A281D",
    }
    first = graphs.edit(**args)
    again = graphs.edit(**args, attempt=1)
    # Darker colours multiply (blonde → dark brown keeps every strand), lighter ones screen.
    assert first["41"]["inputs"]["blend_mode"] == "multiply"
    assert first["41"]["inputs"]["blend_factor"] == 1.0  # darker: multiplied fully
    assert again["41"]["inputs"]["blend_factor"] == 1.0
    assert graphs.tint("#3A281D", 5) == ("multiply", 1.0)
    assert graphs.tint("#E5DCC8") == ("screen", 0.55)  # platinum blonde
    assert graphs.tint("#1B1716")[0] == "multiply" and graphs.tint("#CDAE74")[0] == "screen"
    assert first["40"]["inputs"]["width"] == 896 and first["40"]["inputs"]["height"] == 1152
    # Without a region nothing is tinted or masked (the whole photo is image to image).
    whole = graphs.edit(**{**args, "region": None})
    assert "40" not in whole and whole["12"]["inputs"]["pixels"] == ["11", 0]
    assert "60" not in whole and whole["7"]["inputs"]["model"] == ["1", 0]
    known = (*graphs.NODES_EDIT, *graphs.NODES_ANYPAINT)
    assert all(n["class_type"] in known for n in first.values())


# --- regions from the text and the LoRA identity (pure functions) --------------------------

OWNER_PROMPT = (
    "lmxx2 woman, fully transform the woman's facial identity to match the trained LoRA, "
    "with accurate facial features and natural realistic skin texture. Change her hair to "
    "long, voluminous dark brown hair styled in a high ponytail with soft face-framing "
    "strands. Preserve the exact original body position, lying pose, arm placement, camera "
    "angle, framing, outfit and indoor background. Match the body proportions and buttock "
    "size to the trained LoRA identity, keeping natural and realistic anatomy. Maintain the "
    "original lighting and photographic composition. Photorealistic, highly detailed, "
    "natural skin texture, realistic proportions, seamless identity transformation."
)


def test_regions_from_the_text():
    from app.advanced.compiler import by_rules, detected_targets, validate

    assert set(detected_targets(OWNER_PROMPT)) == {"face", "hair", "body", "hips"}
    # A sentence that keeps something changes nothing; a named hair colour is a change.
    assert detected_targets("Keep her hair and face exactly as they are.") == []
    assert detected_targets("Preserve the body position and the hair.") == []
    assert detected_targets("a woman with platinum blonde hair on a beach") == ["hair"]
    assert detected_targets("Photorealistic, realistic proportions, sharp focus.") == []
    assert detected_targets("Zmeň jej farbu vlasov na tmavohnedú a zatvor jej ústa.") == [
        "hair",
        "mouth",
    ]
    # Clothing, mouth, eyes and expression are found too; a colour belongs to its clause.
    assert detected_targets("change her dress to a black dress") == ["clothing"]
    assert detected_targets("close her mouth") == ["mouth"]
    assert detected_targets("make her smile") == ["expression"]
    assert detected_targets("change her eye colour to blue") == ["eyes"]
    text = "change her hair to dark brown and her red dress to a black dress"
    spec = validate(by_rules(mode="edit", prompt=text, targets=[]), user_prompt=text, adult=False)
    assert [(e.target, e.color) for e in spec.edits] == [
        ("clothing", "#1B1716"),
        ("hair", "#281B14"),
    ]
    # SAM 3.1 looks for every piece of the garment named; its new colour is said first.
    clothing = spec.edits[0]
    assert clothing.region == "dress:3"
    assert clothing.prompt.startswith("black dress, ")
    assert "hair" not in clothing.prompt
    # With the LoRA's character the face step draws the mouth, eyes and expression.
    text = "lmxx2 woman, close her mouth and make her smile. Change her hair to black."
    spec = validate(
        by_rules(mode="edit", prompt=text, targets=[]),
        user_prompt=text,
        adult=False,
        identity=True,
    )
    assert [e.target for e in spec.edits] == ["body", "hair", "face"]
    assert "close her mouth" in spec.edits[-1].prompt
    # A new background is asked in the text; keeping it is not a change.
    assert detected_targets("Change the background to a sunny beach with palm trees.") == [
        "background"
    ]
    assert detected_targets("Zmeň pozadie na pláž.") == ["background"]
    assert detected_targets("Keep the room and the background.") == []
    # Without analysis the chosen "whole" stays first and the text's regions follow, in the
    # owner's order: the figure and its parts, the hair, the face last.
    spec = by_rules(mode="edit", prompt=OWNER_PROMPT, targets=["whole"])
    assert [e.target for e in spec.edits] == ["whole", "body", "hips", "hair", "face"]
    # The figure's parts, in English and Slovak.
    assert detected_targets("Make her breasts bigger and her hips wider.") == ["breasts", "hips"]
    assert detected_targets("Zväčši jej zadok a prsia.") == ["breasts", "hips"]
    assert detected_targets("Make her waist slimmer.") == ["body"]
    assert spec.edits[1].region == "body"
    # Nothing chosen and nothing found: the whole photo, as before.
    assert [e.target for e in by_rules(mode="edit", prompt="a woman", targets=[]).edits] == [
        "whole"
    ]


def test_identity_plan_redraws_the_figure_first_and_the_face_last():
    from app.advanced.compiler import Edit, Spec, by_rules, validate

    spec = validate(
        by_rules(mode="edit", prompt=OWNER_PROMPT, targets=["whole"]),
        user_prompt=OWNER_PROMPT,
        adult=False,
        identity=True,
    )
    # "whole" is not redrawn: the figure with the LoRA, the buttocks the text names, the hair
    # with its colour, the face last.
    assert [e.target for e in spec.edits] == ["body", "hips", "hair", "face"]
    assert [e.region for e in spec.edits] == ["body", "hips and buttocks", "hair", "face"]
    assert spec.edits[2].color == "#281B14"
    # Each region with the request's own sentences about it (not the hair over the face).
    body, hips, hair, face = spec.edits
    assert hips.prompt.startswith("lmxx2 woman, Match the body proportions and buttock size")
    assert face.prompt.startswith("lmxx2 woman, fully transform the woman's facial identity")
    assert "hair" not in face.prompt and "buttock" not in face.prompt
    assert hair.prompt.startswith("very dark brown, almost black hair, one even natural colour")
    assert "lmxx2 woman, Change her hair to long" in hair.prompt
    assert "facial identity" not in hair.prompt
    # The figure: the pose, outfit and proportions sentences, never the hair or the face.
    assert body.prompt == (
        "lmxx2 woman, Preserve the exact original body position, lying pose, arm placement, "
        "camera angle, framing, outfit and indoor background. Match the body proportions and "
        "buttock size to the trained LoRA identity, keeping natural and realistic anatomy. "
        "Photorealistic, highly detailed, natural skin texture, realistic proportions, seamless "
        "identity transformation"
    )
    # A plan the user edited keeps its own figure and face prompts, the figure first.
    edited = Spec(
        mode="edit",
        edits=[
            Edit(target="hair", prompt="red hair"),
            Edit(target="face", prompt="my face prompt"),
            Edit(target="body", prompt="my body prompt"),
        ],
    )
    spec = validate(edited, user_prompt="x", adult=False, identity=True)
    assert [(e.target, e.prompt.split(",")[0]) for e in spec.edits] == [
        ("body", "my body prompt"),
        ("hair", "red hair"),
        ("face", "my face prompt"),
    ]
    # Without identity the plan's own order stays.
    plain = Spec(
        mode="edit", edits=[Edit(target="hair", prompt="a"), Edit(target="body", prompt="b")]
    )
    assert [e.target for e in validate(plain, user_prompt="x", adult=False).edits] == [
        "hair",
        "body",
    ]


async def test_improve_background_redraws_all_but_the_person(user: Api, comfy, tmp_path):
    from app.advanced.compiler import Spec, validate

    # The plan: the figure first, then the background, sharper (0.5), with its own sentences.
    spec = validate(
        Spec(mode="edit", edits=[]),
        user_prompt=OWNER_PROMPT,
        adult=False,
        identity=True,
        improve_background=True,
    )
    assert [e.target for e in spec.edits] == ["body", "background", "face"]
    background = spec.edits[1]
    assert background.denoise == 0.5 and background.region == "background"
    assert background.prompt == "the same place, sharp, detailed, natural light, photorealistic"
    assert "lmxx2" not in background.prompt and "hair" not in background.prompt
    # A new background asked in the text wins (redrawn fully), nothing is added.
    asked = validate(
        Spec(mode="edit", edits=[]),
        user_prompt="Change the background to a sunny beach.",
        adult=False,
        improve_background=True,
    )
    assert [e.target for e in asked.edits] == ["background"]
    # A job with only the switch: no LoRA, no trigger, the people kept by an inverted mask.
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "edit",
            "variant": "turbo",
            "prompt": "a woman in a kitchen",
            "image": {"kind": "reference", "id": ref["id"]},
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "improve_background": True,
        },
    )
    assert job["status"] == "completed", job
    assert job["options"]["improve_background"] is True
    assert [e["target"] for e in job["spec"]["edits"]] == ["background"]
    graph = sent_graphs(comfy)[-1]
    assert graph["31"]["inputs"]["text"] == "person"
    assert graph["45"]["inputs"]["text"] == "reflection of a person"  # people in a mirror too
    # Everything but the people (their exact outline), redrawn by AnyPaint without the
    # character LoRA; the people pasted back as they were, pixel by pixel.
    assert "43" not in graph
    assert graph["44"] == {"class_type": "InvertMask", "inputs": {"mask": ["48", 0]}}
    assert graph["33"]["inputs"]["mask"] == ["44", 0] and graph["33"]["inputs"]["expand"] == 0
    assert graph["7"]["inputs"]["denoise"] == 0.5
    loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert [n["inputs"]["lora_name"] for n in loras] == [anypaint.LORA.filename]
    paste = graph[graph["38"]["inputs"]["mask"][0]]
    blur = graph[paste["inputs"]["image"][0]]
    assert blur["inputs"]["blur_radius"] == 4
    assert graph[blur["inputs"]["image"][0]]["inputs"]["mask"] == ["33", 0]


async def test_text_to_image_with_a_lora_redraws_a_face_short_of_the_goal(
    user: Api, comfy, tmp_path, face_engine, monkeypatch
):
    """A face far from the camera is a few latent pixels: when it is not yet as alike to the
    LoRA's photos as the goal, it is redrawn in its crop like an edit's LoRA face, and the most
    alike image — the generated one included — is the result."""
    from app.advanced import quality

    async def face_box(path: Path):
        return (28.0, 30.0, 6.0, 8.0, 64, 80)

    monkeypatch.setattr(crop, "face_box", face_box)
    scores: list[float] = []

    async def identity(result: Path, references: list[Path]):
        return scores.pop(0), 0.6  # the goal is 0.85 × 0.6 = 0.51

    monkeypatch.setattr(quality, "identity", identity)
    lora = await imported_lora(user)
    for i, colour in enumerate([(200, 40, 40), (201, 41, 40)]):
        files = {"file": (f"t{i}.png", png_bytes(64, 64, colour), "image/png")}
        r = await user.post(f"/api/loras/{lora['id']}/dataset/upload", files=files)
        assert r.status_code == 201, r.text
    conn = await connect(user, tmp_path)
    body = {
        "connection_id": conn["id"],
        "mode": "text-to-image",
        "variant": "turbo",
        "prompt": "lmxx2 woman walking on a city street at night, full body",
        "lora": {"version_id": lora["preferred_version_id"], "strength": 1.0},
        "analysis": False,
        "quality": False,
        "corrections": 0,
    }

    def stage_image(job: dict, key: str) -> str:
        return next(s["image_id"] for s in job["stages"] if s["key"] == key)

    # Short of the goal, then closer, then a match: the third image is the result.
    scores[:] = [0.40, 0.45, 0.55]
    job = await run(user, body)
    assert job["status"] == "completed", job
    keys = [s["key"] for s in job["stages"]]
    assert keys == ["plan", "generate", "edit1", "edit1-try2"]
    face = sent_graphs(comfy)[-2]
    assert face["31"]["inputs"]["text"] == "face" and face["50"]["class_type"] == "ImageCrop"
    assert face["7"]["inputs"]["denoise"] == 0.65
    # The text says nothing about the face: only who it is (never the street or the clothes).
    assert face["4"]["inputs"]["text"] == "lmx, the woman"
    assert any(n["class_type"] == "LoraLoaderModelOnly" for n in face.values())
    assert job["result"]["image_id"] == stage_image(job, "edit1-try2")
    # Every try less alike than the generated face: the generated image stays the result.
    scores[:] = [0.45, 0.30, 0.20, 0.25]
    job = await run(user, body)
    assert [s["key"] for s in job["stages"]][-3:] == ["edit1", "edit1-try2", "edit1-try3"]
    assert job["result"]["image_id"] == stage_image(job, "generate")
    # A try only a little more alike than the first (under 0.08): the first stays (the most
    # alike face tends to the LoRA's neutral look and lost a smile).
    scores[:] = [0.40, 0.44, 0.46, 0.45]
    job = await run(user, body)
    assert job["result"]["image_id"] == stage_image(job, "generate")
    # No score SFace can stand behind (all under its threshold 0.363): no choice by chance,
    # the generated image stays.
    scores[:] = [0.2, 0.3, 0.1, 0.25]
    job = await run(user, body)
    assert job["result"]["image_id"] == stage_image(job, "generate")
    # Already alike enough (or no face to measure): no face step.
    scores[:] = [0.6]
    job = await run(user, body)
    assert [s["key"] for s in job["stages"]] == ["plan", "generate"]


def test_the_main_face_is_the_largest_sure_one():
    """YuNet rows (x, y, w, h, 10 landmark values, score): the lift photo's frontal face in
    the mirror (0.93) wins over the larger cut-off profile at the edge (0.76); among faces
    as sure as each other the largest wins."""

    def row(w: float, h: float, score: float) -> list[float]:
        return [0.0, 0.0, w, h, *([0.0] * 10), score]

    mirror, profile = row(142, 173, 0.93), row(128, 246, 0.759)
    assert faces.main_face([profile, mirror]) is mirror
    big, small = row(200, 200, 0.9), row(100, 100, 0.95)
    assert faces.main_face([small, big]) is big


async def test_without_anypaint_nothing_runs_another_way(user: Api, comfy, tmp_path):
    """Region redraws need Krea 2 AnyPaint: without its nodes or its LoRA the job is refused
    with what is missing — never run with another method instead."""
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    body = {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": "Change her hair color to red.",
        "image": {"kind": "reference", "id": ref["id"]},
        "analysis": False,
        "quality": False,
        "corrections": 0,
    }
    comfy.files["loras"].remove(anypaint.LORA.filename)
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    caps = (await user.get(f"/api/advanced/capabilities?connection_id={conn['id']}")).json()
    assert caps["anypaint"]["missing_files"] == [anypaint.LORA.filename]
    assert caps["anypaint_file"]["size"] == 228_587_752
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "missing_models"
    assert "AnyPaint" in r.json()["error"]["message"]
    comfy.files["loras"].append(anypaint.LORA.filename)
    removed = [n for n in comfy.nodes if n.startswith("Krea2AnyPaint")]
    for name in removed:
        comfy.nodes.remove(name)
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "missing_nodes"
    assert "rent it again" in r.json()["error"]["message"]
    comfy.nodes.extend(removed)


async def test_breasts_bigger_and_light_unified_after_a_new_background(
    user: Api, comfy, tmp_path, face_engine
):
    """The figure's parts: their own SAM 3.1 region, grown 48 px past the outline, the hands
    left alone, the LoRA at 0.65 of its strength. "Unify the light" (off by default): after a
    new background the whole photo once more at 0.2 (plain image to image, no AnyPaint)."""
    lora = await imported_lora(user)
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "edit",
            "variant": "turbo",
            "prompt": "Make her breasts bigger. Change the background to a sunny beach.",
            "image": {"kind": "reference", "id": ref["id"]},
            "lora": {"version_id": lora["preferred_version_id"], "strength": 1.0},
            "identity": False,
            "adult": True,
            "analysis": False,
            "quality": False,
            "corrections": 0,
            "unify_light": True,
        },
    )
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "edit1", "edit2", "edit2-light"]
    breasts, background, light = sent_graphs(comfy)[-3:]
    assert breasts["31"]["inputs"]["text"] == "breasts"
    assert breasts["33"]["inputs"]["expand"] == 48 and breasts["70"]["inputs"]["text"] == "hands:4"
    strengths = [
        n["inputs"]["strength_model"]
        for k, n in sorted(breasts.items(), key=lambda kv: int(kv[0]))
        if n["class_type"] == "LoraLoaderModelOnly"
    ]
    assert strengths == [1.0, 0.65]
    assert breasts["7"]["inputs"]["denoise"] == 0.85
    assert background["7"]["inputs"]["denoise"] == 1.0
    # The light: the whole photo at 0.2, the character LoRA, no mask and no AnyPaint.
    kinds = {n["class_type"] for n in light.values()}
    assert "Krea2AnyPaintPrepare" not in kinds and "SAM3_Detect" not in kinds
    assert light["7"]["inputs"]["denoise"] == 0.2
    assert light["12"]["inputs"]["pixels"] == ["11", 0]
    # Off by default: no light step.
    job = await run(
        user,
        {
            "connection_id": conn["id"],
            "mode": "edit",
            "variant": "turbo",
            "prompt": "Change the background to a sunny beach.",
            "image": {"kind": "reference", "id": ref["id"]},
            "analysis": False,
            "quality": False,
            "corrections": 0,
        },
    )
    assert [s["key"] for s in job["stages"]] == ["plan", "edit1"]


def test_anypaint_download_is_pinned_and_detected():
    """One click downloads AnyPaint's LoRA from its pinned Hugging Face revision; the GPU
    agent renames it into place only when its SHA-256 matches."""
    (item,) = anypaint.downloads()
    assert item == {
        "url": "https://huggingface.co/yijunwang2/krea2-anypaint/resolve/"
        "1a9fb37a304c27523939c44fc2b770c11472451b/krea2_anypaint_rank32.safetensors",
        "folder": "loras",
        "filename": "krea2_anypaint_rank32.safetensors",
        "sha256": "3a7d09f6b27f8ead160d340f2f59c11f4ee635c4a1ee87ffd8b1b9f9ba412f7a",
    }
    info = {
        "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[item["filename"]]]}}},
        **{n: {} for n in anypaint.NODES},
    }
    assert anypaint.detect(info)["ready"] is True
    del info["Krea2AnyPaintEncode"]
    state = anypaint.detect(info)
    assert state["ready"] is False and state["missing_nodes"] == ["Krea2AnyPaintEncode"]


# --- phase 2: a pose from another photo (the skeleton on the Mac, the pose LoRA on the GPU) ---
# The pose models are not run here: `pose.main_person` is replaced by a drawn person.


def person(w: int, h: int, *, turn: float = 0.0, size: float = 1.0) -> pose.Person:
    """A standing person filling a w × h photo (COCO-WholeBody points); `turn` moves the face
    sideways (a head turn), `size` scales the person around the photo's centre."""
    u = h / 10 * size
    cx, top = w / 2, h / 2 - 5 * u
    body = {
        0: (turn, 1.0), 1: (turn - 0.3, 0.8), 2: (turn + 0.3, 0.8), 3: (-0.6, 0.9),
        4: (0.6, 0.9), 5: (-1.2, 2), 6: (1.2, 2), 7: (-1.5, 3.5), 8: (1.5, 3.5),
        9: (-1.6, 5), 10: (1.6, 5), 11: (-0.8, 5), 12: (0.8, 5), 13: (-0.8, 7),
        14: (0.8, 7), 15: (-0.8, 9), 16: (0.8, 9),
    }  # fmt: skip
    pts, conf = np.zeros((133, 2)), np.zeros(133)
    for k, (x, y) in body.items():
        pts[k] = (cx + x * u, top + y * u)
        conf[k] = 0.9
    return pose.Person(pts, conf, (cx - 2 * u, top + 0.5 * u, cx + 2 * u, top + 9.5 * u))


def drawn_rows(png: bytes) -> np.ndarray:
    pixels = np.asarray(PILImage.open(io.BytesIO(png)).convert("L"))
    return np.nonzero(pixels.max(1))[0]


def test_skeleton_maps_and_head_geometry():
    """The whole body or only the head and neck, at the size asked; the reference's head
    moved onto the target's neck; the head's square inside the photo; the sheet side by
    side."""
    p = pose.scaled(person(64, 80), 10, 10)  # 640 × 800, the neck at y 160
    body = pose.render(p, 640, 800)
    head = pose.render(p, 640, 800, "head")
    assert PILImage.open(io.BytesIO(body)).size == (640, 800)
    assert drawn_rows(body).max() > 700  # down to the ankles
    assert drawn_rows(head).max() < 170 and drawn_rows(head).min() >= 55
    # The reference (smaller, turned) onto the target's neck, at the target's head size.
    reference = person(64, 80, turn=0.5, size=0.5)
    target = pose.scaled(person(64, 80), 10, 10)
    moved = pose.head_onto(reference, target)
    neck = (moved.points[5] + moved.points[6]) / 2
    assert np.allclose(neck, (target.points[5] + target.points[6]) / 2)
    assert moved.points[0][0] > target.points[0][0]  # the face turned like the reference
    left, top, side = pose.head_square(moved, 640, 800, graphs.HEAD_FACTOR)
    assert left >= 0 and top >= 0 and left + side <= 640 and top + side <= 800
    assert left <= moved.points[0][0] <= left + side and top <= moved.points[0][1] <= top + side
    # No neck in one photo: refused with a message, never guessed.
    hidden = person(64, 80)
    hidden.conf[5] = 0
    with pytest.raises(pose.PoseError):
        pose.head_onto(reference, hidden)
    buf = io.BytesIO()
    PILImage.new("RGB", (30, 40), (200, 0, 0)).save(buf, "PNG")
    sheet = PILImage.open(io.BytesIO(pose.sheet(buf.getvalue(), head)))
    assert sheet.size == (60, 40) and sheet.getpixel((5, 5)) == (200, 0, 0)


def test_pose_lora_download_is_pinned_and_detected():
    (item,) = pose_control.downloads()
    assert item["url"].startswith("https://huggingface.co/thedeoxen/Krea-2-pose-controlnet/")
    assert pose_control.REVISION in item["url"] and len(item["sha256"]) == 64
    info = {
        "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[item["filename"]]]}}},
        **{n: {} for n in (*pose_control.NODES, "FluxKontextMultiReferenceLatentMethod")},
    }
    assert pose_control.detect(info)["ready"] is True
    # The reference-latent node is built into ComfyUI, but an older one lacks it.
    del info["FluxKontextMultiReferenceLatentMethod"]
    state = pose_control.detect(info)
    assert not state["ready"] and state["missing_nodes"] == [
        "FluxKontextMultiReferenceLatentMethod"
    ]


def test_plan_keeps_only_the_pose_step_asked_for():
    """A pose step only with a pose photo and of the part chosen: an edited plan cannot add
    one or change it; it is the first step."""
    text = "Turn her head to the left. Change her hair to red."
    spec = compiler.by_rules(mode="edit", prompt=text, targets=[])
    plain = compiler.validate(spec.model_copy(deep=True), user_prompt=text, adult=False)
    assert all(e.target not in compiler.POSE_TARGETS for e in plain.edits)
    head = compiler.validate(spec.model_copy(deep=True), user_prompt=text, adult=False, pose="head")
    assert head.edits[0].target == "head_pose" and head.edits[0].region == ""
    assert "red" not in head.edits[0].prompt  # only the head's own sentences
    sneaked = head.model_copy(deep=True)
    assert compiler.validate(sneaked, user_prompt=text, adult=False).edits[0].target == "hair"
    head.checks = [compiler.Check(id="turned", question="Is the head turned?", target="head_pose")]
    body = compiler.validate(head.model_copy(deep=True), user_prompt=text, adult=False, pose="body")
    assert body.checks[0].target is None  # a pose is never redrawn as a correction
    assert [e.target for e in body.edits].count("pose") == 1
    assert "head_pose" not in [e.target for e in body.edits] and body.edits[0].target == "pose"


@pytest.fixture
def found(monkeypatch):
    """The person `pose.main_person` finds in each photo (by what the step asks for)."""
    seen: list[str] = []

    async def main_person(path, what: str = "the pose photo"):
        seen.append(what)
        if what == "the pose photo":
            return person(64, 80, turn=0.5)
        return person(64, 80)

    monkeypatch.setattr(pose, "main_person", main_person)
    return seen


def _edit(conn: dict, ref: dict, pose_ref: dict, part: str, prompt: str, **extra) -> dict:
    return {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": prompt,
        "image": {"kind": "reference", "id": ref["id"]},
        "pose_image": {"kind": "reference", "id": pose_ref["id"]},
        "pose_part": part,
        "analysis": False,
        "quality": False,
        "corrections": 0,
        **extra,
    }


async def test_body_pose_draws_the_photo_anew_in_the_reference_pose(
    user: Api, comfy, tmp_path, found
):
    """The whole body: the skeleton (from the pose photo, on this Mac) is the pose LoRA's
    image 1, the photo drawn anew from the text at 1K on Turbo — as the authors' example
    (Ostris Edit patch, the pose LoRA at 1.0, 10 steps, CFG 1). Nothing else is added."""
    conn = await connect(user, tmp_path)
    ref, pose_ref = await photo(user), await photo(user)
    job = await run(user, _edit(conn, ref, pose_ref, "body", "A woman in a red dress in a park."))
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "edit1"]
    assert job["stages"][1]["label"] == "pose · Pose LoRA"
    assert job["options"]["pose"] == {
        "part": "body",
        "source": {
            k: job["options"]["pose"]["source"][k]
            for k in ("kind", "id", "thumb_url", "url", "width", "height")
        },
    }
    assert "file_path" not in job["options"]["pose"]["source"]
    (graph,) = sent_graphs(comfy)
    assert graph["19"]["class_type"] == "Krea2OstrisEditModelPatch"
    assert graph["19"]["inputs"]["kv_cache"] is True
    assert graph["20"]["inputs"]["lora_name"] == pose_control.LORA.filename
    assert graph["20"]["inputs"]["strength_model"] == 1.0
    assert graph["4"]["class_type"] == "TextEncodeKrea2OstrisEdit"
    assert graph["4"]["inputs"]["image1"] == ["10", 0]
    assert graph["13"]["inputs"]["reference_latents_method"] == "index_timestep_zero"
    sampler = graph["7"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["denoise"]) == (10, 1.0, 1.0)
    assert sampler["latent_image"] == ["6", 0] and sampler["positive"] == ["13", 0]
    size = (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"])
    assert size == krea2.size_for_source(64, 80, "1K")  # about 1 MP in the pose photo's shape
    assert "Krea2AnyPaintPrepare" not in {n["class_type"] for n in graph.values()}


async def test_head_pose_redraws_only_the_head_and_blends_its_edge(
    user: Api, comfy, tmp_path, face_engine, found
):
    """Only the head: the reference's head put on this photo's neck, the photo and that
    skeleton sent as one sheet; the head's square redrawn with the pose LoRA and the
    character LoRA after it, then its edge blended by AnyPaint. The face is redrawn last with
    the LoRA (identity)."""
    lora = await imported_lora(user)
    conn = await connect(user, tmp_path)
    ref, pose_ref = await photo(user), await photo(user)
    job = await run(
        user,
        _edit(
            conn,
            ref,
            pose_ref,
            "head",
            "lmx woman looking over her shoulder",
            lora={"version_id": lora["preferred_version_id"], "strength": 1.0},
            identity=True,
        ),
    )
    assert job["status"] == "completed", job
    keys = [s["key"] for s in job["stages"]]
    assert keys[:4] == ["plan", "edit1", "edit1-seam", "edit2"]
    assert found == ["the pose photo", "the photo to edit"]
    labels = {s["key"]: s["label"] for s in job["stages"]}
    assert labels["edit1"] == "head_pose · Pose LoRA" and labels["edit1-seam"] == "seam · AnyPaint"
    head, seam = sent_graphs(comfy)[:2]
    loras = [head[k]["inputs"] for k in ("20", "21")]
    assert loras[0]["lora_name"] == pose_control.LORA.filename and loras[1]["strength_model"] == 1.0
    # The sheet is cut in two: the photo, then the skeleton beside it.
    assert (
        head["15"]["inputs"]["x"] == 0
        and head["16"]["inputs"]["x"] == head["15"]["inputs"]["width"]
    )
    assert head["4"]["inputs"]["image1"] == ["57", 0]
    assert head["7"]["inputs"]["denoise"] == graphs.HEAD_DENOISE
    assert head["7"]["inputs"]["latent_image"] == ["12", 0]
    left, top, side = (head["50"]["inputs"][k] for k in ("x", "y", "width"))
    assert head["38"]["inputs"]["x"] == left and head["38"]["inputs"]["y"] == top
    width, height = head["11"]["inputs"]["width"], head["11"]["inputs"]["height"]
    assert left + side <= width and top + side <= height
    assert "Krea2AnyPaintPrepare" in {n["class_type"] for n in seam.values()}


async def test_pose_refusals_say_what_is_missing(user: Api, comfy, tmp_path, found):
    """Turbo only; the pose LoRA and its nodes must be there; text to image has no pose —
    each refused with what is missing, never run another way."""
    conn = await connect(user, tmp_path)
    ref, pose_ref = await photo(user), await photo(user)
    body = _edit(conn, ref, pose_ref, "body", "A woman in a park.")
    r = await user.post("/api/advanced/jobs", json={**body, "variant": "raw"})
    assert r.status_code == 400 and "Turbo" in r.json()["error"]["message"]
    r = await user.post("/api/advanced/jobs", json={**body, "mode": "text-to-image"})
    assert r.status_code == 400
    comfy.files["loras"].remove(pose_control.LORA.filename)
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    caps = (await user.get(f"/api/advanced/capabilities?connection_id={conn['id']}")).json()
    assert caps["pose"]["missing_files"] == [pose_control.LORA.filename]
    assert caps["pose_file"]["size"] == 228_587_504
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "missing_models"
    comfy.files["loras"].append(pose_control.LORA.filename)
    comfy.nodes.remove("TextEncodeKrea2OstrisEdit")
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "missing_nodes"
    comfy.nodes.append("TextEncodeKrea2OstrisEdit")


async def test_pose_preview_and_a_photo_without_a_person(user: Api, monkeypatch, found):
    pose_ref = await photo(user)
    r = await user.post(
        "/api/advanced/pose-preview",
        json={"image": {"kind": "reference", "id": pose_ref["id"]}, "part": "head"},
    )
    assert r.status_code == 200, r.text
    data = r.json()["image"]
    assert data.startswith("data:image/png;base64,")
    png = base64.b64decode(data.split(",", 1)[1])
    assert (
        PILImage.open(io.BytesIO(png)).size == (64, 80) == (r.json()["width"], r.json()["height"])
    )

    async def nobody(path, what: str = "the pose photo"):
        raise pose.PoseError(f"No person was found in {what}.")

    monkeypatch.setattr(pose, "main_person", nobody)
    r = await user.post(
        "/api/advanced/pose-preview", json={"image": {"kind": "reference", "id": pose_ref["id"]}}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "pose_failed"


# --- phase 3: Finish (the face with the skin LoRA, a likeness check, SeedVR2) ---------------


def test_finish_downloads_are_pinned_and_detected():
    (skin,) = finish.downloads("skin")
    assert skin["url"] == (
        "https://huggingface.co/inlineresearch/skin-lora-krea-2-raw/resolve/"
        "15fbc074697eed816159dbbc99f9c98170301b14/inline-skin-lora-krea-2-raw.safetensors"
    )
    assert skin["sha256"].startswith("7ac516bc") and skin["folder"] == "loras"
    # SeedVR2: the two files Enhance photo uses, from the same place.
    assert [d["url"] for d in finish.downloads("seedvr2")] == [
        enhance.SEEDVR2.url,
        enhance.SEEDVR2_VAE.url,
    ]
    info = {
        "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[skin["filename"]]]}}},
        "UNETLoader": {"input": {"required": {"unet_name": [[enhance.SEEDVR2.filename]]}}},
        "VAELoader": {"input": {"required": {"vae_name": [[enhance.SEEDVR2_VAE.filename]]}}},
        **{n: {} for n in finish.UPSCALE_NODES},
    }
    state = finish.detect(info)
    assert state["skin"]["ready"] and state["seedvr2"]["ready"]
    del info["SeedVR2Conditioning"]
    info["VAELoader"]["input"]["required"]["vae_name"] = [[]]
    state = finish.detect(info)
    assert state["seedvr2"]["missing_nodes"] == ["SeedVR2Conditioning"]
    assert state["seedvr2"]["missing_files"] == [enhance.SEEDVR2_VAE.filename]


async def test_finish_redraws_the_face_with_the_skin_lora_then_enlarges(
    user: Api, comfy, tmp_path, face_engine, monkeypatch
):
    """At the end: the face in its crop at 0.35 with AnyPaint, the character LoRA and the skin
    LoRA (0.6, its trigger first); kept only when SFace finds it as alike as before (within
    0.05); then SeedVR2 enlarges the result (1 step, "lab" colours) and the likeness of the
    enlarged photo is measured. No face found = no face pass."""
    from app.advanced import quality

    box: list = [(28.0, 30.0, 6.0, 8.0, 64, 80)]

    async def face_box(path: Path):
        return box[0]

    monkeypatch.setattr(crop, "face_box", face_box)
    scores: list[float] = []

    async def identity(result: Path, references: list[Path]):
        return scores.pop(0), None

    monkeypatch.setattr(quality, "identity", identity)
    lora = await imported_lora(user)
    conn = await connect(user, tmp_path)
    ref = await photo(user)
    body = {
        "connection_id": conn["id"],
        "mode": "edit",
        "variant": "turbo",
        "prompt": "Change her hair color to red.",
        "image": {"kind": "reference", "id": ref["id"]},
        "lora": {"version_id": lora["preferred_version_id"], "strength": 1.0},
        "identity": False,
        "analysis": False,
        "quality": False,
        "corrections": 0,
        "finish": True,
        "finish_upscale": 2.0,
    }

    def stage(job: dict, key: str) -> dict:
        return next(s for s in job["stages"] if s["key"] == key)

    async def upscaled_from(job: dict) -> str:
        async with SessionLocal() as db:
            gen = await db.get(Generation, uuid.UUID(stage(job, "finish-upscale")["generation_id"]))
            return gen.comfy_meta["input"]["id"]

    # Before 0.60, after the pass 0.57 (within 0.05): kept; the enlarged photo 0.56.
    scores[:] = [0.60, 0.57, 0.56]
    job = await run(user, body)
    assert job["status"] == "completed", job
    keys = [s["key"] for s in job["stages"]]
    assert keys == [
        "plan",
        "edit1",
        "finish-face",
        "finish-check",
        "finish-upscale",
        "finish-likeness",
    ]
    assert job["options"]["finish"] == {"upscale": 2.0}
    assert stage(job, "finish-face")["label"] == "finish · AnyPaint"
    assert stage(job, "finish-check")["likeness"] == {"before": 0.6, "after": 0.57, "kept": True}
    assert stage(job, "finish-likeness")["likeness"] == {"after": 0.56}
    face, upscaled = sent_graphs(comfy)[-2:]
    loras = [
        (n["inputs"]["lora_name"], n["inputs"]["strength_model"])
        for k, n in sorted(face.items(), key=lambda kv: int(kv[0]))
        if n["class_type"] == "LoraLoaderModelOnly"
    ]
    assert loras[0] == (anypaint.LORA.filename, 1.0) and loras[-1] == (
        finish.SKIN_LORA.filename,
        0.6,
    )
    assert len(loras) == 3 and loras[1][1] == 1.0  # the character LoRA between them
    assert face["4"]["inputs"]["text"].startswith("inline-skin-lora, detailed skin texture, lmx")
    assert face["7"]["inputs"]["denoise"] == 0.35 and face["50"]["class_type"] == "ImageCrop"
    assert upscaled["5"]["inputs"]["scale_by"] == 2.0
    assert upscaled["40"]["inputs"]["unet_name"] == enhance.SEEDVR2.filename
    assert upscaled["45"]["inputs"]["steps"] == 1
    assert upscaled["47"]["inputs"]["color_correction_method"] == "lab"
    assert "LoraLoaderModelOnly" not in {n["class_type"] for n in upscaled.values()}
    assert await upscaled_from(job) == stage(job, "finish-face")["image_id"]
    assert job["result"]["image_id"] == stage(job, "finish-upscale")["image_id"]
    # Less alike by more than 0.05: the face pass is not kept, the image before it is enlarged.
    scores[:] = [0.60, 0.50, 0.59]
    job = await run(user, body)
    assert stage(job, "finish-check")["likeness"]["kept"] is False
    assert await upscaled_from(job) == stage(job, "edit1")["image_id"]
    # No face found: no face pass and no check; the photo is still enlarged.
    box[0] = None
    scores[:] = [0.4]
    job = await run(user, body)
    assert job["status"] == "completed", job
    assert stage(job, "finish-face")["status"] == "skipped"
    assert "finish-check" not in [s["key"] for s in job["stages"]]
    assert await upscaled_from(job) == stage(job, "edit1")["image_id"]


async def test_finish_refused_with_what_is_missing(user: Api, comfy, tmp_path):
    conn = await connect(user, tmp_path)
    body = {
        "connection_id": conn["id"],
        "mode": "text-to-image",
        "variant": "turbo",
        "prompt": "A woman in a park.",
        "analysis": False,
        "quality": False,
        "corrections": 0,
        "finish": True,
    }
    r = await user.post("/api/advanced/jobs", json={**body, "finish_upscale": 2.5})
    assert r.status_code == 400
    comfy.files["loras"].remove(finish.SKIN_LORA.filename)
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    caps = (await user.get(f"/api/advanced/capabilities?connection_id={conn['id']}")).json()
    assert caps["finish"]["skin"]["missing_files"] == [finish.SKIN_LORA.filename]
    assert caps["finish_files"]["skin"]["size"] == 191_871_280 and caps["upscales"] == [
        1.5,
        2.0,
        3.0,
    ]
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and "skin LoRA" in r.json()["error"]["message"]
    comfy.files["loras"].append(finish.SKIN_LORA.filename)
    comfy.files["vae"].remove(enhance.SEEDVR2_VAE.filename)
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    r = await user.post("/api/advanced/jobs", json=body)
    assert r.status_code == 409 and "SeedVR2" in r.json()["error"]["message"]
    comfy.files["vae"].append(enhance.SEEDVR2_VAE.filename)
    # Off: none of it is needed.
    r = await user.post("/api/advanced/jobs", json={**body, "finish": False})
    assert r.status_code == 201, r.text
