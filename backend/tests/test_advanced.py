"""Generate › Advanced: plan → Krea 2 generation or masked edits → checks → corrections, on a
fake ComfyUI and a fake AI provider. The built-in Krea 2 workflow, Enhance photo and LoRA
training are only read; a snapshot proves their graphs did not change."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.advanced import crop, engine
from app.ai.providers.openai_compat import OpenAICompatibleProvider
from app.ai.registry import AI_PROVIDERS, set_adapter
from app.comfy import enhance, krea2
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.db import SessionLocal
from app.models import AdvancedJob, Generation, Image
from app.services import faces
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


def sent_graphs(fake: FakeComfy) -> list[dict]:
    return [p["prompt"] for p in fake.prompts.values()]


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
    assert keys == ["plan", "edit1", "edit2", "check", "fix1-mouth", "check2"]
    graphs_sent = sent_graphs(comfy)
    hair, mouth, fix = graphs_sent[-3:]
    # Only the region is redrawn and pasted back into the photo.
    for graph, region in ((hair, "hair"), (mouth, "mouth and lips"), (fix, "mouth and lips")):
        kinds = {n["class_type"] for n in graph.values()}
        assert {"SAM3_Detect", "SetLatentNoiseMask", "ImageCompositeMasked", "LoadImage"} <= kinds
        assert graph["31"]["inputs"]["text"] == region
    # Hair colour: the plan named "dark brown", so the hair is tinted with it first (only
    # inside the mask) and redrawn less; the colour leads the prompt; a wider mask.
    assert hair["40"]["class_type"] == "EmptyImage"
    assert hair["40"]["inputs"]["color"] == 0x3A281D
    assert hair["41"]["inputs"] == {
        "image1": ["11", 0],
        "image2": ["40", 0],
        "blend_factor": 0.55,
        "blend_mode": "normal",
    }
    assert hair["42"]["inputs"]["mask"] == ["33", 0]
    assert hair["12"]["inputs"]["pixels"] == ["42", 0]
    assert hair["33"]["inputs"]["expand"] == 24 and mouth["33"]["inputs"]["expand"] == 16
    assert hair["4"]["inputs"]["text"].startswith("dark brown hair, ")
    assert hair["7"]["inputs"]["denoise"] == 0.75 and mouth["7"]["inputs"]["denoise"] == 0.7
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
    assert mouth["32"]["inputs"]["image"] == ["51", 0] and mouth["12"]["inputs"]["pixels"] == [
        "51",
        0,
    ]
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
    assert "tongue" not in mouth["4"]["inputs"]["text"]
    assert mouth["4"]["inputs"]["text"].startswith("closed mouth, lips gently together, ")
    assert len(job["reports"]) == 2 and job["reports"][0]["checks"][1]["passed"] is False
    first = job["reports"][0]
    assert first["identity_reference"] == "original" and first["changed_area"] is not None
    # The best attempt is the result; the steps' other images left the library.
    assert job["result"]["image_id"] == job["reports"][1]["image_id"]
    async with SessionLocal() as db:
        images = (await db.scalars(select(Image))).all()
    in_library = {str(i.id): i.in_library for i in images}
    assert in_library[job["result"]["image_id"]] is True
    assert sum(1 for v in in_library.values() if v is False) == 2  # the two first edits
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
    # Nothing to change without analysis or a chosen region.
    assert (await user.post("/api/advanced/jobs", json=body)).status_code == 400
    job = await run(user, {**body, "targets": ["hair"], "denoise": 0.75})
    assert job["status"] == "completed", job
    assert [s["key"] for s in job["stages"]] == ["plan", "edit1", "check"]
    graph = sent_graphs(comfy)[-1]
    assert graph["31"]["inputs"]["text"] == "hair" and graph["7"]["inputs"]["denoise"] == 0.75
    report = job["reports"][0]
    assert report["checks"] == [] and report["ai"] is None
    assert isinstance(report["changed_area"], float) and report["identity"] is not None


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

    assert hair_colour("dark brown hair, lips closed") == ("dark brown", "#3A281D")
    # The new colour, not the old one.
    assert hair_colour("change her hair from blonde to dark brown")[0] == "dark brown"
    assert hair_colour("Zmeň jej farbu vlasov na tmavohnedú a zatvor jej ústa.")[0] == (
        "dark brown"
    )
    assert hair_colour("zmeň jej vlasy z blond na čierne")[0] == "black"
    # Only a colour next to the hair counts.
    assert hair_colour("a blue dress and long wavy blonde hair")[0] == "blonde"
    assert hair_colour("a blue dress, close her mouth") is None
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
    assert spec.edits[0].color == "#3A281D"
    assert spec.edits[0].prompt == "dark brown hair, a woman in a park"
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


def test_hair_tint_grows_with_each_correction():
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
        "tint": "#3A281D",
    }
    first = graphs.edit(**args, tint_strength=graphs.TINT)
    again = graphs.edit(**args, tint_strength=graphs.TINT + graphs.TINT_STEP)
    assert first["41"]["inputs"]["blend_factor"] == 0.55
    assert again["41"]["inputs"]["blend_factor"] == 0.7
    assert first["40"]["inputs"]["width"] == 896 and first["40"]["inputs"]["height"] == 1152
    # Without a region nothing is tinted (the whole photo is image to image).
    whole = graphs.edit(**{**args, "region": None})
    assert "40" not in whole and whole["12"]["inputs"]["pixels"] == ["11", 0]
    assert all(n["class_type"] in graphs.NODES_EDIT for n in first.values())
