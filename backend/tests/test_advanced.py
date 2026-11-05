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
    assert keys == ["plan", "prepare", "edit1", "edit2", "check", "fix1-mouth", "check2"]
    graphs_sent = sent_graphs(comfy)
    prepared, hair, mouth, fix = graphs_sent[-4:]
    # The 64 × 80 photo is first enlarged with Enhance photo (gentlest skin fix, its largest
    # factor: the edit needs far more than 3×).
    assert prepared["24"]["inputs"]["denoise"] == enhance.STRENGTHS[0]
    assert prepared["5"]["inputs"]["scale_by"] == enhance.UPSCALES[-1]
    # Only the region is redrawn and pasted back into the photo.
    for graph, region in ((hair, "hair"), (mouth, "mouth and lips"), (fix, "mouth and lips")):
        kinds = {n["class_type"] for n in graph.values()}
        assert {"SAM3_Detect", "SetLatentNoiseMask", "ImageCompositeMasked", "LoadImage"} <= kinds
        assert graph["31"]["inputs"]["text"] == region
    # Hair colour: the plan named "dark brown", so the hair is tinted with it first (only
    # what SAM 3.1 found, multiplied so the strands keep their light, then half-way to the
    # flat colour so the old highlights go) and redrawn less; the colour leads the prompt;
    # a wider mask with a soft edge redrawn gradually.
    assert hair["40"]["class_type"] == "EmptyImage"
    assert hair["40"]["inputs"]["color"] == 0x3A281D
    assert hair["41"]["inputs"] == {
        "image1": ["11", 0],
        "image2": ["40", 0],
        "blend_factor": 0.85,
        "blend_mode": "multiply",
    }
    assert hair["43"]["inputs"] == {
        "image1": ["41", 0],
        "image2": ["40", 0],
        "blend_factor": 0.5,
        "blend_mode": "normal",
    }
    assert hair["42"]["inputs"]["source"] == ["43", 0]
    assert hair["42"]["inputs"]["mask"] == ["32", 0]
    assert hair["12"]["inputs"]["pixels"] == ["42", 0]
    assert "46" not in hair and "46" not in mouth  # nothing taken out of these regions
    assert hair["33"]["inputs"]["expand"] == 32 and mouth["33"]["inputs"]["expand"] == 16
    for graph in (hair, mouth, fix):
        assert graph["60"]["class_type"] == "DifferentialDiffusion"
        assert graph["7"]["inputs"]["model"] == ["60", 0]
        assert graph["34"]["inputs"]["mask"] == ["37", 0] == graph["38"]["inputs"]["mask"]
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
    # The enlarged photo and the two first edits left the library.
    assert sum(1 for v in in_library.values() if v is False) == 3
    listed = [i["id"] for i in job["images"]]
    assert len(listed) == 4 and listed[-1] == job["result"]["id"]
    assert [i["in_library"] for i in job["images"]] == [False, False, False, True]
    files = [Path(p) for i in images if str(i.id) in listed for p in (i.file_path, i.thumb_path)]
    assert len(files) == 8
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
    assert [s["key"] for s in job["stages"]] == ["plan", "prepare", "edit1", "check"]
    # The region chosen and the one the text asks for are the same: one edit.
    assert [e["target"] for e in job["spec"]["edits"]] == ["hair"]
    graph = sent_graphs(comfy)[-1]
    assert graph["31"]["inputs"]["text"] == "hair" and graph["7"]["inputs"]["denoise"] == 0.75
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


async def test_identity_redraws_person_and_face_and_corrects_the_face(
    user: Api, comfy, tmp_path, face_engine, monkeypatch
):
    from app.advanced import quality

    async def face_box(path: Path):
        return (28.0, 30.0, 6.0, 8.0, 64, 80)  # a small face in a body shot

    monkeypatch.setattr(crop, "face_box", face_box)
    # The face against the LoRA's photos: no match first, a match after the correction.
    measured: list[list[Path]] = []
    scores = [0.21, 0.52]

    async def identity(result: Path, references: list[Path]):
        measured.append(references)
        return scores.pop(0)

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
        "quality": True,
        "corrections": 1,
        "resolution": "2K",
    }
    # The plan before anything runs shows the same steps.
    shown = (await user.post("/api/advanced/plan", json={**body, "targets": []})).json()
    assert [e["target"] for e in shown["edits"]] == ["body", "face", "hair"]
    # No region chosen: with a LoRA the person becomes its character anyway.
    job = await run(user, body)
    assert job["status"] == "completed", job
    assert job["options"]["identity"] is True and job["options"]["prepare"] == "on"
    keys = [s["key"] for s in job["stages"]]
    assert keys == ["plan", "prepare", "edit1", "edit2", "edit3", "check", "fix1-face", "check2"]
    person, face, hair, fix = sent_graphs(comfy)[-4:]
    # The person (SAM 3.1 "person") redrawn with the LoRA, strong enough for its figure;
    # every pixel outside stays the photo's. The face is taken out (SAM 3.1 "face"), so
    # its expression is left for the face edit.
    assert person["31"]["inputs"]["text"] == "person" and "50" not in person
    assert person["45"]["inputs"]["text"] == "face"
    assert person["47"]["inputs"] == {
        "destination": ["32", 0],
        "source": ["46", 0],
        "x": 0,
        "y": 0,
        "operation": "subtract",
    }
    assert person["33"]["inputs"]["mask"] == ["47", 0]
    assert person["7"]["inputs"]["denoise"] == 0.6
    lora_node = next(n for n in person.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora_node["inputs"]["strength_model"] == 1.0
    assert person["4"]["inputs"]["text"].startswith("lmx, lmxx2 woman")
    # The face in its enlarged crop, as a new identity; the hair last with its colour.
    assert face["31"]["inputs"]["text"] == "face" and face["50"]["class_type"] == "ImageCrop"
    assert face["7"]["inputs"]["denoise"] == 0.75
    assert hair["31"]["inputs"]["text"] == "hair" and hair["41"]["inputs"]["blend_mode"] == (
        "multiply"
    )
    # The face did not match the LoRA (measured): only the face again, stronger.
    first, second = job["reports"]
    match = next(c for c in first["checks"] if c["id"] == "face_match")
    assert match["passed"] is False and match["kind"] == "modify" and match["note"] == "21 %"
    assert first["identity_reference"] == "lora" and first["changed_area"] is not None
    assert all(len(refs) == 2 for refs in measured)  # the LoRA's photos, not the original
    assert fix["31"]["inputs"]["text"] == "face"
    assert fix["7"]["inputs"]["denoise"] == pytest.approx(0.85)
    assert next(c for c in second["checks"] if c["id"] == "face_match")["passed"] is True
    assert job["result"]["image_id"] == second["image_id"]
    # Identity off: the LoRA only helps, and a region has to be chosen again.
    r = await user.post("/api/advanced/jobs", json={**body, "identity": False})
    assert r.status_code == 400


def test_small_photo_is_enlarged_to_the_edit_size():
    # The owner's photos (2026-11-05): 548 × 640 at 2K needs 3.2× (SeedVR2's largest, 3×),
    # 1016 × 1410 needs 1.45× (1.5×); one already big enough is not enlarged at all.
    assert engine.prepare_upscale({"width": 548, "height": 640}, "2K") == 3.0
    assert engine.prepare_upscale({"width": 1016, "height": 1410}, "2K") == 1.5
    assert engine.prepare_upscale({"width": 800, "height": 1000}, "2K") == 3.0
    assert engine.prepare_upscale({"width": 1000, "height": 1300}, "2K") == 2.0
    assert engine.small_photo({"width": 1162, "height": 1593}, "2K") is False


async def test_small_photo_without_enhance_models_is_only_resized(user: Api, tmp_path):
    files = {k: list(v) for k, v in KREA_FILES.items()}
    files.setdefault("checkpoints", []).append(enhance.SAM3.filename)  # SAM 3.1 only
    fake = FakeComfy(files=files)
    ComfyClient.transport = fake.transport()
    engine.POLL = 0.01
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    try:
        conn = await connect(user, tmp_path)
        ref = await photo(user)
        job = await run(
            user,
            {
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
        assert job["status"] == "completed", job
        prepare = job["stages"][1]
        assert prepare["key"] == "prepare" and prepare["status"] == "skipped"
        assert "not installed" in prepare["detail"]
        assert len(fake.prompts) == 1  # only the hair edit ran
    finally:
        ComfyClient.transport = None


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
    # The colour it changes to, not the one it has (owner's test, 2026-11-05).
    assert hair_colour(
        "Completely change her platinum blonde hair color to deep natural dark brown, "
        "with no remaining blonde sections."
    ) == ("dark brown", "#3A281D")
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
    assert first["41"]["inputs"]["blend_factor"] == 0.85
    assert again["41"]["inputs"]["blend_factor"] == 0.95
    assert graphs.tint("#3A281D", 5) == ("multiply", 1.0)
    assert graphs.tint("#E5DCC8") == ("screen", 0.55)  # platinum blonde
    assert graphs.tint("#1B1716")[0] == "multiply" and graphs.tint("#CDAE74")[0] == "screen"
    assert first["40"]["inputs"]["width"] == 896 and first["40"]["inputs"]["height"] == 1152
    # Without a region nothing is tinted or masked (the whole photo is image to image).
    whole = graphs.edit(**{**args, "region": None})
    assert "40" not in whole and whole["12"]["inputs"]["pixels"] == ["11", 0]
    assert "60" not in whole and whole["7"]["inputs"]["model"] == ["1", 0]
    assert all(n["class_type"] in graphs.NODES_EDIT for n in first.values())


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
    from app.advanced.compiler import by_rules, detected_targets

    assert set(detected_targets(OWNER_PROMPT)) == {"face", "hair", "body"}
    # A sentence that keeps something changes nothing; a named hair colour is a change.
    assert detected_targets("Keep her hair and face exactly as they are.") == []
    assert detected_targets("Preserve the body position and the hair.") == []
    assert detected_targets("a woman with platinum blonde hair on a beach") == ["hair"]
    assert detected_targets("Photorealistic, realistic proportions, sharp focus.") == []
    assert detected_targets("Zmeň jej farbu vlasov na tmavohnedú a zatvor jej ústa.") == ["hair"]
    # Without analysis the chosen "whole" stays first and the text's regions follow.
    spec = by_rules(mode="edit", prompt=OWNER_PROMPT, targets=["whole"])
    assert [e.target for e in spec.edits] == ["whole", "body", "face", "hair"]
    assert spec.edits[1].region == "person"
    # Nothing chosen and nothing found: the whole photo, as before.
    assert [e.target for e in by_rules(mode="edit", prompt="a woman", targets=[]).edits] == [
        "whole"
    ]


def test_identity_plan_redraws_the_person_then_the_face():
    from app.advanced.compiler import Edit, Spec, by_rules, validate

    spec = validate(
        by_rules(mode="edit", prompt=OWNER_PROMPT, targets=["whole"]),
        user_prompt=OWNER_PROMPT,
        adult=False,
        identity=True,
    )
    # "whole" is not needed: the person, the face, then the hair with its colour first.
    assert [e.target for e in spec.edits] == ["body", "face", "hair"]
    assert [e.region for e in spec.edits] == ["person", "face", "hair"]
    assert spec.edits[2].color == "#3A281D"
    assert spec.edits[2].prompt.startswith("dark brown hair, ")
    # A plan the user edited keeps its own body and face prompts.
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
        ("face", "my face prompt"),
        ("hair", "red hair"),
    ]
    # Without identity the plan's own order stays.
    plain = Spec(
        mode="edit", edits=[Edit(target="hair", prompt="a"), Edit(target="body", prompt="b")]
    )
    assert [e.target for e in validate(plain, user_prompt="x", adult=False).edits] == [
        "hair",
        "body",
    ]
