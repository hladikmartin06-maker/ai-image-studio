"""LoRA studio: projects, dataset tools, captions, training on (fake) RunPod through the real
agent, versions and checkpoints, import, use in generation, tests, jobs, privacy."""

from __future__ import annotations

import asyncio
import io
import json
import struct
import uuid
from pathlib import Path

import pytest
from PIL import Image as PILImage

from app.ai.providers.openai_compat import OpenAICompatibleProvider
from app.ai.registry import AI_PROVIDERS, set_adapter
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.db import SessionLocal
from app.gpu.registry import set_gpu_adapter
from app.lora import aitoolkit
from app.lora.presets import ARCHITECTURES, TrainingConfig, preset_config
from app.models import LoraVersion
from app.services.gpu_runner import gpu_runner
from tests.ai_fakes import AI_KEY, FakeAI
from tests.conftest import Api, make_user, wait_done
from tests.fake_comfy_server import FakeComfy
from tests.fakes import FakeHub
from tests.gpu_fakes import LocalRunPodProvider, SpawningRunPod, make_toolkit
from tests.test_gpu import RUNPOD_KEY

LORA = {
    "name": "Lena character",
    "trigger_word": "lnx",
    "class_word": "woman",
    "base_variant": "raw",
    "adult_confirmed": True,
}


def photo(i: int, size: tuple[int, int] = (640, 800)) -> bytes:
    img = PILImage.new("RGB", size, (20 * i % 255, 90, 160))
    for x in range(0, size[0], 40):
        for y in range(0, size[1], 40):
            if (x // 40 + y // 40 + i) % (2 + i % 3) == 0:
                img.putpixel((x, y), (255, 255, 255))
                img.paste((i * 37 % 255, i * 11 % 255, 50), (x, y, x + 20 + i % 7, y + 13 + i % 5))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def safetensors(meta: dict[str, str], keys: tuple[str, ...] = ("x.lora_A.weight",)) -> bytes:
    header = {"__metadata__": meta}
    for k in keys:
        header[k] = {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 8


async def create(api: Api, **extra) -> dict:
    r = await api.post("/api/loras", json={**LORA, **extra})
    assert r.status_code == 201, r.text
    return r.json()


async def upload_dataset(api: Api, lora_id: str, n: int, start: int = 1) -> dict:
    body = {}
    for i in range(start, start + n):
        files = {"file": (f"img{i}.png", photo(i), "image/png")}
        r = await api.post(f"/api/loras/{lora_id}/dataset/upload", files=files)
        assert r.status_code == 201, r.text
        body = r.json()["lora"]
    return body


async def chunked(api: Api, name: str, data: bytes) -> str:
    r = await api.post("/api/uploads", json={"filename": name, "size": len(data)})
    upload_id = r.json()["id"]
    for offset in range(0, len(data), 1000):
        part = await api.put(
            f"/api/uploads/{upload_id}",
            params={"offset": offset},
            content=data[offset : offset + 1000],
        )
        assert part.status_code == 200, part.text
    return upload_id


def test_presets_are_architecture_aware():
    fast = preset_config("fast_experiment", "raw")
    hq = preset_config("high_quality", "turbo")
    assert fast["steps"] < hq["steps"] and fast["rank"] < hq["rank"]
    assert hq["base_variant"] == "turbo"
    paths = aitoolkit.run_paths("run1")
    cfg = aitoolkit.build(
        dataset_dir=paths["dataset_abs"],
        output_dir=paths["output_abs"],
        architecture="krea2",
        name="run1",
        trigger_word="lnx",
        class_word="woman",
        config=TrainingConfig.model_validate({**hq, "diff_output_preservation": True}),
        sample_prompts=["lnx woman portrait"],
    )
    proc = cfg["config"]["process"][0]
    assert proc["model"]["arch"] == "krea2"
    assert proc["model"]["name_or_path"] == "krea/Krea-2-Turbo"
    assert "krea2_turbo_training_adapter" in proc["model"]["assistant_lora_path"]
    assert proc["train"]["diff_output_preservation_class"] == "woman"
    # ai-toolkit's Turbo sample settings (its "krea2:turbo" entry).
    assert proc["sample"]["sample_steps"] == 9 and proc["sample"]["guidance_scale"] == 1.0
    assert proc["network"] == {"type": "lora", "linear": 64, "linear_alpha": 32}
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({**hq, "resolutions": [2048]})


def test_faster_training_options():
    """Validation images only at the end, output preservation off and gradient checkpointing
    off reach ai-toolkit's config as its own fields."""

    def build(**overrides):
        cfg = TrainingConfig.model_validate(
            {**preset_config("character_identity", "raw"), **overrides}
        )
        return aitoolkit.build(
            dataset_dir="/d",
            output_dir="/o",
            architecture="krea2",
            name="run1",
            trigger_word="lnx",
            class_word="woman",
            config=cfg,
            sample_prompts=["lnx woman portrait"],
        )["config"]["process"][0]

    default = build()
    assert default["sample"]["sample_every"] == 250
    # No baseline images of the untrained model (they came again after every restart).
    assert default["train"]["skip_first_sample"] is True
    assert "disable_sampling" not in default["train"]
    # The interval the user picks (every 100 … 1000 steps) is ai-toolkit's sample_every.
    assert build(sample_every=100)["sample"]["sample_every"] == 100
    # ai-toolkit's own Krea 2 choices: linear timesteps, preservation off unless switched on.
    assert default["train"]["timestep_type"] == "linear"
    assert "diff_output_preservation" not in default["train"]
    assert build(diff_output_preservation=True)["train"]["diff_output_preservation"] is True
    assert default["train"]["gradient_checkpointing"] is True
    fast = build(previews="end", diff_output_preservation=False, gradient_checkpointing=False)
    assert fast["sample"]["sample_every"] == 0 and fast["train"]["skip_first_sample"] is True
    assert "disable_sampling" not in fast["train"]  # the images after the last step stay
    assert "diff_output_preservation" not in fast["train"]
    assert fast["train"]["gradient_checkpointing"] is False
    # Stored configs from before keep working (previews defaults to "during").
    assert TrainingConfig.model_validate({"steps": 1000}).previews == "during"
    assert ARCHITECTURES["krea2"]["vram_no_checkpointing_gb"] == 96
    # A version's saved settings can be used again: its run-only fields are left out, other
    # unknown fields are still refused.
    saved = {"steps": 1000, "image": None, "cloud_type": "SECURE", "continue_from_number": 1}
    assert TrainingConfig.model_validate(saved).steps == 1000
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"steps": 1000, "unknown": 1})


async def test_project_rules_and_dataset_tools(user: Api):
    assert (
        await user.post("/api/loras", json={**LORA, "adult_confirmed": False})
    ).status_code == 400
    assert (
        await user.post("/api/loras", json={**LORA, "trigger_word": "two words"})
    ).status_code == 400
    assert (
        await user.post("/api/loras", json={**LORA, "class_word": "teen girl"})
    ).status_code == 422
    lora = await create(user)
    assert (await user.post("/api/loras", json={**LORA, "trigger_word": "LNX"})).status_code == 409
    lora = await upload_dataset(user, lora["id"], 3)
    codes = {w["code"] for w in lora["analysis"]["warnings"]}
    assert "too_few" in codes and "missing_captions" in codes and not lora["analysis"]["ready"]
    # The same file twice is stored once.
    dup = await user.post(
        f"/api/loras/{lora['id']}/dataset/upload",
        files={"file": ("again.png", photo(1), "image/png")},
    )
    assert dup.json()["added"] == 0
    item = lora["dataset"][0]
    r = await user.patch(
        f"/api/loras/{lora['id']}/dataset/{item['id']}",
        json={
            "caption": "lnx woman, red coat, park",
            "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.6},
        },
    )
    assert r.json()["caption_source"] == "manual" and r.json()["crop"]["w"] == 0.8
    bad = await user.patch(
        f"/api/loras/{lora['id']}/dataset/{item['id']}",
        json={"crop": {"x": 0.5, "y": 0, "w": 0.9, "h": 1}},
    )
    assert bad.status_code == 400
    ids = [i["id"] for i in lora["dataset"]]
    out = await user.post(
        f"/api/loras/{lora['id']}/dataset/bulk", json={"item_ids": ids[1:], "action": "validation"}
    )
    assert out.json()["analysis"]["validation"] == 2
    out = await user.post(
        f"/api/loras/{lora['id']}/dataset/bulk", json={"item_ids": [ids[2]], "action": "remove"}
    )
    assert len(out.json()["dataset"]) == 2
    # Low resolution is reported.
    small = await user.post(
        f"/api/loras/{lora['id']}/dataset/upload",
        files={"file": ("small.png", photo(9, (300, 400)), "image/png")},
    )
    codes = {w["code"] for w in small.json()["lora"]["analysis"]["warnings"]}
    assert "very_low_res" in codes


async def test_copy_lora_with_photos_and_captions(user: Api):
    """A Turbo project copied to a new RAW project: the same photos, marks, crops and
    captions with the new trigger word; nothing trained is copied; deleting the original
    keeps the copy's photos."""
    src = await create(user, name="Taila", trigger_word="tla", base_variant="turbo")
    empty = await user.post(
        f"/api/loras/{src['id']}/copy",
        json={"name": "x", "trigger_word": "tlaraw", "adult_confirmed": True},
    )
    assert empty.status_code == 409 and empty.json()["error"]["code"] == "empty_dataset"
    src = await upload_dataset(user, src["id"], 3)
    first, second, third = src["dataset"]
    await user.patch(
        f"/api/loras/{src['id']}/dataset/{first['id']}",
        json={
            "caption": "tla woman, Tla smiling, red coat, park",
            "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.6},
        },
    )
    await user.patch(f"/api/loras/{src['id']}/dataset/{second['id']}", json={"split": "validation"})
    await user.patch(f"/api/loras/{src['id']}/dataset/{third['id']}", json={"excluded": True})
    body = {
        "name": "Taila RAW",
        "trigger_word": "tlaraw",
        "class_word": "woman",
        "base_variant": "raw",
        "adult_confirmed": True,
    }
    url = f"/api/loras/{src['id']}/copy"
    assert (await user.post(url, json={**body, "adult_confirmed": False})).status_code == 400
    taken = await user.post(url, json={**body, "trigger_word": "TLA"})
    assert taken.status_code == 409 and taken.json()["error"]["code"] == "trigger_taken"
    assert (await user.post(url, json={**body, "class_word": "teen girl"})).status_code == 422
    other = await make_user("Other")
    try:
        assert (await other.post(url, json=body)).status_code == 404
    finally:
        await other.client.aclose()

    r = await user.post(url, json=body)
    assert r.status_code == 201, r.text
    copy = r.json()
    assert copy["id"] != src["id"] and copy["name"] == "Taila RAW"
    assert copy["base_variant"] == "raw" and copy["source"] == "trained"
    assert copy["trigger_word"] == "tlaraw" and copy["versions"] == []
    items = copy["dataset"]
    assert [i["reference_id"] for i in items] == [i["reference_id"] for i in src["dataset"]]
    assert items[0]["caption"] == "tlaraw woman, tlaraw smiling, red coat, park"
    assert items[0]["caption_source"] == "manual" and items[0]["crop"]["w"] == 0.8
    assert items[1]["split"] == "validation" and items[2]["excluded"]
    # The original is unchanged.
    again = (await user.get(f"/api/loras/{src['id']}")).json()
    assert again["base_variant"] == "turbo"
    assert again["dataset"][0]["caption"] == "tla woman, Tla smiling, red coat, park"

    bare = await user.post(url, json={**body, "trigger_word": "tlab", "captions": False})
    assert bare.status_code == 201
    assert {i["caption"] for i in bare.json()["dataset"]} == {""}
    assert {i["caption_source"] for i in bare.json()["dataset"]} == {"empty"}

    # Deleting the original keeps the photos the copies use.
    assert (await user.delete(f"/api/loras/{src['id']}")).status_code == 200
    kept = (await user.get(f"/api/loras/{copy['id']}")).json()
    assert len(kept["dataset"]) == 3
    assert (await user.get(kept["dataset"][0]["url"])).status_code == 200


async def test_likeness_per_step_and_dataset_faces(user: Api, monkeypatch):
    """The face of each training photo against the others (another person stands out), the
    validation images' likeness per step with the best step, and the hair-colour warning."""
    from app.lora import runner
    from app.models import Lora, LoraVersion
    from app.services import faces, storage
    from tests.fakes import png_bytes
    from tests.test_faces import FakeEngine

    engine = FakeEngine()

    async def get_engine():
        return engine

    monkeypatch.setattr(faces, "_get_engine", get_engine)
    faces._cache.clear()
    lora = await create(user, trigger_word="lkn")
    # FakeEngine: the colour of the first pixel is "the face". Ten photos of her, one of
    # someone else, one without a face.
    colours = [(200, 40, 40 + i) for i in range(10)] + [(10, 200, 10), (0, 0, 0)]
    for i, colour in enumerate(colours):
        files = {"file": (f"f{i}.png", png_bytes(64, 64, colour), "image/png")}
        r = await user.post(f"/api/loras/{lora['id']}/dataset/upload", files=files)
        assert r.status_code == 201, r.text
        detail = r.json()["lora"]
    items = detail["dataset"]
    check = await user.post(f"/api/loras/{lora['id']}/dataset/faces")
    assert check.status_code == 200, check.text
    result = {x["item_id"]: x for x in check.json()["items"]}
    assert [result[i["id"]]["outlier"] for i in items] == [False] * 10 + [True, False]
    assert result[items[11]["id"]]["face"] is False and result[items[0]["id"]]["score"] > 0.9

    # Hair colour named on most captions: not learned as hers.
    for item in items[:10]:
        await user.patch(
            f"/api/loras/{lora['id']}/dataset/{item['id']}",
            json={"caption": "lkn woman, long black hair, park"},
        )
    detail = (await user.get(f"/api/loras/{lora['id']}")).json()
    assert "hair_in_captions" in {w["code"] for w in detail["analysis"]["warnings"]}

    # A finished run with validation images at two steps: her face only at step 500.
    async with SessionLocal() as db:
        row = await db.get(Lora, uuid.UUID(lora["id"]))
        v = LoraVersion(
            lora_id=row.id,
            user_id=row.user_id,
            number=1,
            status="completed",
            config={},
            trainer_config={},
            progress={
                "samples": [
                    {"file": "1__000000250_0.jpg", "step": 250, "index": 0},
                    {"file": "2__000000500_0.jpg", "step": 500, "index": 0},
                    {"file": "3__000000500_1.jpg", "step": 500, "index": 1},
                ]
            },
        )
        db.add(v)
        await db.commit()
        folder = storage.absolute(runner.version_dir(v)) / "samples"
        version_id = v.id
    folder.mkdir(parents=True, exist_ok=True)
    for name, colour in (
        ("1__000000250_0.jpg", (10, 200, 10)),
        ("2__000000500_0.jpg", (200, 40, 45)),
        ("3__000000500_1.jpg", (200, 45, 40)),
    ):
        (folder / name).write_bytes(png_bytes(64, 64, colour))
    url = f"/api/loras/{lora['id']}/versions/{version_id}/likeness"
    r = await user.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["best_step"] == 500 and body["reference_faces"] == 11  # 12 photos, 1 without a face
    by_step = {x["step"]: x for x in body["steps"]}
    assert by_step[500]["faces"] == 2 and by_step[500]["score"] > 0.9
    assert by_step[250]["score"] < faces.MATCH_THRESHOLD
    # Kept with the version; another user sees nothing.
    version = next(
        x
        for x in (await user.get(f"/api/loras/{lora['id']}")).json()["versions"]
        if x["id"] == str(version_id)
    )
    assert version["likeness"]["best_step"] == 500
    other = await make_user("likeness-other")
    try:
        assert (await other.get(url)).status_code == 404
        assert (await other.post(f"/api/loras/{lora['id']}/dataset/faces")).status_code == 404
    finally:
        await other.client.aclose()


async def test_ai_captions(user: Api):
    ai = FakeAI(reply="wearing a red coat in a park, soft daylight")
    set_adapter(
        "openai",
        OpenAICompatibleProvider(
            "https://api.openai.fake/v1",
            use_max_completion_tokens=True,
            reasoning_effort_prefixes=("gpt-5",),
            transport=ai.openai_transport(),
        ),
    )
    try:
        lora = await create(user)
        lora = await upload_dataset(user, lora["id"], 1)
        item = lora["dataset"][0]
        none = await user.post(f"/api/loras/{lora['id']}/dataset/{item['id']}/caption", json={})
        assert none.status_code == 409
        r = await user.put("/api/ai/providers/openai", json={"api_key": AI_KEY})
        assert r.status_code == 200
        r = await user.post(f"/api/loras/{lora['id']}/dataset/{item['id']}/caption", json={})
        assert r.status_code == 200, r.text
        assert r.json()["caption"] == "lnx woman, wearing a red coat in a park, soft daylight"
        sent = ai.calls[-1]["body"]["messages"][-1]["content"]
        assert sent[0]["type"] == "image_url" and sent[0]["image_url"]["url"].startswith(
            "data:image/jpeg"
        )
        usage = (await user.get("/api/ai/usage")).json()
        assert usage
        url = f"/api/loras/{lora['id']}/dataset/{item['id']}/caption"
        # A model that reasons first and runs out of tokens: asked again with more room,
        # the caption is whole.
        ai.reply = (
            "standing by a window, cream knit sweater, soft daylight, half-body shot, "
            "neutral expression, plain wall"
        )
        ai.truncate_below = 2000
        r = await user.post(url, json={"overwrite": True})
        assert r.json()["caption"] == "lnx woman, " + ai.reply
        budgets = [c["body"]["max_completion_tokens"] for c in ai.calls[-2:]]
        assert budgets == [1024, 4096]
        # Cut off even then: only the finished phrases are kept, never half a phrase.
        ai.truncate_below = 10_000
        r = await user.post(url, json={"overwrite": True})
        assert r.json()["caption"] == "lnx woman, standing by a window, cream knit sweater"
        # Nothing finished at all: a clear error, nothing saved.
        ai.reply = "standing by a window in a cream knit sweater"
        r = await user.post(url, json={"overwrite": True})
        assert r.status_code == 502 and r.json()["error"]["code"] == "incomplete"
    finally:
        for key in AI_PROVIDERS:
            set_adapter(key, None)


async def test_training_on_runpod_end_to_end(user: Api, tmp_path: Path, hub: FakeHub):
    toolkit = make_toolkit(tmp_path / "image")
    fake = SpawningRunPod(tmp_path / "pods", {"STUDIO_AITK_DIR": str(toolkit)})
    set_gpu_adapter("runpod", LocalRunPodProvider(fake))
    comfy = FakeComfy(models_dir=str(tmp_path / "ComfyUI"))
    (tmp_path / "ComfyUI" / "models").mkdir(parents=True)
    ComfyClient.transport = comfy.transport()
    comfy_runner.POLL_INITIAL = 0.01
    try:
        assert (
            await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
        ).status_code == 200
        lora = await create(user)
        lora = await upload_dataset(user, lora["id"], 10)
        for item in lora["dataset"]:
            await user.patch(
                f"/api/loras/{lora['id']}/dataset/{item['id']}",
                json={"caption": "lnx woman, city street"},
            )
        config = {**preset_config("fast_experiment", "raw"), "steps": 100, "save_every": 50}
        body = {
            "preset": "fast_experiment",
            "config": config,
            "provider": "runpod",
            "gpu_type": "NVIDIA L4",
            "client_request_id": "train-1",
        }
        assert (await user.post(f"/api/loras/{lora['id']}/versions", json=body)).status_code == 400
        gated = await user.post(
            f"/api/loras/{lora['id']}/versions", json={**body, "confirm_billing": True}
        )
        assert gated.status_code == 409 and gated.json()["error"]["code"] == "hf_token_required"
        # Hugging Face: a rejected token is never saved; the license is checked before a GPU.
        empty = (await user.get("/api/secrets/huggingface/status")).json()
        assert not empty["set"] and {r["repo"] for r in empty["repos"]} == {
            "krea/Krea-2-Raw",
            "krea/Krea-2-Turbo",
        }
        hub.invalid_tokens.add("hf_badtoken999")
        bad = await user.put("/api/secrets/huggingface", json={"value": "hf_badtoken999"})
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_key"
        # Hugging Face's own reason is shown.
        assert "Invalid credentials in Authorization header" in bad.json()["error"]["message"]
        hub.no_license["hf_testtoken123"] = {"krea/Krea-2-Raw"}
        assert (
            await user.put("/api/secrets/huggingface", json={"value": "hf_testtoken123"})
        ).status_code == 200
        secrets = (await user.get("/api/secrets")).json()["items"][0]
        assert secrets["set"] and "hf_testtoken123" not in json.dumps(secrets)
        status = (await user.get("/api/secrets/huggingface/status")).json()
        assert status["token"] == "valid" and status["account"] == "studio-user"
        assert not status["ready"] and "hf_testtoken123" not in json.dumps(status)
        assert {r["repo"]: r["access"] for r in status["repos"]} == {
            "krea/Krea-2-Raw": "needs_license",
            "krea/Krea-2-Turbo": "granted",
        }
        raw = next(r for r in status["repos"] if r["repo"] == "krea/Krea-2-Raw")
        assert "Access to model krea/Krea-2-Raw is restricted" in raw["message"]
        blocked = await user.post(
            f"/api/loras/{lora['id']}/versions", json={**body, "confirm_billing": True}
        )
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "hf_license_required"
        assert "Hugging Face says" in blocked.json()["error"]["message"]
        assert fake.pods == {} and not fake.bodies  # nothing rented
        hub.no_license.clear()  # the user clicked "Agree" on huggingface.co
        status = (await user.get("/api/secrets/huggingface/status?refresh=true")).json()
        assert status["ready"]
        r = await user.post(
            f"/api/loras/{lora['id']}/versions", json={**body, "confirm_billing": True}
        )
        assert r.status_code == 201, r.text
        version = r.json()
        again = await user.post(
            f"/api/loras/{lora['id']}/versions", json={**body, "confirm_billing": True}
        )
        assert again.json()["id"] == version["id"]
        for _ in range(300):
            await gpu_runner.tick()
            detail = (await user.get(f"/api/loras/{lora['id']}")).json()
            version = detail["versions"][0]
            if version["status"] in ("completed", "failed", "cancelled", "interrupted"):
                break
            await asyncio.sleep(0.05)
        assert version["status"] == "completed", version
        pod = next(iter(fake.bodies.values()))
        assert pod["env"]["HF_TOKEN"] == "hf_testtoken123"
        assert pod["image"] == "ostris/aitoolkit:latest"
        # Only CUDA 13 versions with a free machine (RunPod's catalog).
        assert pod["gpu"]["allowedCudaVersions"] == ["13.0"]
        assert version["step"] == 100 and version["loss"]
        assert version["checkpoints"][0]["step"] == 50
        assert version["samples"] and version["cost_final"] is not None
        # Only the validation images themselves, not ai-toolkit's thumbnails.
        assert all(x["step"] and not x["file"].endswith(".jpg.jpg") for x in version["samples"])
        assert version["gpu_stopped_at"] and fake.pods == {}  # auto-cleanup terminated it
        assert detail["preferred_version_id"] == version["id"]
        logs = (await user.get(f"/api/loras/{lora['id']}/versions/{version['id']}/logs")).json()
        assert "dataset: 10 images, 10 captions" in logs["text"]
        file = await user.get(version["file_url"])
        assert file.status_code == 200 and b"studio_architecture" in file.content[:600]
        sample = await user.get(version["samples"][0]["url"])
        assert sample.status_code == 200

        # A checkpoint becomes its own version.
        cp = await user.post(
            f"/api/loras/{lora['id']}/versions/{version['id']}/checkpoints/50/as-version"
        )
        assert cp.status_code == 201 and cp.json()["number"] == 2
        blocked = await user.delete(f"/api/loras/{lora['id']}/versions/{version['id']}")
        assert blocked.status_code == 409

        # Continue training from v1 with more photos: v1's file is the starting point and its
        # size is kept (the requested rank is ignored).
        lora = await upload_dataset(user, lora["id"], 2, start=11)
        assert len(lora["dataset"]) == 12
        # Like "Continue training" in the app: v1's saved settings (with its run's own fields:
        # trainer image, cloud) are the starting point.
        assert {"image", "cloud_type"} <= set(version["config"])
        cont = {
            **body,
            "config": {
                **version["config"],
                "continue_from": version["id"],
                "rank": 64,
                "alpha": 64,
            },
            "client_request_id": "train-continue",
            "confirm_billing": True,
        }
        wrong = await user.post(
            f"/api/loras/{lora['id']}/versions",
            json={**cont, "config": {**cont["config"], "continue_from": str(uuid.uuid4())}},
        )
        assert wrong.status_code == 404
        r = await user.post(f"/api/loras/{lora['id']}/versions", json=cont)
        assert r.status_code == 201, r.text
        assert r.json()["config"]["rank"] == config["rank"] == 16
        assert r.json()["config"]["continue_from_number"] == 1
        for _ in range(300):
            await gpu_runner.tick()
            detail = (await user.get(f"/api/loras/{lora['id']}")).json()
            v3 = next(v for v in detail["versions"] if v["id"] == r.json()["id"])
            if v3["status"] in ("completed", "failed", "cancelled", "interrupted"):
                break
            await asyncio.sleep(0.05)
        assert v3["status"] == "completed", v3
        assert v3["dataset_size"] == 12
        logs = (await user.get(f"/api/loras/{lora['id']}/versions/{v3['id']}/logs")).json()
        assert "Using pretrained lora path from config" in logs["text"]
        async with SessionLocal() as db:
            row = await db.get(LoraVersion, uuid.UUID(v3["id"]))
            network = row.trainer_config["config"]["process"][0]["network"]
        assert network["pretrained_lora_path"].endswith("/start_lora.safetensors")
        assert network["linear"] == 16

        # Generate with it on a local ComfyUI folder: copied once, trigger added, strict.
        conn = await user.post(
            "/api/comfy/connections",
            json={
                "name": "Local",
                "kind": "local",
                "base_url": "http://127.0.0.1:8188",
                "models_dir": str(tmp_path / "ComfyUI"),
            },
        )
        conn = conn.json()
        g = await user.post(
            "/api/comfy/generate",
            json={
                "connection_id": conn["id"],
                "variant": "turbo",
                "prompt": "walking in Lisbon",
                "lora": {"version_id": version["id"], "strength": 0.9},
            },
        )
        assert g.status_code == 201, g.text
        gen = await wait_done(user, g.json()["id"])
        assert gen["status"] == "completed", gen
        assert gen["final_prompt"].startswith("lnx, walking in Lisbon")
        assert gen["comfy"]["lora"]["trigger_word"] == "lnx"
        copied = list((tmp_path / "ComfyUI/models/loras/ai-image-studio").iterdir())
        assert len(copied) == 1
        sent = list(comfy.prompts.values())[-1]["prompt"]
        node = next(n for n in sent.values() if n["class_type"] == "LoraLoaderModelOnly")
        assert node["inputs"]["strength_model"] == 0.9
        assert node["inputs"]["lora_name"].startswith("ai-image-studio/")
        lib = (await user.get("/api/images", params={"lora_version": version["id"]})).json()
        assert lib["total"] == 1

        # Side-by-side test: base vs the LoRA, same seed.
        t = await user.post(
            f"/api/loras/{lora['id']}/tests",
            json={
                "connection_id": conn["id"],
                "prompts": ["lnx woman reading"],
                "seeds": [7],
                "version_ids": [version["id"]],
                "confirm_count": 2,
            },
        )
        assert t.status_code == 201, t.text
        await wait_done(
            user, (await user.get("/api/generations", params={"limit": 1})).json()["items"][0]["id"]
        )
        tests = (await user.get(f"/api/loras/{lora['id']}/tests")).json()["items"]
        runs = tests[0]["runs"]
        assert {r["comfy"]["test_label"] for r in runs} == {"base", version["id"]}
        base = next(r for r in runs if r["comfy"]["test_label"] == "base")
        assert base["comfy"]["lora"] is None and "lnx" not in base["prompt"]
        assert {r["settings"]["seed"] for r in runs} == {7}

        # Body consistency test (18+): full-body views, same seed, only with the switch on.
        body_test = {
            "connection_id": conn["id"],
            "prompts": [
                f"full body photo of lnx woman, nude, standing straight, {view}"
                for view in ("front view", "side view", "back view")
            ],
            "seeds": [7],
            "version_ids": [version["id"]],
            "include_base": False,
            "aspect_ratio": "2:3",
            "confirm_count": 3,
        }
        off = await user.post(f"/api/loras/{lora['id']}/tests", json=body_test)
        assert off.status_code == 422 and off.json()["error"]["code"] == "adult_off"
        t = await user.post(f"/api/loras/{lora['id']}/tests", json={**body_test, "adult": True})
        assert t.status_code == 201 and t.json()["count"] == 3, t.text
        for _ in range(400):
            tests = (await user.get(f"/api/loras/{lora['id']}/tests")).json()["items"]
            if all(r["status"] == "completed" for r in tests[0]["runs"]):
                break
            await asyncio.sleep(0.05)
        body_runs = tests[0]["runs"]
        assert len(body_runs) == 3 and all(r["status"] == "completed" for r in body_runs)
        assert all(
            r["settings"]["aspect_ratio"] == "2:3" and r["settings"]["adult"] for r in body_runs
        )
        assert all(r["images"][0]["nsfw"] is True for r in body_runs)

        # The best generated images go into the LoRA's dataset (copied, once).
        best = [r["images"][0]["id"] for r in body_runs[:2]]
        added = await user.post(
            f"/api/loras/{lora['id']}/dataset/add", json={"library_image_ids": best}
        )
        assert added.status_code == 200 and added.json()["added"] == 2
        again = await user.post(
            f"/api/loras/{lora['id']}/dataset/add", json={"library_image_ids": best}
        )
        assert again.json()["added"] == 0

        # Dataset generator: Krea 2 with this LoRA, one run per prompt; the trigger word is
        # added where it is missing; not a test (not listed under Test & compare).
        gen_body = {
            "connection_id": conn["id"],
            "version_id": version["id"],
            "prompts": ["in a sunny park, white sundress", "lnx woman in a cafe, side view", " "],
            "count": 2,
            "confirm_count": 4,
        }
        wrong = await user.post(
            f"/api/loras/{lora['id']}/dataset/generate", json={**gen_body, "confirm_count": 6}
        )
        assert wrong.status_code == 400
        made = await user.post(f"/api/loras/{lora['id']}/dataset/generate", json=gen_body)
        assert made.status_code == 201 and made.json()["runs"] == 2, made.text
        for _ in range(400):
            generated = (await user.get(f"/api/loras/{lora['id']}/dataset/generated")).json()
            if all(g["status"] == "completed" for g in generated["items"]):
                break
            await asyncio.sleep(0.05)
        items = generated["items"]
        assert len(items) == 2 and all(len(g["images"]) == 2 for g in items)
        prompts = sorted(g["prompt"] for g in items)
        assert prompts == [
            "lnx woman in a cafe, side view",
            "photo of lnx woman, in a sunny park, white sundress",
        ]
        assert all(g["comfy"]["lora"]["version_id"] == version["id"] for g in items)
        tests_now = (await user.get(f"/api/loras/{lora['id']}/tests")).json()["items"]
        assert all(len(t["runs"]) != 2 or t["runs"][0]["prompt"] not in prompts for t in tests_now)

        jobs = (await user.get("/api/jobs", params={"type": "training"})).json()
        assert jobs["items"][0]["status"] == "completed" and jobs["items"][0]["progress"] == 100

        listed = (await user.get("/api/loras", params={"usable": True})).json()
        assert listed["items"][0]["usable_versions"]
        other = await make_user("Other")
        assert (await other.get(f"/api/loras/{lora['id']}")).status_code == 404
        assert (await other.get(version["file_url"])).status_code == 404
        await other.client.aclose()
    finally:
        fake.close()
        set_gpu_adapter("runpod", None)
        ComfyClient.transport = None


async def test_training_failure_and_cancel(user: Api, tmp_path: Path):
    toolkit = make_toolkit(tmp_path / "image")
    fake = SpawningRunPod(
        tmp_path / "pods", {"STUDIO_AITK_DIR": str(toolkit), "FAKE_TRAIN_FAIL": "1"}
    )
    set_gpu_adapter("runpod", LocalRunPodProvider(fake))
    try:
        await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
        await user.put("/api/secrets/huggingface", json={"value": "hf_testtoken123"})
        lora = await create(user, trigger_word="mrx")
        await upload_dataset(user, lora["id"], 10, start=20)
        body = {
            "preset": "fast_experiment",
            "config": {**preset_config("fast_experiment", "raw"), "steps": 100},
            "provider": "runpod",
            "gpu_type": "NVIDIA L4",
            "confirm_billing": True,
        }
        r = await user.post(f"/api/loras/{lora['id']}/versions", json=body)
        assert r.status_code == 201, r.text
        for _ in range(300):
            await gpu_runner.tick()
            v = (await user.get(f"/api/loras/{lora['id']}")).json()["versions"][0]
            if v["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)
        assert v["status"] == "failed" and "the dataset loader broke" in v["error_message"]
        assert v["error_code"] == "training_failed" and not v["memory_retries"]
        assert fake.pods == {}  # cleaned up after the failure

        fake.extra_env.pop("FAKE_TRAIN_FAIL")
        r = await user.post(f"/api/loras/{lora['id']}/versions", json=body)
        v2 = r.json()
        await gpu_runner.tick()
        cancel = await user.post(f"/api/loras/{lora['id']}/versions/{v2['id']}/cancel", json={})
        assert cancel.json()["status"] == "cancelled"
        assert fake.pods == {}
    finally:
        fake.close()
        set_gpu_adapter("runpod", None)


def test_memory_fallback_steps():
    from app.lora.aitoolkit import build as build_config
    from app.lora.runner import memory_fallback

    def cfg(**changes):
        config = TrainingConfig.model_validate(
            {**preset_config("character_identity", "raw"), **changes}
        )
        return build_config(
            dataset_dir="/d",
            output_dir="/o",
            architecture="krea2",
            name="run",
            trigger_word="mrx",
            class_word="woman",
            config=config,
            sample_prompts=[],
        )

    # One change at a time, the cheapest for quality first: gradient checkpointing, then
    # batch 1 (same effective batch), then 8-bit quantization, then low-VRAM mode.
    first, changes = memory_fallback(
        cfg(gradient_checkpointing=False, quantize=False, batch_size=2, gradient_accumulation=2)
    )
    assert changes == ["gradient_checkpointing"]
    process = first["config"]["process"][0]
    assert process["train"]["gradient_checkpointing"] is True
    assert process["model"]["quantize"] is False and process["train"]["batch_size"] == 2
    second, changes = memory_fallback(first)
    assert changes == ["batch_size"]
    assert second["config"]["process"][0]["train"]["batch_size"] == 1
    assert second["config"]["process"][0]["train"]["gradient_accumulation"] == 4
    third, changes = memory_fallback(second)
    assert changes == ["quantize"]
    model = third["config"]["process"][0]["model"]
    assert model["quantize"] is True and model["quantize_te"] is True
    fourth, changes = memory_fallback(third)
    assert changes == ["low_vram"] and fourth["config"]["process"][0]["model"]["low_vram"]
    assert memory_fallback(fourth) is None


async def test_out_of_memory_trains_again_with_memory_saving(user: Api, tmp_path: Path):
    """An out-of-memory stop (here: an 80 GB GPU with memory saving off) is not the end: the
    same GPU trains again with memory saving on, and the version shows the new settings.
    When nothing is left to save, the run fails with a clear message."""
    toolkit = make_toolkit(tmp_path / "image")
    fake = SpawningRunPod(
        tmp_path / "pods", {"STUDIO_AITK_DIR": str(toolkit), "FAKE_TRAIN_OOM": "no_checkpointing"}
    )
    set_gpu_adapter("runpod", LocalRunPodProvider(fake))
    try:
        await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
        await user.put("/api/secrets/huggingface", json={"value": "hf_testtoken123"})
        lora = await create(user, trigger_word="mrx")
        await upload_dataset(user, lora["id"], 10, start=20)
        config = {**preset_config("fast_experiment", "raw"), "steps": 100}
        body = {
            "preset": "fast_experiment",
            "config": {**config, "gradient_checkpointing": False},
            "provider": "runpod",
            "gpu_type": "NVIDIA L4",
            "confirm_billing": True,
        }

        async def run(request: dict) -> dict:
            r = await user.post(f"/api/loras/{lora['id']}/versions", json=request)
            assert r.status_code == 201, r.text
            for _ in range(400):
                await gpu_runner.tick()
                v = next(
                    x
                    for x in (await user.get(f"/api/loras/{lora['id']}")).json()["versions"]
                    if x["id"] == r.json()["id"]
                )
                if v["status"] in ("completed", "failed"):
                    return v
                await asyncio.sleep(0.05)
            raise AssertionError(v)

        # An 80 GB GPU trains in full precision; running out of memory turns on gradient
        # checkpointing only, the model stays in full precision.
        v = await run({**body, "gpu_memory_gb": 80})
        assert v["status"] == "completed", v.get("error_message")
        retries = v["memory_retries"]
        assert len(retries) == 1 and retries[0]["changes"] == ["gradient_checkpointing"]
        assert v["config"]["gradient_checkpointing"] is True
        assert v["config"]["quantize"] is False and v["config"]["low_vram"] is False
        pods = [c for c in fake.calls if c[0] == "POST" and c[1] == "/v2/pods"]
        assert len(pods) == 1  # the same GPU, not a second one
        log = (await user.get(f"/api/loras/{lora['id']}/versions/{v['id']}/logs")).json()["text"]
        assert "OOM during training" in log and "starting again with gradient_checkpointing" in log

        # Out of memory even with every saving on: a clear message, the GPU removed.
        fake.extra_env["FAKE_TRAIN_OOM"] = "always"
        v = await run(body)
        assert v["status"] == "failed" and v["error_code"] == "out_of_memory"
        assert v["error_message"].startswith("The GPU ran out of memory, even with memory saving")
        assert [r["changes"] for r in v["memory_retries"]] == [
            ["gradient_checkpointing"],
            ["low_vram"],
        ]
        assert fake.pods == {}
    finally:
        fake.close()
        set_gpu_adapter("runpod", None)


async def test_training_on_modal_waits_for_a_free_gpu(user: Api, monkeypatch):
    """Modal had no free GPU for the training: the version waits (nothing billed) and asks
    again, then rents the GPU once one is free; after the waiting limit it fails clearly."""
    from app.gpu.modal_provider import ModalProvider
    from app.lora import runner
    from tests.test_gpu import FakeModalSDK

    monkeypatch.setattr(runner, "CAPACITY_PAUSES", [0])
    sdk = FakeModalSDK()
    sdk.no_capacity = 2
    set_gpu_adapter("modal", ModalProvider(sdk=sdk))
    try:
        await user.put(
            "/api/gpu/providers/modal",
            json={"token_id": "ak-123456", "token_secret": "as-secret-123456"},
        )
        await user.put("/api/secrets/huggingface", json={"value": "hf_testtoken123"})
        lora = await create(user, trigger_word="mrx")
        await upload_dataset(user, lora["id"], 10, start=20)
        body = {
            "preset": "fast_experiment",
            "config": {**preset_config("fast_experiment", "raw"), "steps": 100},
            "provider": "modal",
            "gpu_type": "L40S",
            "confirm_billing": True,
        }
        r = await user.post(f"/api/loras/{lora['id']}/versions", json=body)
        assert r.status_code == 201, r.text
        details = []
        for _ in range(20):
            await gpu_runner.tick()
            v = (await user.get(f"/api/loras/{lora['id']}")).json()["versions"][0]
            details.append(v["stage_detail"])
            if v["status"] != "queued":
                break
        assert v["status"] == "preparing" and len(sdk.created) == 1, v
        assert any(d.startswith("Modal has no free L40S right now") for d in details)
        await user.post(f"/api/loras/{lora['id']}/versions/{v['id']}/cancel", json={})

        monkeypatch.setattr(runner, "CAPACITY_WAIT", runner.timedelta(0))
        sdk.no_capacity = 1
        r = await user.post(f"/api/loras/{lora['id']}/versions", json=body)
        for _ in range(20):
            await gpu_runner.tick()
            v = next(
                x
                for x in (await user.get(f"/api/loras/{lora['id']}")).json()["versions"]
                if x["id"] == r.json()["id"]
            )
            if v["status"] != "queued":
                break
        assert v["status"] == "failed" and "Modal had no free L40S" in v["error_message"]
        assert len(sdk.created) == 1
    finally:
        set_gpu_adapter("modal", None)


async def test_import_lora_with_architecture_checks(user: Api):
    krea = safetensors(
        {"training_info": json.dumps({"studio_architecture": "krea2"}), "name": "lena"}
    )
    upload = await chunked(user, "lena.safetensors", krea)
    r = await user.post(
        "/api/loras/import",
        json={
            "upload_id": upload,
            "name": "Imported",
            "trigger_word": "imp1",
            "adult_confirmed": True,
        },
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "variant_unknown"
    r = await user.post(
        "/api/loras/import",
        json={
            "upload_id": upload,
            "name": "Imported",
            "trigger_word": "imp1",
            "declared_variant": "raw",
            "adult_confirmed": True,
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "imported" and r.json()["preferred_version_id"]

    flux = await chunked(user, "flux.safetensors", safetensors({"ss_base_model_version": "flux1"}))
    bad = await user.post(
        "/api/loras/import",
        json={
            "upload_id": flux,
            "name": "F",
            "trigger_word": "flx",
            "declared_variant": "raw",
            "adult_confirmed": True,
        },
    )
    assert bad.status_code == 409
    unknown = await chunked(user, "u.safetensors", safetensors({}))
    r = await user.post(
        "/api/loras/import",
        json={
            "upload_id": unknown,
            "name": "U",
            "trigger_word": "unk1",
            "declared_variant": "raw",
            "adult_confirmed": True,
        },
    )
    assert r.json()["error"]["code"] == "architecture_unknown"
    r = await user.post(
        "/api/loras/import",
        json={
            "upload_id": unknown,
            "name": "U",
            "trigger_word": "unk1",
            "declared_variant": "turbo",
            "declared_architecture": "krea2",
            "adult_confirmed": True,
        },
    )
    assert r.status_code == 201 and r.json()["base_variant"] == "turbo"
    not_safetensors = await chunked(user, "x.safetensors", b"hello world, not a model")
    r = await user.post(
        "/api/loras/import",
        json={
            "upload_id": not_safetensors,
            "name": "X",
            "trigger_word": "xx1",
            "adult_confirmed": True,
        },
    )
    assert r.status_code == 400
