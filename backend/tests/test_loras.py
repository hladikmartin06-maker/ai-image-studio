"""LoRA studio: projects, dataset tools, captions, training on (fake) RunPod through the real
agent, versions and checkpoints, import, use in generation, tests, jobs, privacy."""

from __future__ import annotations

import asyncio
import io
import json
import struct
from pathlib import Path

import pytest
from PIL import Image as PILImage

from app.ai.providers.openai_compat import OpenAICompatibleProvider
from app.ai.registry import AI_PROVIDERS, set_adapter
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.gpu.registry import set_gpu_adapter
from app.lora import aitoolkit
from app.lora.presets import ARCHITECTURES, TrainingConfig, preset_config
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
        config=TrainingConfig.model_validate(hq),
        sample_prompts=["lnx woman portrait"],
    )
    proc = cfg["config"]["process"][0]
    assert proc["model"]["arch"] == "krea2"
    assert proc["model"]["name_or_path"] == "krea/Krea-2-Turbo"
    assert "krea2_turbo_training_adapter" in proc["model"]["assistant_lora_path"]
    assert proc["train"]["diff_output_preservation_class"] == "woman"
    assert proc["sample"]["sample_steps"] == 8 and proc["sample"]["guidance_scale"] == 1.0
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
    assert (
        "skip_first_sample" not in default["train"] and "disable_sampling" not in default["train"]
    )
    assert default["train"]["diff_output_preservation"] is True
    assert default["train"]["gradient_checkpointing"] is True
    fast = build(previews="end", diff_output_preservation=False, gradient_checkpointing=False)
    assert fast["sample"]["sample_every"] == 0 and fast["train"]["skip_first_sample"] is True
    assert "disable_sampling" not in fast["train"]  # the images after the last step stay
    assert "diff_output_preservation" not in fast["train"]
    assert fast["train"]["gradient_checkpointing"] is False
    # Stored configs from before keep working (previews defaults to "during").
    assert TrainingConfig.model_validate({"steps": 1000}).previews == "during"
    assert ARCHITECTURES["krea2"]["vram_no_checkpointing_gb"] == 80


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
        assert v["status"] == "failed" and "CUDA out of memory" in v["error_message"]
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
