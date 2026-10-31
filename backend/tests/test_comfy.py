"""ComfyUI: Krea 2 facts, workflows, connections, generation through the job worker,
cancel and recovery, LoRA placement, deployment on RunPod through the real agent."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.comfy import krea2, workflows
from app.comfy import runner as comfy_runner
from app.comfy.client import ComfyClient
from app.db import SessionLocal
from app.gpu.registry import set_gpu_adapter
from app.models import ComfyConnection, Generation, ProviderTask
from app.services.gpu_runner import gpu_runner
from app.services.worker import worker
from tests.conftest import Api, make_user, wait_done
from tests.fake_comfy_server import FakeComfy
from tests.gpu_fakes import LocalRunPodProvider, SpawningRunPod, comfy_env, make_comfy_dir
from tests.test_gpu import RUNPOD_KEY


@pytest.fixture
def comfy():
    fake = FakeComfy()
    ComfyClient.transport = fake.transport()
    comfy_runner.POLL_INITIAL = 0.01
    comfy_runner.POLL_MAX = 0.02
    yield fake
    ComfyClient.transport = None


async def add_local(api: Api, **extra) -> dict:
    body = {"name": "Mac ComfyUI", "kind": "local", "base_url": "http://127.0.0.1:8188", **extra}
    r = await api.post("/api/comfy/connections", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# --- pure functions -----------------------------------------------------------------------


def test_krea2_facts():
    assert krea2.size_for("1:1", "1K") == (1024, 1024)
    w, h = krea2.size_for("9:16", "2K")
    assert w % 16 == 0 and h % 16 == 0 and max(w, h) <= 2048
    assert krea2.VARIANTS["turbo"].steps == 8 and krea2.VARIANTS["turbo"].cfg == 1.0
    assert krea2.VARIANTS["raw"].tiers == ("1K",)
    ok, _ = krea2.lora_compatibility("krea2", "raw", "turbo")
    assert ok
    ok, why = krea2.lora_compatibility("krea2", "turbo", "raw")
    assert not ok and "Turbo" in why
    ok, why = krea2.lora_compatibility("flux", "raw", "turbo")
    assert not ok and "flux" in why
    assert krea2.precision_of("krea2_turbo_int8_convrot.safetensors") == "int8"
    detection = krea2.detect(FakeComfy().object_info())
    assert detection["ready"] and detection["variants"]["turbo"]["installed"]
    assert detection["variants"]["raw"]["files"][0]["precision"] == "bf16"
    bare = FakeComfy(
        files={k: [] for k in ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints")}
    )
    missing = krea2.detect(bare.object_info())
    assert not missing["ready"] and len(missing["missing"]) >= 2


def test_builtin_workflow_matches_the_template_nodes():
    wf = krea2.build_workflow(
        variant="turbo",
        unet="krea2_turbo_fp8_scaled.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        vae="qwen_image_vae.safetensors",
        prompt="p",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        seed=7,
        steps=8,
        cfg=1.0,
        batch=2,
        loras=[("a.safetensors", 0.8)],
    )
    kinds = {n["class_type"] for n in wf.values()}
    assert {"UNETLoader", "CLIPLoader", "ConditioningZeroOut", "LoraLoaderModelOnly"} <= kinds
    assert wf["2"]["inputs"]["type"] == "krea2"
    assert wf["7"]["inputs"]["model"] == ["20", 0]
    assert wf["7"]["inputs"]["sampler_name"] == "euler" and wf["6"]["inputs"]["batch_size"] == 2
    raw = krea2.build_workflow(
        variant="raw",
        unet="u",
        text_encoder="t",
        vae="v",
        prompt="p",
        negative_prompt="blurry",
        width=1024,
        height=1024,
        seed=1,
        steps=52,
        cfg=3.5,
        batch=1,
        loras=[],
    )
    assert raw["5"]["class_type"] == "CLIPTextEncode" and raw["5"]["inputs"]["text"] == "blurry"


def test_workflow_import_bindings_and_lora_injection():
    api = krea2.build_workflow(
        variant="turbo",
        unet="u",
        text_encoder="t",
        vae="v",
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        seed=1,
        steps=8,
        cfg=1.0,
        batch=1,
        loras=[],
    )
    api["30"] = {"class_type": "OpenAIImageNode", "inputs": {"api_key": "sk-secret", "x": 1}}
    clean, removed = workflows.scrub(api)
    assert clean["30"]["inputs"]["api_key"] == "" and removed
    assert workflows.kind_of(clean) == "api"
    assert workflows.kind_of({"nodes": [], "links": []}) == "ui"
    with pytest.raises(workflows.WorkflowError):
        workflows.kind_of({"a": 1})
    bindings = workflows.detect_bindings(clean)
    assert bindings["positive"] == "4" and bindings["latent"] == "6"
    assert bindings["model_loader"] == "1" and "negative" not in bindings  # ConditioningZeroOut
    out = workflows.apply(
        clean,
        bindings,
        {
            "prompt": "new",
            "seed": 9,
            "width": 768,
            "height": 1024,
            "batch": 3,
            "loras": [("x.safetensors", 0.7), ("y.safetensors", 0.5)],
        },
    )
    assert out["4"]["inputs"]["text"] == "new" and out["7"]["inputs"]["seed"] == 9
    loras = [k for k, n in out.items() if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 2
    # The sampler now uses the last LoRA, which uses the first, which uses the loader.
    last = out["7"]["inputs"]["model"][0]
    first = out[last]["inputs"]["model"][0]
    assert out[first]["inputs"]["model"] == ["1", 0]
    assert clean["7"]["inputs"]["model"] == ["1", 0]  # the stored copy is untouched


# --- connections and generation -------------------------------------------------------------


async def test_connection_inspection_and_url_rules(user: Api, comfy: FakeComfy):
    bad = await user.post(
        "/api/comfy/connections",
        json={"name": "x", "kind": "remote", "base_url": "http://example.com:8188"},
    )
    assert bad.status_code == 400
    comfy.require_token = "secret-token-1234"
    conn = await add_local(user, auth_type="bearer", secret="secret-token-1234")
    assert conn["status"] == "ready", conn
    assert "secret-token-1234" not in str(conn) and conn["masked_secret"].endswith("1234")
    info = conn["info"]
    assert info["version"] == "0.43.0" and info["devices"][0]["vram_total"] == 48 * 2**30
    assert info["krea2"]["ready"] and "impact" in info["custom_node_packs"]
    wrong = await add_local(user, auth_type="bearer", secret="wrong-token-9999")
    assert wrong["status"] == "error"
    other = await make_user("Other")
    assert (await other.get("/api/comfy/connections")).json()["items"] == []
    assert (await other.post(f"/api/comfy/connections/{conn['id']}/test")).status_code == 404
    await other.client.aclose()


async def test_krea2_generation_end_to_end(user: Api, comfy: FakeComfy):
    conn = await add_local(user)
    r = await user.post(
        "/api/comfy/generate",
        json={
            "connection_id": conn["id"],
            "variant": "turbo",
            "prompt": "a lighthouse at dusk",
            "negative_prompt": "blurry",
            "aspect_ratio": "3:4",
            "resolution": "1K",
            "count": 2,
            "client_request_id": "req-1",
        },
    )
    assert r.status_code == 201, r.text
    again = await user.post(
        "/api/comfy/generate",
        json={
            "connection_id": conn["id"],
            "variant": "turbo",
            "prompt": "x",
            "client_request_id": "req-1",
        },
    )
    assert again.json()["id"] == r.json()["id"]
    gen = await wait_done(user, r.json()["id"])
    assert gen["status"] == "completed", gen
    assert len(gen["images"]) == 2 and gen["provider_name"] == "ComfyUI"
    assert gen["model_name"] == "Krea 2 Turbo"
    assert gen["settings"]["seed"] is not None and gen["settings"]["steps"] == 8
    assert "negative_prompt" not in gen["settings"]  # Turbo runs with CFG 1
    sent = next(iter(comfy.prompts.values()))["prompt"]
    assert sent["5"]["class_type"] == "ConditioningZeroOut"
    assert sent["6"]["inputs"]["width"] == krea2.size_for("3:4", "1K")[0]
    lib = (await user.get("/api/images", params={"model": "krea-2-turbo"})).json()
    assert lib["total"] == 2 and lib["items"][0]["comfy"]["model_file"]

    raw = await user.post(
        "/api/comfy/generate",
        json={
            "connection_id": conn["id"],
            "variant": "raw",
            "prompt": "a fox",
            "negative_prompt": "blurry",
        },
    )
    raw_gen = await wait_done(user, raw.json()["id"])
    assert raw_gen["settings"]["negative_prompt"] == "blurry" and raw_gen["settings"]["steps"] == 52
    too_big = await user.post(
        "/api/comfy/generate",
        json={"connection_id": conn["id"], "variant": "raw", "prompt": "a fox", "resolution": "2K"},
    )
    assert too_big.status_code == 400

    # Repeat with the same seed and workflow.
    rerun = await user.post(f"/api/generations/{gen['id']}/regenerate", json={})
    assert rerun.status_code == 201
    rerun_gen = await wait_done(user, rerun.json()["id"])
    assert rerun_gen["settings"]["seed"] == gen["settings"]["seed"]


async def test_failures_cancel_and_recovery(user: Api, comfy: FakeComfy):
    conn = await add_local(user)
    comfy.fail_with = "Allocation on device failed"
    r = await user.post(
        "/api/comfy/generate", json={"connection_id": conn["id"], "variant": "turbo", "prompt": "a"}
    )
    gen = await wait_done(user, r.json()["id"])
    assert gen["status"] == "failed" and "Allocation" in gen["error_message"]
    comfy.fail_with = None

    # Missing files: no silent substitution.
    comfy.files["diffusion_models"] = ["krea2_turbo_fp8_scaled.safetensors"]
    await user.post(f"/api/comfy/connections/{conn['id']}/test")
    missing = await user.post(
        "/api/comfy/generate", json={"connection_id": conn["id"], "variant": "raw", "prompt": "a"}
    )
    assert missing.status_code == 409 and missing.json()["error"]["code"] == "missing_models"

    # Cancel while ComfyUI runs it.
    comfy.polls_until_done = 1000
    r = await user.post(
        "/api/comfy/generate", json={"connection_id": conn["id"], "variant": "turbo", "prompt": "b"}
    )
    gen_id = r.json()["id"]
    for _ in range(100):
        g = (await user.get(f"/api/generations/{gen_id}")).json()
        if g["tasks"][0]["provider_task_id"] and g["status"] == "processing":
            break
        await asyncio.sleep(0.02)
    cancel = await user.post(f"/api/generations/{gen_id}/cancel")
    assert cancel.status_code == 200
    g = await wait_done(user, gen_id)
    assert g["status"] == "cancelled"

    # Recovery: a task stuck in "submitting" is looked up, never re-sent.
    comfy.polls_until_done = 1
    await worker.stop()
    try:
        r = await user.post(
            "/api/comfy/generate",
            json={"connection_id": conn["id"], "variant": "turbo", "prompt": "c"},
        )
        gen_id = uuid.UUID(r.json()["id"])
        known = str(uuid.uuid4())
        comfy.prompts[known] = {"prompt": {}, "polls": 5, "done": True}
        comfy.order.append(known)
        async with SessionLocal() as db:
            task = await db.scalar(select(ProviderTask).where(ProviderTask.generation_id == gen_id))
            task.status = "submitting"
            task.provider_task_id = known
            await db.commit()
        r2 = await user.post(
            "/api/comfy/generate",
            json={"connection_id": conn["id"], "variant": "turbo", "prompt": "d"},
        )
        lost = uuid.UUID(r2.json()["id"])
        async with SessionLocal() as db:
            task = await db.scalar(select(ProviderTask).where(ProviderTask.generation_id == lost))
            task.status = "submitting"
            task.provider_task_id = str(uuid.uuid4())
            await db.commit()
        sent_before = len(comfy.prompts)
    finally:
        await worker.start()
    resumed = await wait_done(user, str(gen_id))
    assert resumed["status"] == "completed"
    interrupted = await wait_done(user, str(lost))
    assert interrupted["status"] == "failed"
    assert interrupted["tasks"][0]["error_code"] == "interrupted"
    assert len(comfy.prompts) == sent_before  # nothing was sent again


async def test_custom_workflow_import_export_and_missing_nodes(user: Api, comfy: FakeComfy):
    conn = await add_local(user)
    api_wf = krea2.build_workflow(
        variant="turbo",
        unet="krea2_turbo_fp8_scaled.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        vae="qwen_image_vae.safetensors",
        prompt="template",
        negative_prompt="",
        width=512,
        height=512,
        seed=1,
        steps=6,
        cfg=1.0,
        batch=1,
        loras=[],
    )
    api_wf["40"] = {"class_type": "FaceDetailer", "inputs": {"token": "hf_secret_value"}}
    r = await user.post("/api/comfy/workflows", json={"name": "My Krea", "data": api_wf})
    assert r.status_code == 201, r.text
    wf = r.json()
    assert wf["runnable"] and wf["removed_secrets"]
    ui = await user.post(
        "/api/comfy/workflows", json={"name": "UI file", "data": {"nodes": [], "links": []}}
    )
    assert ui.json()["runnable"] is False
    exported = await user.get(f"/api/comfy/workflows/{wf['id']}/export")
    assert "hf_secret_value" not in exported.text
    builtin = await user.get(
        "/api/comfy/workflows/builtin-krea2/export", params={"variant": "turbo"}
    )
    assert builtin.json()["2"]["inputs"]["type"] == "krea2"
    g = await user.post(
        "/api/comfy/generate",
        json={
            "connection_id": conn["id"],
            "variant": "turbo",
            "prompt": "hello",
            "workflow_id": wf["id"],
        },
    )
    gen = await wait_done(user, g.json()["id"])
    assert gen["status"] == "completed" and gen["comfy"]["workflow"]["name"] == "My Krea"
    sent = list(comfy.prompts.values())[-1]["prompt"]
    assert sent["4"]["inputs"]["text"] == "hello" and sent["7"]["inputs"]["steps"] == 8

    api_wf["41"] = {"class_type": "UnknownCustomNode", "inputs": {}}
    r = await user.post("/api/comfy/workflows", json={"name": "Needs nodes", "data": api_wf})
    blocked = await user.post(
        "/api/comfy/generate",
        json={
            "connection_id": conn["id"],
            "variant": "turbo",
            "prompt": "x",
            "workflow_id": r.json()["id"],
        },
    )
    assert blocked.status_code == 409 and "UnknownCustomNode" in blocked.json()["error"]["message"]


def test_setup_script_and_download_progress():
    from app.comfy.deploy import _download_error, download_progress, model_downloads, setup_error
    from app.gpu.agent.studio_agent import comfyui_script, models_script

    script = comfyui_script([])
    # PyTorch is checked against the driver and ComfyUI's packages are installed again in
    # every new container; nothing is downloaded during setup.
    assert "nvidia-smi" in script and "download.pytorch.org/whl/$TORCH_INDEX" in script
    assert "pip install -r requirements.txt" in script and "curl" not in script
    assert model_downloads([]) == []
    files = model_downloads(["turbo"])
    assert [f["folder"] for f in files] == ["text_encoders", "vae", "diffusion_models"]
    # A failed download stops the job instead of going on without the file.
    assert "download failed" in models_script(files) and "exit 1" in models_script(files)
    assert "Hugging Face" in _download_error("curl: (22) The requested URL returned error: 403")
    assert "too old" in setup_error("... ComfyUI needs 12.8 or newer ...")
    sizes = [4.88, 0.242, 12.24]
    log = (
        "[studio] file 1/3 te\n[studio] file 2/3 vae\n[studio] file 3/3 model\n"
        " 85 12.2G   85 10.4G    0     0  61.4M      0  0:03:23  0:02:53  0:00:30 59.9M\r"
        " 90 12.2G   90 11.0G    0     0  61.3M      0  0:03:24  0:03:04  0:00:20 59.6M\r"
    )
    assert download_progress(log, sizes) == 93
    assert download_progress("", sizes) == 0
    assert download_progress("[studio] file 1/3 te\n", [None, None, None]) == 0


async def test_deploy_on_runpod_through_the_agent(user: Api, tmp_path: Path):
    """The studio deploys "ComfyUI" on a (fake) RunPod pod: the real agent starts it, the
    studio reaches it only through the agent, generates, then stops the pod."""
    comfy_dir = make_comfy_dir(tmp_path / "gpu")
    fake = SpawningRunPod(tmp_path / "pods", comfy_env(comfy_dir))
    set_gpu_adapter("runpod", LocalRunPodProvider(fake))
    comfy_runner.POLL_INITIAL = 0.05
    try:
        r = await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
        assert r.status_code == 200, r.text
        body = {"name": "RunPod Krea", "provider": "runpod", "gpu_type": "NVIDIA L4"}
        assert (await user.post("/api/comfy/deployments", json=body)).status_code == 400
        r = await user.post("/api/comfy/deployments", json={**body, "confirm_billing": True})
        assert r.status_code == 201, r.text
        conn = r.json()
        assert conn["status"] == "deploying" and conn["deploy"]["stage"] == "starting"
        pod = next(iter(fake.bodies.values()))
        assert pod["dockerEntrypoint"] == ["bash"] and pod["ports"] == ["8765/http"]
        assert pod["volumeMountPath"] == "/workspace"
        # PyTorch 2.13 / CUDA 13 image, on hosts whose driver has CUDA 12.8 or newer.
        assert pod["imageName"] == "runpod/pytorch:1.5.1-cu1300-torch2130-ubuntu2404"
        assert pod["allowedCudaVersions"] == ["13.0", "12.9", "12.8"]
        for _ in range(200):
            await gpu_runner.tick()
            conn = (await user.get("/api/comfy/connections")).json()["items"][0]
            if conn["status"] in ("ready", "error"):
                break
            await asyncio.sleep(0.1)
        assert conn["status"] == "ready", (conn["deploy"], conn["last_error"])
        assert conn["info"]["krea2"]["ready"]
        assert conn["info"]["agent"]["ok"]
        # Model download after connecting: files already on the volume are skipped.
        raw = await user.post(
            f"/api/comfy/connections/{conn['id']}/models/install", json={"variant": "raw"}
        )
        assert raw.status_code == 400  # no verified download for RAW
        inst = await user.post(
            f"/api/comfy/connections/{conn['id']}/models/install", json={"variant": "turbo"}
        )
        assert inst.status_code == 200, inst.text
        job = inst.json()["deploy"]["model_job"]
        assert job["status"] == "running" and job["files"] == [
            "qwen3vl_4b_fp8_scaled.safetensors",
            "qwen_image_vae.safetensors",
            "krea2_turbo_fp8_scaled.safetensors",
        ]
        for _ in range(100):
            await gpu_runner.tick()
            conn = (await user.get("/api/comfy/connections")).json()["items"][0]
            if conn["deploy"]["model_job"]["status"] != "running":
                break
            await asyncio.sleep(0.1)
        assert conn["deploy"]["model_job"]["status"] == "completed", conn["deploy"]
        assert conn["deploy"]["model_job"]["progress"] == 100
        g = await user.post(
            "/api/comfy/generate",
            json={"connection_id": conn["id"], "variant": "turbo", "prompt": "on a GPU"},
        )
        assert g.status_code == 201, g.text
        gen = await wait_done(user, g.json()["id"])
        assert gen["status"] == "completed", gen
        jobs = (await user.get("/api/jobs", params={"type": "deployment"})).json()
        assert jobs["items"][0]["status"] == "running" and jobs["items"][0]["cost"] is not None
        stop = await user.post(f"/api/comfy/connections/{conn['id']}/stop")
        assert stop.json()["status"] == "stopped"
        assert next(iter(fake.pods.values()))["desiredStatus"] == "EXITED"
        removed = await user.delete(f"/api/comfy/connections/{conn['id']}?terminate=true")
        assert removed.status_code == 200 and fake.pods == {}
    finally:
        fake.close()
        set_gpu_adapter("runpod", None)
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()
