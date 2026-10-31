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
from app.gpu.agent import COMFY_BOOT
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


def test_boot_script_images_and_download_progress():
    from app.comfy import deploy
    from app.gpu.agent import COMFY_BOOT

    # Nothing is installed when the GPU starts: the image's own start script runs ComfyUI.
    for word in ("pip ", "git clone", "torch"):
        assert word not in COMFY_BOOT
    assert 'exec "${STUDIO_START_SCRIPT:-/start.sh}"' in COMFY_BOOT
    assert "--listen 127.0.0.1" in COMFY_BOOT and "base64 -d" in COMFY_BOOT
    assert deploy.image_plan("runpod", "NVIDIA L4") == [
        ("runpod/comfyui:cuda12.8", ["12.8", "12.9"]),
        ("runpod/comfyui:cuda13.0", ["13.0", "13.1", "13.2", "13.3"]),
    ]
    # The CUDA 12 image does not support Blackwell GPUs (its README).
    assert deploy.image_plan("runpod", "NVIDIA GeForce RTX 5090")[0][0] == (
        "runpod/comfyui:cuda13.0"
    )
    assert deploy.image_plan("modal", "L40S") == [("runpod/comfyui:cuda12.8", [])]
    assert deploy.model_downloads([]) == []
    files = deploy.model_downloads(["turbo"])
    assert [f["folder"] for f in files] == ["text_encoders", "vae", "diffusion_models"]
    assert all(f["url"].startswith("https://huggingface.co/") for f in files)
    sizes = [4.88, 0.242, 12.24]
    job = {
        "files": [
            {"size": 4_880_000_000, "done": 4_880_000_000},
            {"size": 242_000_000, "done": 242_000_000},
            {"size": None, "done": 0},
        ]
    }
    assert deploy.model_progress(job, sizes) == 30
    job["files"][2].update(size=12_240_000_000, done=11_000_000_000)
    assert deploy.model_progress(job, sizes) == 93
    assert deploy.model_progress({"files": []}, sizes) == 0


async def _wait_conn(user: Api, done) -> dict:
    for _ in range(200):
        await gpu_runner.tick()
        conn = (await user.get("/api/comfy/connections")).json()["items"][0]
        if done(conn):
            return conn
        await asyncio.sleep(0.1)
    return conn


@pytest.fixture
def no_poll_wait(monkeypatch):
    from app.comfy import deploy

    monkeypatch.setattr(deploy, "POLL_EVERY", deploy.timedelta(0))


async def test_runpod_falls_back_to_cuda13_and_stops_when_comfyui_never_answers(
    user: Api, monkeypatch, no_poll_wait
):
    """No free host with a CUDA 12.8/12.9 driver: the CUDA 13 image is tried (and the
    documented CUDA list when RunPod refuses newer ones). A ComfyUI that never answers stops
    the GPU after the time limit."""
    import httpx

    from app.comfy import deploy
    from app.gpu.agent_client import AgentClient
    from app.gpu.runpod import RunPodProvider
    from tests.test_gpu import FakeRunPod

    fake = FakeRunPod(no_capacity={"12.8", "12.9"}, strict_cuda=True)
    set_gpu_adapter("runpod", RunPodProvider(transport=fake.transport()))
    AgentClient.transport = httpx.MockTransport(lambda r: httpx.Response(502, text="booting"))
    try:
        await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
        body = {"name": "GPU", "provider": "runpod", "gpu_type": "NVIDIA L4"}
        r = await user.post("/api/comfy/deployments", json={**body, "confirm_billing": True})
        assert r.status_code == 201, r.text
        sent = [b for m, path, b in fake.calls if m == "POST" and path == "/v1/pods"]
        assert [(b["imageName"], b["allowedCudaVersions"]) for b in sent] == [
            ("runpod/comfyui:cuda12.8", ["12.8", "12.9"]),
            ("runpod/comfyui:cuda13.0", ["13.0", "13.1", "13.2", "13.3"]),
            ("runpod/comfyui:cuda13.0", ["13.0"]),
        ]
        assert len(fake.pods) == 1  # only the last request created a pod
        conn = r.json()
        assert conn["deploy"]["image"] == "runpod/comfyui:cuda13.0"
        assert conn["deploy"]["cuda"] == ["13.0"]
        await gpu_runner.tick()
        conn = (await user.get("/api/comfy/connections")).json()["items"][0]
        assert conn["status"] == "deploying", conn["last_error"]
        assert conn["deploy"]["stage"] == "starting"
        monkeypatch.setattr(deploy, "BOOT_LIMIT", {"runpod": deploy.timedelta(0)})
        await gpu_runner.tick()
        conn = (await user.get("/api/comfy/connections")).json()["items"][0]
        assert conn["status"] == "error", conn
        assert "within 0 minutes" in conn["last_error"] and "not billed" in conn["last_error"]
        assert next(iter(fake.pods.values()))["desiredStatus"] == "EXITED"

        # Every host busy: a clear message, nothing created.
        fake.no_capacity = {"12.8", "12.9", "13.0", "13.1", "13.2", "13.3"}
        busy = await user.post("/api/comfy/deployments", json={**body, "confirm_billing": True})
        assert busy.status_code == 409 and busy.json()["error"]["code"] == "unavailable"
        assert "No GPU of this type is free" in busy.json()["error"]["message"]
        assert len(fake.pods) == 1
    finally:
        AgentClient.transport = None
        set_gpu_adapter("runpod", None)
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.commit()


def test_agent_downloads_only_from_hugging_face(tmp_path: Path, monkeypatch):
    """The agent's model download: huggingface.co only (redirects to Hugging Face's CDN
    only, the token never leaves huggingface.co), .part file resumed after a dropped
    connection and renamed when complete, files already there skipped, progress per file."""
    import threading
    import time
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from app.gpu.agent import studio_agent as agent

    assert agent.hf_url("https://huggingface.co/Comfy-Org/Krea-2/resolve/main/a.safetensors")
    for bad in (
        "http://huggingface.co/x",
        "https://huggingface.co.evil.com/x",
        "https://evil.com/huggingface.co/x",
        "https://user@huggingface.co/x",
        "https://cdn-lfs.hf.co/x",
    ):
        with pytest.raises(ValueError):
            agent.hf_url(bad)
    with pytest.raises(ValueError):
        agent.model_targets([{"url": "https://evil.com/x", "folder": "vae", "filename": "a"}])
    with pytest.raises(ValueError):
        agent.model_targets(
            [{"url": "https://huggingface.co/x", "folder": "../etc", "filename": "a"}]
        )

    # Redirects: Hugging Face's CDN yes (without the token), anything else no.
    monkeypatch.setattr(agent, "HF_TOKEN", "hf_secret")
    handler = agent._HFRedirect()
    req = urllib.request.Request("https://huggingface.co/a/resolve/main/f")
    req.add_unredirected_header("Authorization", "Bearer hf_secret")
    cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/f")
    assert cdn is not None and not cdn.has_header("Authorization")
    same = handler.redirect_request(req, None, 307, "", {}, "https://huggingface.co/api/r/f")
    assert same.unredirected_hdrs.get("Authorization") == "Bearer hf_secret"
    with pytest.raises(agent.DownloadError):
        handler.redirect_request(req, None, 302, "Found", {}, "https://evil.com/f")

    payload = bytes(range(256)) * 4000  # ~1 MB
    state = {"dropped": False, "auth": []}

    class Files(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            state["auth"].append(self.headers.get("Authorization"))
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            else:
                self.send_response(200)
            body = payload[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not state["dropped"]:
                state["dropped"] = True  # the first answer stops halfway
                self.wfile.write(body[: len(body) // 2])
                self.close_connection = True
                return
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Files)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    comfy = tmp_path / "ws" / "runpod-slim" / "ComfyUI"
    (comfy / "models" / "vae").mkdir(parents=True)
    (comfy / "models" / "vae" / "have.safetensors").write_bytes(b"x" * 10)
    monkeypatch.setattr(agent, "WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(agent, "COMFY_DIR", str(comfy))
    monkeypatch.setattr(agent.DownloadJob, "RETRIES", 3)
    monkeypatch.setattr(agent.DownloadJob, "RETRY_PAUSE", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/file"
    dest = comfy / "models" / "diffusion_models" / "model.safetensors"
    try:
        job = agent.DownloadJob(
            "models-t",
            "models",
            [
                (url, str(comfy / "models" / "vae" / "have.safetensors"), "have.safetensors"),
                (url, str(dest), "model.safetensors"),
            ],
        )
        for _ in range(200):
            if job.status != "running":
                break
            time.sleep(0.02)
    finally:
        server.shutdown()
    info = job.json()
    assert info["status"] == "completed", (info, Path(job.log_path).read_text())
    assert dest.read_bytes() == payload and not Path(str(dest) + ".part").exists()
    assert info["files"][0]["status"] == "skipped"
    assert info["files"][1] == {
        "name": "model.safetensors",
        "size": len(payload),
        "done": len(payload),
        "status": "done",
    }
    assert info["bytes_done"] == info["bytes_total"] == len(payload) + 10
    # The plain-HTTP test server is not huggingface.co: the token was never sent to it.
    assert state["auth"] == [None, None]
    assert "retrying" in Path(job.log_path).read_text()


async def test_deploy_on_runpod_through_the_agent(user: Api, tmp_path: Path, no_poll_wait):
    """The studio rents a (fake) RunPod pod with RunPod's ComfyUI image: the start command
    starts the real agent and hands over to the image's start script; the studio reaches
    ComfyUI only through the agent, installs the model, generates, then stops the pod."""
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
        assert conn["status"] == "deploying", conn["last_error"]
        assert conn["deploy"]["stage"] == "starting"
        pod = next(iter(fake.bodies.values()))
        assert pod["dockerEntrypoint"] == ["/bin/bash", "-c"] and pod["ports"] == ["8189/http"]
        assert pod["dockerStartCmd"] == [COMFY_BOOT]
        assert pod["volumeMountPath"] == "/workspace"
        # RunPod's ComfyUI image (ComfyUI and PyTorch inside), on CUDA 12.8 / 12.9 hosts.
        assert pod["imageName"] == "runpod/comfyui:cuda12.8"
        assert pod["allowedCudaVersions"] == ["12.8", "12.9"]
        conn = await _wait_conn(user, lambda c: c["status"] in ("ready", "error"))
        assert conn["status"] == "ready", (conn["deploy"], conn["last_error"])
        assert conn["info"]["krea2"]["ready"]
        assert conn["info"]["agent"]["ok"] and conn["info"]["agent"]["comfyui_dir"]
        # The image's first start copied ComfyUI into the volume; it listens on 127.0.0.1.
        workspace = tmp_path / "pods" / conn["workload_id"] / "workspace"
        assert (workspace / "runpod-slim" / "ComfyUI" / "main.py").is_file()
        args = (workspace / "runpod-slim" / "comfyui_args.txt").read_text()
        assert "--listen 127.0.0.1" in args
        log_tail = conn["deploy"]["log_tail"]
        assert "First time setup" in log_tail and "--listen 127.0.0.1" in log_tail
        assert "fake-ssh-secret" not in log_tail  # passwords in the start log are hidden
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
        conn = await _wait_conn(user, lambda c: c["deploy"]["model_job"]["status"] != "running")
        assert conn["deploy"]["model_job"]["status"] == "completed", conn["deploy"]
        assert conn["deploy"]["model_job"]["progress"] == 100
        # Manage › download by link: Hugging Face links only.
        bad = await user.post(
            f"/api/comfy/connections/{conn['id']}/models/download",
            json={
                "url": "https://example.com/x.safetensors",
                "folder": "loras",
                "filename": "x.safetensors",
            },
        )
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_url"
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
        # Started again: the volume keeps ComfyUI and the models, nothing is copied again.
        started = await user.post(
            f"/api/comfy/connections/{conn['id']}/start", json={"confirm_billing": True}
        )
        assert started.status_code == 200, started.text
        conn = await _wait_conn(user, lambda c: c["status"] in ("ready", "error"))
        assert conn["status"] == "ready", (conn["deploy"], conn["last_error"])
        assert "First time setup" not in conn["deploy"]["log_tail"]
        assert conn["info"]["krea2"]["variants"]["turbo"]["installed"]
        removed = await user.delete(f"/api/comfy/connections/{conn['id']}?terminate=true")
        assert removed.status_code == 200 and fake.pods == {}
    finally:
        fake.close()
        set_gpu_adapter("runpod", None)
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.execute(Generation.__table__.delete().where(Generation.provider == "comfyui"))
            await db.commit()


async def test_deploy_on_modal_uses_the_same_image_and_start(user: Api, no_poll_wait):
    """Modal: RunPod's ComfyUI image with its ENTRYPOINT cleared, the same start command and
    agent, a Volume at /workspace, and a 45-minute limit for the first start."""
    from app.comfy import deploy
    from app.gpu.modal_provider import ModalProvider
    from tests.test_gpu import FakeModalSDK

    sdk = FakeModalSDK()
    set_gpu_adapter("modal", ModalProvider(sdk=sdk))
    try:
        r = await user.put(
            "/api/gpu/providers/modal",
            json={"token_id": "ak-123456", "token_secret": "as-secret-123456"},
        )
        assert r.status_code == 200, r.text
        body = {"name": "Modal", "provider": "modal", "gpu_type": "L40S", "confirm_billing": True}
        r = await user.post("/api/comfy/deployments", json=body)
        assert r.status_code == 201, r.text
        spec = sdk.created[-1]
        assert spec.image == "runpod/comfyui:cuda12.8" and spec.clear_entrypoint
        assert [*spec.entrypoint, *spec.command] == ["/bin/bash", "-c", COMFY_BOOT]
        assert spec.ports == [8189] and spec.volume_name and spec.volume_gb == 0
        assert "STUDIO_AGENT_TOKEN" in spec.secret_env and "STUDIO_AGENT_TOKEN" not in spec.env
        assert deploy.BOOT_LIMIT["modal"] == deploy.timedelta(minutes=45)
    finally:
        set_gpu_adapter("modal", None)
        async with SessionLocal() as db:
            await db.execute(ComfyConnection.__table__.delete())
            await db.commit()


def test_modal_image_clears_the_entrypoint():
    """The real SDK call path builds modal.Image.from_registry(...).entrypoint([])."""
    import modal

    image = modal.Image.from_registry("runpod/comfyui:cuda12.8").entrypoint([])
    assert isinstance(image, modal.Image)
