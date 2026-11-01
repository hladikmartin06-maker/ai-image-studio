"""GPU compute: RunPod (fake REST + GraphQL) and Modal (fake SDK facade)."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.gpu.base import GPUError, WorkloadSpec
from app.gpu.modal_provider import ModalProvider
from app.gpu.registry import set_gpu_adapter
from app.gpu.runpod import RunPodProvider
from app.models import GPUIntegration, GPUWorkload
from tests.conftest import Api, make_user

RUNPOD_KEY = "rp_test_key_123456"


# RunPod's catalog in the fake: GPU types with prices per cloud.
CATALOG = [
    {
        "id": "NVIDIA L4",
        "name": "L4",
        "manufacturer": "NVIDIA",
        "memory": 24,
        "secure": True,
        "community": True,
        "price": {"secure": 0.43, "community": 0.39},
    },
    {
        "id": "NVIDIA H100 80GB HBM3",
        "name": "H100 SXM",
        "manufacturer": "NVIDIA",
        "memory": 80,
        "secure": True,
        "community": False,
        "price": {"secure": 2.99, "community": 0},
    },
    {
        "id": "AMD Instinct MI300X OAM",
        "name": "MI300X",
        "manufacturer": "AMD",
        "memory": 192,
        "secure": True,
        "community": False,
        "price": {"secure": 2.49, "community": 0},
    },
]
OFFERED_CUDA = ["12.4", "12.8", "12.9", "13.0"]
# CreatePodRequest (v2) closes its properties (unevaluatedProperties: false).
POD_KEYS = {
    "name", "image", "templateId", "cloud", "cpu", "gpu", "dataCenterIds", "globalNetworking",
    "mounts", "startJupyter", "startSsh", "args", "cmd", "entrypoint", "disk", "env", "ports",
    "registry",
}  # fmt: skip
GPU_KEYS = {"id", "count", "allowedCudaVersions", "minCudaVersion", "minRamPerGpu",
            "minVcpuCountPerGpu"}  # fmt: skip


def problem(status: int, detail: str, errors: list[str] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"title": httpx.codes.get_reason_phrase(status), "status": status,
                            "detail": detail}  # fmt: skip
    if errors:
        body["errors"] = errors
    return httpx.Response(status, json=body, headers={"content-type": "application/problem+json"})


@dataclass
class FakeRunPod:
    """RunPod's REST API v2 (pods, actions, catalog with availability, serverless) and the
    GraphQL balance query. Free machines per (GPU, cloud) are listed by CUDA version; a
    create whose allowedCudaVersions match no free machine is refused like RunPod does."""

    pods: dict[str, dict[str, Any]] = field(default_factory=dict)
    endpoints: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "ep1": {
                "id": "ep1",
                "name": "sdxl",
                "gpu": {"pools": ["ADA_24"]},
                "workers": {"min": 0, "max": 2},
            }
        }
    )
    calls: list[tuple[str, str, Any]] = field(default_factory=list)
    # CUDA versions with a free machine, per (GPU id, cloud); default: 12.8, 12.9 and 13.0.
    free: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Versions the catalog lists as free but whose last machine is taken at create time.
    taken_at_create: set[str] = field(default_factory=set)
    catalog: list[dict[str, Any]] = field(default_factory=lambda: list(CATALOG))
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def authorized(self, request: httpx.Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {RUNPOD_KEY}"

    def new_id(self) -> str:
        return f"pod{next(self._ids)}"

    def free_versions(self, gpu: dict[str, Any], cloud: str) -> list[str]:
        if not gpu["secure" if cloud == "SECURE" else "community"]:
            return []
        return self.free.get((gpu["id"], cloud), ["12.8", "12.9", "13.0"])

    def catalog_item(self, gpu: dict[str, Any], cloud: str) -> dict[str, Any]:
        free = self.free_versions(gpu, cloud)
        return {
            **gpu,
            "pool": None,
            "maxCount": {"secure": 8, "community": 4},
            "availability": "HIGH" if free else "NONE",
            "dataCenters": [{"id": "EU-RO-1", "name": "Europe Romania 1", "availability": "HIGH"}]
            if free
            else [],
            "cudaVersions": [{"version": v, "available": v in free} for v in OFFERED_CUDA],
        }

    def create(self, body: dict[str, Any]) -> httpx.Response:
        unknown = sorted(set(body) - POD_KEYS) + sorted(set(body.get("gpu") or {}) - GPU_KEYS)
        if unknown or "name" not in body or "image" not in body or "gpu" not in body:
            return problem(422, "The request body is invalid.", [f"unknown: {unknown}"])
        cloud = body.get("cloud", "SECURE")
        gpu = next((g for g in self.catalog if g["id"] == body["gpu"]["id"]), None)
        if gpu is None:
            return problem(400, f"Unknown GPU type {body['gpu']['id']}")
        allowed = body["gpu"].get("allowedCudaVersions") or []
        free = [v for v in self.free_versions(gpu, cloud) if v not in self.taken_at_create]
        if not (set(allowed) & set(free) if allowed else free):
            return problem(409, "There are no instances currently available")
        pod_id = self.new_id()
        self.pods[pod_id] = {
            "id": pod_id,
            "name": body["name"],
            "status": "RUNNING",
            "actions": ["stop", "terminate"],
            "image": body["image"],
            "gpu": {"id": gpu["id"], "count": body["gpu"].get("count", 1)},
            "cloud": cloud,
            "cost": gpu["price"]["secure" if cloud == "SECURE" else "community"],
            "ports": body.get("ports", []),
            "cudaVersion": next(v for v in free if not allowed or v in allowed),
            "createdAt": "2026-11-01T10:00:00Z",
        }
        return httpx.Response(201, json=self.pods[pod_id])

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.authorized(request):
            return problem(401, "Unauthorized")
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        host, path, query = request.url.host, request.url.path, request.url.params
        if host == "api.runpod.io" and path == "/graphql":
            return httpx.Response(
                200, json={"data": {"myself": {"clientBalance": 12.5, "currentSpendPerHr": 0.43}}}
            )
        if host == "api.runpod.ai":
            return httpx.Response(
                200, json={"jobs": {"inQueue": 0}, "workers": {"idle": 0, "running": 0}}
            )
        empty_page = {"nextCursor": None, "hasNextPage": False}
        if path.startswith("/v2/catalog/gpus"):
            if query.get("include") != "AVAILABILITY" or query.get("product") != "POD":
                return problem(400, "include=AVAILABILITY needs product")
            cloud = query.get("cloud", "SECURE")
            if path == "/v2/catalog/gpus":
                return httpx.Response(
                    200, json={"gpus": [self.catalog_item(g, cloud) for g in self.catalog]}
                )
            gpu_id = path.removeprefix("/v2/catalog/gpus/")
            gpu = next((g for g in self.catalog if g["id"] == gpu_id), None)
            if gpu is None:
                return problem(404, "GPU type not found")
            return httpx.Response(200, json=self.catalog_item(gpu, cloud))
        if path == "/v2/pods" and request.method == "GET":
            return httpx.Response(
                200, json={"pods": list(self.pods.values()), "pagination": empty_page}
            )
        if path == "/v2/pods" and request.method == "POST":
            return self.create(body or {})
        if path.startswith("/v2/pods/"):
            parts = path.split("/")
            pod_id = parts[3]
            if pod_id not in self.pods:
                return problem(404, "The requested Pod does not exist.")
            if request.method == "GET" and len(parts) == 4:
                return httpx.Response(200, json=self.pods[pod_id])
            if request.method == "DELETE":
                self.pods.pop(pod_id)
                return httpx.Response(204)
            if parts[-1] == "action":
                status = {"stop": "EXITED", "start": "RUNNING", "terminate": "TERMINATED"}
                self.pods[pod_id]["status"] = status[body["action"]]
                return httpx.Response(200, json=self.pods[pod_id])
            return problem(404, "Not found")
        if path == "/v2/serverless":
            return httpx.Response(
                200, json={"endpoints": list(self.endpoints.values()), "pagination": empty_page}
            )
        if path.startswith("/v2/serverless/"):
            endpoint_id = path.split("/")[3]
            if request.method == "PATCH":
                self.endpoints[endpoint_id].update(body)
                return httpx.Response(200, json=self.endpoints[endpoint_id])
            self.endpoints.pop(endpoint_id, None)
            return httpx.Response(204)
        return problem(404, "unknown")


class FakeSandbox:
    def __init__(self, object_id: str, tags: dict[str, str] | None = None):
        self.object_id = object_id
        self.code: int | None = None
        self.tags = tags or {}
        self.output = ""


class FakeModalSDK:
    def __init__(self):
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.created: list[WorkloadSpec] = []
        self._ids = itertools.count(1)

    async def client(self, creds):
        if creds.get("token_secret") != "as-secret-123456":
            raise RuntimeError("AuthError: invalid token")
        return object()

    async def app(self, client):
        return type("App", (), {"app_id": "ap-1"})()

    async def list(self, client, app, tags=None):
        return [
            sb
            for sb in self.sandboxes.values()
            if all(sb.tags.get(k) == v for k, v in (tags or {}).items())
        ]

    async def create(self, client, app, spec):
        sb = FakeSandbox(f"sb-{next(self._ids)}", spec.tags)
        self.sandboxes[sb.object_id] = sb
        self.created.append(spec)
        return sb

    async def from_id(self, client, sandbox_id):
        if sandbox_id not in self.sandboxes:
            raise RuntimeError("NotFoundError: sandbox not found")
        return self.sandboxes[sandbox_id]

    async def poll(self, sandbox):
        return sandbox.code

    async def terminate(self, sandbox):
        sandbox.code = 137

    async def tunnels(self, sandbox):
        return {8000: f"https://{sandbox.object_id}.modal.run"}

    async def output(self, sandbox):
        return sandbox.output

    async def close(self, client):
        return None


@pytest.fixture
def runpod():
    fake = FakeRunPod()
    set_gpu_adapter("runpod", RunPodProvider(transport=fake.transport()))
    yield fake
    set_gpu_adapter("runpod", None)


@pytest.fixture
def modal_sdk():
    sdk = FakeModalSDK()
    set_gpu_adapter("modal", ModalProvider(sdk=sdk))
    yield sdk
    set_gpu_adapter("modal", None)


async def test_runpod_credentials(user: Api, runpod: FakeRunPod):
    bad = await user.put("/api/gpu/providers/runpod", json={"api_key": "rp_wrong_000000"})
    assert bad.status_code == 400
    ok = await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
    assert ok.status_code == 200 and ok.json()["status"] == "connected"
    assert RUNPOD_KEY not in ok.text
    async with SessionLocal() as db:
        row = (await db.scalars(select(GPUIntegration))).all()[-1]
        assert RUNPOD_KEY.encode() not in row.encrypted_credentials
    # GPU compute is separate from image and AI integrations.
    images = (await user.get("/api/integrations")).json()["integrations"]
    assert all(i["status"] == "not_connected" for i in images)


async def test_runpod_workload_lifecycle(user: Api, runpod: FakeRunPod):
    await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
    types = (await user.get("/api/gpu/runpod/gpu-types")).json()["items"]
    assert types[0]["id"] == "NVIDIA L4" and types[0]["price_secure"] == 0.43
    spec = {
        "name": "comfy",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpu_type": "NVIDIA L4",
        "ports": [8188],
    }
    unconfirmed = await user.post("/api/gpu/runpod/workloads", json=spec)
    assert unconfirmed.status_code == 400
    assert unconfirmed.json()["error"]["code"] == "confirmation_required"
    created = await user.post("/api/gpu/runpod/workloads", json={**spec, "confirm_billing": True})
    assert created.status_code == 201, created.text
    pod_id = created.json()["id"]
    sent = next(c for c in runpod.calls if c[0] == "POST" and c[1] == "/v2/pods")[2]
    assert sent["gpu"] == {"id": "NVIDIA L4", "count": 1} and sent["ports"] == ["8188/http"]
    listing = (await user.get("/api/gpu/runpod/workloads")).json()["items"]
    assert listing[0]["status"] == "running" and listing[0]["from_studio"] is True
    assert listing[0]["urls"] == [f"https://{pod_id}-8188.proxy.runpod.net"]
    assert (await user.post(f"/api/gpu/runpod/workloads/{pod_id}/stop")).status_code == 200
    stopped = (await user.get("/api/gpu/runpod/workloads")).json()["items"][0]
    assert stopped["status"] == "stopped" and stopped["can_start"] is True
    no_confirm = await user.post(f"/api/gpu/runpod/workloads/{pod_id}/start", json={})
    assert no_confirm.status_code == 400
    started = await user.post(
        f"/api/gpu/runpod/workloads/{pod_id}/start", json={"confirm_billing": True}
    )
    assert started.status_code == 200
    usage = (await user.get("/api/gpu/runpod/usage")).json()
    assert usage["balance"] == 12.5 and usage["spend_per_hour"] == 0.43
    assert usage["studio_workloads"] == 1
    assert (await user.delete(f"/api/gpu/runpod/workloads/{pod_id}")).status_code == 200
    async with SessionLocal() as db:
        record = (
            await db.scalars(select(GPUWorkload).where(GPUWorkload.external_id == pod_id))
        ).one()
        assert record.status == "terminated" and record.stopped_at is not None


async def test_runpod_endpoints(user: Api, runpod: FakeRunPod):
    await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
    items = (await user.get("/api/gpu/runpod/endpoints")).json()["items"]
    assert items[0]["id"] == "ep1" and items[0]["health"]["workers"]["idle"] == 0
    always_on = await user.patch(
        "/api/gpu/runpod/endpoints/ep1", json={"workers_min": 1, "workers_max": 2}
    )
    assert always_on.status_code == 400
    r = await user.patch(
        "/api/gpu/runpod/endpoints/ep1",
        json={"workers_min": 1, "workers_max": 3, "confirm_billing": True},
    )
    assert r.status_code == 200
    assert runpod.endpoints["ep1"]["workers"]["max"] == 3
    assert (await user.delete("/api/gpu/runpod/endpoints/ep1")).status_code == 200


async def test_modal_sandboxes(user: Api, modal_sdk: FakeModalSDK):
    bad = await user.put(
        "/api/gpu/providers/modal", json={"token_id": "ak-123456", "token_secret": "wrong-secret"}
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_credentials"
    ok = await user.put(
        "/api/gpu/providers/modal",
        json={"token_id": "ak-123456", "token_secret": "as-secret-123456"},
    )
    assert ok.status_code == 200
    types = (await user.get("/api/gpu/modal/gpu-types")).json()["items"]
    assert {"T4", "H100", "B200"} <= {t["id"] for t in types}
    created = await user.post(
        "/api/gpu/modal/workloads",
        json={
            "name": "notebook",
            "image": "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            "gpu_type": "H100",
            "gpu_count": 2,
            "ports": [8000],
            "timeout_minutes": 30,
            "confirm_billing": True,
        },
    )
    assert created.status_code == 201, created.text
    assert modal_sdk.created[0].gpu_count == 2 and modal_sdk.created[0].timeout_minutes == 30
    items = (await user.get("/api/gpu/modal/workloads")).json()["items"]
    assert items[0]["status"] == "running" and items[0]["gpu_type"] == "H100"
    assert items[0]["urls"] == ["https://sb-1.modal.run"]
    stop = await user.post(f"/api/gpu/modal/workloads/{items[0]['id']}/stop")
    assert stop.status_code == 400 and stop.json()["error"]["code"] == "unsupported"
    assert (await user.delete(f"/api/gpu/modal/workloads/{items[0]['id']}")).status_code == 200
    after = (await user.get("/api/gpu/modal/workloads")).json()["items"]
    assert after[0]["status"] == "exited"
    assert (await user.get("/api/gpu/modal/endpoints")).json()["supported"] is False


async def test_gpu_is_private(user: Api, runpod: FakeRunPod):
    await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
    other = await make_user("Other")
    r = await other.get("/api/gpu/runpod/workloads")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_connected"
    await other.client.aclose()


async def test_runpod_requests_get_through_cloudflare():
    """RunPod's API sits behind Cloudflare, which refuses library User-Agents: the studio
    sends its own, and a Cloudflare block page is reported as such (not as a wrong key)."""
    fake = FakeRunPod()
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return fake.handle(request)

    creds = {"api_key": RUNPOD_KEY}
    types = await RunPodProvider(transport=httpx.MockTransport(answer)).gpu_types(creds)
    assert all(r.headers["user-agent"].startswith("ai-image-studio/") for r in seen)
    assert seen[0].url.path == "/v2/catalog/gpus"
    assert types[0].manufacturer == "NVIDIA" and types[0].price_secure == 0.43

    def blocked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"server": "cloudflare", "content-type": "text/plain; charset=UTF-8"},
            text="error code: 1010",
        )

    with pytest.raises(GPUError) as err:
        await RunPodProvider(transport=httpx.MockTransport(blocked)).gpu_types(creds)
    assert err.value.code == "provider_error"
    assert "Cloudflare" in err.value.message and "1010" in err.value.message


async def test_runpod_catalog_shows_only_free_gpus(user: Api, runpod: FakeRunPod):
    """The GPU list comes from RunPod's catalog per cloud: price, stock and the CUDA versions
    with free machines. A GPU counts as free only when a machine with a driver the task's
    image runs on is free; AMD GPUs are never offered."""
    await user.put("/api/gpu/providers/runpod", json={"api_key": RUNPOD_KEY})
    # Secure Cloud: only old-driver machines free; Community: 12.8 free.
    runpod.free[("NVIDIA L4", "SECURE")] = ["12.4"]
    runpod.free[("NVIDIA L4", "COMMUNITY")] = ["12.8"]
    runpod.free[("NVIDIA H100 80GB HBM3", "SECURE")] = ["13.0"]
    items = (await user.get("/api/gpu/runpod/gpu-types?purpose=comfyui")).json()["items"]
    by_id = {g["id"]: g for g in items}
    assert "AMD Instinct MI300X OAM" not in by_id
    l4 = by_id["NVIDIA L4"]
    assert l4["stock_secure"] == "NONE" and l4["stock_community"] == "HIGH"
    assert l4["price_community"] == 0.39 and l4["cuda_community"] == ["12.8"]
    h100 = by_id["NVIDIA H100 80GB HBM3"]
    assert h100["stock_secure"] == "HIGH" and h100["price_community"] is None
    # Training needs a CUDA 13 machine: the L4 in Community (12.8 only) does not fit.
    training = (await user.get("/api/gpu/runpod/gpu-types?purpose=training")).json()["items"]
    l4t = next(g for g in training if g["id"] == "NVIDIA L4")
    assert l4t["stock_community"] == "NONE"
