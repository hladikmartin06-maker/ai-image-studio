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
from app.gpu.base import WorkloadSpec
from app.gpu.modal_provider import ModalProvider
from app.gpu.registry import set_gpu_adapter
from app.gpu.runpod import RunPodProvider
from app.models import GPUIntegration, GPUWorkload
from tests.conftest import Api, make_user

RUNPOD_KEY = "rp_test_key_123456"


@dataclass
class FakeRunPod:
    pods: dict[str, dict[str, Any]] = field(default_factory=dict)
    endpoints: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "ep1": {
                "id": "ep1",
                "name": "sdxl",
                "gpuTypeIds": ["NVIDIA L4"],
                "workersMin": 0,
                "workersMax": 2,
            }
        }
    )
    calls: list[tuple[str, str, Any]] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {RUNPOD_KEY}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        host, path = request.url.host, request.url.path
        if host == "api.runpod.io":
            query = body["query"]
            if "gpuTypes" in query:
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "gpuTypes": [
                                {
                                    "id": "NVIDIA L4",
                                    "displayName": "L4",
                                    "memoryInGb": 24,
                                    "securePrice": 0.43,
                                    "communityPrice": 0.39,
                                },
                                {
                                    "id": "NVIDIA H100 80GB HBM3",
                                    "displayName": "H100 SXM",
                                    "memoryInGb": 80,
                                    "securePrice": 2.99,
                                    "communityPrice": None,
                                },
                            ]
                        }
                    },
                )
            return httpx.Response(
                200, json={"data": {"myself": {"clientBalance": 12.5, "currentSpendPerHr": 0.43}}}
            )
        if host == "api.runpod.ai":
            return httpx.Response(
                200, json={"jobs": {"inQueue": 0}, "workers": {"idle": 0, "running": 0}}
            )
        if path == "/v1/pods" and request.method == "GET":
            return httpx.Response(200, json=list(self.pods.values()))
        if path == "/v1/pods" and request.method == "POST":
            pod_id = f"pod{next(self._ids)}"
            self.pods[pod_id] = {
                "id": pod_id,
                "name": body["name"],
                "desiredStatus": "RUNNING",
                "imageName": body["imageName"],
                "gpuCount": body["gpuCount"],
                "costPerHr": 0.43,
                "ports": body["ports"],
                "gpu": {"id": body["gpuTypeIds"][0], "displayName": "L4"},
            }
            return httpx.Response(200, json=self.pods[pod_id])
        if path.startswith("/v1/pods/"):
            parts = path.split("/")
            pod_id = parts[3]
            if pod_id not in self.pods:
                return httpx.Response(404, json={"error": "pod not found"})
            if request.method == "DELETE":
                self.pods.pop(pod_id)
            elif parts[-1] == "stop":
                self.pods[pod_id]["desiredStatus"] = "EXITED"
            elif parts[-1] == "start":
                self.pods[pod_id]["desiredStatus"] = "RUNNING"
            return httpx.Response(200, json={})
        if path == "/v1/endpoints":
            return httpx.Response(200, json=list(self.endpoints.values()))
        if path.startswith("/v1/endpoints/"):
            endpoint_id = path.split("/")[3]
            if request.method == "PATCH":
                self.endpoints[endpoint_id].update(body)
            else:
                self.endpoints.pop(endpoint_id, None)
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": "unknown"})


class FakeSandbox:
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.code: int | None = None


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

    async def list(self, client, app):
        return list(self.sandboxes.values())

    async def create(self, client, app, spec):
        sb = FakeSandbox(f"sb-{next(self._ids)}")
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
    sent = next(c for c in runpod.calls if c[0] == "POST" and c[1] == "/v1/pods")[2]
    assert sent["gpuTypeIds"] == ["NVIDIA L4"] and sent["ports"] == ["8188/http"]
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
    assert runpod.endpoints["ep1"]["workersMax"] == 3
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
