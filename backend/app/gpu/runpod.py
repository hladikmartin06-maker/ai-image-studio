"""RunPod: GPU pods and serverless endpoints.

REST API (https://docs.runpod.io/api-reference), "Authorization: Bearer <key>":
  GET    /v1/pods                       list pods
  POST   /v1/pods                       create a pod {name, imageName, gpuTypeIds, gpuCount,
                                        cloudType, containerDiskInGb, volumeInGb, ports, env,
                                        allowedCudaVersions (driver's CUDA, e.g. "13.0")}
  POST   /v1/pods/{id}/start | /stop    start / stop
  DELETE /v1/pods/{id}                  terminate
  GET    /v1/endpoints                  serverless endpoints
  PATCH  /v1/endpoints/{id}             {workersMin, workersMax}
  DELETE /v1/endpoints/{id}
Serverless health: GET https://api.runpod.ai/v2/{endpoint_id}/health
GraphQL (https://api.runpod.io/graphql): `gpuTypes {id displayName memoryInGb securePrice
communityPrice}` for the GPU catalog; `myself {clientBalance currentSpendPerHr}` for the
balance (shown only when the API returns it).
Pods are reached at https://{podId}-{port}.proxy.runpod.net for HTTP ports.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import get_settings
from app.gpu.base import Endpoint, GPUError, GpuType, Workload, WorkloadSpec


def _rest() -> str:
    return get_settings().runpod_rest_url.rstrip("/")


def _graphql_url() -> str:
    return get_settings().runpod_graphql_url


def _serverless() -> str:
    return get_settings().runpod_serverless_url.rstrip("/")


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:300]
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            if isinstance(data.get(key), str):
                return data[key]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("error") or data[0].get("message") or "")
    return ""


# How RunPod says that no machine is free for the request (phrases from RunPod's own pod
# tests in runpod-workers/comfyui-base, tests/runpod_smoke/pod.py).
UNAVAILABLE_RE = re.compile(
    r"no\s+longer\s+any\s+instances\s+available|does\s+not\s+have\s+the\s+resources"
    r"|no\s+(?:machines|capacity|hosts|gpus|instances)\s+(?:currently\s+)?available"
    r"|insufficient\s+capacity|out\s+of\s+stock|sold\s+out"
    # A stopped pod whose machine has no free GPU left to start it again (wording as
    # RunPod's console shows it; not checked against the API).
    r"|not\s+enough\s+free\s+gpus",
    re.IGNORECASE,
)


# Driver CUDA versions for allowedCudaVersions. RunPod's published API (runpod/docs,
# api-reference/openapi.json) lists versions up to "13.0"; newer drivers are asked for too,
# and if RunPod refuses them the request is sent again with CUDA13_DOCUMENTED (a refused
# request created no pod, so nothing can be billed twice).
CUDA12 = ["12.8", "12.9"]
CUDA13 = ["13.0", "13.1", "13.2", "13.3"]
CUDA13_DOCUMENTED = ["13.0"]


def refused_cuda(exc: GPUError) -> bool:
    """RunPod rejected the allowedCudaVersions values themselves."""
    return exc.code == "invalid_request" and "cuda" in exc.detail.lower()


def _error(response: httpx.Response) -> GPUError:
    status = response.status_code
    message = _message(response)
    lowered = message.lower()
    if UNAVAILABLE_RE.search(message):
        return GPUError("unavailable", message)
    if status in (401, 403):
        return GPUError("invalid_credentials", message)
    if status == 402 or "balance" in lowered or "insufficient" in lowered:
        return GPUError("insufficient_credits", message)
    if status == 404:
        return GPUError("not_found", message)
    if status == 429:
        return GPUError("rate_limited", message)
    if status >= 500:
        return GPUError("provider_error", message)
    return GPUError("invalid_request", message)


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class RunPodProvider:
    key = "runpod"
    supports_endpoints = True
    supports_stop_start = True

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    def _client(self, creds: dict[str, str]) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {creds['api_key']}"},
            timeout=httpx.Timeout(60, connect=15),
            transport=self.transport,
        )

    async def _call(self, creds: dict[str, str], method: str, url: str, **kwargs: Any) -> Any:
        try:
            async with self._client(creds) as client:
                response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise GPUError("network_error", type(exc).__name__) from exc
        if response.status_code >= 400:
            raise _error(response)
        if not response.content:
            return None
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            return None

    async def _graphql(self, creds: dict[str, str], query: str) -> dict[str, Any]:
        data = await self._call(creds, "POST", _graphql_url(), json={"query": query})
        if not isinstance(data, dict):
            raise GPUError("provider_error", "Unexpected GraphQL response")
        if data.get("errors"):
            first = data["errors"][0]
            raise GPUError("invalid_request", str(first.get("message", "")))
        return data.get("data") or {}

    async def check(self, creds: dict[str, str]) -> None:
        await self._call(creds, "GET", f"{_rest()}/pods")

    async def gpu_types(self, creds: dict[str, str]) -> list[GpuType]:
        data = await self._graphql(
            creds,
            "query GpuTypes { gpuTypes { id displayName memoryInGb securePrice communityPrice } }",
        )
        out = [
            GpuType(
                id=str(g.get("id")),
                name=str(g.get("displayName") or g.get("id")),
                memory_gb=g.get("memoryInGb"),
                price_secure=_num(g.get("securePrice")),
                price_community=_num(g.get("communityPrice")),
                available=bool(g.get("securePrice") or g.get("communityPrice")),
            )
            for g in data.get("gpuTypes") or []
            if g.get("id")
        ]
        return sorted(out, key=lambda g: (g.memory_gb or 0, g.name))

    def _workload(self, pod: dict[str, Any]) -> Workload:
        pod_id = str(pod.get("id"))
        status = str(pod.get("desiredStatus") or pod.get("status") or "unknown").lower()
        status = {"running": "running", "exited": "stopped", "terminated": "terminated"}.get(
            status, status
        )
        gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
        machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
        ports = pod.get("ports") or []
        if isinstance(ports, str):
            ports = [p.strip() for p in ports.split(",") if p.strip()]
        urls = [
            f"https://{pod_id}-{str(p).split('/')[0]}.proxy.runpod.net"
            for p in ports
            if str(p).endswith("/http")
        ]
        return Workload(
            id=pod_id,
            name=str(pod.get("name") or pod_id),
            kind="pod",
            status=status,
            gpu_type=str(
                gpu.get("displayName") or gpu.get("id") or machine.get("gpuDisplayName") or ""
            ),
            gpu_count=int(pod.get("gpuCount") or gpu.get("count") or 1),
            image=str(pod.get("imageName") or pod.get("image") or ""),
            cost_per_hour=_num(pod.get("costPerHr") or pod.get("adjustedCostPerHr")),
            urls=urls if status == "running" else [],
            created_at=pod.get("createdAt"),
            dashboard_url="https://console.runpod.io/pods",
            can_start=status in ("stopped", "exited"),
            can_stop=status == "running",
        )

    async def list_workloads(self, creds: dict[str, str]) -> list[Workload]:
        data = await self._call(creds, "GET", f"{_rest()}/pods")
        pods = data if isinstance(data, list) else (data or {}).get("pods", [])
        return [self._workload(p) for p in pods if isinstance(p, dict)]

    async def create_workload(self, creds: dict[str, str], spec: WorkloadSpec) -> Workload:
        body: dict[str, Any] = {
            "name": spec.name,
            "imageName": spec.image,
            "gpuTypeIds": [spec.gpu_type],
            "gpuCount": spec.gpu_count,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.container_disk_gb,
            "volumeInGb": spec.volume_gb,
            "ports": [f"{p}/http" for p in spec.ports],
        }
        env = {**spec.env, **spec.secret_env}
        if env:
            body["env"] = env
        if spec.entrypoint:
            body["dockerEntrypoint"] = list(spec.entrypoint)
            body["dockerStartCmd"] = list(spec.command)
        elif spec.command:
            # Entrypoint + arguments, so the image's own entrypoint does not wrap them.
            body["dockerEntrypoint"] = spec.command[:1]
            body["dockerStartCmd"] = spec.command[1:]
        if spec.volume_gb:
            body["volumeMountPath"] = "/workspace"
        if spec.allowed_cuda_versions:
            body["allowedCudaVersions"] = list(spec.allowed_cuda_versions)
        data = await self._call(creds, "POST", f"{_rest()}/pods", json=body)
        if not isinstance(data, dict) or not data.get("id"):
            raise GPUError("provider_error", "No pod id in the response")
        return self._workload({**{k: v for k, v in body.items() if k != "env"}, **data})

    async def workload(self, creds: dict[str, str], workload_id: str) -> Workload | None:
        try:
            data = await self._call(creds, "GET", f"{_rest()}/pods/{workload_id}")
        except GPUError as exc:
            if exc.code == "not_found":
                return None
            raise
        return self._workload(data) if isinstance(data, dict) and data.get("id") else None

    async def port_url(self, creds: dict[str, str], workload_id: str, port: int) -> str | None:
        # RunPod's HTTP proxy (documented): https://{podId}-{port}.proxy.runpod.net
        base = get_settings().runpod_proxy_url
        return base.format(pod_id=workload_id, port=port)

    async def start(self, creds: dict[str, str], workload_id: str) -> None:
        await self._call(creds, "POST", f"{_rest()}/pods/{workload_id}/start")

    async def stop(self, creds: dict[str, str], workload_id: str) -> None:
        await self._call(creds, "POST", f"{_rest()}/pods/{workload_id}/stop")

    async def terminate(self, creds: dict[str, str], workload_id: str) -> None:
        await self._call(creds, "DELETE", f"{_rest()}/pods/{workload_id}")

    async def list_endpoints(self, creds: dict[str, str]) -> list[Endpoint]:
        data = await self._call(creds, "GET", f"{_rest()}/endpoints")
        items = data if isinstance(data, list) else (data or {}).get("endpoints", [])
        out: list[Endpoint] = []
        for e in items:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            endpoint_id = str(e["id"])
            health = None
            try:
                health = await self._call(creds, "GET", f"{_serverless()}/{endpoint_id}/health")
            except GPUError:
                health = None
            out.append(
                Endpoint(
                    id=endpoint_id,
                    name=str(e.get("name") or endpoint_id),
                    gpu_types=[str(g) for g in e.get("gpuTypeIds") or []],
                    workers_min=int(e.get("workersMin") or 0),
                    workers_max=int(e.get("workersMax") or 0),
                    url=f"{_serverless()}/{endpoint_id}",
                    health=health if isinstance(health, dict) else None,
                )
            )
        return out

    async def scale_endpoint(
        self, creds: dict[str, str], endpoint_id: str, workers_min: int, workers_max: int
    ) -> None:
        await self._call(
            creds,
            "PATCH",
            f"{_rest()}/endpoints/{endpoint_id}",
            json={"workersMin": workers_min, "workersMax": workers_max},
        )

    async def delete_endpoint(self, creds: dict[str, str], endpoint_id: str) -> None:
        await self._call(creds, "DELETE", f"{_rest()}/endpoints/{endpoint_id}")

    async def account(self, creds: dict[str, str]) -> dict[str, Any]:
        """Balance and current spend; spend falls back to the running pods' own rates."""
        result: dict[str, Any] = {"balance": None, "spend_per_hour": None, "source": None}
        try:
            data = await self._graphql(
                creds, "query Myself { myself { clientBalance currentSpendPerHr } }"
            )
            me = data.get("myself") or {}
            result["balance"] = _num(me.get("clientBalance"))
            result["spend_per_hour"] = _num(me.get("currentSpendPerHr"))
            result["source"] = "account"
        except GPUError as exc:
            if exc.code in ("invalid_credentials", "network_error"):
                raise
        if result["spend_per_hour"] is None:
            pods = await self.list_workloads(creds)
            rates = [p.cost_per_hour for p in pods if p.status == "running" and p.cost_per_hour]
            result["spend_per_hour"] = round(sum(rates), 4) if rates else 0.0
            result["source"] = result["source"] or "pods"
        return result
