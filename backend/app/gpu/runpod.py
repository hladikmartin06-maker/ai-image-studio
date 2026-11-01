"""RunPod: GPU pods and serverless endpoints, through RunPod's REST API v2.

REST API v2 (https://api.runpod.io/v2, docs.runpod.io/api-reference-v2; v1 at
rest.runpod.io/v1 is retired on 2026-11-15), "Authorization: Bearer <key>":
  GET    /v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE|COMMUNITY
                                    GPU types: price per cloud, availability (NONE/LOW/
                                    MEDIUM/HIGH), dataCenters, cudaVersions [{version,
                                    available}] — what is actually free right now
  GET    /v2/catalog/gpus/{id}?…    the same for one GPU type
  GET    /v2/pods                   {pods: [...], pagination: {nextCursor, hasNextPage}}
  POST   /v2/pods                   create: {name, image, gpu: {id, count,
                                    allowedCudaVersions}, cloud, disk, ports, env,
                                    entrypoint, cmd, mounts: {persistent: {size, path}}}
  GET    /v2/pods/{id}              status PROVISIONING|STARTING|RUNNING|EXITED|ERROR|
                                    TERMINATED, cost (USD/h), ports, gpu, cudaVersion
  POST   /v2/pods/{id}/action       {"action": "start"|"stop"|"restart"|"terminate"}
  DELETE /v2/pods/{id}
  GET    /v2/serverless, PATCH /v2/serverless/{id} {workers: {min, max}}, DELETE …
Errors are RFC 9457 objects {title, status, detail, errors}. A capacity refusal is a 400 or
409 (no pod was created; runpodctl's pods create treats them the same way), a 422 is a
malformed body, and a 5xx or no answer may have created a pod — never retried.
allowedCudaVersions match exactly: a version no free machine reports yields a capacity
error, so the versions are taken from the catalog's cudaVersions.
Serverless health: GET https://api.runpod.ai/v2/{endpoint_id}/health
GraphQL (https://api.runpod.io/graphql) `myself {clientBalance currentSpendPerHr}` for the
balance (shown only when the API returns it).
api.runpod.io is behind Cloudflare: requests carry the studio's own User-Agent.
Pods are reached at https://{podId}-{port}.proxy.runpod.net for HTTP ports.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

import httpx

from app.config import USER_AGENT, get_settings
from app.gpu.base import Endpoint, GPUError, GpuType, Workload, WorkloadSpec

Cloud = str  # "SECURE" | "COMMUNITY"
CLOUDS = ("SECURE", "COMMUNITY")
STOCK = ("NONE", "LOW", "MEDIUM", "HIGH")


def _api() -> str:
    return get_settings().runpod_api_url.rstrip("/")


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
        parts = [
            str(data[key])
            for key in ("detail", "error", "message", "title")
            if isinstance(data.get(key), str) and data[key]
        ][:1]
        errors = data.get("errors")
        if isinstance(errors, list):
            parts += [str(e) for e in errors[:5]]
        return "; ".join(parts)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("error") or data[0].get("message") or "")
    return ""


# How RunPod says that no machine is free for the request (v1 wording "create pod: There are
# no instances currently available", and phrases from RunPod's own pod tests in
# runpod-workers/comfyui-base, tests/runpod_smoke/pod.py).
UNAVAILABLE_RE = re.compile(
    r"no\s+longer\s+any\s+instances\s+available|does\s+not\s+have\s+the\s+resources"
    r"|no\s+(?:machines|capacity|hosts|gpus|instances)\s+(?:currently\s+)?available"
    r"|insufficient\s+capacity|out\s+of\s+stock|sold\s+out"
    # A stopped pod whose machine has no free GPU left to start it again (wording as
    # RunPod's console shows it; not checked against the API).
    r"|not\s+enough\s+free\s+gpus",
    re.IGNORECASE,
)


def _cloudflare(response: httpx.Response) -> bool:
    """A block page from Cloudflare in front of RunPod (not RunPod's own answer)."""
    return "cloudflare" in response.headers.get("server", "").lower() and "json" not in (
        response.headers.get("content-type", "")
    )


def _cloudflare_code(response: httpx.Response) -> str:
    match = re.search(r"error code:?\s*(\d{3,4})", response.text[:2000], re.IGNORECASE)
    return f"error {match.group(1)}" if match else ""


def _error(response: httpx.Response) -> GPUError:
    status = response.status_code
    message = _message(response)
    lowered = message.lower()
    if status in (403, 429, 503) and _cloudflare(response):
        return GPUError(
            "provider_error",
            "RunPod's firewall (Cloudflare) blocked the request"
            + (f": {_cloudflare_code(response)}" if _cloudflare_code(response) else ""),
        )
    if status == 409 or (status < 500 and UNAVAILABLE_RE.search(message)):
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


def _stock(value: Any) -> str:
    level = str(value or "NONE").upper()
    return level if level in STOCK else "NONE"


def free_cuda(gpu: dict[str, Any]) -> list[str]:
    """CUDA versions of machines with this GPU that have free capacity now (catalog)."""
    out = []
    for item in gpu.get("cudaVersions") or []:
        if isinstance(item, dict) and item.get("available") and item.get("version"):
            out.append(str(item["version"]))
    return out


def _cuda_key(version: str) -> tuple[int, int]:
    major, _, minor = version.partition(".")
    try:
        return int(major), int(minor or 0)
    except ValueError:
        return 0, 0


class RunPodProvider:
    key = "runpod"
    supports_endpoints = True
    supports_stop_start = True

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    def _client(self, creds: dict[str, str]) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {creds['api_key']}", "User-Agent": USER_AGENT},
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

    async def _all_pages(self, creds: dict[str, str], path: str, key: str) -> list[Any]:
        """Every item of a paginated v2 list (cursor pagination)."""
        items: list[Any] = []
        params: dict[str, Any] = {"limit": 1000}
        seen: set[str] = set()
        while True:
            data = await self._call(creds, "GET", f"{_api()}{path}", params=params)
            if isinstance(data, list):  # tolerate a bare list
                return items + data
            data = data or {}
            items += list(data.get(key) or [])
            page = data.get("pagination") or {}
            cursor = page.get("nextCursor")
            if not page.get("hasNextPage") or not cursor or cursor in seen:
                return items
            seen.add(cursor)
            params = {"limit": 1000, "cursor": cursor}

    async def check(self, creds: dict[str, str]) -> None:
        await self._call(creds, "GET", f"{_api()}/pods", params={"limit": 1})

    # --- catalog ----------------------------------------------------------------------

    async def _catalog(self, creds: dict[str, str], cloud: Cloud) -> list[dict[str, Any]]:
        data = await self._call(
            creds,
            "GET",
            f"{_api()}/catalog/gpus",
            params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud},
        )
        return [g for g in (data or {}).get("gpus") or [] if isinstance(g, dict)]

    async def gpu_types(self, creds: dict[str, str]) -> list[GpuType]:
        """NVIDIA GPU types with each cloud's price, stock and the CUDA versions that have
        free machines right now."""
        per_cloud = {cloud: await self._catalog(creds, cloud) for cloud in CLOUDS}
        merged: dict[str, GpuType] = {}
        for cloud, gpus in per_cloud.items():
            for g in gpus:
                gpu_id = str(g.get("id") or "")
                if not gpu_id or gpu_id == "unknown":
                    continue
                if str(g.get("manufacturer") or "").upper() not in ("NVIDIA", ""):
                    continue
                price = g.get("price") or {}
                item = merged.setdefault(
                    gpu_id,
                    GpuType(
                        id=gpu_id,
                        name=str(g.get("name") or gpu_id),
                        memory_gb=g.get("memory"),
                        price_secure=_num(price.get("secure")) if g.get("secure") else None,
                        price_community=(
                            _num(price.get("community")) if g.get("community") else None
                        ),
                        manufacturer=str(g.get("manufacturer") or "") or None,
                    ),
                )
                offered = bool(g.get("secure") if cloud == "SECURE" else g.get("community"))
                stock = _stock(g.get("availability")) if offered else "NONE"
                cuda = free_cuda(g) if offered else []
                if cloud == "SECURE":
                    item.stock_secure, item.cuda_secure = stock, cuda
                else:
                    item.stock_community, item.cuda_community = stock, cuda
        out = list(merged.values())
        for g in out:
            g.available = (g.stock_secure or "NONE") != "NONE" or (
                g.stock_community or "NONE"
            ) != "NONE"
        return sorted(out, key=lambda g: (g.memory_gb or 0, g.name))

    async def catalog_gpu(
        self, creds: dict[str, str], gpu_id: str, cloud: Cloud
    ) -> dict[str, Any] | None:
        """One GPU type in this cloud right now (price, availability, cudaVersions), or None
        when the catalog cannot be read."""
        try:
            data = await self._call(
                creds,
                "GET",
                f"{_api()}/catalog/gpus/{quote(gpu_id, safe='')}",
                params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud},
            )
        except GPUError as exc:
            if exc.code in ("invalid_credentials", "not_found"):
                raise
            return None
        return data if isinstance(data, dict) else None

    async def offer(
        self, creds: dict[str, str], gpu_id: str, cloud: Cloud
    ) -> tuple[list[str] | None, float | None]:
        """(CUDA versions with a free machine, oldest first; the list price per GPU per hour)
        for this GPU and cloud; (None, None) when the catalog cannot be read."""
        item = await self.catalog_gpu(creds, gpu_id, cloud)
        if item is None:
            return None, None
        offered = item.get("secure") if cloud == "SECURE" else item.get("community")
        price = (item.get("price") or {}).get("secure" if cloud == "SECURE" else "community")
        free = sorted(free_cuda(item), key=_cuda_key) if offered else []
        return free, (_num(price) if offered else None)

    # --- pods -------------------------------------------------------------------------

    def _workload(self, pod: dict[str, Any]) -> Workload:
        pod_id = str(pod.get("id"))
        raw = str(pod.get("status") or pod.get("desiredStatus") or "unknown").upper()
        status = {
            "RUNNING": "running",
            "EXITED": "stopped",
            "TERMINATED": "terminated",
            "PROVISIONING": "starting",
            "STARTING": "starting",
            "ERROR": "error",
        }.get(raw, raw.lower())
        gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
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
            gpu_type=str(gpu.get("id") or ""),
            gpu_count=int(gpu.get("count") or 1),
            image=str(pod.get("image") or ""),
            cost_per_hour=_num(pod.get("cost")),
            urls=urls if status == "running" else [],
            created_at=pod.get("createdAt"),
            dashboard_url="https://console.runpod.io/pods",
            can_start=status == "stopped",
            can_stop=status == "running",
        )

    async def list_workloads(self, creds: dict[str, str]) -> list[Workload]:
        pods = await self._all_pages(creds, "/pods", "pods")
        return [self._workload(p) for p in pods if isinstance(p, dict)]

    @staticmethod
    def pod_body(spec: WorkloadSpec) -> dict[str, Any]:
        """CreatePodRequest (v2). Only what the pod needs: no data center, vCPU, RAM or
        network volume constraints, so any free machine can take it."""
        gpu: dict[str, Any] = {"id": spec.gpu_type, "count": spec.gpu_count}
        if spec.allowed_cuda_versions:
            gpu["allowedCudaVersions"] = list(spec.allowed_cuda_versions)
        body: dict[str, Any] = {
            "name": spec.name,
            "image": spec.image,
            "gpu": gpu,
            "cloud": spec.cloud_type,
            "disk": spec.container_disk_gb,
            "ports": [f"{p}/http" for p in spec.ports],
        }
        env = {**spec.env, **spec.secret_env}
        if env:
            body["env"] = env
        if spec.entrypoint:
            body["entrypoint"] = list(spec.entrypoint)
            body["cmd"] = list(spec.command)
        elif spec.command:
            # Entrypoint + arguments, so the image's own entrypoint does not wrap them.
            body["entrypoint"] = spec.command[:1]
            body["cmd"] = spec.command[1:]
        if spec.volume_gb:
            body["mounts"] = {"persistent": {"size": spec.volume_gb, "path": "/workspace"}}
        return body

    async def create_workload(self, creds: dict[str, str], spec: WorkloadSpec) -> Workload:
        body = self.pod_body(spec)
        data = await self._call(creds, "POST", f"{_api()}/pods", json=body)
        if not isinstance(data, dict) or not data.get("id"):
            raise GPUError("provider_error", "No pod id in the response")
        return self._workload(
            {**{k: v for k, v in body.items() if k not in ("env", "cmd")}, **data}
        )

    async def workload(self, creds: dict[str, str], workload_id: str) -> Workload | None:
        try:
            data = await self._call(creds, "GET", f"{_api()}/pods/{workload_id}")
        except GPUError as exc:
            if exc.code == "not_found":
                return None
            raise
        return self._workload(data) if isinstance(data, dict) and data.get("id") else None

    async def port_url(self, creds: dict[str, str], workload_id: str, port: int) -> str | None:
        # RunPod's HTTP proxy (documented): https://{podId}-{port}.proxy.runpod.net
        base = get_settings().runpod_proxy_url
        return base.format(pod_id=workload_id, port=port)

    async def _action(self, creds: dict[str, str], workload_id: str, action: str) -> None:
        await self._call(
            creds, "POST", f"{_api()}/pods/{workload_id}/action", json={"action": action}
        )

    async def start(self, creds: dict[str, str], workload_id: str) -> None:
        await self._action(creds, workload_id, "start")

    async def stop(self, creds: dict[str, str], workload_id: str) -> None:
        await self._action(creds, workload_id, "stop")

    async def terminate(self, creds: dict[str, str], workload_id: str) -> None:
        await self._call(creds, "DELETE", f"{_api()}/pods/{workload_id}")

    # --- serverless ---------------------------------------------------------------------

    async def list_endpoints(self, creds: dict[str, str]) -> list[Endpoint]:
        items = await self._all_pages(creds, "/serverless", "endpoints")
        out: list[Endpoint] = []
        for e in items:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            endpoint_id = str(e["id"])
            try:
                health = await self._call(creds, "GET", f"{_serverless()}/{endpoint_id}/health")
            except GPUError:
                health = None
            workers = e.get("workers") or {}
            gpu = e.get("gpu") or {}
            out.append(
                Endpoint(
                    id=endpoint_id,
                    name=str(e.get("name") or endpoint_id),
                    gpu_types=[str(g) for g in gpu.get("pools") or []],
                    workers_min=int(workers.get("min") or 0),
                    workers_max=int(workers.get("max") or 0),
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
            f"{_api()}/serverless/{endpoint_id}",
            json={"workers": {"min": workers_min, "max": workers_max}},
        )

    async def delete_endpoint(self, creds: dict[str, str], endpoint_id: str) -> None:
        await self._call(creds, "DELETE", f"{_api()}/serverless/{endpoint_id}")

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
