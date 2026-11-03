"""Modal: GPU sandboxes through the official `modal` Python SDK.

Credentials are a token id + token secret (modal.com › Settings › API tokens), used with
`modal.Client.from_credentials` — the documented way to manage Modal on behalf of a user.
Workloads are Modal Sandboxes created in one app ("ai-image-studio-gpu") and tagged
`created-by=ai-image-studio`:
  modal.Sandbox.create(*command, app=…, image=modal.Image.from_registry(…).entrypoint([]),
                       gpu="H100:2",
                       timeout=…, encrypted_ports=[…], tags=…, client=…)
  modal.Sandbox.list(app_id=…, tags=…, client=…), Sandbox.from_id(…).poll()/.terminate()
  sandbox.tunnels() -> {port: Tunnel(url)} for exposed ports (the workload's endpoints)
  sandbox.logs.tail(n) -> the last stdout / stderr / system log entries (also after it ended)
  modal.Workspace.from_context(client=…).billing.rates() -> the workspace's current pricing
      rates, a mapping of rate name -> Decimal (modal SDK 1.6, `modal billing rates`)
Sandboxes cannot be paused: they run until they finish, time out or are terminated.
Modal has no balance API; usage is shown from the workloads started here.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from app.gpu.base import Endpoint, GPUError, GpuType, Workload, WorkloadSpec

APP_NAME = "ai-image-studio-gpu"
TAGS = {"created-by": "ai-image-studio"}

log = logging.getLogger(__name__)

# GPU names accepted by Modal's `gpu=` argument (modal.com/docs/guide/gpu; the same names are
# used in modal-labs/modal-examples), VRAM in GB, and Modal's list price in USD per hour
# (billed per second): the owner's last known values from https://modal.com/pricing
# (2026-11-01; the page could not be opened from the build sandbox). The live price comes from
# the workspace's rates (`live_rates`); this table is the fallback, shown as an estimate with
# that link, and the yardstick that rejects a rate read the wrong way. H200: no known value.
PRICING_URL = "https://modal.com/pricing"
GPU_TYPES: list[tuple[str, str, int, float | None]] = [
    ("T4", "NVIDIA T4", 16, 0.59),
    ("L4", "NVIDIA L4", 24, 0.80),
    ("A10G", "NVIDIA A10G", 24, 1.10),
    ("L40S", "NVIDIA L40S", 48, 1.95),
    ("A100-40GB", "NVIDIA A100 40 GB", 40, 2.10),
    ("A100-80GB", "NVIDIA A100 80 GB", 80, 2.50),
    ("H100", "NVIDIA H100", 80, 3.95),
    ("H200", "NVIDIA H200", 141, None),
    ("B200", "NVIDIA B200", 180, 6.25),
]
# Modal answers "Insufficient capacity to create sandbox" (ResourceExhaustedError) when it has
# no free GPU of the type for the workspace right now. Nothing exists or is billed then, so
# the studio asks again after these pauses (seconds; the last one repeats) for up to
# CAPACITY_WAIT, then gives up with a clear message (comfy/deploy.py, lora/runner.py).
CAPACITY_PAUSES = [20, 40, 60]
CAPACITY_WAIT = timedelta(minutes=20)

# CPU and memory are billed on top (last known: per physical core-hour, per GiB-hour).
CPU_CORE_HOUR = 0.0472
MEMORY_GIB_HOUR = 0.008
# The studio's sandboxes (ComfyUI, training): 4 cores and 32 GiB of memory.
SANDBOX_CPU = 4
SANDBOX_MEMORY_MB = 32768
SANDBOX_EXTRA_PER_HOUR = round(
    SANDBOX_CPU * CPU_CORE_HOUR + SANDBOX_MEMORY_MB / 1024 * MEMORY_GIB_HOUR, 4
)


# --- live rates (Workspace.billing.rates) ---------------------------------------------------
#
# The SDK documents only "a single mapping containing cost values" (Decimal), not the names of
# the rates or their unit. So the names are matched loosely (GPU names with or without
# "gpu"/"nvidia", memory, punctuation), the unit comes from the name when it says so ("sec",
# "hour") and otherwise from what fits the known list prices, and a value that is far from the
# list price is ignored (that GPU then falls back to the table). Nothing is ever made up: a
# price is either Modal's own rate or the labelled list price.

RATES_TTL = 600  # seconds; failures are retried after RATES_RETRY
RATES_RETRY = 120
_NOISE = {
    "gpu", "gpus", "nvidia", "price", "prices", "rate", "rates", "cost", "usd", "per", "unit",
    "sec", "secs", "second", "seconds", "hour", "hours", "hr", "hrs", "hourly",
}  # fmt: skip
_PER_SECOND = {"sec", "secs", "second", "seconds"}
_PER_HOUR = {"hour", "hours", "hr", "hrs", "hourly"}
_CPU_NAMES = {
    "cpu",
    "cpus",
    "core",
    "cores",
    "cpucore",
    "cpucores",
    "physicalcore",
    "cpuphysicalcore",
}
_MEMORY_NAMES = {"memory", "mem", "ram", "memorygib", "memorygb", "gibmemory", "memorygibs"}


def _aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for gpu_id, _, memory, _ in GPU_TYPES:
        base = re.sub(r"[^a-z0-9]", "", gpu_id.lower())
        out[base] = gpu_id
        out[f"{base}{memory}gb"] = gpu_id
        out[base.removesuffix("gb")] = gpu_id  # a10040gb -> a10040
    out["a10"] = "A10G"  # Modal's pricing page says "A10"; `gpu=` takes "A10G"
    out["a1024gb"] = "A10G"
    return out


_GPU_ALIASES = _aliases()


@dataclass
class LiveRates:
    """Modal's current rates for the workspace, in USD per hour."""

    gpu_hour: dict[str, float] = field(default_factory=dict)
    # CPU + memory of the studio's sandbox per hour, when both rates were found.
    extra_hour: float | None = None
    # Which rate name each price came from, and the unit used (for the logs and the docs).
    keys: dict[str, str] = field(default_factory=dict)
    unit: str = ""


def _number(value: Any) -> float | None:
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _plausible(hourly: float, reference: float | None) -> bool:
    if reference is None:
        return 0.05 <= hourly <= 100
    return reference / 3 <= hourly <= reference * 3


def parse_rates(raw: dict[str, Any]) -> LiveRates | None:
    """GPU, CPU and memory prices per hour from the workspace's rates, or None when nothing
    can be read safely."""
    table = {i: p for i, _, _, p in GPU_TYPES}
    found: list[tuple[str, str, float, str | None]] = []  # (what, key, value, unit hint)
    for key, value in raw.items():
        number = _number(value)
        if number is None:
            continue
        tokens = re.findall(r"[a-z0-9]+", str(key).lower())
        hint = (
            "second" if _PER_SECOND & set(tokens) else "hour" if _PER_HOUR & set(tokens) else None
        )
        name = "".join(t for t in tokens if t not in _NOISE)
        if name in _GPU_ALIASES:
            found.append((_GPU_ALIASES[name], str(key), number, hint))
        elif name in _CPU_NAMES:
            found.append(("cpu", str(key), number, hint))
        elif name in _MEMORY_NAMES:
            found.append(("memory", str(key), number, hint))

    # The unit of rates whose name does not say it: the one that fits the known list prices.
    votes = {"second": 0, "hour": 0}
    for what, _, number, hint in found:
        reference = table.get(what)
        if hint is None and reference:
            votes["second"] += _plausible(number * 3600, reference)
            votes["hour"] += _plausible(number, reference)
    default_unit = (
        "second"
        if votes["second"] > votes["hour"]
        else "hour"
        if votes["hour"] > votes["second"]
        else None
    )

    rates = LiveRates(unit=default_unit or "")
    extras: dict[str, float] = {}
    for what, key, number, hint in found:
        unit = hint or default_unit
        if unit is None:
            continue
        hourly = number * 3600 if unit == "second" else number
        if what == "cpu":
            if _plausible(hourly, CPU_CORE_HOUR):
                extras["cpu"] = hourly
                rates.keys["cpu"] = key
        elif what == "memory":
            if _plausible(hourly, MEMORY_GIB_HOUR):
                extras["memory"] = hourly
                rates.keys["memory"] = key
        elif what not in rates.gpu_hour and _plausible(hourly, table.get(what)):
            rates.gpu_hour[what] = round(hourly, 4)
            rates.keys[what] = key
    if "cpu" in extras and "memory" in extras:
        rates.extra_hour = round(
            SANDBOX_CPU * extras["cpu"] + SANDBOX_MEMORY_MB / 1024 * extras["memory"], 4
        )
    return rates if rates.gpu_hour else None


def hourly_estimate(gpu: str) -> float | None:
    """GPU + CPU + memory of the studio's sandbox, USD per hour (estimate), or None."""
    price = next((p for i, _, _, p in GPU_TYPES if i == gpu), None)
    return round(price + SANDBOX_EXTRA_PER_HOUR, 4) if price is not None else None


# Lines that could carry a secret are left out of what the studio shows.
_SECRET_LINE = re.compile(r"password|token|secret|authorization|api[_-]?key", re.IGNORECASE)


def _clean_output(text: str, limit: int = 1500) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not _SECRET_LINE.search(ln)]
    return "\n".join(lines)[-limit:]


def _error(exc: Exception) -> GPUError:
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
    # "Insufficient capacity to create sandbox" (ResourceExhaustedError): Modal has no free
    # GPU of this type for the workspace right now. Nothing was created or billed.
    if "resourceexhausted" in lowered or "insufficient capacity" in lowered:
        return GPUError("unavailable", text or name)
    if "auth" in lowered or "token" in lowered or "unauthenticated" in lowered:
        return GPUError("invalid_credentials", text)
    if "notfound" in lowered or "not found" in lowered:
        return GPUError("not_found", text)
    if "connection" in lowered or "timeout" in lowered:
        return GPUError("network_error", name)
    if "invalid" in lowered:
        return GPUError("invalid_request", text)
    return GPUError("provider_error", text or name)


class ModalSDK:
    """The few SDK calls used, in one place (tests replace this class with a fake)."""

    async def report(self, client: Any, start: Any) -> float:
        """The workspace's cost since `start` (UTC) up to the last full hour, from Modal's
        billing report (`Workspace.billing.report`, SDK 1.6.3; partial intervals are left out
        by Modal)."""
        import modal

        workspace = modal.Workspace.from_context(client=client)
        items = await workspace.billing.report.aio(start=start, resolution="h")
        return float(sum((item.cost for item in items), Decimal(0)))

    async def rates(self, client: Any) -> dict[str, str]:
        """The workspace's current pricing rates (`modal billing rates`)."""
        import modal

        workspace = modal.Workspace.from_context(client=client)
        rates = await workspace.billing.rates.aio()
        return {str(k): str(v) for k, v in rates.items()}

    async def client(self, creds: dict[str, str]) -> Any:
        import modal

        client = await modal.Client.from_credentials.aio(creds["token_id"], creds["token_secret"])
        await client.hello.aio()
        return client

    async def app(self, client: Any) -> Any:
        import modal

        return await modal.App.lookup.aio(APP_NAME, create_if_missing=True, client=client)

    async def list(self, client: Any, app: Any, tags: dict[str, str] | None = None) -> list[Any]:
        import modal

        return [
            sb
            async for sb in modal.Sandbox.list.aio(
                app_id=app.app_id, tags={**TAGS, **(tags or {})}, client=client
            )
        ]

    async def create(self, client: Any, app: Any, spec: WorkloadSpec) -> Any:
        import modal

        gpu = spec.gpu_type if spec.gpu_count <= 1 else f"{spec.gpu_type}:{spec.gpu_count}"
        command = [*spec.entrypoint, *spec.command] or ["sleep", "infinity"]
        volumes: dict[str, Any] = {}
        if spec.volume_name:
            volumes["/workspace"] = modal.Volume.from_name(
                spec.volume_name, create_if_missing=True, client=client
            )
        secrets = [modal.Secret.from_dict(spec.secret_env)] if spec.secret_env else None
        image = (
            modal.Image.from_registry(spec.image, add_python=spec.image_python)
            if spec.image_python
            else modal.Image.from_registry(spec.image)
        )
        if spec.image_apt:
            image = image.apt_install(*spec.image_apt)
        if spec.clear_entrypoint:
            image = image.entrypoint([])
        extra: dict[str, Any] = {}
        if spec.cpu:
            extra["cpu"] = spec.cpu
        if spec.memory_mb:
            extra["memory"] = spec.memory_mb
        if spec.workdir:
            extra["workdir"] = spec.workdir
        return await modal.Sandbox.create.aio(
            *command,
            **extra,
            app=app,
            name=spec.name or None,
            tags={**TAGS, **spec.tags},
            image=image,
            env=spec.env or None,
            secrets=secrets,
            volumes=volumes,
            gpu=gpu,
            timeout=spec.timeout_minutes * 60,
            encrypted_ports=spec.ports,
            client=client,
        )

    async def from_id(self, client: Any, sandbox_id: str) -> Any:
        import modal

        return await modal.Sandbox.from_id.aio(sandbox_id, client=client)

    async def poll(self, sandbox: Any) -> int | None:
        return await sandbox.poll.aio()

    async def terminate(self, sandbox: Any) -> None:
        await sandbox.terminate.aio()

    async def tunnels(self, sandbox: Any) -> dict[int, str]:
        tunnels = await sandbox.tunnels.aio(timeout=5)
        return {port: t.url for port, t in tunnels.items()}

    async def output(self, sandbox: Any, entries: int = 40) -> str:
        """The sandbox's last log lines (its own output and Modal's system messages, e.g. why
        the container could not start)."""
        lines = []
        async for entry in sandbox.logs.tail.aio(entries):
            text = str(entry.message).rstrip()
            lines.append(f"[modal] {text}" if entry.source == "system" else text)
        return "\n".join(lines)

    async def close(self, client: Any) -> None:
        await client.__aexit__(None, None, None)


class ModalProvider:
    key = "modal"
    supports_endpoints = False
    supports_stop_start = False

    def __init__(self, sdk: ModalSDK | None = None):
        self.sdk = sdk or ModalSDK()
        # Live rates per token id: (time read, rates or None when they could not be read).
        self._rates: dict[str, tuple[float, LiveRates | None]] = {}

    async def live_rates(self, creds: dict[str, str]) -> LiveRates | None:
        """Modal's current rates for this workspace (cached), or None (then the list prices
        are used). Never raises: prices must not block the GPU list."""
        token_id = creds.get("token_id", "")
        cached = self._rates.get(token_id)
        if cached and time.monotonic() - cached[0] < (RATES_TTL if cached[1] else RATES_RETRY):
            return cached[1]
        rates: LiveRates | None = None
        client = None
        try:
            client = await self._client(creds)
            rates = parse_rates(await self.sdk.rates(client))
            if rates is None:
                log.warning("Modal rates: no GPU price could be read; using list prices")
        except Exception as exc:  # noqa: BLE001 - SDK raises its own error classes
            log.warning("Modal rates could not be read (%s); using list prices", type(exc).__name__)
        finally:
            if client is not None:
                await self.sdk.close(client)
        self._rates[token_id] = (time.monotonic(), rates)
        return rates

    async def hourly(self, creds: dict[str, str], gpu: str, count: int = 1) -> float | None:
        """USD per hour of the studio's sandbox with this GPU: Modal's live rates where they
        were read, the list prices otherwise (an estimate either way: billed per second)."""
        rates = await self.live_rates(creds)
        live = rates.gpu_hour.get(gpu) if rates else None
        price = (
            live if live is not None else next((p for i, _, _, p in GPU_TYPES if i == gpu), None)
        )
        if price is None:
            return None
        extra = rates.extra_hour if rates and rates.extra_hour is not None else None
        return round(
            price * max(1, count) + (extra if extra is not None else SANDBOX_EXTRA_PER_HOUR), 4
        )

    async def spent_since(self, creds: dict[str, str], start: Any) -> float:
        """What the whole Modal workspace cost since `start` (Modal's own billing report)."""
        client = await self._client(creds)
        try:
            return await self.sdk.report(client, start)
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises its own error classes
            raise _error(exc) from exc
        finally:
            await self.sdk.close(client)

    async def offer(
        self, creds: dict[str, str], gpu_id: str, cloud: str
    ) -> tuple[None, float | None]:
        """The GPU's live price per hour (for "the price changed since you confirmed it"), or
        None when Modal's rates cannot be read (then nothing is compared)."""
        rates = await self.live_rates(creds)
        return None, (rates.gpu_hour.get(gpu_id) if rates else None)

    async def _client(self, creds: dict[str, str]) -> Any:
        try:
            return await self.sdk.client(creds)
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises its own error classes
            raise _error(exc) from exc

    async def check(self, creds: dict[str, str]) -> None:
        client = await self._client(creds)
        await self.sdk.close(client)

    async def gpu_types(self, creds: dict[str, str]) -> list[GpuType]:
        """Modal's GPUs with the workspace's live rates; the list price where a rate is missing
        (labelled an estimate)."""
        rates = await self.live_rates(creds)
        extra = rates.extra_hour if rates and rates.extra_hour is not None else None
        out = []
        for i, n, m, listed in GPU_TYPES:
            live = rates.gpu_hour.get(i) if rates else None
            out.append(
                GpuType(
                    id=i,
                    name=n,
                    memory_gb=m,
                    price_secure=live if live is not None else listed,
                    price_estimate=live is None or extra is None,
                    price_extra=extra if extra is not None else SANDBOX_EXTRA_PER_HOUR,
                    price_source="api" if live is not None else "list",
                    manufacturer="NVIDIA",
                )
            )
        return out

    async def _describe(self, sandbox: Any, records: dict[str, dict[str, Any]]) -> Workload:
        sandbox_id = sandbox.object_id
        record = records.get(sandbox_id, {})
        code = await self.sdk.poll(sandbox)
        running = code is None
        urls: list[str] = []
        if running and record.get("ports"):
            try:
                urls = list((await self.sdk.tunnels(sandbox)).values())
            except Exception:  # noqa: BLE001 - tunnels may not be ready yet
                urls = []
        return Workload(
            id=sandbox_id,
            name=record.get("name") or sandbox_id,
            kind="sandbox",
            status="running" if running else "exited",
            gpu_type=record.get("gpu_type", ""),
            gpu_count=record.get("gpu_count", 1),
            image=record.get("image", ""),
            urls=urls,
            created_at=record.get("created_at"),
            dashboard_url="https://modal.com/apps",
            can_terminate=running,
        )

    async def list_workloads(
        self, creds: dict[str, str], records: dict[str, dict[str, Any]] | None = None
    ) -> list[Workload]:
        client = await self._client(creds)
        try:
            app = await self.sdk.app(client)
            sandboxes = await self.sdk.list(client, app)
            return [await self._describe(sb, records or {}) for sb in sandboxes]
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _error(exc) from exc
        finally:
            await self.sdk.close(client)

    async def find_workloads(self, creds: dict[str, str], tags: dict[str, str]) -> list[str]:
        """Running sandboxes of the studio's app carrying these tags."""
        client = await self._client(creds)
        try:
            app = await self.sdk.app(client)
            out = []
            for sb in await self.sdk.list(client, app, tags):
                if await self.sdk.poll(sb) is None:
                    out.append(sb.object_id)
            return out
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _error(exc) from exc
        finally:
            await self.sdk.close(client)

    async def create_workload(self, creds: dict[str, str], spec: WorkloadSpec) -> Workload:
        client = await self._client(creds)
        try:
            app = await self.sdk.app(client)
            sandbox = await self.sdk.create(client, app, spec)
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _error(exc) from exc
        finally:
            await self.sdk.close(client)
        return Workload(
            id=sandbox.object_id,
            name=spec.name or sandbox.object_id,
            kind="sandbox",
            status="starting",
            gpu_type=spec.gpu_type,
            gpu_count=spec.gpu_count,
            image=spec.image,
            # Modal's live rates (list prices where a rate is missing), GPU + CPU + memory.
            cost_per_hour=await self.hourly(creds, spec.gpu_type, spec.gpu_count),
            dashboard_url="https://modal.com/apps",
        )

    async def start(self, creds: dict[str, str], workload_id: str) -> None:
        raise GPUError("unsupported", "Modal sandboxes cannot be paused or resumed")

    async def stop(self, creds: dict[str, str], workload_id: str) -> None:
        raise GPUError("unsupported", "Modal sandboxes cannot be paused or resumed")

    async def terminate(self, creds: dict[str, str], workload_id: str) -> None:
        client = await self._client(creds)
        try:
            sandbox = await self.sdk.from_id(client, workload_id)
            await self.sdk.terminate(sandbox)
        except GPUError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _error(exc) from exc
        finally:
            await self.sdk.close(client)

    async def list_endpoints(self, creds: dict[str, str]) -> list[Endpoint]:
        return []

    async def scale_endpoint(
        self, creds: dict[str, str], endpoint_id: str, workers_min: int, workers_max: int
    ) -> None:
        raise GPUError("unsupported")

    async def delete_endpoint(self, creds: dict[str, str], endpoint_id: str) -> None:
        raise GPUError("unsupported")

    async def account(self, creds: dict[str, str]) -> dict[str, Any]:
        return {"balance": None, "spend_per_hour": None, "source": None}

    async def workload(self, creds: dict[str, str], workload_id: str) -> Workload | None:
        client = await self._client(creds)
        try:
            try:
                sandbox = await self.sdk.from_id(client, workload_id)
                code = await self.sdk.poll(sandbox)
            except Exception as exc:  # noqa: BLE001
                error = exc if isinstance(exc, GPUError) else _error(exc)
                if error.code == "not_found":
                    return None
                raise error from exc
            running = code is None
            detail = None if running else await self._exit_detail(sandbox, code)
        finally:
            await self.sdk.close(client)
        return Workload(
            id=workload_id,
            name=workload_id,
            kind="sandbox",
            status="running" if running else "exited",
            dashboard_url="https://modal.com/apps",
            can_terminate=running,
            detail=detail,
        )

    async def _exit_detail(self, sandbox: Any, code: int | None) -> str:
        """Exit code and last output of an ended sandbox (best effort, a few seconds)."""
        try:
            text = await asyncio.wait_for(self.sdk.output(sandbox), timeout=10)
        except Exception:  # noqa: BLE001 - the logs are only an explanation
            text = ""
        out = _clean_output(text)
        return f"exit code {code}" + (f"\n{out}" if out else "")

    async def port_url(self, creds: dict[str, str], workload_id: str, port: int) -> str | None:
        client = await self._client(creds)
        try:
            sandbox = await self.sdk.from_id(client, workload_id)
            tunnels = await self.sdk.tunnels(sandbox)
        except Exception:  # noqa: BLE001 - not ready yet
            return None
        finally:
            await self.sdk.close(client)
        return tunnels.get(port)
