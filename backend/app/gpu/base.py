"""The contract of a GPU compute provider, kept apart from image and chat providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

GPUErrorCode = Literal[
    "invalid_credentials",
    "unavailable",
    "price_changed",
    "insufficient_credits",
    "rate_limited",
    "not_found",
    "invalid_request",
    "unsupported",
    "provider_error",
    "network_error",
]

MESSAGES = {
    "invalid_credentials": "The credentials were rejected. Update them in Integrations › GPU "
    "compute.",
    "insufficient_credits": "The account has no credit left for this.",
    "unavailable": "No GPU of this type is free right now. Pick another GPU or try again in a "
    "few minutes.",
    "price_changed": "The price of this GPU changed since you confirmed it. Check the new price "
    "and confirm again.",
    "rate_limited": "The provider is limiting requests. Try again in a moment.",
    "not_found": "The resource no longer exists at the provider.",
    "invalid_request": "The provider rejected the request.",
    "unsupported": "This provider does not support that action.",
    "provider_error": "The provider had an internal problem. Try again later.",
    "network_error": "Could not reach the provider. Check your internet connection.",
}


@dataclass(eq=False)
class GPUError(Exception):
    code: GPUErrorCode
    detail: str = ""

    def __str__(self) -> str:
        return self.message

    @property
    def message(self) -> str:
        base = MESSAGES.get(self.code, "The GPU provider request failed.")
        if self.detail and self.code not in ("invalid_credentials", "network_error"):
            return f"{base} ({self.detail[:600]})"
        return base


@dataclass
class GpuType:
    id: str
    name: str
    memory_gb: int | None = None
    # USD per hour as reported by the provider (None when it reports none).
    price_secure: float | None = None
    price_community: float | None = None
    available: bool | None = None
    manufacturer: str | None = None
    # RunPod catalog, per cloud: stock right now (NONE/LOW/MEDIUM/HIGH) and the CUDA versions
    # of machines with free capacity. None when the provider reports no availability.
    # Modal: list prices from a fixed table (no price API): price_secure is the GPU's price,
    # price_extra the CPU and memory the studio's sandbox adds on top (USD per hour).
    price_estimate: bool = False
    price_extra: float | None = None
    stock_secure: str | None = None
    stock_community: str | None = None
    cuda_secure: list[str] = field(default_factory=list)
    cuda_community: list[str] = field(default_factory=list)

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Workload:
    id: str
    name: str
    kind: Literal["pod", "sandbox"]
    status: str  # running | starting | stopped | exited | terminated | unknown
    gpu_type: str = ""
    gpu_count: int = 1
    image: str = ""
    cost_per_hour: float | None = None
    urls: list[str] = field(default_factory=list)
    created_at: str | None = None
    dashboard_url: str | None = None
    can_start: bool = False
    can_stop: bool = False
    can_terminate: bool = True

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Endpoint:
    id: str
    name: str
    gpu_types: list[str]
    workers_min: int
    workers_max: int
    url: str | None = None
    health: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadSpec:
    name: str
    image: str
    gpu_type: str
    gpu_count: int = 1
    ports: list[int] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # RunPod
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    container_disk_gb: int = 20
    volume_gb: int = 0
    # The container's start command (RunPod: entrypoint + args; Modal: sandbox command).
    command: list[str] = field(default_factory=list)
    # When set, RunPod's dockerEntrypoint (command is then dockerStartCmd); Modal runs
    # entrypoint + command.
    entrypoint: list[str] = field(default_factory=list)
    # Modal: clear the registry image's ENTRYPOINT (modal.Image.entrypoint([])).
    clear_entrypoint: bool = False
    # Modal: sandbox time limit; RunPod pods have none.
    timeout_minutes: int = 60
    # Sensitive environment (tokens): RunPod env, a Modal Secret on Modal.
    secret_env: dict[str, str] = field(default_factory=dict)
    # Modal: a persistent Volume mounted at /workspace (RunPod: volume_gb at /workspace).
    volume_name: str | None = None
    # Modal: add Python / apt packages to a registry image (modal.Image methods).
    image_python: str | None = None
    image_apt: list[str] = field(default_factory=list)
    # RunPod: only hosts whose NVIDIA driver supports one of these CUDA versions.
    allowed_cuda_versions: list[str] = field(default_factory=list)
    # Modal: extra sandbox tags (e.g. the studio connection, to find the sandbox again).
    tags: dict[str, str] = field(default_factory=dict)
    # Modal: CPU cores and memory (MiB) of the sandbox; None = Modal's defaults.
    cpu: float | None = None
    memory_mb: int | None = None


class GPUProvider(Protocol):
    key: str
    supports_endpoints: bool
    supports_stop_start: bool

    async def check(self, creds: dict[str, str]) -> None: ...

    async def gpu_types(self, creds: dict[str, str]) -> list[GpuType]: ...

    async def list_workloads(self, creds: dict[str, str]) -> list[Workload]: ...

    async def create_workload(self, creds: dict[str, str], spec: WorkloadSpec) -> Workload: ...

    async def start(self, creds: dict[str, str], workload_id: str) -> None: ...

    async def stop(self, creds: dict[str, str], workload_id: str) -> None: ...

    async def terminate(self, creds: dict[str, str], workload_id: str) -> None: ...

    async def list_endpoints(self, creds: dict[str, str]) -> list[Endpoint]: ...

    async def scale_endpoint(
        self, creds: dict[str, str], endpoint_id: str, workers_min: int, workers_max: int
    ) -> None: ...

    async def delete_endpoint(self, creds: dict[str, str], endpoint_id: str) -> None: ...

    async def account(self, creds: dict[str, str]) -> dict[str, Any]: ...

    async def workload(self, creds: dict[str, str], workload_id: str) -> Workload | None:
        """One workload, or None when it no longer exists."""
        ...

    async def port_url(self, creds: dict[str, str], workload_id: str, port: int) -> str | None:
        """The HTTPS URL of an exposed port, or None while it is not reachable yet."""
        ...
