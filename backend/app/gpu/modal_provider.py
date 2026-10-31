"""Modal: GPU sandboxes through the official `modal` Python SDK.

Credentials are a token id + token secret (modal.com › Settings › API tokens), used with
`modal.Client.from_credentials` — the documented way to manage Modal on behalf of a user.
Workloads are Modal Sandboxes created in one app ("ai-image-studio-gpu") and tagged
`created-by=ai-image-studio`:
  modal.Sandbox.create(*command, app=…, image=modal.Image.from_registry(…), gpu="H100:2",
                       timeout=…, encrypted_ports=[…], tags=…, client=…)
  modal.Sandbox.list(app_id=…, tags=…, client=…), Sandbox.from_id(…).poll()/.terminate()
  sandbox.tunnels() -> {port: Tunnel(url)} for exposed ports (the workload's endpoints)
Sandboxes cannot be paused: they run until they finish, time out or are terminated.
Modal has no balance API; usage is shown from the workloads started here.
"""

from __future__ import annotations

from typing import Any

from app.gpu.base import Endpoint, GPUError, GpuType, Workload, WorkloadSpec

APP_NAME = "ai-image-studio-gpu"
TAGS = {"created-by": "ai-image-studio"}

# GPU names accepted by Modal's `gpu=` argument (modal.com/docs/guide/gpu). Prices are not
# returned by any Modal API, so none are shown.
GPU_TYPES: list[tuple[str, str, int]] = [
    ("T4", "NVIDIA T4", 16),
    ("L4", "NVIDIA L4", 24),
    ("A10", "NVIDIA A10", 24),
    ("L40S", "NVIDIA L40S", 48),
    ("A100-40GB", "NVIDIA A100 40 GB", 40),
    ("A100-80GB", "NVIDIA A100 80 GB", 80),
    ("H100", "NVIDIA H100", 80),
    ("H200", "NVIDIA H200", 141),
    ("B200", "NVIDIA B200", 180),
]


def _error(exc: Exception) -> GPUError:
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
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

    async def client(self, creds: dict[str, str]) -> Any:
        import modal

        client = await modal.Client.from_credentials.aio(creds["token_id"], creds["token_secret"])
        await client.hello.aio()
        return client

    async def app(self, client: Any) -> Any:
        import modal

        return await modal.App.lookup.aio(APP_NAME, create_if_missing=True, client=client)

    async def list(self, client: Any, app: Any) -> list[Any]:
        import modal

        return [
            sb async for sb in modal.Sandbox.list.aio(app_id=app.app_id, tags=TAGS, client=client)
        ]

    async def create(self, client: Any, app: Any, spec: WorkloadSpec) -> Any:
        import modal

        gpu = spec.gpu_type if spec.gpu_count <= 1 else f"{spec.gpu_type}:{spec.gpu_count}"
        command = spec.command or ["sleep", "infinity"]
        return await modal.Sandbox.create.aio(
            *command,
            app=app,
            name=spec.name or None,
            tags=TAGS,
            image=modal.Image.from_registry(spec.image),
            env=spec.env or None,
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

    async def close(self, client: Any) -> None:
        await client.__aexit__(None, None, None)


class ModalProvider:
    key = "modal"
    supports_endpoints = False
    supports_stop_start = False

    def __init__(self, sdk: ModalSDK | None = None):
        self.sdk = sdk or ModalSDK()

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
        return [GpuType(id=i, name=n, memory_gb=m) for i, n, m in GPU_TYPES]

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
