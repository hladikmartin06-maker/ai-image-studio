"""GPU compute providers. To add one: implement `GPUProvider` (base.py) and register it."""

from __future__ import annotations

from typing import Any

from app.gpu.base import GPUProvider
from app.gpu.modal_provider import ModalProvider
from app.gpu.runpod import RunPodProvider

GPU_PROVIDERS: dict[str, dict[str, Any]] = {
    "runpod": {
        "key": "runpod",
        "name": "RunPod",
        "description": "GPU pods you can start, stop and terminate, plus serverless endpoints.",
        "credential_fields": ["api_key"],
        "keys_url": "https://console.runpod.io/user/settings",
        "docs_url": "https://docs.runpod.io",
        "dashboard_url": "https://console.runpod.io",
        "supports_endpoints": True,
        "supports_stop_start": True,
        "default_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    },
    "modal": {
        "key": "modal",
        "name": "Modal",
        "description": "Serverless GPU sandboxes billed per second; exposed ports become "
        "HTTPS URLs.",
        "credential_fields": ["token_id", "token_secret"],
        "keys_url": "https://modal.com/settings/tokens",
        "docs_url": "https://modal.com/docs/guide/sandbox",
        "dashboard_url": "https://modal.com/apps",
        "supports_endpoints": False,
        "supports_stop_start": False,
        "default_image": "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
    },
}

_ADAPTERS: dict[str, GPUProvider] = {}


def get_gpu_adapter(key: str) -> GPUProvider:
    if key not in _ADAPTERS:
        _ADAPTERS[key] = RunPodProvider() if key == "runpod" else ModalProvider()
    return _ADAPTERS[key]


def set_gpu_adapter(key: str, adapter: GPUProvider | None) -> None:
    if adapter is None:
        _ADAPTERS.pop(key, None)
    else:
        _ADAPTERS[key] = adapter
