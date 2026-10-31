"""Reads the header of a .safetensors file (8-byte length + JSON) without loading tensors:
validates uploads and finds which architecture a LoRA was trained for."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

MAX_HEADER = 100 * 1024 * 1024


class InvalidSafetensors(ValueError):
    pass


def read_header(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise InvalidSafetensors("Not a safetensors file.")
        (length,) = struct.unpack("<Q", raw)
        if length <= 1 or length > MAX_HEADER:
            raise InvalidSafetensors("Not a safetensors file.")
        try:
            header: dict[str, Any] = json.loads(f.read(length))
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidSafetensors("Not a safetensors file.") from exc
    if not isinstance(header, dict):
        raise InvalidSafetensors("Not a safetensors file.")
    metadata = header.pop("__metadata__", {}) or {}
    return list(header.keys()), {str(k): str(v) for k, v in metadata.items()}


def lora_architecture(keys: list[str], metadata: dict[str, str]) -> tuple[str | None, str]:
    """(architecture, evidence). Only metadata written by the trainer counts as evidence:
    key names alone cannot tell apart models that share layer names."""
    if not any("lora" in k.lower() for k in keys):
        return None, "The file contains no LoRA weights."
    blob = " ".join(f"{k}={v}" for k, v in metadata.items()).lower()
    if "krea2" in blob or "krea-2" in blob or "krea_2" in blob:
        return "krea2", "The file's training metadata names Krea 2."
    for arch in ("flux", "qwen_image", "sdxl", "wan", "z_image", "hidream", "chroma"):
        if f'"arch": "{arch}' in blob or f"ss_base_model_version={arch}" in blob:
            return arch, f"The file's training metadata names {arch}."
    return None, "The file has no metadata that names its base model."


def lora_variant(metadata: dict[str, str]) -> str | None:
    blob = " ".join(metadata.values()).lower()
    if "krea-2-turbo" in blob or "krea2_turbo_training_adapter" in blob:
        return "turbo"
    if "krea-2-raw" in blob:
        return "raw"
    return None
