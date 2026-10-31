"""Krea 2 (open weights, June 2026) as a ComfyUI model family.

Verified (2026-10-31):
- github.com/krea-ai/krea-2: two checkpoints. RAW = undistilled pretrained base for
  fine-tuning / LoRA training (recommended 52 steps, CFG 3.5, generation up to 1k);
  Turbo = 8-step distilled (CFG off), generates 1k to 2k. "LoRAs trained on RAW will work
  well on Turbo." HF repos krea/Krea-2-Raw, krea/Krea-2-Turbo. Community license.
- ComfyUI (comfyanonymous/ComfyUI): native support; supported_models.Krea2, CLIPLoader type
  "krea2" (Qwen3-VL-4B text encoder), sampling shift 1.15, latent format Wan21 (16 ch).
- Comfy-Org workflow template "image_krea2_turbo_t2i" (min ComfyUI 0.24.1): UNETLoader
  krea2_turbo_fp8_scaled.safetensors (12.24 GB), CLIPLoader qwen3vl_4b_fp8_scaled
  .safetensors (4.88 GB, type krea2), VAELoader qwen_image_vae.safetensors (242 MB),
  LoraLoaderModelOnly, CLIPTextEncode, ConditioningZeroOut as negative, EmptyLatentImage,
  KSampler 8 steps / cfg 1 / euler / simple, VAEDecode, SaveImage. Int8 variant:
  krea2_turbo_int8_convrot.safetensors (template image_krea2_turbo_t2i_int8).
- ostris/ai-toolkit: arch "krea2" (single-stream MMDiT: features 6144, 28 layers;
  Qwen3-VL-4B; Qwen-Image VAE), RAW and Turbo share it; Turbo training uses the adapter
  ostris/krea2_turbo_training_adapter; LoRA keys are saved with ComfyUI's
  "diffusion_model." prefix.
Not verified: the ComfyUI file names of the RAW checkpoint (detected by name instead),
VRAM needs (not published), sampler settings of RAW inside ComfyUI (studio default).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

FAMILY = "krea2"
HF_BASE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"

ARCHITECTURE: dict[str, Any] = {
    "family": FAMILY,
    "name": "Krea 2",
    "type": "Single-stream MMDiT, rectified flow (28 blocks, width 6144)",
    "text_encoder": "Qwen3-VL-4B (CLIPLoader type “krea2”)",
    "vae": "Qwen-Image VAE (16 latent channels)",
    "license": "Krea 2 community license — check it on Hugging Face before commercial use",
    "min_comfyui": "0.24.1",
    "lora_format": "safetensors, ComfyUI “diffusion_model.” keys (ai-toolkit arch krea2)",
    "sources": [
        "https://github.com/krea-ai/krea-2",
        "https://huggingface.co/Comfy-Org/Krea-2",
        "https://github.com/ostris/ai-toolkit",
    ],
}


@dataclass(frozen=True)
class KnownFile:
    filename: str
    folder: str
    precision: str
    size_gb: float | None
    url: str


@dataclass(frozen=True)
class Krea2Variant:
    key: str
    model_key: str
    name: str
    description: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    negative_prompt: bool
    tiers: tuple[str, ...]
    settings_source: str
    known_files: tuple[KnownFile, ...] = field(default_factory=tuple)


VARIANTS: dict[str, Krea2Variant] = {
    "turbo": Krea2Variant(
        key="turbo",
        model_key="krea-2-turbo",
        name="Krea 2 Turbo",
        description="8-step distilled checkpoint for fast, high-quality text to image.",
        steps=8,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        # CFG 1 ignores a negative prompt; the official workflow zeroes it out.
        negative_prompt=False,
        tiers=("1K", "2K"),
        settings_source="Comfy-Org template image_krea2_turbo_t2i",
        known_files=(
            KnownFile(
                "krea2_turbo_fp8_scaled.safetensors",
                "diffusion_models",
                "fp8 (scaled)",
                12.24,
                f"{HF_BASE}/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
            ),
            KnownFile(
                "krea2_turbo_int8_convrot.safetensors",
                "diffusion_models",
                "int8 (convrot)",
                None,
                f"{HF_BASE}/diffusion_models/krea2_turbo_int8_convrot.safetensors",
            ),
        ),
    ),
    "raw": Krea2Variant(
        key="raw",
        model_key="krea-2-raw",
        name="Krea 2 RAW",
        description="Undistilled base checkpoint, meant for fine-tuning and LoRA training.",
        steps=52,
        cfg=3.5,
        sampler="euler",
        scheduler="simple",
        negative_prompt=True,
        tiers=("1K",),
        settings_source="steps and CFG from Krea's GitHub; sampler is the studio's default",
    ),
}

TEXT_ENCODER = KnownFile(
    "qwen3vl_4b_fp8_scaled.safetensors",
    "text_encoders",
    "fp8 (scaled)",
    4.88,
    f"{HF_BASE}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
)
VAE = KnownFile(
    "qwen_image_vae.safetensors", "vae", "bf16", 0.242, f"{HF_BASE}/vae/qwen_image_vae.safetensors"
)

MODEL_KEYS = {v.model_key: v for v in VARIANTS.values()}
ASPECT_RATIOS = ("1:1", "4:5", "3:4", "2:3", "9:16", "5:4", "4:3", "3:2", "16:9")
TIER_EDGE = {"1K": 1024, "2K": 2048}
SEED_MAX = 2**53 - 1

_TE_RE = re.compile(r"qwen3[_-]?vl[_-]?4b", re.IGNORECASE)
_VAE_RE = re.compile(r"qwen[_-]?image.*vae", re.IGNORECASE)


def precision_of(filename: str) -> str:
    name = filename.lower()
    for token, label in (
        ("nvfp4", "nvfp4"),
        ("int8", "int8"),
        ("fp8", "fp8"),
        ("bf16", "bf16"),
        ("fp16", "fp16"),
        ("fp32", "fp32"),
    ):
        if token in name:
            return label
    return "unknown"


def variant_of(filename: str) -> str | None:
    name = filename.lower().replace("-", "_")
    if "krea2" not in name and "krea_2" not in name:
        return None
    if "turbo" in name:
        return "turbo"
    if "raw" in name:
        return "raw"
    return None


def _choices(object_info: dict[str, Any], node: str, field_name: str) -> list[str]:
    spec = (object_info.get(node) or {}).get("input", {})
    for group in ("required", "optional"):
        entry = (spec.get(group) or {}).get(field_name)
        if isinstance(entry, list) and entry:
            first = entry[0]
            if isinstance(first, list):
                return [str(x) for x in first]
            # Newer schema: ["COMBO", {"options": [...]}]
            if len(entry) > 1 and isinstance(entry[1], dict):
                options = entry[1].get("options")
                if isinstance(options, list):
                    return [str(x) for x in options]
    return []


def installed_files(object_info: dict[str, Any]) -> dict[str, list[str]]:
    """Model files ComfyUI reports through its loader nodes."""
    return {
        "diffusion_models": _choices(object_info, "UNETLoader", "unet_name"),
        "text_encoders": _choices(object_info, "CLIPLoader", "clip_name"),
        "vae": _choices(object_info, "VAELoader", "vae_name"),
        "loras": _choices(object_info, "LoraLoaderModelOnly", "lora_name"),
        "checkpoints": _choices(object_info, "CheckpointLoaderSimple", "ckpt_name"),
    }


REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "LoraLoaderModelOnly",
)


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """What this ComfyUI has for Krea 2: support in CLIPLoader, files per variant, missing."""
    files = installed_files(object_info)
    clip_types = _choices(object_info, "CLIPLoader", "type")
    supports = FAMILY in clip_types
    missing_nodes = [n for n in REQUIRED_NODES if n not in object_info]
    variants: dict[str, Any] = {}
    for key, variant in VARIANTS.items():
        found = [f for f in files["diffusion_models"] if variant_of(f) == key]
        variants[key] = {
            "name": variant.name,
            "installed": bool(found),
            "files": [{"filename": f, "precision": precision_of(f)} for f in found],
        }
    encoders = [f for f in files["text_encoders"] if _TE_RE.search(f)]
    vaes = [f for f in files["vae"] if _VAE_RE.search(f)]
    missing: list[str] = []
    if not supports:
        missing.append("ComfyUI without Krea 2 support (CLIPLoader has no “krea2” type) — update")
    if missing_nodes:
        missing.append("Missing nodes: " + ", ".join(missing_nodes))
    if not encoders:
        missing.append(f"Text encoder ({TEXT_ENCODER.filename}) in models/text_encoders")
    if not vaes:
        missing.append(f"VAE ({VAE.filename}) in models/vae")
    if not any(v["installed"] for v in variants.values()):
        missing.append("A Krea 2 diffusion model in models/diffusion_models")
    return {
        "supported": supports,
        "missing_nodes": missing_nodes,
        "variants": variants,
        "text_encoders": encoders,
        "vaes": vaes,
        "loras": files["loras"],
        "ready": supports and not missing_nodes and bool(encoders) and bool(vaes),
        "missing": missing,
    }


def size_for(aspect: str, tier: str) -> tuple[int, int]:
    """Width/height with the tier's pixel area, multiples of 16, long side <= 2048."""
    if aspect not in ASPECT_RATIOS or tier not in TIER_EDGE:
        raise ValueError("Unsupported size")
    rw, rh = (int(x) for x in aspect.split(":"))
    area = TIER_EDGE[tier] ** 2
    width = math.sqrt(area * rw / rh)
    height = area / width
    scale = min(1.0, 2048 / max(width, height))
    return (
        max(256, int(round(width * scale / 16)) * 16),
        max(256, int(round(height * scale / 16)) * 16),
    )


def size_table() -> dict[str, dict[str, list[int]]]:
    return {tier: {a: list(size_for(a, tier)) for a in ASPECT_RATIOS} for tier in ("1K", "2K")}


# --- LoRA compatibility -----------------------------------------------------------------


def lora_compatibility(architecture: str, trained_on: str, target: str) -> tuple[bool, str]:
    """Strict: other architectures never load; RAW -> Turbo is documented by Krea."""
    if architecture != FAMILY:
        return False, (
            f"This LoRA was made for “{architecture}”, not Krea 2. It cannot be loaded into "
            "Krea 2 — train or import a Krea 2 LoRA."
        )
    if trained_on == target:
        return True, ""
    if trained_on == "raw" and target == "turbo":
        return True, "Trained on RAW: Krea documents that RAW LoRAs work on Turbo."
    if trained_on == "turbo" and target == "raw":
        return False, (
            "Trained on Turbo with the training adapter: use it with Turbo. Only RAW → Turbo "
            "is documented; train on RAW to use a LoRA with both."
        )
    return False, "The base variant of this LoRA is unknown."


# --- workflow -------------------------------------------------------------------------------


def build_workflow(
    *,
    variant: str,
    unet: str,
    text_encoder: str,
    vae: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    batch: int,
    loras: list[tuple[str, float]],
    prefix: str = "ai-image-studio",
) -> dict[str, Any]:
    """The API-format workflow (same nodes as Comfy-Org's Krea 2 template)."""
    spec = VARIANTS[variant]
    wf: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": FAMILY, "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model_ref: list[Any] = ["1", 0]
    for index, (lora_name, strength) in enumerate(loras):
        node_id = str(20 + index)
        wf[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": lora_name, "strength_model": strength},
        }
        model_ref = [node_id, 0]
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    if spec.negative_prompt:
        wf["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }
    else:
        wf["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
    wf["6"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": batch},
    }
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": spec.sampler,
            "scheduler": spec.scheduler,
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


def catalog_json() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "variants": [
            {
                "key": v.key,
                "model_key": v.model_key,
                "name": v.name,
                "description": v.description,
                "steps": v.steps,
                "cfg": v.cfg,
                "sampler": v.sampler,
                "scheduler": v.scheduler,
                "negative_prompt": v.negative_prompt,
                "tiers": list(v.tiers),
                "settings_source": v.settings_source,
                "known_files": [f.__dict__ for f in v.known_files],
            }
            for v in VARIANTS.values()
        ],
        "text_encoder": TEXT_ENCODER.__dict__,
        "vae": VAE.__dict__,
        "aspect_ratios": list(ASPECT_RATIOS),
        "sizes": size_table(),
        "seed_max": SEED_MAX,
    }
