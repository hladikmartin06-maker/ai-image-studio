"""Krea 2 AnyPaint: Advanced's region redraw (the owner's request, 2026-11-06, phase 1).

Sources (checked 2026-11-06):
- Model card huggingface.co/yijunwang2/krea2-anypaint (revision 1a9fb37a): a functional
  adapter for arbitrary-mask inpainting, outpainting and image editing, trained at rank/alpha
  32/32 against Krea 2 Raw and intended for Krea 2 Turbo; recommended 8 steps, guidance 0
  (CFG 1 in ComfyUI), LoRA scale 1.0, a 384 px semantic reference with VLM encoding, K/V
  cache, a 32 px boundary band redrawn to blend; known pixels are kept at every flow step.
  Weights: Krea 2 Community License; pipeline code Apache 2.0. Ordinary LoRA importers
  outside ComfyUI cannot use it; in ComfyUI it is loaded with LoraLoaderModelOnly and needs
  the nodes below.
- ComfyUI nodes github.com/alexw5702-afk/krea2-anypaint (MIT, v0.1.0, commit 675be5a9;
  comfy/custom_nodes.py): Krea2AnyPaintPrepare (source, left, top, right, bottom,
  reference_max_edge, boundary_redraw_px, optional generated_mask → semantic_reference,
  known_image, generated_mask, keep_mask, canvas_width, canvas_height), Krea2AnyPaintEncode
  (clip, prompt, vae, semantic_reference, known_image, keep_mask, vlm_reference → positive,
  latent with its noise mask), Krea2AnyPaintModelPatch (model, kv_cache). Its example graph:
  UNETLoader → LoraLoaderModelOnly (AnyPaint 1.0) → ModelPatch → KSampler (8 steps, CFG 1,
  euler, simple, denoise 1.0), negative an empty CLIPTextEncode; "the output is the raw model
  decode". It imports einops, numpy and Pillow, which ComfyUI v0.35.0 already requires.
White mask pixels are redrawn, black ones kept. It reads the prompt with Krea 2's own text
encoder (qwen3vl_4b) — no other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAMILY = "anypaint"
REVISION = "1a9fb37a304c27523939c44fc2b770c11472451b"


@dataclass(frozen=True)
class ModelFile:
    folder: str
    filename: str
    size: int  # bytes (Hugging Face's file list at REVISION)
    sha256: str  # Hugging Face's LFS hash at REVISION; the GPU agent checks it

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/yijunwang2/krea2-anypaint/resolve/{REVISION}/{self.filename}"
        )


LORA = ModelFile(
    "loras",
    "krea2_anypaint_rank32.safetensors",
    228_587_752,
    "3a7d09f6b27f8ead160d340f2f59c11f4ee635c4a1ee87ffd8b1b9f9ba412f7a",
)
FILES = (LORA,)
NODES = ("Krea2AnyPaintPrepare", "Krea2AnyPaintEncode", "Krea2AnyPaintModelPatch")
# The authors' values.
STRENGTH = 1.0
REFERENCE_EDGE = 384
BOUNDARY_PX = 32


def downloads() -> list[dict[str, str]]:
    """The model files for the GPU agent (huggingface.co only, checked by SHA-256)."""
    return [
        {"url": f.url, "folder": f.folder, "filename": f.filename, "sha256": f.sha256}
        for f in FILES
    ]


def _choices(object_info: dict[str, Any], node: str, field: str) -> set[str]:
    spec = ((object_info.get(node) or {}).get("input") or {}).get("required") or {}
    options = (spec.get(field) or [None])[0]
    return set(options) if isinstance(options, list) else set()


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Whether this ComfyUI has the AnyPaint nodes and its LoRA file."""
    loras = _choices(object_info, "LoraLoaderModelOnly", "lora_name")
    files = [
        {
            "folder": f.folder,
            "filename": f.filename,
            "size": f.size,
            "installed": f.filename in loras,
        }
        for f in FILES
    ]
    missing_nodes = [n for n in NODES if n not in object_info]
    missing_files = [f["filename"] for f in files if not f["installed"]]
    return {
        "files": files,
        "missing_files": missing_files,
        "missing_nodes": missing_nodes,
        "ready": not missing_files and not missing_nodes,
    }
