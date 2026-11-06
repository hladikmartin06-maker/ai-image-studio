"""Advanced's "Finish" (the owner's plan, phase 3, approved 2026-11-06): one gentle face pass
with the character LoRA and the skin LoRA, a likeness check, then SeedVR2.

Sources (checked 2026-11-06):
- Model card huggingface.co/inlineresearch/skin-lora-krea-2-raw (revision 15fbc074, Krea 2
  Community License): "a photorealistic skin texture LoRA", "a texture and realism adapter,
  not a specific person"; trained on Krea 2 RAW, it "carries over to Krea 2 Turbo"; strength
  1.0; `inline-skin-lora, detailed skin texture` at the front of the prompt; a PEFT adapter
  (rank 16) loaded as a standard LoRA. NSFW is not mentioned.
- SeedVR2: ComfyUI's own nodes and the two files Enhance photo already uses (its settings:
  1 step, tiles of 512, the original's colours by "lab"); 3× is very slow (48 GB VRAM
  recommended by the template's README).
"""

from __future__ import annotations

from typing import Any

from app.comfy import anypaint, enhance

SKIN = "skin"
UPSCALE = "seedvr2"
SKIN_REVISION = "15fbc074697eed816159dbbc99f9c98170301b14"


class ModelFile(anypaint.ModelFile):
    @property
    def url(self) -> str:
        return (
            "https://huggingface.co/inlineresearch/skin-lora-krea-2-raw/resolve/"
            f"{SKIN_REVISION}/{self.filename}"
        )


SKIN_LORA = ModelFile(
    "loras",
    "inline-skin-lora-krea-2-raw.safetensors",
    191_871_280,
    "7ac516bc335953be4afffa53411e38a012e91eaff56afc16c79232438b7040ac",
)
SKIN_TRIGGER = "inline-skin-lora, detailed skin texture"  # the authors'
SKIN_STRENGTH = 0.6  # the owner's estimate (the authors': 1.0)
FACE_DENOISE = 0.35  # the owner's 0.3–0.4 (estimate)
LIKENESS_DROP = 0.05  # a face pass less alike by more than this is not kept (estimate)
UPSCALE_FILES = (enhance.SEEDVR2, enhance.SEEDVR2_VAE)
UPSCALE_NODES = (
    "ImageScaleBy",
    "SeedVR2Preprocess",
    "VAEEncodeTiled",
    "SeedVR2Conditioning",
    "VAEDecodeTiled",
    "SeedVR2PostProcessing",
)
UPSCALES = enhance.UPSCALES
DEFAULT_UPSCALE = enhance.DEFAULT_UPSCALE


def downloads(variant: str) -> list[dict[str, str]]:
    """The skin LoRA (checked by SHA-256 on the GPU), or SeedVR2's two files from the Enhance
    photo repository, as Enhance photo downloads them."""
    if variant == SKIN:
        f = SKIN_LORA
        return [{"url": f.url, "folder": f.folder, "filename": f.filename, "sha256": f.sha256}]
    return [{"url": f.url, "folder": f.folder, "filename": f.filename} for f in UPSCALE_FILES]


def _state(files: list[dict[str, Any]], nodes: tuple[str, ...], object_info: dict) -> dict:
    missing_nodes = [n for n in nodes if n not in object_info]
    missing_files = [f["filename"] for f in files if not f["installed"]]
    return {
        "files": files,
        "missing_files": missing_files,
        "missing_nodes": missing_nodes,
        "ready": not missing_files and not missing_nodes,
    }


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Whether this ComfyUI has the skin LoRA, and SeedVR2's nodes and files."""
    listed = {
        "loras": anypaint._choices(object_info, "LoraLoaderModelOnly", "lora_name"),
        "diffusion_models": anypaint._choices(object_info, "UNETLoader", "unet_name"),
        "vae": anypaint._choices(object_info, "VAELoader", "vae_name"),
    }

    def row(f: Any) -> dict[str, Any]:
        installed = f.filename in listed[f.folder]
        return {"folder": f.folder, "filename": f.filename, "size": f.size, "installed": installed}

    return {
        SKIN: _state([row(SKIN_LORA)], (), object_info),
        UPSCALE: _state([row(f) for f in UPSCALE_FILES], UPSCALE_NODES, object_info),
    }
