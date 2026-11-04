"""The Advanced engine's own ComfyUI graphs (API format).

Separate from the built-in Krea 2 workflow (`comfy/krea2.build_workflow`), which is neither
called nor changed here: the verified Krea 2 loaders and sampler settings are read from
`krea2.VARIANTS`, the graphs are built here. Built-in ComfyUI nodes only, all present in
ComfyUI v0.35.0 (the pinned rented image; checked in its nodes.py and comfy_extras):

- text to image: UNETLoader → LoraLoaderModelOnly → CLIPLoader (krea2) → CLIPTextEncode →
  (ConditioningZeroOut | negative) → EmptyLatentImage → KSampler → VAEDecode → SaveImage;
- masked edit: LoadImage → ImageScale → SAM 3.1 (CheckpointLoaderSimple + CLIPTextEncode with
  its own text model + SAM3_Detect, as in Enhance photo) → GrowMask → the latent noise mask
  (SetLatentNoiseMask) so Krea 2 redraws only that region → VAEDecode → ImageCompositeMasked
  with a feathered mask, so every pixel outside the region is the photo's own;
- whole photo: the same without a mask (image to image at `denoise`).
"""

from __future__ import annotations

from typing import Any

from app.comfy import enhance, krea2

INPUT = krea2.INPUT_PLACEHOLDER  # the runner puts the uploaded photo's name here
SAM3_FILE = enhance.SAM3.filename

NODES_T2I = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)
NODES_EDIT = (
    *NODES_T2I,
    "LoadImage",
    "ImageScale",
    "VAEEncode",
    "CheckpointLoaderSimple",
    "SAM3_Detect",
    "GrowMask",
    "MaskToImage",
    "ImageBlur",
    "ImageToMask",
    "SetLatentNoiseMask",
    "ImageCompositeMasked",
)

# Region edits: how much of the region is redrawn by default (the studio's values; Krea
# publishes none). A colour change needs more than closing the lips.
DENOISE = {
    "hair": 0.85,
    "mouth": 0.7,
    "eyes": 0.6,
    "face": 0.55,
    "expression": 0.6,
    "clothing": 0.85,
    "background": 0.9,
    "whole": 0.5,
}
# SAM 3.1's text prompt per target when the plan gives none.
REGIONS = {
    "hair": "hair",
    "mouth": "mouth and lips",
    "eyes": "eyes",
    "face": "face",
    "expression": "face",
    "clothing": "clothing",
    "background": "background",
}
DENOISE_MIN = 0.2
DENOISE_MAX = 0.95


def _krea(
    wf: dict[str, Any],
    *,
    unet: str,
    text_encoder: str,
    vae: str,
    loras: list[tuple[str, float]],
) -> list[Any]:
    wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    wf["2"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": text_encoder, "type": krea2.FAMILY, "device": "default"},
    }
    wf["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    model: list[Any] = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        node = str(20 + index)
        wf[node] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model, "lora_name": name, "strength_model": strength},
        }
        model = [node, 0]
    return model


def _conditioning(wf: dict[str, Any], variant: str, prompt: str, negative: str) -> None:
    spec = krea2.VARIANTS[variant]
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    if spec.negative_prompt:
        wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}}
    else:  # Turbo: CFG 1 ignores a negative prompt (as in the official template)
        wf["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}


def _sampler(
    variant: str,
    model: list[Any],
    latent: list[Any],
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
) -> dict[str, Any]:
    spec = krea2.VARIANTS[variant]
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": model,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": spec.sampler,
            "scheduler": spec.scheduler,
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": latent,
            "denoise": denoise,
        },
    }


def text_to_image(
    *,
    variant: str,
    unet: str,
    text_encoder: str,
    vae: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    loras: list[tuple[str, float]],
    prefix: str = "advanced",
) -> dict[str, Any]:
    wf: dict[str, Any] = {}
    model = _krea(wf, unet=unet, text_encoder=text_encoder, vae=vae, loras=loras)
    _conditioning(wf, variant, prompt, negative)
    wf["6"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    wf["7"] = _sampler(variant, model, ["6", 0], seed, steps, cfg, 1.0)
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


def edit(
    *,
    variant: str,
    unet: str,
    text_encoder: str,
    vae: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    loras: list[tuple[str, float]],
    denoise: float,
    region: str | None,
    grow: int = 16,
    feather: int = 9,
    prefix: str = "advanced",
) -> dict[str, Any]:
    """Redraws only `region` (SAM 3.1's text prompt) of the photo; None = the whole photo.
    The result outside the region is the photo itself (pasted back with a soft edge)."""
    wf: dict[str, Any] = {}
    model = _krea(wf, unet=unet, text_encoder=text_encoder, vae=vae, loras=loras)
    _conditioning(wf, variant, prompt, negative)
    wf["10"] = {"class_type": "LoadImage", "inputs": {"image": INPUT}}
    wf["11"] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["10", 0],
            "upscale_method": "lanczos",
            "width": width,
            "height": height,
            "crop": "disabled",
        },
    }
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
    latent: list[Any] = ["12", 0]
    if region:
        wf.update(
            {
                "30": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_FILE}},
                "31": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["30", 1], "text": region},
                },
                "32": {
                    "class_type": "SAM3_Detect",
                    "inputs": {
                        "model": ["30", 0],
                        "image": ["11", 0],
                        "conditioning": ["31", 0],
                        "threshold": 0.4,
                        "refine_iterations": 2,
                        "individual_masks": False,
                    },
                },
                "33": {
                    "class_type": "GrowMask",
                    "inputs": {"mask": ["32", 0], "expand": grow, "tapered_corners": True},
                },
                "34": {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["12", 0], "mask": ["33", 0]},
                },
                # A soft edge for pasting the redrawn region back into the photo.
                "35": {"class_type": "MaskToImage", "inputs": {"mask": ["33", 0]}},
                "36": {
                    "class_type": "ImageBlur",
                    "inputs": {"image": ["35", 0], "blur_radius": feather, "sigma": feather / 2},
                },
                "37": {
                    "class_type": "ImageToMask",
                    "inputs": {"image": ["36", 0], "channel": "red"},
                },
            }
        )
        latent = ["34", 0]
    wf["7"] = _sampler(variant, model, latent, seed, steps, cfg, denoise)
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    out: list[Any] = ["8", 0]
    if region:
        wf["38"] = {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["11", 0],
                "source": ["8", 0],
                "mask": ["37", 0],
                "x": 0,
                "y": 0,
                "resize_source": False,
            },
        }
        out = ["38", 0]
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": prefix}}
    return wf
