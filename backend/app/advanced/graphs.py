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
- whole photo: the same without a mask (image to image at `denoise`);
- colour change (hair, clothing): before the redraw the region is tinted towards the new
  colour (EmptyImage in that colour, ImageBlend "normal", ImageCompositeMasked inside the
  mask), so Krea 2 starts from the right colour instead of fighting the old one (Turbo has
  CFG 1, so a negative prompt cannot push the old colour away);
- face regions (mouth, eyes, face, expression): "crop and stitch" — ImageCrop around the face
  (found on the Mac, `advanced/crop.py`), ImageScale to 1024 px, the same masked redraw
  there, ImageScale back and ImageCompositeMasked into the photo at that place, inside the
  mask only.
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
    "EmptyImage",
    "ImageBlend",
    "ImageCrop",
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

# Colour changes (the studio's values, to be tuned on a GPU): the region is tinted first, so
# less has to be redrawn — the hairstyle and its light stay closer to the photo.
COLOUR_TARGETS = ("hair", "clothing")
DENOISE_COLOUR = {"hair": 0.75, "clothing": 0.75}
TINT = 0.55  # how much of the region's pixels become the new colour before the redraw
TINT_STEP = 0.15  # more on every correction
TINT_MAX = 0.85

# How far the mask grows past what SAM 3.1 found (px at the edit size) — hair gets more, so
# loose strands and the edge are redrawn too and no rim of the old colour stays.
GROW = {"hair": 24}
GROW_DEFAULT = 16


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
    grow: int = GROW_DEFAULT,
    feather: int = 9,
    tint: str | None = None,
    tint_strength: float = TINT,
    crop: tuple[int, int, int] | None = None,
    crop_size: int = 1024,
    prefix: str = "advanced",
) -> dict[str, Any]:
    """Redraws only `region` (SAM 3.1's text prompt) of the photo; None = the whole photo.
    The result outside the region is the photo itself (pasted back with a soft edge).
    `tint` ("#RRGGBB"): the region is first tinted towards that colour (`tint_strength`).
    `crop` (left, top, side at width × height): only that square is redrawn, at `crop_size`."""
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
    # What is redrawn: the photo, or for a face edit a square around the face, enlarged.
    base: list[Any] = ["11", 0]
    size_w, size_h = width, height
    if region and crop:
        left, top, side = crop
        wf["50"] = {
            "class_type": "ImageCrop",
            "inputs": {"image": ["11", 0], "width": side, "height": side, "x": left, "y": top},
        }
        wf["51"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["50", 0],
                "upscale_method": "lanczos",
                "width": crop_size,
                "height": crop_size,
                "crop": "disabled",
            },
        }
        base = ["51", 0]
        size_w = size_h = crop_size
    pixels: list[Any] = base
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
                        "image": base,
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
        if tint:
            # The colour first: the region (only inside the mask) blended towards it.
            wf["40"] = {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": size_w,
                    "height": size_h,
                    "batch_size": 1,
                    "color": colour_int(tint),
                },
            }
            wf["41"] = {
                "class_type": "ImageBlend",
                "inputs": {
                    "image1": base,
                    "image2": ["40", 0],
                    "blend_factor": round(tint_strength, 2),
                    "blend_mode": "normal",
                },
            }
            wf["42"] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": base,
                    "source": ["41", 0],
                    "mask": ["33", 0],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                },
            }
            pixels = ["42", 0]
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": pixels, "vae": ["3", 0]}}
    wf["7"] = _sampler(variant, model, latent, seed, steps, cfg, denoise)
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    out: list[Any] = ["8", 0]
    if region:
        source: list[Any] = ["8", 0]
        x = y = 0
        if crop:
            # Back to the crop's size, pasted where it came from (the mask scales with it).
            left, top, side = crop
            wf["52"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["8", 0],
                    "upscale_method": "lanczos",
                    "width": side,
                    "height": side,
                    "crop": "disabled",
                },
            }
            source, x, y = ["52", 0], left, top
        wf["38"] = {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["11", 0],
                "source": source,
                "mask": ["37", 0],
                "x": x,
                "y": y,
                "resize_source": False,
            },
        }
        out = ["38", 0]
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": prefix}}
    return wf


def colour_int(hex_colour: str) -> int:
    """ "#RRGGBB" → EmptyImage's `color` (0xRRGGBB)."""
    return int(hex_colour.lstrip("#"), 16)
