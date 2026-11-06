"""The Advanced engine's own ComfyUI graphs (API format).

Separate from the built-in Krea 2 workflow (`comfy/krea2.build_workflow`), which is neither
called nor changed here: the verified Krea 2 loaders and sampler settings are read from
`krea2.VARIANTS`, the graphs are built here. Built-in ComfyUI nodes only, all present in
ComfyUI v0.35.0 (the pinned rented image; checked in its nodes.py and comfy_extras):

- text to image: UNETLoader → LoraLoaderModelOnly → CLIPLoader (krea2) → CLIPTextEncode →
  (ConditioningZeroOut | negative) → EmptyLatentImage → KSampler → VAEDecode → SaveImage;
- masked edit: LoadImage → ImageScale → SAM 3.1 (CheckpointLoaderSimple + CLIPTextEncode with
  its own text model + SAM3_Detect, as in Enhance photo) → GrowMask → a feathered mask as the
  latent noise mask (SetLatentNoiseMask) with DifferentialDiffusion on the model, so Krea 2
  redraws the region fully and its soft edge gradually (no seam) → VAEDecode →
  ImageCompositeMasked with the same mask, so every pixel outside the region is the photo's;
- whole photo: the same without a mask (image to image at `denoise`);
- colour change (hair, clothing): before the redraw the region is tinted towards the new
  colour (EmptyImage in that colour, ImageBlend, ImageCompositeMasked inside the mask), so
  Krea 2 starts from the right colour instead of fighting the old one (Turbo has CFG 1, so a
  negative prompt cannot push the old colour away). "multiply" for a darker colour, "screen"
  for a lighter one: both keep the strands' light and shadow, a flat "normal" blend washed
  them out to grey;
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
    "DifferentialDiffusion",
    "MaskComposite",
    "InvertMask",
    "SolidMask",
)

# Region edits: how much of the region is redrawn by default (the studio's values; Krea
# publishes none). A colour change needs more than closing the lips; the person ("body")
# is redrawn gently: the pose comes first (the owner's tests, 2026-11-05: at 0.7 a foot in a
# pillow and once a whole new close-up with holes of the old background in the face; at 0.6
# an arm lying on the bed became a hand propping up the head), so mostly the skin tone and
# surface follow the LoRA, its figure only a little.
DENOISE = {
    "hair": 0.85,
    "mouth": 0.7,
    "eyes": 0.6,
    "face": 0.55,
    "expression": 0.6,
    "clothing": 0.85,
    "background": 0.9,
    "body": 0.45,
    "whole": 0.5,
}
# The face when the person becomes the LoRA's character: redrawn gently, and when it is not
# yet alike enough, again at the same strength from the last result (a new seed each time).
# The owner's tests (2026-11-05, his three photos, SFace against the LoRA's photos, which are
# 64 % alike): one pass at 0.75 or 0.85 reached 59–65 % on a car and a bed photo, two passes
# at 0.6 the same, and only the gentle passes kept a tilted selfie's tongue-out expression —
# at 0.85 it was lost, at 0.75 hair was drawn over the face.
DENOISE_IDENTITY_FACE = 0.65
# The figure when the person becomes the LoRA's character (the owner's request, 2026-11-06:
# body, breasts, waist and hips like the LoRA's): redrawn strongly enough for the LoRA's
# proportions, while the face, hair, hands and feet stay the photo's and hold the pose
# (EXCLUDE), and the mask grows so the outline can change (GROW). Tried 0.6–0.85 on the
# owner's six photos (2026-11-06): the pose held throughout; at 0.8 a lying woman's buttocks
# came out misshapen and once at 0.75 a second hand appeared on a chest; 0.7 was clean.
DENOISE_IDENTITY_BODY = 0.7
# "Improve the background": the same place redrawn sharper (a new background is 0.9).
DENOISE_BACKGROUND_SHARPER = 0.5
# SAM 3.1's text prompt per target when the plan gives none.
REGIONS = {
    "hair": "hair",
    "mouth": "mouth and lips",
    "eyes": "eyes",
    "face": "face",
    "expression": "face",
    "clothing": "clothing",
    "background": "background",
    "body": "person",
}
# What a region's redraw leaves alone (SAM 3.1 finds each, grown a little, and they are taken
# out of the grown mask; nothing found = nothing taken out): the figure without the face (its
# expression stays for the face edit), the hair (its own step) and the hands and feet, which
# pin the pose (the owner's tests, 2026-11-05: with only the face left out an arm on a bed
# became a hand propping the head, and a foot went into a pillow); the hair without the eyes
# and brows, which a whole-photo redraw would deform (an eye after the hair, 2026-11-05).
# The head too: SAM 3.1's "face" missed the profile of a woman at the edge of a mirror photo
# (2026-11-06), whose head the figure then turned towards the camera.
# SAM 3.1 returns only its best match per text unless the text says how many ("hands:4";
# ComfyUI v0.35.0 `comfy/text_encoders/sam3_clip.py`, `_parse_prompts`): one "hands" pinned
# one hand. Anything found outside the region does nothing.
EXCLUDE: dict[str, tuple[str, ...]] = {
    "body": ("face:2", "head:2", "hair:2", "hands:4", "feet:4"),
    "hair": ("eyes and eyebrows:2",),
}
EXCLUDE_GROW = 8
# Left alone too, except where SAM 3.1 found the region itself: the face during the hair
# step (its grown rim reached the cheeks and smeared the face of the woman on the floor and
# drew a second hand on the lift photo's cheek, 2026-11-06); bangs over the forehead are
# hair and still change colour. The face is redrawn after the hair anyway.
BESIDE: dict[str, str] = {"hair": "face:2"}
# What a region's redraw changes only partly (DifferentialDiffusion reads the mask as a
# per-pixel strength, and the result is pasted back with it): the figure's clothing at this
# share, so its shape can follow the LoRA's figure while its colour and kind stay (at full
# strength grey sweatpants came out beige and pink lace shorts a white thong, 2026-11-06).
SOFTEN: dict[str, tuple[str, float]] = {"body": ("clothing:4", 0.5)}
DENOISE_MIN = 0.2
DENOISE_MAX = 0.95

# Colour changes (the studio's values, to be tuned on a GPU): the region is tinted first, so
# less has to be redrawn — the hairstyle and its light stay closer to the photo. Darker
# colours multiply (blonde → dark brown keeps every strand), lighter ones screen.
COLOUR_TARGETS = ("hair", "clothing")
# Hair at 0.85 since 2026-11-05: at 0.75 Krea 2 kept the old colour's light strands as
# streaks (blonde → dark brown came out with caramel highlights, three times).
DENOISE_COLOUR = {"hair": 0.85, "clothing": 0.75}
# How far the region goes towards the colour. Multiply at full strength since the owner's car
# photo (2026-11-05): at 0.85 the old blonde strands stayed as lighter streaks (hair
# measured 42/28/23, highlights 54), at 1.0 darker and more even (32/21/17, 43).
TINT = {"multiply": 1.0, "screen": 0.55}
TINT_STEP = 0.1  # more on every correction
TINT_MAX = 1.0
DARK = 0.5  # a colour's luma (0–1) below this is "darker": multiply

# How far the mask grows past what SAM 3.1 found (px at the edit size) — hair gets more, so
# loose strands and the edge are redrawn too and no rim of the old colour stays — and how
# soft its edge is (ImageBlur radius, at most 31).
GROW = {"hair": 32, "body": 40, "background": 0}
# The background is everything but the people: SAM 3.1's "person", grown this much (their
# outline stays theirs), inverted. A photo without a person = all of it.
BACKGROUND_PERSON = "person"
# SAM 3.1's "person" leaves out a person in a mirror (the owner's lift photo, 2026-11-06:
# only the woman at the edge was found, not her reflection, the photo's subject); this text
# finds the reflection, and in photos without a mirror the person again. Both are added.
REFLECTION = "reflection of a person"
# Regions found on every person of the photo's people (up to this many matches, kept only
# on what "person" and REFLECTION found): a woman and her reflection have two heads of hair
# (the lift photo kept the real woman's blonde hair beside her brown-haired reflection),
# while a stranger in the background is left alone.
SEVERAL = {"hair": 2}
PERSON_GROW = 8
GROW_DEFAULT = 16
FEATHER = 16


def tint(hex_colour: str, attempt: int = 0) -> tuple[str, float]:
    """The blend mode for this colour and how strong it is on this attempt."""
    r, g, b = (int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    mode = "multiply" if 0.299 * r + 0.587 * g + 0.114 * b < DARK else "screen"
    return mode, round(min(TINT_MAX, TINT[mode] + TINT_STEP * attempt), 2)


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
    exclude: tuple[str, ...] = (),
    soften: tuple[str, float] | None = None,
    grow: int = GROW_DEFAULT,
    feather: int = FEATHER,
    colour: str | None = None,
    attempt: int = 0,
    crop: tuple[int, int, int] | None = None,
    crop_size: int = 1024,
    prefix: str = "advanced",
) -> dict[str, Any]:
    """Redraws only `region` (SAM 3.1's text prompt) of the photo; None = the whole photo;
    "background" = everything but the people. The result outside the region is the photo
    itself (pasted back with a soft edge). `exclude` (SAM 3.1's text prompts): taken out.
    `colour` ("#RRGGBB"): the region is first tinted towards it (stronger on every
    `attempt`). `crop` (left, top, side at width × height): only that square is redrawn, at
    `crop_size`."""
    wf: dict[str, Any] = {}
    model = _krea(wf, unet=unet, text_encoder=text_encoder, vae=vae, loras=loras)
    if region:
        # The soft edge of the mask is redrawn gradually instead of all or nothing.
        wf["60"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": model}}
        model = ["60", 0]
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
        redrawn: list[Any] = ["33", 0]  # the grown mask, before its soft edge
        tinted: list[Any] = ["32", 0]  # what SAM 3.1 found: the colour goes only there
        background = region == REGIONS["background"]
        wf["30"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_FILE}}
        text = BACKGROUND_PERSON if background else region
        several = SEVERAL.get(region)
        wf.update(_detect("31", "32", f"{text}:{several}" if several else text, base))
        found: list[Any] = ["32", 0]
        if text == BACKGROUND_PERSON or several:
            # The people: SAM 3.1's "person" and a person in a mirror.
            if several:
                wf.update(_detect("53", "54", BACKGROUND_PERSON, base))
            wf.update(_detect("45", "46", REFLECTION, base))
            wf["48"] = _combine(["54" if several else "32", 0], ["46", 0], "add")
            found = tinted = ["48", 0]
            if several:
                wf["55"] = {
                    "class_type": "GrowMask",
                    "inputs": {"mask": ["48", 0], "expand": PERSON_GROW, "tapered_corners": True},
                }
                wf["56"] = _combine(["32", 0], ["55", 0], "and")
                found = tinted = ["56", 0]
        if background:
            wf["43"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": found, "expand": PERSON_GROW, "tapered_corners": True},
            }
            wf["44"] = {"class_type": "InvertMask", "inputs": {"mask": ["43", 0]}}
            found = tinted = ["44", 0]
        wf["33"] = {
            "class_type": "GrowMask",
            "inputs": {"mask": found, "expand": grow, "tapered_corners": True},
        }
        taken: list[Any] | None = None
        for index, anchor in enumerate(exclude):
            node = 70 + 4 * index
            wf.update(_detect(str(node), str(node + 1), anchor, base))
            wf[str(node + 2)] = {
                "class_type": "GrowMask",
                "inputs": {
                    "mask": [str(node + 1), 0],
                    "expand": EXCLUDE_GROW,
                    "tapered_corners": True,
                },
            }
            grown = [str(node + 2), 0]
            if taken is None:
                taken = grown
            else:
                wf[str(node + 3)] = _combine(taken, grown, "add")
                taken = [str(node + 3), 0]
        beside = BESIDE.get(region)
        if beside:
            wf.update(_detect("61", "62", beside, base))
            wf["63"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": ["62", 0], "expand": EXCLUDE_GROW, "tapered_corners": True},
            }
            wf["64"] = _combine(["63", 0], found, "subtract")
            if taken is None:
                taken = ["64", 0]
            else:
                wf["65"] = _combine(taken, ["64", 0], "add")
                taken = ["65", 0]
        if taken is not None:
            wf["47"] = _combine(["33", 0], taken, "subtract")
            redrawn = ["47", 0]
            if colour:
                wf["49"] = _combine(tinted, taken, "subtract")
                tinted = ["49", 0]
        if soften:
            part, share = soften
            wf.update(_detect("90", "91", part, base))
            wf["92"] = {
                "class_type": "SolidMask",
                "inputs": {"value": round(1 - share, 2), "width": size_w, "height": size_h},
            }
            wf["93"] = _combine(["91", 0], ["92", 0], "multiply")
            wf["94"] = _combine(redrawn, ["93", 0], "subtract")
            redrawn = ["94", 0]
        wf.update(
            {
                # The soft mask decides what is redrawn and what is pasted back.
                "34": {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["12", 0], "mask": ["37", 0]},
                },
                "35": {"class_type": "MaskToImage", "inputs": {"mask": redrawn}},
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
        if colour:
            # The colour first: the region (only inside the mask) blended towards it.
            mode, strength = tint(colour, attempt)
            wf["40"] = {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": size_w,
                    "height": size_h,
                    "batch_size": 1,
                    "color": colour_int(colour),
                },
            }
            wf["41"] = {
                "class_type": "ImageBlend",
                "inputs": {
                    "image1": base,
                    "image2": ["40", 0],
                    "blend_factor": strength,
                    "blend_mode": mode,
                },
            }
            # Only what SAM 3.1 found is tinted (not the grown rim), so no coloured halo
            # stays on the skin or the background around it.
            wf["42"] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": base,
                    "source": ["41", 0],
                    "mask": tinted,
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


def _detect(text_node: str, node: str, text: str, image: list[Any]) -> dict[str, Any]:
    """SAM 3.1 finds `text` in `image` (one mask of all it found; empty when nothing)."""
    return {
        text_node: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["30", 1], "text": text}},
        node: {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["30", 0],
                "image": image,
                "conditioning": [text_node, 0],
                "threshold": 0.4,
                "refine_iterations": 2,
                "individual_masks": False,
            },
        },
    }


def _combine(mask: list[Any], other: list[Any], operation: str) -> dict[str, Any]:
    """`mask` with `other` added or taken out (MaskComposite clamps the result to 0–1)."""
    return {
        "class_type": "MaskComposite",
        "inputs": {"destination": mask, "source": other, "x": 0, "y": 0, "operation": operation},
    }


def colour_int(hex_colour: str) -> int:
    """ "#RRGGBB" → EmptyImage's `color` (0xRRGGBB)."""
    return int(hex_colour.lstrip("#"), 16)
