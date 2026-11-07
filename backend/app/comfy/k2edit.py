"""Advanced (the owner's request of 2026-11-07): Krea 2 Turbo edits a photo from a plain
instruction, with the owner's own Krea 2 LoRA for the character.

Verified (2026-11-07):
- huggingface.co/conradlocke/krea2-identity-edit (revision f33b2629): Krea 2 in-context edit
  LoRAs, "Requires the ComfyUI-Krea2Edit node pack" (dual conditioning: the source as VAE
  tokens + the instruction encoded together with the source by Qwen3-VL). Rank-128 files
  "rank-reduced from the full files, near-identical quality". Recommended: Turbo, 8 steps,
  CFG 1, ref_boost 4 (General Edit V2) / 2 (Identity Edit V2), grounding_px 768, ≤ 2 MP.
  Licenses: V2 files free for personal / non-commercial use only (commercial use needs the
  author's license); v1.2 under the Krea 2 Community License Agreement. Both LoRAs at 1.0
  together give noise ("use about 0.6 each").
- github.com/lbouaraba/comfyui-krea2edit v2.0.0 (comfy/custom_nodes.py): Krea2EditModelPatch
  (model, source_latent; optional vae + source_image, target_latent, ref_boost, fit_mode,
  ref_position) and Krea2EditGroundedEncode (clip, prompt; optional image, grounding_px).
  Its example workflow: UNETLoader krea2_turbo_fp8_scaled, CLIPLoader qwen3vl_4b_fp8_scaled
  (type krea2), VAELoader qwen_image_vae, LoraLoaderModelOnly, EmptySD3LatentImage, KSampler
  euler / simple, cfg 1.
Measured on the studio's own runs: see docs/PROVIDERS.md (Advanced).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.comfy import krea2

FAMILY = "k2edit"
MODEL_KEY = "krea2-advanced"
NAME = "Krea 2 Advanced"
HF_REPO = "conradlocke/krea2-identity-edit"
HF_REVISION = "f33b26291c640e8ec50b0da420bfee3c9a660daa"


@dataclass(frozen=True)
class EditLora:
    key: str
    filename: str
    size: int
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{self.filename}"


EDIT_LORAS = {
    lora.key: lora
    for lora in (
        EditLora(
            "general_v2",
            "krea2_general_edit_v2_r128.safetensors",
            914_188_760,
            "fbca6ab74b9bef3db8de1d4a49b93d4fe7a778f5d4ae6995e0769ea8de736f00",
            "Krea 2 Identity Edit V2 License (non-commercial) + Krea 2 Community License",
        ),
        EditLora(
            "identity_v2",
            "krea2_identity_edit_v2_r128.safetensors",
            914_188_768,
            "2082515f1eb66532f5efb5343eb15048a29e3abe9f356e35e784cc60d00b45f7",
            "Krea 2 Identity Edit V2 License (non-commercial) + Krea 2 Community License",
        ),
        EditLora(
            "identity_v1_2",
            "krea2_identity_edit_v1_2_r128.safetensors",
            914_159_744,
            "f53db0bb4b081d638f196865cbc9f055379704fafb788336784fc1ccde18d825",
            "Krea 2 Community License",
        ),
    )
}
# The edit LoRA Advanced runs with, and its settings (the studio's own tests, PROVIDERS.md):
# ref_boost 4 keeps the photo best; with a character LoRA 2 lets the face become the
# character's (likeness up, the pose, clothes and background still kept).
DEFAULT_EDIT = "general_v2"


def settings_for(character: bool) -> dict[str, Any]:
    return {"steps": 8, "ref_boost": 2.0 if character else 4.0, "grounding_px": 768}


NODES = ("Krea2EditModelPatch", "Krea2EditGroundedEncode")
REQUIRED_NODES = (*NODES, "EmptySD3LatentImage", "LoraLoaderModelOnly", "ImageScale")
TURBO = krea2.VARIANTS["turbo"].known_files[0]  # krea2_turbo_fp8_scaled.safetensors
REF_POSITIONS = ("aligned (trained)", "beside (right of target)")
MAX_PROMPT = 2000


def _files(keys: tuple[str, ...]) -> list[tuple[str, str, str, float, str | None]]:
    """(url, folder, filename, size in GB, sha256) of everything Advanced needs."""
    out = [
        (f.url, f.folder, f.filename, f.size_gb, None)
        for f in (krea2.TEXT_ENCODER, krea2.VAE, TURBO)
    ]
    for key in keys:
        e = EDIT_LORAS[key]
        out.append((e.url, "loras", e.filename, e.size / 1e9, e.sha256))
    return out


def downloads(keys: tuple[str, ...] = (DEFAULT_EDIT,)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for url, folder, filename, _, sha in _files(keys):
        item: dict[str, Any] = {"url": url, "folder": folder, "filename": filename}
        if sha:
            item["sha256"] = sha
        out.append(item)
    return out


def sizes_gb(keys: tuple[str, ...] = (DEFAULT_EDIT,)) -> dict[str, float]:
    return {filename: size for _, _, filename, size, _ in _files(keys)}


def catalog_json() -> dict[str, Any]:
    edit = EDIT_LORAS[DEFAULT_EDIT]
    return {
        "name": NAME,
        "model_key": MODEL_KEY,
        "files": [
            {"folder": folder, "filename": filename, "size": round(size * 1e9)}
            for _, folder, filename, size, _ in _files((DEFAULT_EDIT,))
        ],
        "total_bytes": round(sum(sizes_gb().values()) * 1e9),
        "edit_lora": edit.filename,
        "edit_license": edit.license,
        "repo": f"https://huggingface.co/{HF_REPO}",
        "max_prompt": MAX_PROMPT,
    }


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Which of Advanced's nodes and files this ComfyUI has."""

    def choices(node: str, field: str) -> list[str]:
        spec = (object_info.get(node) or {}).get("input", {}).get("required", {}).get(field)
        return list(spec[0]) if spec and isinstance(spec[0], list) else []

    unets = choices("UNETLoader", "unet_name")
    encoders = choices("CLIPLoader", "clip_name")
    vaes = choices("VAELoader", "vae_name")
    loras = choices("LoraLoaderModelOnly", "lora_name")
    edit = {k: e.filename in loras for k, e in EDIT_LORAS.items()}
    missing_nodes = [n for n in REQUIRED_NODES if n not in object_info]
    missing_files = [
        f.filename
        for f, found in (
            (TURBO, unets),
            (krea2.TEXT_ENCODER, encoders),
            (krea2.VAE, vaes),
        )
        if f.filename not in found
    ]
    return {
        "missing_nodes": missing_nodes,
        "missing_files": missing_files,
        "edit_loras": edit,
        "ready": not missing_nodes and not missing_files and edit[DEFAULT_EDIT],
    }


def names(prompt: str, trigger: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", prompt, re.IGNORECASE) is not None


def with_character(prompt: str, trigger: str, class_word: str) -> str:
    """The instruction with the LoRA's character named, so the LoRA's trigger word is in it."""
    if names(prompt, trigger):
        return prompt
    return f"{prompt.rstrip()} The {class_word or 'person'} is {trigger} {class_word}.".strip()


def build_edit(
    *,
    image: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    edit_lora: str,
    edit_strength: float = 1.0,
    loras: list[tuple[str, float]] | None = None,
    steps: int = 8,
    ref_boost: float = 4.0,
    grounding_px: int = 768,
    image_b: str | None = None,
    ref_position: str = REF_POSITIONS[0],
    prefix: str = "ai-image-studio-advanced",
) -> dict[str, Any]:
    """The API-format edit graph: the photo (scaled to width x height) is the reference the
    edit LoRA keeps, the instruction is encoded together with it, the owner's LoRAs follow the
    edit LoRA. An optional second photo (image_b) is the person for two-image edits (the first
    photo is then the scene, as the LoRAs were trained)."""
    if ref_position not in REF_POSITIONS:
        raise ValueError("Unknown reference position")
    wf: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": TURBO.filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": krea2.TEXT_ENCODER.filename,
                "type": krea2.FAMILY,
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": krea2.VAE.filename}},
        "20": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": EDIT_LORAS[edit_lora].filename,
                "strength_model": edit_strength,
            },
        },
    }
    model_ref: list[Any] = ["20", 0]
    for index, (name, strength) in enumerate(loras or []):
        node_id = str(21 + index)
        wf[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": name, "strength_model": strength},
        }
        model_ref = [node_id, 0]

    def scaled(node: str, name: str, w: int, h: int) -> list[Any]:
        wf[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[node + "1"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": [node, 0],
                "upscale_method": "lanczos",
                "width": w,
                "height": h,
                "crop": "center",
            },
        }
        return [node + "1", 0]

    src = scaled("10", image, width, height)
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": src, "vae": ["3", 0]}}
    wf["6"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    patch: dict[str, Any] = {
        "model": model_ref,
        "source_latent": ["12", 0],
        "ref_boost": ref_boost,
        "ref_boost_a": 1.0,
        "fit_mode": "fit",
        "vae": ["3", 0],
        "source_image": src,
        "target_latent": ["6", 0],
        "ref_position": ref_position,
        "ref_position_a": REF_POSITIONS[0],
    }
    encode: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "image": src,
        "grounding_px": grounding_px,
        "system_prompt": "",
    }
    if image_b:
        # Two images: the scene first (aligned), the person second; ref_boost acts on the last.
        person = scaled("30", image_b, width, height)
        wf["32"] = {"class_type": "VAEEncode", "inputs": {"pixels": person, "vae": ["3", 0]}}
        patch |= {"source_latent_b": ["32", 0], "source_image_b": person}
        encode["image_b"] = person
    wf["13"] = {"class_type": "Krea2EditModelPatch", "inputs": patch}
    wf["4"] = {"class_type": "Krea2EditGroundedEncode", "inputs": encode}
    # CFG 1 (Turbo): the negative is not used, so it is not encoded.
    wf["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["13", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


# --- keep what the edit did not change ------------------------------------------------------
#
# The model redraws the whole photo at about 1 MP; where it changed nothing, the original's
# own pixels (at the original's resolution) are put back, so untouched parts stay exactly as
# they were. The change mask: per-pixel difference of the two (both lightly blurred, so a
# one-pixel shift or VAE texture is not a change), opened to drop thin lines and specks,
# grown and feathered. Thresholds from the studio's own runs (docs/PROVIDERS.md, Advanced).
KEEP_THRESHOLD = 0.07
KEEP_OPEN = 9
KEEP_GROW = 16
KEEP_FEATHER = 8.0


def keep_unchanged(original: bytes, edited: bytes) -> tuple[bytes, float]:
    """The edited photo with the unchanged parts taken from the original, as PNG, and the
    share of the picture the edit changed (0–1)."""
    import io

    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    src = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGB")
    out = Image.open(io.BytesIO(edited)).convert("RGB")
    w, h = out.size

    def soft(im: Image.Image) -> Any:
        return np.asarray(im.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255

    diff = np.abs(soft(src.resize((w, h), Image.LANCZOS)) - soft(out)).mean(axis=2)
    mask = Image.fromarray(((diff > KEEP_THRESHOLD) * 255).astype(np.uint8), "L")
    mask = mask.filter(ImageFilter.MinFilter(KEEP_OPEN)).filter(ImageFilter.MaxFilter(KEEP_OPEN))
    mask = mask.filter(ImageFilter.MaxFilter(2 * KEEP_GROW + 1))
    mask = mask.filter(ImageFilter.MinFilter(2 * (KEEP_GROW // 2) + 1))
    mask = mask.filter(ImageFilter.GaussianBlur(KEEP_FEATHER))
    changed = float(np.asarray(mask, dtype=np.float32).mean() / 255)
    # The original's size, but never smaller than the edit and at most 1.5x it (a huge upload
    # would make the redrawn parts look soft next to the untouched ones).
    long_src, long_out = max(src.size), max(w, h)
    scale = min(max(long_src, long_out), round(1.5 * long_out)) / long_src
    size = (round(src.width * scale), round(src.height * scale))
    if size != src.size:
        src = src.resize(size, Image.LANCZOS)
    result = Image.composite(
        out.resize(size, Image.LANCZOS), src, mask.resize(size, Image.BILINEAR)
    )
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return buf.getvalue(), changed


# --- the character's face, redrawn close up ----------------------------------------------
#
# In a whole-body photo the face is small, so the owner's LoRA has few pixels to work with.
# With a character LoRA, the main face (YuNet on the Mac, services/faces.py) is cropped as a
# square 2.2x its size, redrawn at 1024 px by Krea 2 Turbo image to image with the LoRA at
# denoise 0.4 (keeps expression and angle), and pasted back through a feathered ellipse.
# Faces already large (over FACE_MAX_SHARE of the shorter side) are left alone. Measured on
# the studio's runs: face likeness to the character's photos up (docs/PROVIDERS.md).
FACE_EXPAND = 2.2
FACE_PX = 1024
FACE_DENOISE = 0.4
FACE_MAX_SHARE = 0.3
FACE_MIN_PX = 24


def face_square(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int] | None:
    """(left, top, side) of the square to redraw around a face box, or None to leave it."""
    x, y, w, h = box
    size = max(w, h)
    if size < FACE_MIN_PX or size > FACE_MAX_SHARE * min(width, height):
        return None
    side = int(round(min(size * FACE_EXPAND, width, height)))
    cx, cy = x + w / 2, y + h / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, side


def face_crop(data: bytes, square: tuple[int, int, int]) -> bytes:
    import io

    from PIL import Image

    left, top, side = square
    img = Image.open(io.BytesIO(data)).convert("RGB")
    crop = img.crop((left, top, left + side, top + side)).resize((FACE_PX, FACE_PX), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    return buf.getvalue()


def face_paste(data: bytes, patch: bytes, square: tuple[int, int, int]) -> bytes:
    import io

    from PIL import Image, ImageDraw, ImageFilter

    left, top, side = square
    img = Image.open(io.BytesIO(data)).convert("RGB")
    face = Image.open(io.BytesIO(patch)).convert("RGB").resize((side, side), Image.LANCZOS)
    mask = Image.new("L", (side, side), 0)
    pad = int(side * 0.12)
    ImageDraw.Draw(mask).ellipse((pad, pad, side - pad, side - pad), fill=255)
    img.paste(face, (left, top), mask.filter(ImageFilter.GaussianBlur(side * 0.06)))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def face_graph(*, image: str, lora: str, strength: float, who: str, seed: int) -> dict[str, Any]:
    return krea2.build_workflow(
        variant="turbo",
        unet=TURBO.filename,
        text_encoder=krea2.TEXT_ENCODER.filename,
        vae=krea2.VAE.filename,
        prompt=f"close-up photo of {who}, natural skin texture",
        negative_prompt="",
        width=FACE_PX,
        height=FACE_PX,
        seed=seed,
        steps=8,
        cfg=1.0,
        batch=1,
        loras=[(lora, strength)],
        prefix="ai-image-studio-advanced-face",
        image=image,
        denoise=FACE_DENOISE,
    )
