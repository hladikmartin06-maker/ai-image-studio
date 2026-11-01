"""Enhance photo: PGUpscaler, the workflow of the free RunPod template "PG - Free Upscaler"
(console.runpod.io/hub/template/okugktg6fs), whose files are in the public Hugging Face
repository AIPRIVATEGROUP/pgupscaler. It fixes plastic-looking AI skin and then upscales and
sharpens the photo:

  Z-Image Turbo (img2img, 5 steps, denoise = strength) on the photo scaled to 2 MP — on the
  whole photo, or only on the face that SAM 3.1 finds (soft mask, pasted back) — then
  SeedVR2 7B (1 step) upscales it by 1.5×, 2× or 3× with the original's colours (lab).

Verified 2026-11-01 from the repository: workflows/PGUpscaler - EN.json (nodes, links and
widget values), README.md (settings) and setup.sh (models/<folder>/ go to ComfyUI/models).
The workflow uses only ComfyUI's own nodes (SAM3_Detect, SeedVR2*, ModelSamplingAuraFlow, …;
present since ComfyUI 0.30.0); the template's custom nodes (Pixaroma, rgthree) only draw
its controls and previews, so the studio builds the same graph in API format and needs no
custom node. Not run on a real GPU here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.comfy.krea2 import _choices

FAMILY = "pgupscaler"
NAME = "PGUpscaler"
MODEL_KEY = "pgupscaler"
REPO = "AIPRIVATEGROUP/pgupscaler"
REPO_URL = f"https://huggingface.co/{REPO}"
TEMPLATE_URL = "https://console.runpod.io/hub/template/okugktg6fs"
INPUT_PLACEHOLDER = "__studio_input__.png"


@dataclass(frozen=True)
class ModelFile:
    folder: str
    filename: str
    size: int  # bytes, from the repository's file list

    @property
    def url(self) -> str:
        return f"{REPO_URL}/resolve/main/models/{self.folder}/{self.filename}"


Z_IMAGE = ModelFile("diffusion_models", "z_image_turbo_int8_convrot.safetensors", 6_201_001_296)
Z_TEXT_ENCODER = ModelFile("text_encoders", "qwen_3_4b_fp8_mixed.safetensors", 5_631_994_051)
Z_VAE = ModelFile("vae", "ae.safetensors", 335_304_388)
SAM3 = ModelFile("checkpoints", "sam3.1_multiplex_fp16.safetensors", 1_745_546_848)
SEEDVR2 = ModelFile("diffusion_models", "seedvr2_7b_int8_convrot.safetensors", 8_334_897_976)
SEEDVR2_VAE = ModelFile("vae", "seedvr2_ema_vae_fp16.safetensors", 501_324_814)
FILES = (Z_IMAGE, Z_TEXT_ENCODER, Z_VAE, SAM3, SEEDVR2, SEEDVR2_VAE)
TOTAL_BYTES = sum(f.size for f in FILES)

REQUIRED_NODES = (
    "LoadImage",
    "ImageScaleToTotalPixels",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "ModelSamplingAuraFlow",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "VAEEncode",
    "KSampler",
    "VAEDecode",
    "CheckpointLoaderSimple",
    "SAM3_Detect",
    "GrowMask",
    "MaskToImage",
    "ImageBlur",
    "ImageToMask",
    "ImageCompositeMasked",
    "ImageScaleBy",
    "SeedVR2Preprocess",
    "VAEEncodeTiled",
    "SeedVR2Conditioning",
    "VAEDecodeTiled",
    "SeedVR2PostProcessing",
    "SaveImage",
)

# The template's settings (README): strength 0.2 subtle · 0.3 default · 0.4 strong;
# upscale 1.5 · 2 default · 3 (very slow, 48 GB VRAM recommended).
STRENGTHS = (0.2, 0.3, 0.4)
UPSCALES = (1.5, 2.0, 3.0)
DEFAULT_STRENGTH = 0.3
DEFAULT_UPSCALE = 2.0
MIN_VRAM_GB = 24
MIN_VRAM_GB_3X = 48
SKIN_PROMPT = (
    "Photorealistic photo with natural, realistic skin: visible pores, fine skin texture, "
    "subtle natural imperfections and an even, natural skin tone, no airbrushing, no beauty "
    "filter. Sharp focus, natural lighting, RAW photo quality."
)


def downloads() -> list[dict[str, str]]:
    """The model files for the GPU agent (huggingface.co only)."""
    return [{"url": f.url, "folder": f.folder, "filename": f.filename} for f in FILES]


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Which PGUpscaler files and nodes this ComfyUI has."""
    listed = {
        "diffusion_models": _choices(object_info, "UNETLoader", "unet_name"),
        "text_encoders": _choices(object_info, "CLIPLoader", "clip_name"),
        "vae": _choices(object_info, "VAELoader", "vae_name"),
        "checkpoints": _choices(object_info, "CheckpointLoaderSimple", "ckpt_name"),
    }
    files = [
        {
            "folder": f.folder,
            "filename": f.filename,
            "size": f.size,
            "installed": f.filename in listed[f.folder],
        }
        for f in FILES
    ]
    missing_nodes = [n for n in REQUIRED_NODES if n not in object_info]
    missing_files = [f["filename"] for f in files if not f["installed"]]
    return {
        "files": files,
        "missing_files": missing_files,
        "missing_nodes": missing_nodes,
        "ready": not missing_files and not missing_nodes,
    }


def catalog_json() -> dict[str, Any]:
    return {
        "name": NAME,
        "model_key": MODEL_KEY,
        "repo": REPO_URL,
        "template": TEMPLATE_URL,
        "files": [{"folder": f.folder, "filename": f.filename, "size": f.size} for f in FILES],
        "total_bytes": TOTAL_BYTES,
        "strengths": list(STRENGTHS),
        "upscales": list(UPSCALES),
        "default_strength": DEFAULT_STRENGTH,
        "default_upscale": DEFAULT_UPSCALE,
        "min_vram_gb": MIN_VRAM_GB,
        "min_vram_gb_3x": MIN_VRAM_GB_3X,
    }


def build_workflow(
    *, image: str, strength: float, face_only: bool, upscale: float, seed: int
) -> dict[str, Any]:
    """The PGUpscaler graph in ComfyUI's API format. Node ids are the template's own. With
    face_only the skin fix is pasted back only where SAM 3.1 finds the face; without it the
    face branch is left out (the template switches between both with ComfySwitchNode)."""
    wf: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "20": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "megapixels": 2.0,
                "resolution_steps": 16,
            },
        },
        "10": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": Z_IMAGE.filename, "weight_dtype": "default"},
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": Z_TEXT_ENCODER.filename,
                "type": "lumina2",
                "device": "default",
            },
        },
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": Z_VAE.filename}},
        "13": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["10", 0], "shift": 3}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": SKIN_PROMPT}},
        "21": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["12", 0]}},
        "24": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["13", 0],
                "positive": ["6", 0],
                "negative": ["21", 0],
                "latent_image": ["22", 0],
                "seed": seed,
                "steps": 5,
                "cfg": 1,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "beta",
                "denoise": strength,
            },
        },
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["12", 0]}},
    }
    fixed = ["25", 0]
    if face_only:
        wf.update(
            {
                "30": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": SAM3.filename},
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["30", 1], "text": "face"},
                },
                "31": {
                    "class_type": "SAM3_Detect",
                    "inputs": {
                        "model": ["30", 0],
                        "image": ["20", 0],
                        "conditioning": ["7", 0],
                        "threshold": 0.5,
                        "refine_iterations": 2,
                        "individual_masks": False,
                    },
                },
                "32": {
                    "class_type": "GrowMask",
                    "inputs": {"mask": ["31", 0], "expand": 16, "tapered_corners": True},
                },
                "33": {"class_type": "MaskToImage", "inputs": {"mask": ["32", 0]}},
                "34": {
                    "class_type": "ImageBlur",
                    "inputs": {"image": ["33", 0], "blur_radius": 31, "sigma": 10.0},
                },
                "35": {
                    "class_type": "ImageToMask",
                    "inputs": {"image": ["34", 0], "channel": "red"},
                },
                "36": {
                    "class_type": "ImageCompositeMasked",
                    "inputs": {
                        "destination": ["20", 0],
                        "source": ["25", 0],
                        "mask": ["35", 0],
                        "x": 0,
                        "y": 0,
                        "resize_source": False,
                    },
                },
            }
        )
        fixed = ["36", 0]
    tiles = {"tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}
    wf.update(
        {
            "5": {
                "class_type": "ImageScaleBy",
                "inputs": {"image": fixed, "upscale_method": "lanczos", "scale_by": upscale},
            },
            "40": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": SEEDVR2.filename, "weight_dtype": "default"},
            },
            "41": {"class_type": "VAELoader", "inputs": {"vae_name": SEEDVR2_VAE.filename}},
            "42": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["5", 0]}},
            "43": {
                "class_type": "VAEEncodeTiled",
                "inputs": {"pixels": ["42", 0], "vae": ["41", 0], **tiles},
            },
            "44": {
                "class_type": "SeedVR2Conditioning",
                "inputs": {"model": ["40", 0], "vae_conditioning": ["43", 0]},
            },
            "45": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["40", 0],
                    "positive": ["44", 0],
                    "negative": ["44", 1],
                    "latent_image": ["43", 0],
                    "seed": seed,
                    "steps": 1,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                },
            },
            "46": {
                "class_type": "VAEDecodeTiled",
                "inputs": {"samples": ["45", 0], "vae": ["41", 0], **tiles},
            },
            "47": {
                "class_type": "SeedVR2PostProcessing",
                "inputs": {
                    "images": ["46", 0],
                    "original_resized_images": ["5", 0],
                    "color_correction_method": "lab",
                },
            },
            "50": {
                "class_type": "SaveImage",
                "inputs": {"images": ["47", 0], "filename_prefix": "ai-image-studio/enhanced"},
            },
        }
    )
    return wf


def set_input(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    """The workflow with the uploaded photo's name in its LoadImage node."""
    out = {k: {**v, "inputs": dict(v["inputs"])} for k, v in workflow.items()}
    for node in out.values():
        if node["class_type"] == "LoadImage":
            node["inputs"]["image"] = name
    return out
