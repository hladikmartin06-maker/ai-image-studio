"""The Krea 2 pose LoRA (the owner's plan, phase 2, approved 2026-11-06).

Sources (checked 2026-11-06):
- Model card huggingface.co/thedeoxen/Krea-2-pose-controlnet (revision 1da9762c, Apache 2.0):
  an OpenPose ControlNet-style LoRA for Krea 2 Turbo — "not a reference + pose fusion
  model": the pose map is image 1 of the Krea2 Ostris Edit text encoder; appearance, outfit,
  light and place come from the prompt. Weight 0.8–1.0; its example workflow
  (krea2_controlnet_pose.json): UNETLoader → Krea2OstrisEditModelPatch (kv_cache on) →
  LoraLoaderModelOnly (1.0) → KSampler (10 steps, CFG 1, euler, simple, denoise 1.0) on an
  empty latent of the pose map's size; positive and negative from TextEncodeKrea2OstrisEdit
  (clip, prompt, vae, image1 = the map; the negative with an empty prompt) through
  FluxKontextMultiReferenceLatentMethod "index_timestep_zero" (a built-in ComfyUI conditioning
  node, `comfy_extras/nodes_flux.py` — only its name says Flux; no Flux model is used).
- Nodes github.com/ostris/ComfyUI-Krea2-Ostris-Edit (MIT, commit 7756566160c4;
  comfy/custom_nodes.py): TextEncodeKrea2OstrisEdit (clip, prompt; optional vae, image1–3),
  Krea2OstrisEditModelPatch (model, kv_cache). No extra packages.
"""

from __future__ import annotations

from typing import Any

from app.comfy import anypaint
from app.comfy.anypaint import ModelFile as _HFFile

FAMILY = "pose"
REVISION = "1da9762ca46e6c9e4014b309bf0147889f10982a"


class ModelFile(_HFFile):
    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/thedeoxen/Krea-2-pose-controlnet/resolve/{REVISION}/"
            f"{self.filename}"
        )


LORA = ModelFile(
    "loras",
    "krea2_turbo_openpose_controlnet.safetensors",
    228_587_504,
    "0ddc3aafce4abdf7af3309b2f00c1bacdf15df1f2b4fb7adc9ff71795da90ecf",
)
FILES = (LORA,)
NODES = ("TextEncodeKrea2OstrisEdit", "Krea2OstrisEditModelPatch")
# The authors' values.
STRENGTH = 1.0
STEPS = 10
REFERENCE_METHOD = "index_timestep_zero"


def downloads() -> list[dict[str, str]]:
    return [
        {"url": f.url, "folder": f.folder, "filename": f.filename, "sha256": f.sha256}
        for f in FILES
    ]


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Whether this ComfyUI has the Ostris Edit nodes, the built-in reference node and the
    pose LoRA."""
    loras = anypaint._choices(object_info, "LoraLoaderModelOnly", "lora_name")
    files = [
        {
            "folder": f.folder,
            "filename": f.filename,
            "size": f.size,
            "installed": f.filename in loras,
        }
        for f in FILES
    ]
    needed = (*NODES, "FluxKontextMultiReferenceLatentMethod")
    missing_nodes = [n for n in needed if n not in object_info]
    missing_files = [f["filename"] for f in files if not f["installed"]]
    return {
        "files": files,
        "missing_files": missing_files,
        "missing_nodes": missing_nodes,
        "ready": not missing_files and not missing_nodes,
    }
