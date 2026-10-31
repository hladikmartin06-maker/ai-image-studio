"""ComfyUI workflow files: import (API vs UI format), secret scrubbing, which node inputs
the studio fills ("bindings"), and filling them for a run.

Only the API format ({node_id: {class_type, inputs}} — ComfyUI "Export (API)") can be sent
to POST /prompt. A UI-format file (with "nodes" and "links") is stored for export only:
turning it into the API format needs the frontend's widget logic, so the studio asks for
the API export instead of guessing.
"""

from __future__ import annotations

import copy
import re
from typing import Any

SAMPLERS = ("KSampler", "KSamplerAdvanced")
TEXT_NODES = ("CLIPTextEncode",)
LATENT_NODES = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo")
MODEL_LOADERS = ("UNETLoader", "CheckpointLoaderSimple")
LORA_NODES = ("LoraLoaderModelOnly", "LoraLoader")
SAVE_NODES = ("SaveImage",)
_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|auth)", re.IGNORECASE)


class WorkflowError(ValueError):
    pass


def kind_of(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return "ui"
    if (
        isinstance(data, dict)
        and data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    ):
        return "api"
    raise WorkflowError("This is not a ComfyUI workflow file.")


def scrub(data: Any) -> tuple[Any, list[str]]:
    """Removes values of inputs/widgets that look like credentials (API nodes keep keys in
    their inputs); returns the cleaned copy and what was removed."""
    removed: list[str] = []
    clean = copy.deepcopy(data)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if _SECRET_KEY.search(str(key)) and isinstance(value, str) and value:
                    node[key] = ""
                    removed.append(f"{path}{key}")
                else:
                    walk(value, f"{path}{key}.")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}{index}.")

    walk(clean, "")
    return clean, removed


def node_types(api: dict[str, Any]) -> list[str]:
    return sorted({str(n.get("class_type")) for n in api.values() if isinstance(n, dict)})


def _ref(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
        return str(value[0])
    return None


def detect_bindings(api: dict[str, Any]) -> dict[str, Any]:
    """Finds the nodes the studio can fill: prompt, negative, seed, steps, cfg, size,
    batch, model file, LoRA loader, save node."""
    bindings: dict[str, Any] = {}
    sampler_id = next((k for k, n in api.items() if n.get("class_type") in SAMPLERS), None)
    if sampler_id:
        sampler = api[sampler_id]
        inputs = sampler.get("inputs", {})
        bindings["sampler"] = sampler_id
        for name in ("positive", "negative"):
            source = _ref(inputs.get(name))
            if source and api.get(source, {}).get("class_type") in TEXT_NODES:
                bindings[name] = source
        latent = _ref(inputs.get("latent_image"))
        if latent and api.get(latent, {}).get("class_type") in LATENT_NODES:
            bindings["latent"] = latent
        seed_field = "noise_seed" if "noise_seed" in inputs else "seed"
        if seed_field in inputs and not _ref(inputs[seed_field]):
            bindings["seed_field"] = seed_field
    loader = next((k for k, n in api.items() if n.get("class_type") in MODEL_LOADERS), None)
    if loader:
        bindings["model_loader"] = loader
    lora = next((k for k, n in api.items() if n.get("class_type") in LORA_NODES), None)
    if lora:
        bindings["lora"] = lora
    save = [k for k, n in api.items() if n.get("class_type") in SAVE_NODES]
    if save:
        bindings["save"] = save
    return bindings


def missing_nodes(api: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    return [t for t in node_types(api) if t not in object_info]


def _next_id(api: dict[str, Any]) -> str:
    numbers = [int(k) for k in api if str(k).isdigit()]
    return str((max(numbers) if numbers else 0) + 1)


def apply(api: dict[str, Any], bindings: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """A copy of the workflow with the run's values filled into the bound inputs.

    params: prompt, negative_prompt, seed, steps, cfg, width, height, batch, unet,
    loras [(file, strength)]. A LoRA without a bound loader is inserted right after the
    model loader, so every node that used the model now uses the model with the LoRA.
    """
    wf = copy.deepcopy(api)
    if "sampler" not in bindings or "positive" not in bindings:
        raise WorkflowError("The workflow has no KSampler with a text prompt the studio can fill.")
    wf[bindings["positive"]]["inputs"]["text"] = params["prompt"]
    if bindings.get("negative") and params.get("negative_prompt") is not None:
        wf[bindings["negative"]]["inputs"]["text"] = params["negative_prompt"]
    sampler = wf[bindings["sampler"]]["inputs"]
    if bindings.get("seed_field") and params.get("seed") is not None:
        sampler[bindings["seed_field"]] = params["seed"]
    for key in ("steps", "cfg"):
        if params.get(key) is not None and key in sampler and not _ref(sampler[key]):
            sampler[key] = params[key]
    if bindings.get("latent"):
        latent = wf[bindings["latent"]]["inputs"]
        for key in ("width", "height"):
            if params.get(key) and not _ref(latent.get(key)):
                latent[key] = params[key]
        if params.get("batch") and "batch_size" in latent and not _ref(latent["batch_size"]):
            latent["batch_size"] = params["batch"]
    loader = bindings.get("model_loader")
    if loader and params.get("unet") and wf[loader]["class_type"] == "UNETLoader":
        wf[loader]["inputs"]["unet_name"] = params["unet"]
    loras: list[tuple[str, float]] = params.get("loras") or []
    if loras:
        bound = bindings.get("lora")
        if bound:
            first, strength = loras[0]
            node = wf[bound]
            node["inputs"]["lora_name"] = first
            node["inputs"]["strength_model"] = strength
            if node["class_type"] == "LoraLoader":
                node["inputs"].setdefault("strength_clip", strength)
            rest = loras[1:]
            source = bound
        else:
            if not loader:
                raise WorkflowError("No model loader found to attach the LoRA to.")
            rest = loras
            source = loader
        for lora_file, strength in rest:
            new_id = _next_id(wf)
            for node in wf.values():
                for key, value in node.get("inputs", {}).items():
                    if _ref(value) == source and value[1] == 0 and key == "model":
                        node["inputs"][key] = [new_id, 0]
            wf[new_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [source, 0],
                    "lora_name": lora_file,
                    "strength_model": strength,
                },
            }
            source = new_id
    return wf
