"""The ai-toolkit job file for one training run (JSON: ai-toolkit reads .json configs).

Layout inside the GPU workspace (/workspace/runs/<run>/):
  dataset/<n>.<ext> + <n>.txt   images and captions (ai-toolkit: jpg/jpeg/png + .txt)
  config.json                   this file
  start_lora.safetensors        when continuing: the earlier version the training starts from
  output/<run>/                 <run>.safetensors, <run>_<step>.safetensors checkpoints,
                                samples/*.jpg
"""

from __future__ import annotations

from typing import Any

from app.lora.presets import ARCHITECTURES, TrainingConfig

WORKSPACE = "/workspace"


def run_paths(run: str) -> dict[str, str]:
    """Relative (agent) and absolute (container) paths of a run."""
    rel = f"runs/{run}"
    return {
        "rel": rel,
        "dataset_rel": f"{rel}/dataset",
        "config_rel": f"{rel}/config.json",
        "start_rel": f"{rel}/start_lora.safetensors",
        "start_abs": f"{WORKSPACE}/{rel}/start_lora.safetensors",
        "output_rel": f"{rel}/output/{run}",
        "dataset_abs": f"{WORKSPACE}/{rel}/dataset",
        "output_abs": f"{WORKSPACE}/{rel}/output",
    }


def run_name(lora_id: str, number: int) -> str:
    return f"lora_{lora_id.replace('-', '')[:12]}_v{number}"


def build(
    *,
    dataset_dir: str,
    output_dir: str,
    architecture: str,
    name: str,
    trigger_word: str,
    class_word: str,
    config: TrainingConfig,
    sample_prompts: list[str],
    start_lora: str | None = None,
) -> dict[str, Any]:
    arch = ARCHITECTURES[architecture]
    variant = arch["variants"][config.base_variant]
    model: dict[str, Any] = {
        "name_or_path": variant["name_or_path"],
        "arch": architecture,
        "quantize": config.quantize,
        "quantize_te": config.quantize,
        "low_vram": config.low_vram,
    }
    if variant["assistant_lora_path"]:
        model["assistant_lora_path"] = variant["assistant_lora_path"]
    train: dict[str, Any] = {
        "batch_size": config.batch_size,
        "steps": config.steps,
        "gradient_accumulation": config.gradient_accumulation,
        "train_unet": True,
        "train_text_encoder": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "noise_scheduler": "flowmatch",
        "optimizer": config.optimizer,
        "lr": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "dtype": config.mixed_precision,
        "timestep_type": config.timestep_type,
    }
    if config.diff_output_preservation and class_word:
        train.update(
            diff_output_preservation=True,
            diff_output_preservation_multiplier=config.diff_output_preservation_multiplier,
            diff_output_preservation_class=class_word,
        )
    at_end = config.previews == "end"
    if not sample_prompts or (config.sample_every == 0 and not at_end):
        train["disable_sampling"] = True
    elif at_end:
        # ai-toolkit still samples once after the final step (BaseSDTrainProcess).
        train["skip_first_sample"] = True
    process: dict[str, Any] = {
        "type": "sd_trainer",
        "training_folder": output_dir,
        "device": "cuda:0",
        "trigger_word": trigger_word,
        "network": {
            "type": "lora",
            "linear": config.rank,
            "linear_alpha": config.alpha,
            # Continue training: ai-toolkit loads these weights when the run has no
            # checkpoint of its own yet (BaseSDTrainProcess.get_latest_save_path).
            **({"pretrained_lora_path": start_lora} if start_lora else {}),
        },
        "save": {
            "dtype": config.save_dtype,
            "save_every": config.save_every,
            "max_step_saves_to_keep": config.max_checkpoints,
        },
        "datasets": [
            {
                "folder_path": dataset_dir,
                "caption_ext": "txt",
                "caption_dropout_rate": config.caption_dropout_rate,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": config.resolutions,
            }
        ],
        "train": train,
        "model": model,
    }
    if not train.get("disable_sampling"):
        process["sample"] = {
            "sampler": "flowmatch",
            "sample_every": 0 if at_end else config.sample_every,
            "width": 1024,
            "height": 1024,
            "prompts": sample_prompts,
            "neg": "",
            "seed": config.seed,
            "walk_seed": False,
            "guidance_scale": variant["sample_guidance"],
            "sample_steps": variant["sample_steps"],
        }
    return {
        "job": "extension",
        "config": {"name": name, "process": [process]},
        # Saved into the LoRA's safetensors metadata by ai-toolkit.
        "meta": {
            "name": "[name]",
            "version": "1.0",
            "studio_architecture": architecture,
            "studio_base_model": variant["name_or_path"],
            "studio_base_variant": config.base_variant,
            "studio_trigger_word": trigger_word,
        },
    }


def default_sample_prompts(trigger_word: str, class_word: str) -> list[str]:
    subject = f"{trigger_word} {class_word}".strip()
    return [
        f"portrait photo of {subject}, soft window light, 85mm",
        f"{subject} walking in a city street at golden hour, full body",
        f"close-up of {subject} smiling, studio lighting, neutral background",
    ]
