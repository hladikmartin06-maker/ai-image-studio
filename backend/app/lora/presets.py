"""Training settings per architecture: what the trainer supports, limits and presets.

Trainer: ostris/ai-toolkit (verified for Krea 2: arch "krea2", RAW base repo
krea/Krea-2-Raw — gated on Hugging Face —, Turbo krea/Krea-2-Turbo with the training adapter
ostris/krea2_turbo_training_adapter). Options exposed are ai-toolkit config fields
(toolkit/config_modules.py); what ai-toolkit cannot do for Krea 2 is listed as unavailable.
The presets are the studio's starting points, not values published by Krea or ai-toolkit.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "krea2": {
        "key": "krea2",
        "name": "Krea 2",
        "trainer": "ai-toolkit",
        "trainer_image": "ostris/aitoolkit:latest",
        "variants": {
            "raw": {
                "name": "Krea 2 RAW",
                "name_or_path": "krea/Krea-2-Raw",
                "gated": True,
                "assistant_lora_path": None,
                "usable_on": ["raw", "turbo"],
                "note": "Recommended by Krea for training; the LoRA also works on Turbo.",
                "sample_steps": 30,
                "sample_guidance": 3.5,
            },
            "turbo": {
                "name": "Krea 2 Turbo (training adapter)",
                "name_or_path": "krea/Krea-2-Turbo",
                "gated": True,
                "assistant_lora_path": (
                    "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1"
                    ".safetensors"
                ),
                "usable_on": ["turbo"],
                "note": "Trains on the distilled model through ai-toolkit's adapter; use the "
                "LoRA with Turbo only.",
                "sample_steps": 8,
                "sample_guidance": 1.0,
            },
        },
        "resolutions": [512, 768, 1024],
        "optimizers": ["adamw8bit", "adamw", "adafactor", "prodigy"],
        "schedulers": ["constant", "cosine", "linear", "constant_with_warmup"],
        "timestep_types": ["sigmoid", "linear"],
        "unavailable": {
            "text_encoder_training": "ai-toolkit trains only the Krea 2 transformer.",
            "validation_loss": "ai-toolkit reports training loss only; held-out images are "
            "kept out of training for your own visual comparison.",
            "regularization_images": "Not offered in the studio; differential output "
            "preservation protects the base model instead.",
            "fixed_training_seed": "ai-toolkit's config has no training seed; the seed "
            "applies to the validation images.",
        },
        # Estimates for choosing a GPU (Krea publishes no training VRAM figures). Without
        # gradient checkpointing all activations stay in memory: an 80 GB GPU ran out of
        # memory in the owner's run (2026-11-02), so the studio asks for more than that (still
        # an estimate). If it runs out anyway, training starts again with memory saving on.
        "vram_quantized_gb": 24,
        "vram_full_gb": 48,
        "vram_no_checkpointing_gb": 96,
        # From this much GPU memory the model trains in full precision (no 8-bit
        # quantization, no low-VRAM mode): the owner's rule for 80 GB GPUs (2026-11-03).
        "full_precision_gb": 80,
        "bucketing": "ai-toolkit buckets images by aspect ratio automatically.",
    }
}


# Stored next to the training settings of a version (routes/loras.py), not settings.
RUN_FIELDS = ("image", "cloud_type", "continue_from_number")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_variant: Literal["raw", "turbo"] = "raw"
    rank: int = Field(default=32, ge=4, le=128)
    alpha: int = Field(default=32, ge=1, le=128)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-3)
    optimizer: Literal["adamw8bit", "adamw", "adafactor", "prodigy"] = "adamw8bit"
    lr_scheduler: Literal["constant", "cosine", "linear", "constant_with_warmup"] = "constant"
    steps: int = Field(default=2500, ge=100, le=20000)
    batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation: int = Field(default=1, ge=1, le=16)
    gradient_checkpointing: bool = True
    mixed_precision: Literal["bf16", "fp16"] = "bf16"
    resolutions: list[int] = Field(default_factory=lambda: [768, 1024])
    quantize: bool = True
    low_vram: bool = False
    timestep_type: Literal["sigmoid", "linear"] = "sigmoid"
    caption_dropout_rate: float = Field(default=0.05, ge=0.0, le=0.5)
    diff_output_preservation: bool = True
    diff_output_preservation_multiplier: float = Field(default=1.0, ge=0.1, le=3.0)
    save_every: int = Field(default=250, ge=50, le=5000)
    max_checkpoints: int = Field(default=6, ge=1, le=12)
    sample_every: int = Field(default=250, ge=0, le=5000)
    # "end": no in-between validation images, only the ones after the last step
    # (sample_every 0 never matches a step). No run makes ai-toolkit's "baseline" images of
    # the untrained model (skip_first_sample). The images do not change the LoRA; they cost
    # time while the GPU is billed.
    previews: Literal["during", "end"] = "during"
    # Continue an earlier version: training starts from that version's LoRA weights
    # (ai-toolkit network.pretrained_lora_path) instead of from zero, on the current photos.
    # Its size (rank/alpha) is kept, so the weights fit.
    continue_from: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    validation_prompts: list[str] = Field(default_factory=list, max_length=6)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    save_dtype: Literal["float16", "bf16"] = "float16"

    @model_validator(mode="before")
    @classmethod
    def _without_run_fields(cls, data: Any) -> Any:
        # A version's saved settings also hold what belongs to that run only (trainer image,
        # cloud, the version it continued). Copying them into a new run must not fail;
        # anything else unknown is still refused.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in RUN_FIELDS}
        return data

    @field_validator("resolutions")
    @classmethod
    def _resolutions(cls, value: list[int]) -> list[int]:
        allowed = set(ARCHITECTURES["krea2"]["resolutions"])
        clean = sorted({v for v in value if v in allowed})
        if not clean:
            raise ValueError("Choose at least one training resolution")
        return clean

    @field_validator("validation_prompts")
    @classmethod
    def _prompts(cls, value: list[str]) -> list[str]:
        out = [p.strip()[:500] for p in value if p.strip()]
        return out


_BASE = TrainingConfig().model_dump()

PRESETS: dict[str, dict[str, Any]] = {
    "character_identity": {
        "name": "Character Identity",
        "summary": "Keeps a recognizable face and look. Output preservation keeps the base "
        "model's other abilities.",
        "tradeoff": "Balanced time and cost; the default for a new character.",
        "config": {**_BASE},
    },
    "photorealistic": {
        "name": "Photorealistic Character",
        "summary": "Trains at 1024 px only with a gentler learning rate and cosine decay for "
        "photographic detail.",
        "tradeoff": "About 20% more steps than Character Identity, so longer and costlier.",
        "config": {
            **_BASE,
            "alpha": 16,
            "learning_rate": 8e-5,
            "lr_scheduler": "cosine",
            "steps": 3000,
            "resolutions": [1024],
        },
    },
    "fast_experiment": {
        "name": "Fast Experiment",
        "summary": "Short, low-rank run at 512–768 px to check a dataset and captions.",
        "tradeoff": "About a third of the steps: cheapest, weakest identity.",
        "config": {
            **_BASE,
            "rank": 16,
            "alpha": 16,
            "learning_rate": 2e-4,
            "steps": 800,
            "resolutions": [512, 768],
            "diff_output_preservation": False,
            "save_every": 200,
            "max_checkpoints": 4,
            "sample_every": 200,
        },
    },
    "high_quality": {
        "name": "High Quality",
        "summary": "Higher rank, lower learning rate, more steps and gradient accumulation.",
        "tradeoff": "Longest and most expensive; watch the samples for overtraining.",
        "config": {
            **_BASE,
            "rank": 64,
            "alpha": 32,
            "learning_rate": 6e-5,
            "lr_scheduler": "cosine",
            "steps": 4000,
            "gradient_accumulation": 2,
            "save_every": 250,
            "max_checkpoints": 8,
            "sample_every": 500,
        },
    },
}


def preset_config(preset: str, base_variant: str) -> dict[str, Any]:
    config = dict(PRESETS[preset]["config"])
    config["base_variant"] = base_variant
    return TrainingConfig.model_validate(config).model_dump()


def catalog() -> dict[str, Any]:
    return {
        "architectures": list(ARCHITECTURES.values()),
        "presets": [
            {"key": k, **{f: v[f] for f in ("name", "summary", "tradeoff")}, "config": v["config"]}
            for k, v in PRESETS.items()
        ],
        "defaults": _BASE,
    }
