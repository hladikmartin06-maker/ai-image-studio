"""JSON shapes for LoRAs, dataset items and versions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.gpu.service import run_cost
from app.models import Lora, LoraDatasetItem, LoraVersion
from app.services.serializers import iso

ACTIVE = ("queued", "preparing", "uploading", "running", "downloading")


def item_json(item: LoraDatasetItem) -> dict[str, Any]:
    ref = item.reference
    return {
        "id": str(item.id),
        "reference_id": str(item.reference_image_id),
        "url": f"/api/reference-images/{item.reference_image_id}/file",
        "thumb_url": f"/api/reference-images/{item.reference_image_id}/thumb",
        "original_name": ref.original_name if ref else "",
        "mime_type": ref.mime_type if ref else None,
        "width": ref.width if ref else None,
        "height": ref.height if ref else None,
        "size_bytes": ref.size_bytes if ref else None,
        "caption": item.caption,
        "caption_source": item.caption_source,
        "split": item.split,
        "excluded": item.excluded,
        "crop": item.crop,
        "position": item.position,
    }


def version_json(v: LoraVersion) -> dict[str, Any]:
    progress = dict(v.progress or {})
    end = v.gpu_stopped_at or (datetime.now(UTC) if v.gpu_started_at else None)
    cost = run_cost(v.cost_per_hour, v.gpu_started_at, end) if end else None
    base = f"/api/loras/{v.lora_id}/versions/{v.id}"
    return {
        "id": str(v.id),
        "lora_id": str(v.lora_id),
        "number": v.number,
        "status": v.status,
        "stage_detail": v.stage_detail,
        "preset": v.preset,
        "config": v.config,
        "trainer_config": v.trainer_config,
        "gpu_provider": v.gpu_provider,
        "gpu_type": v.gpu_type,
        "gpu_count": v.gpu_count,
        "workload_id": v.workload_id,
        "auto_cleanup": v.auto_cleanup,
        "cost_per_hour": v.cost_per_hour,
        "cost_estimate": cost,
        "cost_final": cost if v.gpu_stopped_at else None,
        "step": progress.get("step"),
        "total_steps": progress.get("total") or (v.config or {}).get("steps"),
        "loss": progress.get("loss", []),
        # Out-of-memory restarts on the same GPU, each with what was changed.
        "memory_retries": progress.get("memory_retries", []),
        "gpu": progress.get("gpu"),
        "samples": [
            {**s, "url": f"{base}/samples/{s['file']}"} for s in progress.get("samples", [])
        ],
        # Face likeness of the validation images per step (GET …/likeness), when measured.
        "likeness": progress.get("likeness"),
        "file_url": f"{base}/file" if v.file_path else None,
        "file_size": v.file_size,
        "file_sha256": v.file_sha256,
        "checkpoints": [
            {"step": c["step"], "size": c.get("size"), "url": f"{base}/file?step={c['step']}"}
            for c in v.checkpoints or []
        ],
        "deployments": v.deployments or {},
        "dataset_size": len((v.dataset_manifest or {}).get("items", [])),
        "source": (v.dataset_manifest or {}).get("source", "trained"),
        "notes": v.notes,
        "error_code": v.error_code,
        "error_message": v.error_message,
        "created_at": iso(v.created_at),
        "started_at": iso(v.started_at),
        "completed_at": iso(v.completed_at),
        "gpu_started_at": iso(v.gpu_started_at),
        "gpu_stopped_at": iso(v.gpu_stopped_at),
    }


def lora_json(
    lora: Lora,
    versions: list[LoraVersion],
    *,
    persona_name: str | None = None,
    dataset_count: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(versions, key=lambda v: v.number, reverse=True)
    preferred = next((v for v in ordered if v.id == lora.preferred_version_id), None)
    usable = [v for v in ordered if v.status == "completed" and v.file_path]
    return {
        "id": str(lora.id),
        "name": lora.name,
        "trigger_word": lora.trigger_word,
        "class_word": lora.class_word,
        "architecture": lora.architecture,
        "base_variant": lora.base_variant,
        "source": lora.source,
        "persona_id": str(lora.persona_id) if lora.persona_id else None,
        "persona_name": persona_name,
        "caption_template": lora.caption_template,
        "tags": lora.tags or [],
        "notes": lora.notes,
        "preferred_version_id": str(preferred.id) if preferred else None,
        "preferred_version": version_json(preferred) if preferred else None,
        "latest_version": version_json(ordered[0]) if ordered else None,
        "usable_versions": [
            {"id": str(v.id), "number": v.number, "preferred": v.id == lora.preferred_version_id}
            for v in usable
        ],
        "active": any(v.status in ACTIVE for v in ordered),
        "version_count": len(ordered),
        "dataset_count": dataset_count,
        "created_at": iso(lora.created_at),
        "updated_at": iso(lora.updated_at),
    }
