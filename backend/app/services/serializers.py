"""JSON shapes shared by the routes and the live event stream."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.catalog import MODELS, PROVIDERS, get_model, resolve_variant, size_label
from app.models import Generation, Image, ProviderTask, ReferenceImage


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def model_name(key: str) -> str:
    model = MODELS.get(key)
    return model.name if model else key


def provider_name(key: str) -> str:
    return PROVIDERS.get(key, {}).get("name", key)


def image_json(image: Image, generation: Generation | None = None) -> dict[str, Any]:
    gen = generation or image.generation
    base = f"/api/images/{image.id}"
    return {
        "id": str(image.id),
        "generation_id": str(image.generation_id),
        "index": image.index,
        "url": f"{base}/file",
        "thumb_url": f"{base}/thumb",
        "download_url": f"{base}/download",
        "width": image.width,
        "height": image.height,
        "mime_type": image.mime_type,
        "size_bytes": image.size_bytes,
        "nsfw": image.nsfw,
        "in_library": image.in_library,
        "is_favorite": image.is_favorite,
        "favorited_at": iso(image.favorited_at),
        "created_at": iso(image.created_at),
        "model_key": gen.model_key,
        "model_name": model_name(gen.model_key),
        "provider": gen.provider,
        "provider_name": provider_name(gen.provider),
        "mode": gen.mode,
        "prompt": gen.prompt,
        "settings": gen.settings,
        "seed": gen.settings.get("seed"),
        "status": gen.status,
    }


def task_json(task: ProviderTask) -> dict[str, Any]:
    return {
        "index": task.index,
        "status": task.status,
        "images_requested": task.images_requested,
        "provider_task_id": task.provider_task_id,
        "error_code": task.error_code,
        "error_message": task.error_message,
    }


def generation_json(gen: Generation, *, images: list[Image] | None = None) -> dict[str, Any]:
    if images is not None:
        image_rows = images
    else:
        # Relationships are only read when they were loaded (no lazy I/O in async code).
        image_rows = list(gen.images) if "images" in gen.__dict__ else []
    target = None
    try:
        variant = resolve_variant(gen.model_key, gen.provider, gen.mode == "image-edit")
        target = size_label(variant, gen.settings)
    except ValueError:
        pass
    try:
        name = get_model(gen.model_key).name
    except ValueError:
        name = gen.model_key
    return {
        "id": str(gen.id),
        "model_key": gen.model_key,
        "model_name": name,
        "provider": gen.provider,
        "provider_name": provider_name(gen.provider),
        "mode": gen.mode,
        "provider_model_id": gen.provider_model_id,
        "prompt": gen.prompt,
        "settings": gen.settings,
        "target_size": target,
        "request_payload": gen.request_payload,
        "reference_image_ids": gen.reference_image_ids,
        "reference_images": [
            {
                "id": ref_id,
                "url": f"/api/reference-images/{ref_id}/file",
                "thumb_url": f"/api/reference-images/{ref_id}/thumb",
            }
            for ref_id in gen.reference_image_ids
        ],
        "image_count": gen.image_count,
        "status": gen.status,
        "error_code": gen.error_code,
        "error_message": gen.error_message,
        "source_generation_id": str(gen.source_generation_id) if gen.source_generation_id else None,
        "created_at": iso(gen.created_at),
        "started_at": iso(gen.started_at),
        "completed_at": iso(gen.completed_at),
        "duration_seconds": (gen.completed_at - gen.started_at).total_seconds()
        if gen.completed_at and gen.started_at
        else None,
        "tasks": [task_json(t) for t in gen.tasks] if "tasks" in gen.__dict__ else [],
        "images": [image_json(i, gen) for i in image_rows],
    }


def reference_json(ref: ReferenceImage) -> dict[str, Any]:
    return {
        "id": str(ref.id),
        "url": f"/api/reference-images/{ref.id}/file",
        "thumb_url": f"/api/reference-images/{ref.id}/thumb",
        "original_name": ref.original_name,
        "mime_type": ref.mime_type,
        "width": ref.width,
        "height": ref.height,
        "size_bytes": ref.size_bytes,
        "created_at": iso(ref.created_at),
    }
