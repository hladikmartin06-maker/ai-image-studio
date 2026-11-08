"""Runs a ComfyUI generation inside the job worker.

One prompt per generation (batch_size = image count). The prompt id is chosen by the
studio and saved before sending, so after a restart a run that may have been accepted is
looked up in ComfyUI's queue/history instead of being sent twice. Progress comes from
GET /queue (position) and, for direct connections, the /ws "progress" events.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from app.comfy import enhance, k2edit
from app.comfy.client import (
    ComfyClient,
    ComfyError,
    history_error,
    history_outputs,
    queue_position,
)
from app.comfy.service import client_for, file_capability, place_lora
from app.config import get_settings
from app.db import SessionLocal
from app.models import ComfyConnection, Generation, Lora, LoraVersion
from app.services import storage
from app.services.events import broker
from app.services.serializers import generation_json

log = logging.getLogger("studio.comfy")

# Tests make polling instant.
POLL_INITIAL = 1.0
POLL_MAX = 3.0
ABSENT_LIMIT = 6


class RunFailed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


async def _publish(gen_id: uuid.UUID) -> None:
    from app.services.worker import publish

    await publish(gen_id)


async def _connection(gen: Generation) -> ComfyConnection:
    conn_id = (gen.comfy_meta or {}).get("connection_id")
    async with SessionLocal() as db:
        conn = (
            await db.scalar(
                select(ComfyConnection).where(
                    ComfyConnection.id == uuid.UUID(conn_id), ComfyConnection.user_id == gen.user_id
                )
            )
            if conn_id
            else None
        )
    if conn is None:
        raise RunFailed("not_found", "The ComfyUI connection of this run was removed.")
    return conn


async def _ensure_lora(gen: Generation, conn: ComfyConnection) -> None:
    lora_meta = (gen.comfy_meta or {}).get("lora")
    if not lora_meta or file_capability(conn) == "none":
        return
    async with SessionLocal() as db:
        version = await db.scalar(
            select(LoraVersion).where(
                LoraVersion.id == uuid.UUID(lora_meta["version_id"]),
                LoraVersion.user_id == gen.user_id,
            )
        )
        lora = await db.get(Lora, version.lora_id) if version else None
        if version is None or lora is None:
            raise RunFailed("not_found", "The LoRA of this run was deleted.")
        try:
            name = await place_lora(db, conn, lora, version)
        except ComfyError as exc:
            raise RunFailed("lora_upload_failed", exc.message) from exc
    if name != lora_meta.get("file"):
        raise RunFailed("lora_upload_failed", "The LoRA file name changed; start the run again.")


async def _with_input(client: ComfyClient, gen: Generation) -> dict[str, Any]:
    """Enhance photo: uploads the user's photo to ComfyUI's input folder (POST /upload/image,
    through the agent on a rented GPU) and puts its name into the LoadImage node."""
    source = (gen.comfy_meta or {}).get("input")
    if not source:
        return gen.request_payload
    try:
        data = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
    except OSError as exc:
        raise RunFailed("not_found", "The photo to enhance is no longer on this Mac.") from exc
    ext = {"image/png": "png", "image/webp": "webp"}.get(source.get("mime_type"), "jpg")
    mime = source.get("mime_type") or "image/jpeg"
    meta = gen.comfy_meta or {}
    square = meta.get("face_only") or meta.get("head_crop")
    if square:
        # Advanced, face only: just the face, cropped on this Mac (k2edit.face_crop).
        data = await asyncio.to_thread(k2edit.face_crop, data, tuple(square))
        ext, mime = "png", "image/png"
    if meta.get("family") == k2edit.FAMILY:
        # Advanced scales the photo to the edit's size anyway: send it at that size (a 4 MP
        # PNG timed out on its way to a rented GPU).
        settings = gen.settings or {}
        longest = max(int(settings.get("width") or 0), int(settings.get("height") or 0))
        data = await asyncio.to_thread(k2edit.for_upload, data, longest or k2edit.FACE_PX)
        ext, mime = "jpg", "image/jpeg"
    name = await client.upload_image(data, f"studio-{gen.id.hex[:12]}.{ext}", mime)
    return enhance.set_input(gen.request_payload, name)


async def _known(client: ComfyClient, prompt_id: str) -> bool:
    with contextlib.suppress(ComfyError):
        if await client.history(prompt_id):
            return True
        state, _ = queue_position(await client.queue(), prompt_id)
        return state != "absent"
    return False


async def _set_progress(task_id: uuid.UUID, gen_id: uuid.UUID, progress: dict[str, Any]) -> None:
    from app.models import ProviderTask

    async with SessionLocal() as db:
        await db.execute(
            update(ProviderTask).where(ProviderTask.id == task_id).values(progress=progress)
        )
        await db.commit()
    await _publish(gen_id)


async def _listen(conn: ComfyConnection, client: ComfyClient, gen: Generation, state: dict) -> None:
    """Sampler progress from ComfyUI's websocket (direct connections only)."""
    if conn.auth_type == "agent":
        return
    from websockets.asyncio.client import connect

    url = client.base_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{url}/ws?clientId={gen.id}"
    try:
        async with connect(url, additional_headers=client.headers, open_timeout=10) as ws:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                with contextlib.suppress(ValueError, TypeError):
                    message = json.loads(raw)
                    if message.get("type") == "progress":
                        data = message.get("data") or {}
                        if data.get("prompt_id") in (None, state.get("prompt_id")):
                            state["value"] = data.get("value")
                            state["max"] = data.get("max")
    except Exception:  # noqa: BLE001 - progress is optional
        return


async def _cancel_requested(gen_id: uuid.UUID) -> bool:
    async with SessionLocal() as db:
        meta = await db.scalar(select(Generation.comfy_meta).where(Generation.id == gen_id))
    return bool((meta or {}).get("cancel_requested"))


async def _poll(client: ComfyClient, gen: Generation, task: Any, prompt_id: str) -> dict:
    deadline = (task.created_at or datetime.now(UTC)) + timedelta(
        seconds=get_settings().comfy_timeout_seconds
    )
    delay = POLL_INITIAL
    absent = 0
    failures = 0
    live: dict[str, Any] = {"prompt_id": prompt_id}
    last: dict[str, Any] = {}
    last_write = 0.0
    while True:
        if await _cancel_requested(gen.id):
            with contextlib.suppress(ComfyError):
                await client.delete_queued(prompt_id)
                await client.interrupt(prompt_id)
            raise RunFailed("cancelled", "Cancelled.")
        try:
            entry = await client.history(prompt_id)
            failures = 0
        except ComfyError as exc:
            if exc.code == "unauthorized":
                raise RunFailed(exc.code, exc.message) from exc
            failures += 1
            if failures > 20:
                raise RunFailed(exc.code, exc.message) from exc
            entry = None
        if entry:
            error = history_error(entry)
            if error:
                raise RunFailed("generation_failed", error)
            status = entry.get("status") or {}
            if status.get("completed", True) and entry.get("outputs") is not None:
                return entry
        else:
            try:
                state, position = queue_position(await client.queue(), prompt_id)
            except ComfyError:
                state, position = "unknown", None
            if state == "absent":
                absent += 1
                if absent >= ABSENT_LIMIT:
                    raise RunFailed(
                        "lost", "ComfyUI no longer knows this job (it may have restarted)."
                    )
            else:
                absent = 0
            progress = {"state": state, "queue_position": position}
            if live.get("max"):
                progress.update({"value": live.get("value"), "max": live.get("max")})
            now = time.monotonic()
            if progress != last and now - last_write > 0.8:
                last, last_write = progress, now
                await _set_progress(task.id, gen.id, progress)
        if datetime.now(UTC) > deadline:
            with contextlib.suppress(ComfyError):
                await client.interrupt(prompt_id)
            raise RunFailed("timeout", "ComfyUI did not finish in time.")
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, POLL_MAX)


async def run(worker: Any, gen: Generation) -> None:
    task = gen.tasks[0] if gen.tasks else None
    if task is None or task.status in ("completed", "failed"):
        return
    listener: asyncio.Task | None = None
    try:
        conn = await _connection(gen)
        client = client_for(conn)
        prompt_id = task.provider_task_id
        if task.status == "submitting":
            # The app stopped while sending: resume only if ComfyUI has the job.
            if not prompt_id or not await _known(client, prompt_id):
                raise RunFailed(
                    "interrupted",
                    "The app stopped while this run was being sent; it was not sent again.",
                )
            await worker.set_task(task.id, status="processing")
        elif task.status == "pending":
            await _ensure_lora(gen, conn)
            workflow = await _with_input(client, gen)
            prompt_id = str(uuid.uuid4())
            await worker.set_task(task.id, status="submitting", provider_task_id=prompt_id)
            try:
                returned = await client.submit(workflow, prompt_id, str(gen.id))
            except ComfyError as exc:
                if exc.code == "unreachable" and await _known(client, prompt_id):
                    returned = prompt_id
                else:
                    raise
            await worker.set_task(task.id, status="processing", provider_task_id=returned)
            prompt_id = returned
            await _publish(gen.id)
        assert prompt_id
        state: dict[str, Any] = {"prompt_id": prompt_id}
        listener = asyncio.create_task(_listen(conn, client, gen, state))
        entry = await _poll(client, gen, task, prompt_id)
        images = history_outputs(entry)
        if not images:
            raise RunFailed("generation_failed", "The workflow finished without saving an image.")
        datas = [await client.view(i["filename"], i["subfolder"], i["type"]) for i in images]
        datas = await _pose_fallback(client, gen, task, workflow_sent(gen, entry), datas)
        datas = await _advanced_finish(client, gen, task, datas, entry)
        # 18+ runs (Krea 2 switch, or an 18+ photo) are marked, so they can be blurred.
        adult = bool((gen.comfy_meta or {}).get("adult"))
        await worker.store_outputs(gen, task, datas, [True] * len(datas) if adult else [])
        await worker.set_task(
            task.id,
            status="completed",
            outputs=[f"{i['subfolder']}/{i['filename']}".lstrip("/") for i in images],
            progress={},
        )
    except RunFailed as exc:
        await worker.set_task(
            task.id, status="failed", error_code=exc.code, error_message=exc.message, progress={}
        )
    except ComfyError as exc:
        await worker.set_task(
            task.id, status="failed", error_code=exc.code, error_message=exc.message, progress={}
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("ComfyUI run %s failed", gen.id)
        await worker.set_task(
            task.id,
            status="failed",
            error_code="provider_error",
            error_message=f"Unexpected error: {type(exc).__name__}",
            progress={},
        )
    finally:
        if listener is not None:
            listener.cancel()
    await _publish(gen.id)


async def _extra_pass(
    client: ComfyClient, gen: Generation, task: Any, data: bytes, instruction: str, index: int
) -> bytes:
    """Advanced: the edited photo edited once more (another face of her, a reflection), with
    the same edit LoRA and character LoRA; the first result is kept if this fails."""
    workflow = dict(gen.request_payload)
    try:
        name = await client.upload_image(
            data, f"studio-{gen.id.hex[:12]}-x{index}.png", "image/png"
        )
        graph = k2edit.without_masks(enhance.set_input(workflow, name))
        graph["4"]["inputs"]["prompt"] = instruction
        graph["13"]["inputs"]["ref_boost"] = 4.0  # no region: the whole photo, held as at 4
        graph["7"]["inputs"]["seed"] = int(graph["7"]["inputs"]["seed"]) + 1
        prompt_id = str(uuid.uuid4())
        await client.submit(graph, prompt_id, str(gen.id))
        images = history_outputs(await _poll(client, gen, task, prompt_id))
        if not images:
            return data
        out = await client.view(images[0]["filename"], images[0]["subfolder"], images[0]["type"])
    except RunFailed as exc:
        if exc.code in ("cancelled", "timeout"):
            raise
        log.warning("Advanced second pass failed for %s: %s", gen.id, exc.message)
        return data
    except ComfyError as exc:
        log.warning("Advanced second pass failed for %s: %s", gen.id, exc.message)
        return data
    # Her own face as the first pass left it: the second pass redraws the whole photo and closed
    # a smile with teeth into a pout (the owner's lift photo, 2026-11-08); it is for the
    # reflection only.
    try:
        box = await _face_at(data)
    except Exception:  # noqa: BLE001 - no face models: the second pass as it came
        return out
    if box is None:
        return out
    return await asyncio.to_thread(k2edit.hold_face, data, out, box[0])


async def _refine_face(
    client: ComfyClient, gen: Generation, task: Any, data: bytes, face: dict, index: int
) -> bytes:
    """Advanced with a character LoRA: the main face redrawn close up (k2edit.face_*)."""
    from PIL import Image as PILImage

    try:
        level = await _level_face(data)
        if level is None:
            return data
        work, angle, box = level
        size = await asyncio.to_thread(lambda: PILImage.open(io.BytesIO(work)).size)
    except Exception:  # noqa: BLE001 - no face models or an unreadable image: keep the edit
        log.warning("Advanced face pass skipped for %s", gen.id, exc_info=True)
        return data
    # Every size: a big face kept the photo's own face too much (likeness 0.29 vs 0.67).
    square = k2edit.face_square(box, *size, any_size=True)
    if square is None:
        return data
    crop = await asyncio.to_thread(k2edit.face_crop, work, square)
    try:
        name = await client.upload_image(
            crop, f"studio-{gen.id.hex[:12]}-face{index}.png", "image/png"
        )
        workflow = k2edit.face_graph(
            image=name,
            lora=face["lora"],
            strength=float(face["strength"]),
            who=face["who"],
            seed=int((gen.settings or {}).get("seed") or 0) + index,
            note=str(face.get("note") or ""),
            keep_mouth=bool(face.get("keep_mouth")),
        )
        prompt_id = str(uuid.uuid4())
        await client.submit(workflow, prompt_id, str(gen.id))
        entry = await _poll(client, gen, task, prompt_id)
        images = history_outputs(entry)
        if not images:
            return data
        patch = await client.view(images[0]["filename"], images[0]["subfolder"], images[0]["type"])
    except RunFailed as exc:
        if exc.code in ("cancelled", "timeout"):
            raise
        log.warning("Advanced face pass failed for %s: %s", gen.id, exc.message)
        return data
    except ComfyError as exc:
        log.warning("Advanced face pass failed for %s: %s", gen.id, exc.message)
        return data
    pasted = await asyncio.to_thread(k2edit.face_paste, work, patch, square, None, box)
    if angle:
        pasted = await asyncio.to_thread(k2edit.unturn, data, pasted, angle, box)
    return pasted


async def _level_face(data: bytes) -> tuple[bytes, float, tuple[float, ...]] | None:
    """(photo, angle, face box): the photo as it is when YuNet finds the face, else turned
    (k2edit.turn by angle) the way YuNet is surest of it, and the face's box there. None
    without a face."""
    from app.services import faces

    box = await faces.face_box(data)
    if box:
        return data, 0.0, box
    found: dict[float, tuple[bytes, tuple[float, ...], float]] = {}
    for angle in k2edit.TURNS:
        image = await asyncio.to_thread(k2edit.turn, data, angle)
        face = await faces.face_box_score(image)
        if face:
            found[angle] = (image, *face)
    if not found:
        return None
    best = max(found, key=lambda a: found[a][2])
    image, box, _ = found[best]
    return image, best, box


def workflow_sent(gen: Generation, entry: dict[str, Any]) -> dict[str, Any]:
    """The graph ComfyUI ran (its history keeps it), else the stored one."""
    prompt = entry.get("prompt")
    if isinstance(prompt, list) and len(prompt) > 2 and isinstance(prompt[2], dict):
        return prompt[2]
    return gen.request_payload


async def _pose_fallback(
    client: ComfyClient, gen: Generation, task: Any, sent: dict[str, Any], datas: list[bytes]
) -> list[bytes]:
    """Advanced Body: when the person moved in the edit (k2edit.moved), it is run again with
    everything but the body part held, on the same uploaded photo."""
    meta = gen.comfy_meta or {}
    fallback, source = meta.get("pose_fallback"), meta.get("input")
    if not fallback or not source or not datas:
        return datas
    try:
        original = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        upright = await asyncio.to_thread(k2edit.upright, original)
    except OSError:
        return datas
    if not await _moved(upright, datas[0]):
        return datas
    name = next(
        (n["inputs"]["image"] for n in sent.values() if n.get("class_type") == "LoadImage"),
        None,
    )
    if not name:
        return datas
    try:
        prompt_id = str(uuid.uuid4())
        await client.submit(enhance.set_input(fallback, name), prompt_id, str(gen.id))
        images = history_outputs(await _poll(client, gen, task, prompt_id))
        if not images:
            return datas
        log.info("Advanced %s: the person moved, redone with the pose held", gen.id)
        return [await client.view(i["filename"], i["subfolder"], i["type"]) for i in images]
    except RunFailed as exc:
        if exc.code in ("cancelled", "timeout"):
            raise
        return datas
    except ComfyError:
        return datas


async def _face_at(data: bytes) -> tuple[tuple[float, ...], tuple[int, int]] | None:
    """The main face's box in the picture and the picture's size; a tilted face (a woman
    lying down) is found turned (_level_face) and its box put back where it is."""
    from PIL import Image as PILImage

    level = await _level_face(data)
    if level is None:
        return None
    _, angle, (x, y, w, h) = level
    size = await asyncio.to_thread(lambda: PILImage.open(io.BytesIO(data)).size)
    if angle:
        cx, cy = k2edit.unturn_point(x + w / 2, y + h / 2, angle, size)
        x, y = cx - w / 2, cy - h / 2
    return (x, y, w, h), size


async def _moved(before: bytes, after: bytes) -> bool:
    """Whether the person moved from one picture to the other (k2edit.moved). Her face lost
    in the second counts as moved (a woman lying in a tilted selfie stood up and turned to
    the camera: YuNet, looking only level, found neither face); no face in the first, not."""
    try:
        a = await _face_at(before)
        b = await _face_at(after) if a else None
    except Exception:  # noqa: BLE001 - no face models
        return False
    if a is None:
        return False
    return b is None or k2edit.moved(a[0], b[0], a[1], b[1])


async def _body_pass(
    client: ComfyClient, gen: Generation, task: Any, data: bytes, index: int
) -> tuple[bytes, bytes | None]:
    """Advanced, the whole photo with a body change: the edited photo's body changed with the
    body words alone, everything else held (k2edit.body_change, keep_limbs); her arms, hands,
    face and hair are then put back (k2edit.paste_held). Also the part it changed, for
    keep_unchanged. The edit is kept as it was if this fails."""
    spec = (gen.comfy_meta or {})["body_pass"]
    try:
        name = await client.upload_image(
            data, f"studio-{gen.id.hex[:12]}-b{index}.png", "image/png"
        )
        prompt_id = str(uuid.uuid4())
        await client.submit(enhance.set_input(spec["workflow"], name), prompt_id, str(gen.id))
        entry = await _poll(client, gen, task, prompt_id)
        images = history_outputs(entry)
        if not images:
            return data, None
        out = await client.view(images[0]["filename"], images[0]["subfolder"], images[0]["type"])
        held = await _preview(client, entry, k2edit.HELD_NODE)
        part = await _preview(client, entry, k2edit.PART_NODE)
    except RunFailed as exc:
        if exc.code in ("cancelled", "timeout"):
            raise
        log.warning("Advanced body pass failed for %s: %s", gen.id, exc.message)
        return data, None
    except ComfyError as exc:
        log.warning("Advanced body pass failed for %s: %s", gen.id, exc.message)
        return data, None
    if held is None:
        return data, None  # what it held is unknown: the edit as it was
    return await asyncio.to_thread(k2edit.paste_held, data, out, held), part


async def _preview(
    client: ComfyClient, entry: dict[str, Any], node: str = k2edit.HANDS_NODE
) -> bytes | None:
    """A mask the run showed as a preview: the hands SAM found in the face square (Face only,
    Head), the changed part of a body pass; None if there is none."""
    images = ((entry.get("outputs") or {}).get(node) or {}).get("images") or []
    if not images:
        return None
    try:
        return await client.view(
            images[0]["filename"], images[0].get("subfolder", ""), images[0].get("type", "temp")
        )
    except ComfyError:
        return None


async def _face_box(original: bytes) -> tuple[float, float, float, float] | None:
    """The main face's box in the upright photo (YuNet on the Mac), or None."""
    from app.services import faces

    try:
        return await faces.face_box(await asyncio.to_thread(k2edit.upright, original))
    except Exception:  # noqa: BLE001 - no face models: the square's own ellipse
        return None


async def _advanced_finish(
    client: ComfyClient,
    gen: Generation,
    task: Any,
    datas: list[bytes],
    entry: dict[str, Any] | None = None,
) -> list[bytes]:
    """Advanced: the character's face redrawn close up, then the untouched parts of the
    photo put back from the original (k2edit)."""
    meta = gen.comfy_meta or {}
    if meta.get("family") != k2edit.FAMILY:
        return datas
    source = meta.get("input")
    head = meta.get("head_crop")
    if head and source:
        # Head: what changed in the edited square goes back into the original photo.
        try:
            original = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        except OSError as exc:
            raise RunFailed("not_found", "The photo is no longer on this Mac.") from exc
        hands = await _preview(client, entry or {})
        return [
            await asyncio.to_thread(k2edit.crop_paste, original, d, tuple(head), hands)
            for d in datas
        ]
    square = meta.get("face_only")
    if square and source:
        # Face only: the edited face goes back into the original photo; nothing else changes.
        try:
            original = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
        except OSError as exc:
            raise RunFailed("not_found", "The photo is no longer on this Mac.") from exc
        hands = await _preview(client, entry or {})
        box = await _face_box(original)
        return [
            await asyncio.to_thread(
                k2edit.face_paste, original, d, tuple(square), hands, box, k2edit.FACE_ONLY_GROW
            )
            for d in datas
        ]
    # Where the edit was asked to change something: always the edit's pixels (keep_unchanged).
    changed = [
        m
        for m in [
            await _preview(client, entry or {}, k2edit.HAIR_NODE),
            await _preview(client, entry or {}, k2edit.PART_NODE),
        ]
        if m
    ]
    if meta.get("body_pass"):
        passes = [await _body_pass(client, gen, task, d, i) for i, d in enumerate(datas)]
        datas = [d for d, _ in passes]
        changed += [p for _, p in passes[:1] if p]
    extra = meta.get("extra_pass")
    if extra:
        datas = [await _extra_pass(client, gen, task, d, extra, i) for i, d in enumerate(datas)]
    face = meta.get("face")
    if face:
        datas = [await _refine_face(client, gen, task, d, face, i) for i, d in enumerate(datas)]
    datas = await _keep_original(gen, datas, changed)
    if meta.get("detail"):
        datas = [await _detail(client, gen, task, d, i) for i, d in enumerate(datas)]
    return datas


async def _keep_original(gen: Generation, datas: list[bytes], changed: list[bytes]) -> list[bytes]:
    """Untouched parts from the original (k2edit.keep_unchanged), and the face held exactly
    (Hair and Body without a character)."""
    meta = gen.comfy_meta or {}
    source = meta.get("input")
    if not source or not (meta.get("keep_unchanged") or meta.get("hold_face")):
        return datas
    try:
        original = await asyncio.to_thread(storage.absolute(source["file_path"]).read_bytes)
    except OSError:
        return datas  # the photo was deleted meanwhile: the edit as it came
    if meta.get("keep_unchanged"):
        datas = [
            (await asyncio.to_thread(k2edit.keep_unchanged, original, d, changed))[0] for d in datas
        ]
    if meta.get("hold_face"):
        box = await _face_box(original)
        if box:
            datas = [await asyncio.to_thread(k2edit.hold_face, original, d, box) for d in datas]
    return datas


async def _detail(
    client: ComfyClient, gen: Generation, task: Any, data: bytes, index: int
) -> bytes:
    """Advanced with her LoRA: the finished photo once more at 2K (k2edit.detail_graph); the
    photo as it was if this fails."""
    try:
        name = await client.upload_image(
            data, f"studio-{gen.id.hex[:12]}-d{index}.png", "image/png"
        )
        prompt_id = str(uuid.uuid4())
        graph = enhance.set_input((gen.comfy_meta or {})["detail"], name)
        await client.submit(graph, prompt_id, str(gen.id))
        images = history_outputs(await _poll(client, gen, task, prompt_id))
        if not images:
            return data
        return await client.view(images[0]["filename"], images[0]["subfolder"], images[0]["type"])
    except RunFailed as exc:
        if exc.code in ("cancelled", "timeout"):
            raise
        log.warning("Advanced detail pass failed for %s: %s", gen.id, exc.message)
        return data
    except ComfyError as exc:
        log.warning("Advanced detail pass failed for %s: %s", gen.id, exc.message)
        return data


async def after_finish(gen_id: uuid.UUID) -> None:
    """A cancelled run ends as "cancelled" rather than "failed"."""
    async with SessionLocal() as db:
        gen = await db.get(Generation, gen_id)
        if gen is None or not (gen.comfy_meta or {}).get("cancel_requested"):
            return
        if gen.status == "failed":
            gen.status = "cancelled"
            gen.error_code = None
            gen.error_message = "Cancelled while running on ComfyUI."
            await db.commit()
    from app.services.worker import load_generation

    loaded = await load_generation(gen_id)
    if loaded is not None:
        broker.publish(loaded.user_id, "generation", generation_json(loaded))


async def request_cancel(gen: Generation) -> None:
    """Best effort right away; the running job loop also sees the flag."""
    task = gen.tasks[0] if gen.tasks else None
    if task is None or not task.provider_task_id:
        return
    try:
        conn = await _connection(gen)
        client = client_for(conn)
        await client.delete_queued(task.provider_task_id)
        await client.interrupt(task.provider_task_id)
    except (RunFailed, ComfyError):
        return
