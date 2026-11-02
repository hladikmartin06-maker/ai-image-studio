"""The generation job runner.

Lifecycle of a generation (see docs/ARCHITECTURE.md):
  queued -> processing -> completed | failed        (cancelled only while still queued)
Each generation has one or more provider tasks:
  pending -> submitting -> processing -> completed | failed
ComfyUI runs (Krea 2, Enhance photo) wait in a queue per ComfyUI connection: one run at a
time on a GPU, the next starts when the previous ended (files such as the LoRA are uploaded
just before each run), outside the worker slots of the hosted models.
A task is sent to the provider at most once. If the app stops while a task is
"submitting" (the provider may already have accepted and charged it), the task is
failed as "interrupted" instead of being sent again. Tasks that have a provider id are
polled again after a restart, so no finished image is lost.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import delete, select, update
from sqlalchemy.orm import selectinload

from app.catalog import MODELS, REFERENCE_MODES, build_payload, resolve_variant
from app.config import get_settings
from app.db import SessionLocal
from app.models import (
    AuthSession,
    Generation,
    Image,
    Integration,
    ProviderTask,
    ReferenceImage,
    UserSettings,
)
from app.providers import get_adapter
from app.providers.base import ProviderError, failure_from_message
from app.security import decrypt_secret
from app.services import preferences, references, storage
from app.services.events import broker
from app.services.serializers import generation_json

log = logging.getLogger("studio.worker")

TRANSIENT = ("network_error", "rate_limited", "provider_error")
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
# A Wan 3.0 video (up to 30 s at 1080p) is far larger than an image.
MAX_VIDEO_BYTES = 1024 * 1024 * 1024
VIDEO_MODES = ("text-to-video", "image-to-video")


def utcnow() -> datetime:
    return datetime.now(UTC)


async def load_generation(gen_id: uuid.UUID) -> Generation | None:
    async with SessionLocal() as db:
        return await db.scalar(
            select(Generation)
            .where(Generation.id == gen_id)
            .options(selectinload(Generation.tasks), selectinload(Generation.images))
        )


async def publish(gen_id: uuid.UUID) -> None:
    gen = await load_generation(gen_id)
    if gen is not None:
        broker.publish(gen.user_id, "generation", generation_json(gen))


class JobWorker:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[uuid.UUID] = asyncio.Queue()
        self._running: set[uuid.UUID] = set()
        self._tasks: list[asyncio.Task[None]] = []
        # ComfyUI: one lane per connection (the runs in it, the first one is running).
        self._lanes: dict[str, list[uuid.UUID]] = {}
        self._lane_locks: dict[str, asyncio.Lock] = {}
        self._comfy: dict[uuid.UUID, asyncio.Task[None]] = {}
        # Overridable in tests to make polling instant.
        self.poll_initial = 1.5
        self.poll_max = 6.0
        self.download_transport: httpx.AsyncBaseTransport | None = None

    # --- lifecycle -----------------------------------------------------------------

    async def start(self) -> None:
        if self._tasks:
            return
        for _ in range(max(1, get_settings().worker_concurrency)):
            self._tasks.append(asyncio.create_task(self._consume()))
        self._tasks.append(asyncio.create_task(self._janitor()))
        await self.recover()

    async def stop(self) -> None:
        tasks = [*self._tasks, *self._comfy.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._comfy.clear()
        self._lanes.clear()
        self._lane_locks.clear()

    def enqueue(self, gen_id: uuid.UUID) -> None:
        self._queue.put_nowait(gen_id)

    async def recover(self) -> None:
        """Picks up generations that were queued or running when the app stopped."""
        async with SessionLocal() as db:
            ids = (
                await db.scalars(
                    select(Generation.id)
                    .where(Generation.status.in_(("queued", "processing")))
                    .order_by(Generation.created_at)
                )
            ).all()
        for gen_id in ids:
            self.enqueue(gen_id)
        if ids:
            log.info("Resuming %d generation(s)", len(ids))

    async def drain(self) -> None:
        """Waits until every queued generation has finished (used by tests)."""
        while not self._queue.empty() or self._running or self._comfy:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

    async def _consume(self) -> None:
        while True:
            gen_id = await self._queue.get()
            if gen_id in self._running:
                continue
            self._running.add(gen_id)
            try:
                await self.run(gen_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one broken job must never stop the worker
                log.exception("Generation %s crashed", gen_id)
                await self._fail(gen_id, "provider_error", "Unexpected error while generating.")
            finally:
                self._running.discard(gen_id)

    async def _janitor(self) -> None:
        while True:
            try:
                await cleanup()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Cleanup failed")
            await asyncio.sleep(3600)

    # --- one generation --------------------------------------------------------------

    async def run(self, gen_id: uuid.UUID) -> None:
        gen = await load_generation(gen_id)
        if gen is None or gen.status not in ("queued", "processing"):
            return
        if gen.provider == "comfyui":
            self._queue_comfy(gen)
            return

        async with SessionLocal() as db:
            result = await db.execute(
                update(Generation)
                .where(Generation.id == gen_id, Generation.status.in_(("queued", "processing")))
                .values(status="processing", started_at=gen.started_at or utcnow())
            )
            await db.commit()
        if result.rowcount == 0:  # cancelled in the meantime
            return
        await publish(gen_id)

        try:
            api_key = await self._api_key(gen)
            variant = resolve_variant(gen.model_key, gen.provider, gen.mode in REFERENCE_MODES)
            image_urls = await self._reference_urls(gen, api_key)
        except ProviderError as exc:
            await self._fail_open_tasks(gen_id, exc.code, exc.user_message)
            await self._finish(gen_id)
            return
        except ValueError as exc:
            await self._fail_open_tasks(gen_id, "unsupported_model", str(exc))
            await self._finish(gen_id)
            return

        adapter = get_adapter(gen.provider)
        await asyncio.gather(
            *(
                self._run_task(gen, task, adapter, api_key, variant, image_urls)
                for task in gen.tasks
                if task.status not in ("completed", "failed")
            )
        )
        await self._finish(gen_id)

    # --- ComfyUI queue -------------------------------------------------------------------

    def _queue_comfy(self, gen: Generation) -> None:
        if gen.id in self._comfy:
            return
        lane = str((gen.comfy_meta or {}).get("connection_id") or "")
        self._lanes.setdefault(lane, []).append(gen.id)
        task = asyncio.create_task(self._run_comfy(gen.id, lane))
        self._comfy[gen.id] = task

    async def _run_comfy(self, gen_id: uuid.UUID, lane: str) -> None:
        from app.comfy import runner as comfy_runner

        lock = self._lane_locks.setdefault(lane, asyncio.Lock())
        try:
            await self._show_lane(lane)
            async with lock:  # asyncio.Lock wakes its waiters in order
                gen = await load_generation(gen_id)
                if gen is None or gen.status not in ("queued", "processing"):
                    return  # cancelled while waiting
                async with SessionLocal() as db:
                    result = await db.execute(
                        update(Generation)
                        .where(
                            Generation.id == gen_id,
                            Generation.status.in_(("queued", "processing")),
                        )
                        .values(status="processing", started_at=gen.started_at or utcnow())
                    )
                    await db.commit()
                if result.rowcount == 0:
                    return
                async with SessionLocal() as db:  # its place in the queue is over
                    await db.execute(
                        update(ProviderTask)
                        .where(
                            ProviderTask.generation_id == gen_id, ProviderTask.status == "pending"
                        )
                        .values(progress={})
                    )
                    await db.commit()
                await publish(gen_id)
                gen = await load_generation(gen_id) or gen
                try:
                    await comfy_runner.run(self, gen)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - one broken run must never stop the queue
                    log.exception("ComfyUI run %s crashed", gen_id)
                    await self._fail_open_tasks(
                        gen_id, "provider_error", "Unexpected error while generating."
                    )
                await self._finish(gen_id)
                await comfy_runner.after_finish(gen_id)
        finally:
            waiting = self._lanes.get(lane, [])
            if gen_id in waiting:
                waiting.remove(gen_id)
            if not waiting:
                self._lanes.pop(lane, None)
            self._comfy.pop(gen_id, None)
            if waiting:
                with contextlib.suppress(Exception):
                    await self._show_lane(lane)

    async def _show_lane(self, lane: str) -> None:
        """Runs waiting for an earlier one on the same GPU show their place in the queue."""
        for position, gen_id in enumerate(list(self._lanes.get(lane, []))):
            if position == 0:
                continue
            async with SessionLocal() as db:
                await db.execute(
                    update(ProviderTask)
                    .where(ProviderTask.generation_id == gen_id, ProviderTask.status == "pending")
                    .values(progress={"state": "waiting", "queue_position": position})
                )
                await db.commit()
            await publish(gen_id)

    async def _api_key(self, gen: Generation) -> str:
        async with SessionLocal() as db:
            integration = await db.scalar(
                select(Integration).where(
                    Integration.user_id == gen.user_id, Integration.provider == gen.provider
                )
            )
        if integration is None:
            raise ProviderError("invalid_credentials", gen.provider, "Not connected")
        return decrypt_secret(integration.encrypted_key, associated=f"{gen.user_id}:{gen.provider}")

    async def _reference_urls(self, gen: Generation, api_key: str) -> list[str]:
        """Uploads reference images to the provider once and reuses the URLs (renewed
        after references.UPLOAD_REUSE)."""
        if not gen.reference_image_ids:
            return []
        adapter = get_adapter(gen.provider)
        urls: list[str] = []
        async with SessionLocal() as db:
            for ref_id in gen.reference_image_ids:
                ref = await db.scalar(
                    select(ReferenceImage).where(
                        ReferenceImage.id == uuid.UUID(ref_id),
                        ReferenceImage.user_id == gen.user_id,
                    )
                )
                if ref is None:
                    raise ProviderError(
                        "invalid_parameters", gen.provider, "A reference image was deleted"
                    )
                url = references.reusable_url(ref, gen.provider)
                if not url:
                    data = await asyncio.to_thread(storage.absolute(ref.file_path).read_bytes)
                    data, mime, ext = await asyncio.to_thread(
                        references.for_provider,
                        data,
                        ref.mime_type,
                        ref.file_path.rsplit(".", 1)[-1],
                    )
                    url = await adapter.upload(api_key, data, f"{ref.id}.{ext}", mime)
                    references.remember_upload(ref, gen.provider, url)
                    await db.commit()
                urls.append(url)
            row = await db.get(Generation, gen.id)
            if row is not None:
                # History shows the request as sent (a video: first and last frame).
                if gen.mode in VIDEO_MODES:
                    sent = {"image": urls[0], **({"last_image": urls[1]} if len(urls) > 1 else {})}
                else:
                    sent = {"images": urls}
                row.request_payload = {**row.request_payload, **sent}
                await db.commit()
        return urls

    async def _run_task(
        self,
        gen: Generation,
        task: ProviderTask,
        adapter: Any,
        api_key: str,
        variant: Any,
        image_urls: list[str],
    ) -> None:
        provider_task_id = task.provider_task_id
        try:
            if task.status == "submitting":
                raise ProviderError("interrupted", gen.provider)
            if task.status == "pending":
                await self._set_task(task.id, status="submitting")
                payload = build_payload(
                    variant,
                    prompt=gen.final_prompt or gen.prompt,
                    settings=gen.settings,
                    image_urls=image_urls,
                    count=task.images_requested,
                )
                try:
                    provider_task_id = await adapter.submit(api_key, variant.model_id, payload)
                except ProviderError as exc:
                    # A network error while sending leaves it unknown whether the provider
                    # accepted the request: never resend it.
                    if exc.code == "network_error":
                        exc.detail = "The request may not have reached the provider"
                    raise
                await self._set_task(
                    task.id, status="processing", provider_task_id=provider_task_id
                )
                await publish(gen.id)
            if not provider_task_id:
                raise ProviderError("interrupted", gen.provider)
            state = await self._poll(gen, adapter, api_key, provider_task_id, task)
            await self._save_outputs(gen, task, state.outputs, state.nsfw_flags)
            await self._set_task(
                task.id, status="completed", outputs=state.outputs, nsfw_flags=state.nsfw_flags
            )
        except ProviderError as exc:
            await self._set_task(
                task.id, status="failed", error_code=exc.code, error_message=exc.user_message
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Task %s failed", task.id)
            await self._set_task(
                task.id,
                status="failed",
                error_code="provider_error",
                error_message=f"Unexpected error: {type(exc).__name__}",
            )
        await publish(gen.id)

    async def _poll(
        self, gen: Generation, adapter: Any, api_key: str, provider_task_id: str, task: ProviderTask
    ):
        deadline = (task.created_at or utcnow()) + timedelta(
            seconds=get_settings().job_timeout_seconds
        )
        delay = self.poll_initial
        transient_failures = 0
        while True:
            try:
                state = await adapter.get_task(api_key, provider_task_id)
                transient_failures = 0
            except ProviderError as exc:
                if exc.code not in TRANSIENT or transient_failures >= 20:
                    raise
                transient_failures += 1
                state = None
            if state is not None and state.status == "completed":
                return state
            if state is not None and state.status == "failed":
                raise failure_from_message(gen.provider, state.error)
            if utcnow() > deadline:
                raise ProviderError("timeout", gen.provider)
            await asyncio.sleep(delay)
            delay = min(delay * 1.3, self.poll_max)

    async def _download(self, gen: Generation, output: str) -> bytes:
        limit = MAX_VIDEO_BYTES if gen.mode in VIDEO_MODES else MAX_DOWNLOAD_BYTES
        if output.startswith("data:"):
            try:
                return base64.b64decode(output.split(",", 1)[1], validate=False)
            except (IndexError, binascii.Error) as exc:
                raise ProviderError("provider_error", gen.provider, "Bad image data") from exc
        if not output.startswith(("http://", "https://")):
            try:
                return base64.b64decode(output, validate=True)
            except binascii.Error as exc:
                raise ProviderError("provider_error", gen.provider, "Bad image output") from exc
        # Output URLs are public CDN links: the API key is never sent with them.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600 if gen.mode in VIDEO_MODES else 120, connect=20),
            follow_redirects=True,
            transport=self.download_transport,
        ) as client:
            for attempt in range(3):
                try:
                    async with client.stream("GET", output) as response:
                        if response.status_code >= 400:
                            raise ProviderError(
                                "provider_error",
                                gen.provider,
                                f"Image download failed (HTTP {response.status_code})",
                            )
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > limit:
                                raise ProviderError(
                                    "provider_error", gen.provider, "Image is too large"
                                )
                            chunks.append(chunk)
                        return b"".join(chunks)
                except httpx.HTTPError as exc:
                    if attempt == 2:
                        raise ProviderError(
                            "network_error", gen.provider, "Image download failed"
                        ) from exc
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise ProviderError("network_error", gen.provider, "Image download failed")

    async def _save_outputs(
        self, gen: Generation, task: ProviderTask, outputs: list[str], nsfw_flags: list[bool]
    ) -> None:
        datas = [await self._download(gen, output) for output in outputs]
        if (gen.settings or {}).get("adult"):
            # Run with "18+ content" on: every image counts as 18+ (blurred per Settings).
            nsfw_flags = [True] * len(datas)
        await self.store_outputs(gen, task, datas, nsfw_flags)

    async def store_outputs(
        self, gen: Generation, task: ProviderTask, datas: list[bytes], nsfw_flags: list[bool]
    ) -> None:
        """Stores finished images (originals byte for byte + thumbnails) and videos (as
        delivered) in the library."""
        async with SessionLocal() as db:
            prefs = await preferences.load(db, gen.user_id)
        for i, data in enumerate(datas):
            image_id = uuid.uuid4()
            try:
                if gen.mode in VIDEO_MODES:
                    stored = await storage.store_video(
                        gen.user_id, image_id, data, storage.expected_size(gen.settings)
                    )
                else:
                    stored = await storage.store_image(gen.user_id, image_id, data)
            except storage.InvalidImage as exc:
                raise ProviderError("provider_error", gen.provider, str(exc)) from exc
            async with SessionLocal() as db:
                db.add(
                    Image(
                        id=image_id,
                        user_id=gen.user_id,
                        generation_id=gen.id,
                        index=task.index * max(1, task.images_requested) + i,
                        file_path=stored.file_path,
                        thumb_path=stored.thumb_path,
                        mime_type=stored.mime_type,
                        width=stored.width,
                        height=stored.height,
                        size_bytes=stored.size_bytes,
                        nsfw=nsfw_flags[i] if i < len(nsfw_flags) else None,
                        in_library=prefs.auto_save,
                    )
                )
                await db.commit()
            await publish(gen.id)

    # --- state helpers -----------------------------------------------------------------

    async def set_task(self, task_id: uuid.UUID, **values: Any) -> None:
        await self._set_task(task_id, **values)

    async def _set_task(self, task_id: uuid.UUID, **values: Any) -> None:
        async with SessionLocal() as db:
            await db.execute(
                update(ProviderTask).where(ProviderTask.id == task_id).values(**values)
            )
            await db.commit()

    async def _fail_open_tasks(self, gen_id: uuid.UUID, code: str, message: str) -> None:
        async with SessionLocal() as db:
            await db.execute(
                update(ProviderTask)
                .where(
                    ProviderTask.generation_id == gen_id,
                    ProviderTask.status.in_(("pending", "submitting", "processing")),
                )
                .values(status="failed", error_code=code, error_message=message)
            )
            await db.commit()

    async def _fail(self, gen_id: uuid.UUID, code: str, message: str) -> None:
        await self._fail_open_tasks(gen_id, code, message)
        await self._finish(gen_id)

    async def _finish(self, gen_id: uuid.UUID) -> None:
        gen = await load_generation(gen_id)
        if gen is None:
            return
        if any(t.status not in ("completed", "failed") for t in gen.tasks):
            return
        failed = [t for t in gen.tasks if t.status == "failed"]
        produced = len(gen.images)
        async with SessionLocal() as db:
            row = await db.get(Generation, gen_id)
            if row is None:
                return
            row.completed_at = utcnow()
            if produced > 0:
                row.status = "completed"
                if failed:
                    row.error_code = failed[0].error_code
                    row.error_message = (
                        f"{len(failed)} of {len(gen.tasks)} requests failed: "
                        f"{failed[0].error_message}"
                    )
            else:
                row.status = "failed"
                first = failed[0] if failed else None
                row.error_code = first.error_code if first else "generation_failed"
                row.error_message = (
                    first.error_message if first else "The provider returned no image."
                )
            await db.commit()
        await publish(gen_id)


async def cleanup() -> None:
    """Hourly: expired sessions, unfinished uploads older than a day, and failed generations
    past the user's retention."""
    from app.services import uploads

    await uploads.cleanup()
    async with SessionLocal() as db:
        await db.execute(delete(AuthSession).where(AuthSession.expires_at < utcnow()))
        await db.commit()
        rows = (await db.scalars(select(UserSettings))).all()
        for row in rows:
            days = int((row.data or {}).get("keep_failed_days") or 0)
            if days <= 0:
                continue
            await db.execute(
                delete(Generation).where(
                    Generation.user_id == row.user_id,
                    Generation.status.in_(("failed", "cancelled")),
                    Generation.created_at < utcnow() - timedelta(days=days),
                )
            )
        await db.commit()


def model_allows_mature(model_key: str) -> bool:
    model = MODELS.get(model_key)
    return bool(model and model.content == "sfw+nsfw")


worker = JobWorker()
