"""Background loop that advances studio GPU work: LoRA training runs and ComfyUI
deployments (each item at most one step at a time; all state lives in the database)."""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from app.db import SessionLocal
from app.models import ComfyConnection

log = logging.getLogger("studio.gpu")


class GPURunner:
    def __init__(self) -> None:
        self.interval = 5.0
        self._task: asyncio.Task[None] | None = None
        self._busy: set[uuid.UUID] = set()
        self._running: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for task in list(self._running):
            task.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick(wait=False)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must survive any single failure
                log.exception("GPU runner tick failed")
            await asyncio.sleep(self.interval)

    def _spawn(self, key: uuid.UUID, coro) -> asyncio.Task[None] | None:
        if key in self._busy:
            coro.close()
            return None
        self._busy.add(key)

        async def run() -> None:
            try:
                await coro
            except Exception:  # noqa: BLE001
                log.exception("GPU step for %s failed", key)
            finally:
                self._busy.discard(key)

        task = asyncio.create_task(run())
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return task

    async def tick(self, wait: bool = True) -> None:
        """One pass over every active item (tests call this directly)."""
        from app.comfy import deploy
        from app.lora import runner as training

        tasks = []
        for version_id in await training.active_ids():
            task = self._spawn(version_id, training.advance(version_id))
            if task:
                tasks.append(task)
        async with SessionLocal() as db:
            conns = (
                await db.scalars(
                    select(ComfyConnection.id).where(
                        ComfyConnection.auth_type == "agent",
                        ComfyConnection.status.in_(("deploying", "ready")),
                    )
                )
            ).all()
        for conn_id in conns:
            task = self._spawn(conn_id, deploy.advance(conn_id))
            if task:
                tasks.append(task)
        if wait and tasks:
            await asyncio.gather(*tasks)


gpu_runner = GPURunner()
