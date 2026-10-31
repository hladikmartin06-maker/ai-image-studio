"""In-process publish/subscribe for live job updates (served as server-sent events).

Each open browser tab subscribes with its own queue; events go only to the queues of
the user who owns the job.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections import defaultdict
from typing import Any


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def subscribe(self, user_id: uuid.UUID) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._subscribers[user_id].add(queue)
        return queue

    def unsubscribe(self, user_id: uuid.UUID, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subscribers = self._subscribers.get(user_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(user_id, None)

    def publish(self, user_id: uuid.UUID, event: str, data: dict[str, Any]) -> None:
        for queue in list(self._subscribers.get(user_id, ())):
            # A stalled tab drops events; the page refetches when it reconnects.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait({"event": event, "data": data})

    def subscriber_count(self, user_id: uuid.UUID) -> int:
        return len(self._subscribers.get(user_id, ()))


broker = EventBroker()
