from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventBus:
    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[Event]] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> bool:
        """Remove a subscriber queue; safe to call more than once."""
        try:
            self._queues.remove(queue)
        except ValueError:
            return False
        return True

    async def publish(self, event_type: str, **data: Any) -> None:
        event = Event(event_type, data)
        dead: list[asyncio.Queue[Event]] = []

        for queue in tuple(self._queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(queue)

        for queue in dead:
            self.unsubscribe(queue)

    def subscriber_count(self) -> int:
        return len(self._queues)
