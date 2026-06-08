"""Per-engagement event bus using in-process asyncio primitives.

Producers: BaseAgent.step, stage controller, approval persistence, tool runs.
Consumers: SSE endpoint streams to the UI, abort consumer.

Replaces the former Redis Streams implementation with in-process
``asyncio.Queue`` fan-out. Each topic (keyed by engagement_id) maintains
a list of subscriber queues and a bounded history buffer for replay.

No external dependencies. No network I/O. Single-process only.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

log = structlog.get_logger(__name__)

_MAX_HISTORY = 10_000  # cap per-topic history so it doesn't grow unbounded


class EventBus:
    """In-process pub/sub event bus with history replay.

    Each engagement gets its own topic (keyed by ``str(engagement_id)``).
    Subscribers receive a dedicated ``asyncio.Queue``; ``publish()`` fans
    out each event to every live subscriber queue for that topic.

    History is kept per-topic so late-joining subscribers can replay from
    the beginning (``from_id="0"``) or start live (``from_id="$"``).
    """

    def __init__(self) -> None:
        # topic -> list of subscriber queues
        self._topics: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        # topic -> ordered list of historical events
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # monotonic counter for event IDs
        self._counter: int = 0
        # Guards the counter bump + history append/trim + subscriber-list
        # mutation. ``publish`` holds it for its (await-free) critical
        # section; ``subscribe`` holds it only while registering its queue
        # and snapshotting history, never across the yield loop. This is a
        # threading.Lock (not asyncio.Lock) on purpose: the critical
        # sections contain no awaits, so a non-async guard is sufficient and
        # is immune to the event-loop-affinity footgun an asyncio.Lock has
        # when the bus singleton is touched from more than one loop.
        self._lock: threading.Lock = threading.Lock()

    async def aclose(self) -> None:
        """Clean up resources. For API compatibility with the old Redis-backed bus."""
        with self._lock:
            self._topics.clear()
            self._history.clear()

    async def publish(
        self,
        *,
        engagement_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> str:
        """Publish an event; returns a monotonic string ID.

        Best-effort: failures are logged but never propagated to callers
        so the agent loop is never blocked by event bus issues.
        """
        try:
            topic = str(engagement_id)
            # The entire id-allocation + history-mutation + fan-out section
            # is await-free and must run atomically w.r.t. concurrent
            # ``publish`` / ``subscribe`` callers. Without the lock two
            # publishes scheduled on the same tick boundary can read the same
            # counter value and emit DUPLICATE event ids (the abort consumer
            # and SSE consumers order/dedupe on id), and a ``subscribe``
            # finally-clause ``remove`` can race the fan-out's index math.
            with self._lock:
                self._counter += 1
                event_id = str(self._counter)

                # Serialize payload the same way the old bus did so downstream
                # consumers see consistent types (payload is always a dict after
                # deserialization, timestamps are ISO strings).
                event: dict[str, Any] = {
                    "id": event_id,
                    "kind": kind,
                    "ts": datetime.now(UTC).isoformat(),
                    "payload": json.loads(json.dumps(payload, default=str)),
                }

                # Append to history (bounded)
                history = self._history[topic]
                history.append(event)
                if len(history) > _MAX_HISTORY:
                    # Trim the oldest 10% to avoid trimming on every publish
                    trim = _MAX_HISTORY // 10
                    del history[:trim]

                # Fan out to all live subscriber queues (non-blocking put)
                dead: list[int] = []
                subscribers = self._topics[topic]
                for idx, q in enumerate(subscribers):
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        # Slow consumer; drop oldest events from its queue to
                        # make room, then retry. If still full, mark as dead.
                        try:
                            q.get_nowait()
                            q.put_nowait(event)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            dead.append(idx)

                # Remove dead subscriber queues (reverse order keeps indices valid)
                for idx in reversed(dead):
                    subscribers.pop(idx)

            return event_id
        except Exception as e:
            # Best-effort; event bus failures must not block the agent loop.
            log.warning("eventbus.publish_failed", err=str(e), kind=kind)
            return ""

    async def subscribe(
        self,
        engagement_id: UUID,
        *,
        from_id: str = "$",
        idle_timeout_s: float = 30.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields events for the given engagement.

        ``from_id="$"`` starts from the newest entry (live mode).
        ``from_id="0"`` or ``from_id="0-0"`` replays the whole history first.

        Heartbeat dicts (``kind="_heartbeat"``) are yielded on idle timeout
        to keep SSE connections alive, matching the old Redis Streams behavior.
        """
        topic = str(engagement_id)
        replay = from_id not in ("$",)

        # Create a dedicated queue for this subscriber
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_HISTORY)

        # Register the queue AND snapshot history atomically. Registering
        # before snapshotting (the old order) let a ``publish`` that landed
        # in the window between the two deliver the SAME event twice: once
        # via the history snapshot and once via the freshly-registered live
        # queue. Doing both under the bus lock — and recording the id of the
        # last replayed event — lets us skip any live event the snapshot
        # already covered, so each event is delivered exactly once in order.
        with self._lock:
            history_snapshot = list(self._history.get(topic, [])) if replay else []
            self._topics[topic].append(q)
        last_replayed_id = (
            int(history_snapshot[-1]["id"]) if history_snapshot else None
        )

        try:
            # Replay history if requested
            for event in history_snapshot:
                yield event

            # Live tail
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=idle_timeout_s)
                    # Drop any live event already covered by the replayed
                    # snapshot (it was appended to history and fanned out to
                    # this queue in the same publish that ran while we were
                    # registering). Guards against double-delivery.
                    if last_replayed_id is not None:
                        try:
                            if int(event["id"]) <= last_replayed_id:
                                continue
                        except (KeyError, TypeError, ValueError):
                            pass
                    yield event
                except TimeoutError:
                    # Heartbeat to keep SSE connection alive. Carry an ``id``
                    # (the current counter, or "" before any publish) so
                    # consumers that record ``last_id = event["id"]`` for
                    # Last-Event-ID resume don't KeyError on a heartbeat.
                    yield {
                        "id": str(self._counter) if self._counter else "",
                        "kind": "_heartbeat",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {},
                    }
        finally:
            # Unregister on exit (consumer cancelled, connection closed, etc.)
            with self._lock:
                try:
                    self._topics[topic].remove(q)
                except ValueError:
                    pass  # already removed (e.g. dead-subscriber cleanup)


_bus_singleton: EventBus | None = None


def get_bus() -> EventBus:
    global _bus_singleton
    if _bus_singleton is None:
        _bus_singleton = EventBus()
    return _bus_singleton
