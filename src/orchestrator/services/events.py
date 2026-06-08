"""In-process event subscription — thin wrapper around EventBus.

Extracted from ``orchestrator.api.events``.  No SSE, no FastAPI.
The TUI subscribes to the async iterator directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from orchestrator.events.bus import get_bus


async def subscribe(
    engagement_id: UUID, *, from_id: str = "$"
) -> AsyncIterator[dict[str, Any]]:
    """Yield events for the given engagement.

    ``from_id="$"`` — live tail only.
    ``from_id="0"`` — replay full history, then live.
    """
    bus = get_bus()
    async for event in bus.subscribe(engagement_id, from_id=from_id):
        yield event
