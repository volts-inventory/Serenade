"""Abort-event consumer that terminates in-flight containers.

Subscribes to ``engagement.abort_requested`` events on the in-process
:class:`orchestrator.events.bus.EventBus` for a single engagement, looks
up the running container set in the shared
:class:`sandbox.runner.InFlightContainerRegistry`, and fires
``SandboxRunner.terminate(container_name, timeout_s=30)`` against every
entry in parallel. Each terminate call SIGTERMs the container and waits
up to 30s before SIGKILLing. ``snapshot_and_clear`` drains the registry
inside its own lock, so a second publish for the same engagement is a
no-op (the first call already terminated and removed every container).

Before entering the subscribe loop, the consumer does a one-time DB
poll for ``engagement.status == 'aborted'`` to catch any abort that was
committed before the subscriber attached.

In addition to the bus-driven consumer, this module exposes two simple
functions for cooperative abort signalling between coroutines in the
same process:

- ``request_abort(engagement_id)`` -- sets an ``asyncio.Event`` for the
  given engagement.
- ``is_abort_requested(engagement_id)`` -- returns True if the event
  has been set.

These complement the bus-driven consumer: the agent loop checks
``is_abort_requested`` at each step boundary for an immediate,
zero-latency abort check, while the bus consumer handles the
container-termination side.

Wiring: the worker process owns the subscriber lifecycle. The
``start_abort_consumer`` factory takes an engagement_id and a
``SandboxRunner`` (so the terminate call uses the same runner instance
the rest of the worker is using). It returns an :class:`asyncio.Task`
the worker can ``cancel()`` on engagement completion / teardown.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from orchestrator.events.bus import EventBus, get_bus

if TYPE_CHECKING:
    from sandbox.runner import InFlightContainerRegistry, SandboxRunner

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight cooperative abort signals (asyncio.Event per engagement)
# ---------------------------------------------------------------------------

_abort_events: dict[str, asyncio.Event] = {}


def request_abort(engagement_id: UUID) -> None:
    """Signal an abort for the given engagement.

    Safe to call from any coroutine in the same event loop. The
    corresponding ``is_abort_requested`` check is O(1) with no I/O.
    """
    key = str(engagement_id)
    if key not in _abort_events:
        _abort_events[key] = asyncio.Event()
    _abort_events[key].set()


def is_abort_requested(engagement_id: UUID) -> bool:
    """Return True if ``request_abort`` has been called for this engagement."""
    key = str(engagement_id)
    ev = _abort_events.get(key)
    return ev.is_set() if ev is not None else False


def clear_abort(engagement_id: UUID) -> None:
    """Remove the abort signal for a completed/cleaned-up engagement."""
    _abort_events.pop(str(engagement_id), None)


# ---------------------------------------------------------------------------
# Container termination logic (unchanged from the Redis era)
# ---------------------------------------------------------------------------


async def _terminate_all(
    *,
    runner: SandboxRunner,
    registry: InFlightContainerRegistry,
    engagement_id: UUID,
    timeout_s: int = 30,
) -> int:
    """Snapshot every in-flight container for ``engagement_id`` and
    ``terminate()`` them in parallel. Returns the number of containers
    actually terminated (0 when the registry was empty -- a published
    abort with no in-flight work is a legitimate no-op, not an error).

    Idempotency: ``snapshot_and_clear`` atomically removes the
    engagement's bucket inside the registry's lock, so a re-publish of
    the same abort event sees an empty snapshot and does not re-call
    terminate on already-killed containers.
    """
    containers = await registry.snapshot_and_clear(str(engagement_id))
    if not containers:
        log.info(
            "abort_consumer.no_inflight",
            engagement_id=str(engagement_id),
            advice="abort published but registry was empty (work already done)",
        )
        return 0
    log.warning(
        "abort_consumer.terminating",
        engagement_id=str(engagement_id),
        count=len(containers),
        containers=sorted(containers),
    )
    results = await asyncio.gather(
        *(runner.terminate(name, timeout_s=timeout_s) for name in containers),
        return_exceptions=True,
    )
    failures = [r for r in results if isinstance(r, BaseException) or r is False]
    if failures:
        log.warning(
            "abort_consumer.partial_terminate",
            engagement_id=str(engagement_id),
            failed=len(failures),
            total=len(containers),
        )
    return len(containers)


async def _poll_engagement_status_aborted(engagement_id: UUID) -> bool:
    """Return True if the engagement row is currently ``status='aborted'``.

    Closes the residual startup-window gap where an abort is published
    between the worker spawning the consumer task and the bus subscriber
    actually issuing its first read. Best-effort: a DB outage here
    returns False and the loop keeps going.
    """
    try:
        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import EngagementRow  # noqa: PLC0415

        async with session_scope() as s:
            row = await s.get(EngagementRow, engagement_id)
            if row is not None and row.status == "aborted":
                return True
    except Exception as e:
        log.warning(
            "abort_consumer.startup_poll_failed",
            engagement_id=str(engagement_id),
            err=str(e),
        )
    return False


async def run_abort_consumer(
    engagement_id: UUID,
    *,
    runner: SandboxRunner,
    bus: EventBus | None = None,
    registry: InFlightContainerRegistry | None = None,
    timeout_s: int = 30,
) -> None:
    """Long-running coroutine: subscribe to the per-engagement bus and
    react to every ``engagement.abort_requested`` event by terminating
    in-flight containers.

    Designed to be wrapped in :func:`asyncio.create_task` by the worker
    so the subscriber loop runs in parallel with the agent graph. The
    worker cancels the task when the engagement reaches a terminal
    status (``completed`` / ``failed`` / ``aborted``).

    Before entering the subscribe loop, does a one-time DB poll for
    ``engagement.status == 'aborted'`` so any abort that landed before
    the subscriber was attached is still honored.
    """
    from sandbox.runner import get_inflight_registry  # noqa: PLC0415

    actual_bus = bus or get_bus()
    actual_registry = registry or get_inflight_registry()
    log.info(
        "abort_consumer.start", engagement_id=str(engagement_id)
    )
    # One-time DB poll BEFORE the subscribe loop. If the engagement was
    # aborted before the consumer got here, trigger _terminate_all
    # immediately. Idempotent: the eventual bus delivery sees an
    # already-drained registry and no-ops.
    if await _poll_engagement_status_aborted(engagement_id):
        log.warning(
            "abort_consumer.startup_already_aborted",
            engagement_id=str(engagement_id),
            advice="engagement was aborted before consumer started; firing terminate",
        )
        try:
            await _terminate_all(
                runner=runner,
                registry=actual_registry,
                engagement_id=engagement_id,
                timeout_s=timeout_s,
            )
        except Exception as e:
            log.warning(
                "abort_consumer.startup_terminate_failed",
                engagement_id=str(engagement_id),
                err=str(e),
            )
    try:
        async for event in actual_bus.subscribe(engagement_id, from_id="$"):
            if event.get("kind") != "engagement.abort_requested":
                continue
            # Also set the cooperative abort flag so the agent loop's
            # fast-path check picks it up immediately.
            request_abort(engagement_id)
            try:
                await _terminate_all(
                    runner=runner,
                    registry=actual_registry,
                    engagement_id=engagement_id,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                log.warning(
                    "abort_consumer.terminate_failed",
                    engagement_id=str(engagement_id),
                    err=str(e),
                )
    except asyncio.CancelledError:
        log.info("abort_consumer.cancelled", engagement_id=str(engagement_id))
        raise


def start_abort_consumer(
    engagement_id: UUID,
    *,
    runner: SandboxRunner,
    bus: EventBus | None = None,
    registry: InFlightContainerRegistry | None = None,
    timeout_s: int = 30,
) -> asyncio.Task[None]:
    """Spawn :func:`run_abort_consumer` as an asyncio task.

    Returned task can be ``cancel()``-ed by the worker on engagement
    teardown. Callers SHOULD await the cancelled task with
    ``with contextlib.suppress(asyncio.CancelledError):`` to avoid the
    "Task was destroyed but it is pending!" warning at shutdown.
    """
    return asyncio.create_task(
        run_abort_consumer(
            engagement_id,
            runner=runner,
            bus=bus,
            registry=registry,
            timeout_s=timeout_s,
        ),
        name=f"abort-consumer:{engagement_id}",
    )
