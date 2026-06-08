"""Worker process entry point.

The worker is the process that actually *drives* engagements: it polls the
engagement table for rows in ``status='running'`` and dispatches each one to
``worker.runtime.run_engagement`` as a concurrent asyncio task. Without this
dispatch loop the runtime (the agent graph, abort consumer, network
teardown, ``clear_abort``) is unreachable — flipping an engagement row to
``running`` via the API would never spawn any work (async-06).

The loop:

  * polls every ``_POLL_INTERVAL_S`` for ``running`` engagements;
  * spawns ``run_engagement`` exactly once per engagement id (tracked in
    ``_inflight``) so a row that stays ``running`` across polls isn't
    double-dispatched;
  * reaps finished tasks, logging failures loudly rather than swallowing
    them;
  * on SIGINT/SIGTERM sets ``stop`` and cancels every in-flight engagement
    task, awaiting them so containers/networks tear down cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections import OrderedDict
from uuid import UUID

import structlog

log = structlog.get_logger()

# How often the dispatch loop polls the engagement table for new
# ``running`` rows. Cheap indexed query; the agent loop itself is the
# expensive part and runs in its own task.
_POLL_INTERVAL_S = 2.0


async def _running_engagement_ids() -> list[UUID]:
    """Return ids of engagements currently in ``status='running'``.

    Best-effort: a DB error logs and yields an empty list so a transient
    outage doesn't kill the dispatch loop.
    """
    try:
        from sqlalchemy import select  # noqa: PLC0415

        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import EngagementRow  # noqa: PLC0415

        async with session_scope() as s:
            rows = (
                (
                    await s.execute(
                        select(EngagementRow.id).where(EngagementRow.status == "running")
                    )
                )
                .scalars()
                .all()
            )
        return [r if isinstance(r, UUID) else UUID(str(r)) for r in rows]
    except Exception as e:
        log.warning("worker.poll_failed", err=str(e))
        return []


def _task_is_terminal(task: asyncio.Task[object]) -> bool:
    """Decide whether a finished task ended in a TERMINAL outcome.

    Terminal = ``completed`` / ``failed`` / ``aborted`` (pin in ``finished``,
    never re-dispatch). NON-terminal = ``paused`` / cancelled-on-shutdown
    (stay re-dispatchable so an operator ``resume`` works — async3-01).

    Decided from the task's own result, NOT a DB re-read, so the belt-and-
    braces guard still holds even if the durable terminal write was lost:

      * raised  -> ``failed``  -> terminal.
      * cancelled (worker shutdown) -> treated like a pause -> NOT terminal.
      * returned a state with ``completed`` truthy -> ``completed``/``aborted``
        (an abort sets both ``should_abort`` and ``completed``) -> terminal.
      * returned ``None`` / a state with ``completed`` falsy -> the graph
        interrupted (``paused``) -> NOT terminal.
    """
    from worker.runtime import state_flag  # noqa: PLC0415

    if task.cancelled():
        return False
    if task.exception() is not None:
        return True
    result = task.result()
    if result is None:
        return False
    # ``graph.ainvoke`` returns the final state as a plain dict, so read flags
    # via ``state_flag`` (dict-or-attribute) — ``getattr`` alone always saw
    # False and mis-classified every completed run as paused (async4-01).
    # ``should_abort`` is OR-ed in defensively (async5-01): an aborted run is
    # terminal even on the off chance the controller didn't also set
    # ``completed`` alongside it.
    return state_flag(result, "completed") or state_flag(result, "should_abort")


# Cap the in-process ``finished`` guard so it can't grow without bound for the
# worker's process lifetime (async3-02). It is only a belt-and-braces guard
# for the rare lost-terminal-write race; the durable terminal status is the
# real source of truth, so evicting the oldest ids when the cap is hit is
# safe (a re-dispatch of an evicted-but-truly-terminal id is then blocked by
# the ``running`` poll no longer selecting it).
_FINISHED_CAP = 4096


async def _dispatch_loop(stop: asyncio.Event) -> None:
    """Poll for ``running`` engagements and dispatch each one exactly once.

    Idempotency has two layers (asyncN-01):

      * ``run_engagement`` persists a terminal/parked status in its finally
        (``completed``/``failed``/``aborted``/``paused``), so a finished
        engagement drops out of the ``running`` poll result and is not
        re-selected.
      * The loop *also* records every id whose task finished in a TERMINAL
        status this process-lifetime in ``finished`` and refuses to
        re-dispatch it, so even if the terminal write lost a race or failed
        (it is best-effort) a completed/failed/aborted run is never re-run
        from a fresh state.

    Pause/resume (async3-01): a ``paused`` run is NOT terminal, so it is
    never pinned in ``finished``. When an operator resumes it
    (``status``→``running``) the next poll re-selects it, and because a
    LangGraph checkpoint exists for it the loop dispatches with
    ``resume=True`` so the graph continues past the interrupt instead of
    restarting from recon.

    ``finished`` is bounded (``_FINISHED_CAP``) so it can't grow without
    limit (async3-02); the durable terminal status remains the source of
    truth so evicting old ids is safe.
    """
    from worker.runtime import has_checkpoint, run_engagement  # noqa: PLC0415

    inflight: dict[UUID, asyncio.Task[object]] = {}
    # OrderedDict used as a bounded FIFO set of terminally-finished ids.
    finished: OrderedDict[UUID, None] = OrderedDict()

    def _mark_finished(eid: UUID) -> None:
        finished[eid] = None
        finished.move_to_end(eid)
        while len(finished) > _FINISHED_CAP:
            finished.popitem(last=False)

    try:
        while not stop.is_set():
            # Reap completed tasks; log failures rather than swallow them.
            # Only pin an id in ``finished`` if its row reached a TERMINAL
            # status — a ``paused`` run must stay re-dispatchable on resume
            # (async3-01).
            for eid in [e for e, t in inflight.items() if t.done()]:
                task = inflight.pop(eid)
                exc = task.exception() if not task.cancelled() else None
                if exc is not None:
                    log.error("worker.engagement_failed", engagement_id=str(eid), err=str(exc))
                else:
                    log.info("worker.engagement_finished", engagement_id=str(eid))
                if _task_is_terminal(task):
                    _mark_finished(eid)

            for eid in await _running_engagement_ids():
                if eid in inflight or eid in finished:
                    continue
                # Resume past a prior interrupt/pause when a checkpoint exists
                # for this engagement; otherwise start a fresh run (async3-01).
                resume = await has_checkpoint(eid)
                log.info("worker.dispatch", engagement_id=str(eid), resume=resume)
                inflight[eid] = asyncio.create_task(
                    run_engagement(eid, resume=resume), name=f"engagement:{eid}"
                )

            # Wait for the poll interval OR an early stop signal.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=_POLL_INTERVAL_S)
    finally:
        # Shutdown: cancel every in-flight engagement and await teardown so
        # docker networks/containers and ``clear_abort`` run.
        for task in inflight.values():
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight.values(), return_exceptions=True)
        log.info("worker.dispatch_drained", count=len(inflight))


async def _run() -> None:
    log.info("worker.start")
    stop = asyncio.Event()

    def _shutdown() -> None:
        log.info("worker.shutdown_signal")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await _dispatch_loop(stop)
    log.info("worker.stop")


def run() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    run()
