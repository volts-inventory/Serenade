"""async-06 regression — the worker entrypoint must actually dispatch
engagements.

Before the fix ``worker._run()`` only did ``await stop.wait()`` and never
invoked ``run_engagement``, so the whole runtime (agent graph, abort
consumer, network teardown, ``clear_abort``) was unreachable. These tests
drive the dispatch loop directly with a stubbed ``run_engagement`` and a
stubbed running-engagement query.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from worker import main as worker_main
from worker import runtime as worker_runtime


@pytest.fixture(autouse=True)
def _stub_has_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatch loop calls ``has_checkpoint`` (a real checkpointer probe)
    each time it considers dispatching a not-yet-inflight engagement. These
    tests stub ``run_engagement`` and the running-ids query but must also stub
    the checkpoint probe; otherwise the loop hits the real LangGraph SQLite
    checkpointer (no DB wired in these unit tests) and the loop wedges. Default
    to "no prior checkpoint" -> fresh dispatch (resume=False)."""

    async def _no_checkpoint(_engagement_id: object) -> bool:
        return False

    monkeypatch.setattr(worker_runtime, "has_checkpoint", _no_checkpoint)


@pytest.mark.asyncio
async def test_dispatch_loop_runs_each_running_engagement_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``running`` engagement is dispatched to ``run_engagement`` exactly
    once even though it stays in the poll result across multiple polls."""
    eid = uuid4()
    dispatched: list[str] = []
    started = asyncio.Event()

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        dispatched.append(str(engagement_id))
        started.set()
        # Stay alive until cancelled by the loop's shutdown path.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    async def _fake_running_ids():
        # Always report the same engagement as running.
        return [eid]

    # Patch the runtime import target + the poll helper. ``run_engagement``
    # is imported inside the loop, so patch it on the runtime module.
    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    # Wait for the first dispatch, then let a couple more polls elapse.
    await asyncio.wait_for(started.wait(), timeout=2.0)
    await asyncio.sleep(0.05)

    # Signal shutdown; the loop must cancel the in-flight engagement and
    # drain.
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # Dispatched exactly once despite repeated polls.
    assert dispatched == [str(eid)]


@pytest.mark.asyncio
async def test_dispatch_loop_logs_engagement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed engagement task is reaped and surfaced (the loop reads
    ``task.exception()`` rather than silently swallowing it)."""
    eid = uuid4()
    polled = {"n": 0}

    async def _boom(engagement_id, resume: bool = False):
        raise RuntimeError("engagement blew up")

    async def _running_ids_once():
        polled["n"] += 1
        # Report the engagement only on the first poll so it's dispatched
        # exactly once; subsequent polls return nothing.
        return [eid] if polled["n"] == 1 else []

    monkeypatch.setattr(worker_runtime, "run_engagement", _boom)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _running_ids_once)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    errors: list[tuple] = []
    real_error = worker_main.log.error
    monkeypatch.setattr(
        worker_main.log,
        "error",
        lambda *a, **k: errors.append((a, k)) or real_error(*a, **k),
    )

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    # Let it dispatch, fail, and reap on the next poll.
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    assert any(a and a[0] == "worker.engagement_failed" for a, _ in errors), (
        "a failed engagement must be logged loudly, not swallowed"
    )


# --------------------------------------------------------------------------- #
# asyncN-01 regression — a finished engagement must NOT be re-dispatched.     #
#                                                                             #
# The original async-06 dispatch fix had no terminal-status write, so once    #
# ``run_engagement`` RETURNED the row stayed ``running`` and the next poll    #
# re-dispatched it from a fresh state forever. These tests use a              #
# ``run_engagement`` stub that actually RETURNS (success / failure) and       #
# flips a shared fake "DB" via ``record_terminal_status``; the poll helper    #
# reads that same status. We assert the status becomes terminal and the       #
# dispatch count stays 1 across many polls.                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_completed_engagement_not_redispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    # Shared in-memory engagement status, starts ``running``.
    status = {eid: "running"}
    dispatched: list[str] = []

    async def _record_terminal_status(engagement_id, new_status):
        # Mirror the real helper's non-clobbering contract closely enough:
        # only move a still-``running`` row to a terminal state.
        if status.get(engagement_id) == "running":
            status[engagement_id] = new_status

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        # A REAL run that RETURNS: do work, then persist the terminal status
        # exactly like ``worker.runtime.run_engagement``'s finally does.
        dispatched.append(str(engagement_id))
        await _record_terminal_status(engagement_id, "completed")
        return object()  # an EngagementState stand-in; the loop ignores it

    async def _fake_running_ids():
        # The real poll selects ``status == 'running'`` rows.
        return [e for e, s in status.items() if s == "running"]

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    # Let many poll cycles elapse — far more than one.
    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # The run flipped the row to a terminal status...
    assert status[eid] == "completed"
    # ...and despite ~20 poll cycles it was dispatched exactly once.
    assert dispatched == [str(eid)]


@pytest.mark.asyncio
async def test_failed_engagement_marked_failed_and_not_redispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    status = {eid: "running"}
    dispatched: list[str] = []

    async def _record_terminal_status(engagement_id, new_status):
        if status.get(engagement_id) == "running":
            status[engagement_id] = new_status

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        # A run that RAISES: like the real runtime, persist ``failed`` in the
        # finally before the exception propagates, then re-raise.
        dispatched.append(str(engagement_id))
        try:
            raise RuntimeError("graph blew up")
        finally:
            await _record_terminal_status(engagement_id, "failed")

    async def _fake_running_ids():
        return [e for e, s in status.items() if s == "running"]

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # Failure recorded as ``failed`` (NOT left ``running``)...
    assert status[eid] == "failed"
    # ...and not re-dispatched on subsequent polls.
    assert dispatched == [str(eid)]


@pytest.mark.asyncio
async def test_loop_does_not_redispatch_even_if_status_write_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-braces: the terminal-status write is best-effort. Even if the
    row is somehow still reported ``running`` after the task finishes (lost
    write / race), the loop's in-process ``finished`` set must stop a
    re-dispatch from a fresh state."""
    eid = uuid4()
    dispatched: list[str] = []

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        # A genuinely COMPLETED run. Returns the state as a plain ``dict`` —
        # the shape ``graph.ainvoke`` ACTUALLY returns (async4-01) — so this
        # exercises the real dict-keyed terminal detection, not an
        # attribute-bearing stub that would mask the bug. It DELIBERATELY never
        # persisted its terminal status (lost write / race); the in-process
        # ``finished`` set must still stop a re-dispatch.
        dispatched.append(str(engagement_id))
        return {"completed": True}

    async def _fake_running_ids():
        # Always reports the engagement as running (the lost-write case).
        return [eid]

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # Dispatched exactly once even though the poll keeps returning it.
    assert dispatched == [str(eid)]


# --------------------------------------------------------------------------- #
# asyncN-01 — ``record_terminal_status`` itself flips the row and is          #
# non-clobbering (a terminal row wins over a late ``completed``/``paused``).  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_terminal_status_flips_running_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime, timedelta

    from orchestrator import crypto
    from orchestrator.models import AuditLogRow, EngagementRow
    from orchestrator.services.engagements import record_terminal_status
    from orchestrator.settings import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()

    now = datetime.now(UTC)
    row = EngagementRow(
        id=uuid4(),
        name="terminal-test",
        scope={"cidrs": ["10.0.0.0/24"], "domains": []},
        objectives=[],
        mode="guided",
        time_window_start=now - timedelta(hours=1),
        time_window_end=now + timedelta(hours=8),
        token_budget=1_000_000,
        wallclock_hours=8,
        authorization_artifact_sha256="a" * 64,
        per_tool_token_budget={},
        status="running",
    )

    class _Session:
        """AsyncSession stand-in: mutates the row in-place, supports the
        ledger audit path (``begin_nested`` SAVEPOINT + empty chain-tail
        read)."""

        def __init__(self, row):
            self._row = row
            self.added: list = []

        async def get(self, _model, pk):
            return self._row if pk == self._row.id else None

        def add(self, obj):
            self.added.append(obj)

        async def execute(self, *_a, **_k):
            class _Empty:
                def scalars(self_inner):
                    class _S:
                        def first(self_s):
                            return None

                    return _S()

            return _Empty()

        async def flush(self):
            return None

        def begin_nested(self):
            class _Savepoint:
                async def __aenter__(self_inner):
                    return None

                async def __aexit__(self_inner, *_exc):
                    return False

            return _Savepoint()

    session = _Session(row)

    @asynccontextmanager
    async def _scope():
        yield session

    monkeypatch.setattr("orchestrator.services.engagements.session_scope", _scope)

    # running -> completed
    await record_terminal_status(row.id, "completed")
    assert row.status == "completed"
    assert any(
        isinstance(r, AuditLogRow) and r.action == "engagement.terminal"
        for r in session.added
    )

    # Non-clobbering: an already-terminal row is NOT overwritten by a late
    # ``paused`` (operator abort must win over a late completion).
    row.status = "aborted"
    await record_terminal_status(row.id, "paused")
    assert row.status == "aborted"
