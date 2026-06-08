"""async3-01/02/03 regression — the pause/resume lifecycle of the worker
dispatch loop.

Background: the asyncN-01 fix added a process-lifetime ``finished`` set so a
completed/failed/aborted engagement is never re-dispatched from a fresh state
even if its durable terminal-status write lost a race. That fix OVER-corrected:

- async3-01 (high): the ``finished`` set also pinned ``paused`` engagements,
  so an operator ``resume`` (status -> ``running``) was silently ignored; and
  the loop only ever called ``run_engagement(resume=False)`` so even a
  re-dispatch would restart the graph from recon instead of continuing past
  the LangGraph interrupt.
- async3-02 (low): the ``finished`` set grew unbounded for the process
  lifetime.
- async3-03 (low): the per-engagement threshold lock + abort event leaked for
  every paused engagement (eviction was gated on ``not interrupted``).

These tests drive ``_dispatch_loop`` directly with stubbed ``run_engagement`` /
``has_checkpoint`` / poll helpers, and exercise the runtime teardown to assert
the lock + abort-event eviction now also runs on the interrupt/pause path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from worker import main as worker_main
from worker import runtime as worker_runtime


# --------------------------------------------------------------------------- #
# async3-01 — a paused engagement set back to ``running`` IS re-dispatched,    #
# and with ``resume=True`` because a checkpoint exists for it.                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_paused_then_resumed_is_redispatched_with_resume_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    status = {eid: "running"}
    # Record (engagement_id, resume) for every dispatch.
    dispatches: list[tuple[str, bool]] = []
    # Gate the operator "resume": flip back to running only after the first
    # run has paused.
    paused = asyncio.Event()
    resumed = asyncio.Event()

    async def _record_terminal_status(engagement_id, new_status):
        # Non-clobbering: only move a still-``running`` row.
        if status.get(engagement_id) == "running":
            status[engagement_id] = new_status

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        dispatches.append((str(engagement_id), resume))
        # First dispatch: the graph interrupts -> ``paused`` (NOT terminal).
        # Second dispatch (after resume): completes.
        if not resume:
            await _record_terminal_status(engagement_id, "paused")
            paused.set()
            # Returning a state with ``completed`` falsy mimics an interrupt.
            return SimpleNamespace(completed=False, should_abort=False)
        await _record_terminal_status(engagement_id, "completed")
        resumed.set()
        return SimpleNamespace(completed=True, should_abort=False)

    async def _fake_running_ids():
        return [e for e, s in status.items() if s == "running"]

    async def _fake_has_checkpoint(engagement_id):
        # A checkpoint exists once the engagement has paused at least once.
        return status.get(engagement_id) == "running" and paused.is_set()

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_runtime, "has_checkpoint", _fake_has_checkpoint)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    # Wait for the first run to pause and persist ``paused``.
    await asyncio.wait_for(paused.wait(), timeout=2.0)
    # The row must NOT be pinned in ``finished`` — a paused run is re-dispatchable.
    assert status[eid] == "paused"

    # Operator resume: flip the row back to ``running``.
    status[eid] = "running"

    # The loop must re-select it and dispatch with ``resume=True``.
    await asyncio.wait_for(resumed.wait(), timeout=2.0)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    assert status[eid] == "completed"
    # Exactly two dispatches: the first a fresh run, the second a resume.
    assert len(dispatches) == 2, dispatches
    assert dispatches[0] == (str(eid), False)
    assert dispatches[1] == (str(eid), True)


# --------------------------------------------------------------------------- #
# async3-01 — an ABORTED run (state.completed=True via abort) is TERMINAL:     #
# pinned in ``finished`` and never re-dispatched even on a lost-write poll.    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aborted_engagement_not_redispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    dispatched: list[str] = []

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        dispatched.append(str(engagement_id))
        # An abort sets BOTH should_abort and completed; the row is ``aborted``.
        # DELIBERATELY never records terminal status -> exercise the
        # ``finished`` belt-and-braces guard.
        return SimpleNamespace(completed=True, should_abort=True)

    async def _fake_running_ids():
        # Lost-write: always report it running.
        return [eid]

    async def _fake_has_checkpoint(engagement_id):
        # Even if a checkpoint somehow exists, a terminal abort must not re-run.
        return True

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_runtime, "has_checkpoint", _fake_has_checkpoint)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.01)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # Dispatched exactly once across ~20 polls — the abort is terminal.
    assert dispatched == [str(eid)]


# --------------------------------------------------------------------------- #
# async3-01 — ``_task_is_terminal`` classifies each finish correctly.         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_task_is_terminal_classification() -> None:
    # completed/aborted (state.completed truthy) -> terminal. Returns are
    # plain dicts — the shape ``graph.ainvoke`` actually produces (async4-01) —
    # so this exercises dict-keyed detection, not an attribute-bearing stub.
    async def _completed():
        return {"completed": True, "should_abort": False}

    async def _aborted():
        return {"completed": True, "should_abort": True}

    # paused (state.completed falsy / None) -> NOT terminal.
    async def _paused():
        return {"completed": False, "should_abort": False}

    async def _paused_none():
        return None

    # raised -> terminal (failed).
    async def _failed():
        raise RuntimeError("boom")

    # cancelled (worker shutdown) -> NOT terminal.
    async def _forever():
        await asyncio.Event().wait()

    t_completed = asyncio.create_task(_completed())
    t_aborted = asyncio.create_task(_aborted())
    t_paused = asyncio.create_task(_paused())
    t_paused_none = asyncio.create_task(_paused_none())
    t_failed = asyncio.create_task(_failed())
    await asyncio.gather(
        t_completed, t_aborted, t_paused, t_paused_none, t_failed,
        return_exceptions=True,
    )

    t_cancelled = asyncio.create_task(_forever())
    await asyncio.sleep(0)
    t_cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t_cancelled

    assert worker_main._task_is_terminal(t_completed) is True
    assert worker_main._task_is_terminal(t_aborted) is True
    assert worker_main._task_is_terminal(t_failed) is True
    assert worker_main._task_is_terminal(t_paused) is False
    assert worker_main._task_is_terminal(t_paused_none) is False
    assert worker_main._task_is_terminal(t_cancelled) is False


# --------------------------------------------------------------------------- #
# async3-02 — the ``finished`` set is bounded; it does NOT grow without limit  #
# across many engagements. With the cap lowered, the OLDEST terminal ids are   #
# evicted (and would be re-dispatched only if still reported ``running``),     #
# while RECENT terminal ids stay pinned.                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_finished_set_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = 4
    monkeypatch.setattr(worker_main, "_FINISHED_CAP", cap)

    # ``n_total`` distinct engagements all complete (terminal). The poll keeps
    # reporting EVERY id as running (lost-write for all) so the ONLY thing that
    # can stop a re-dispatch is the bounded ``finished`` set.
    n_total = cap + 3  # 7 ids, cap 4 -> 3 oldest must be evictable
    ids = [uuid4() for _ in range(n_total)]
    # Dispatch them in a deterministic order, one new id per poll, so eviction
    # order (FIFO) is predictable.
    dispatch_counts: dict[str, int] = {}
    # ``available`` controls which ids the poll reports; we feed them in waves.
    available: list = []
    wave = {"i": 0}

    async def _fake_run_engagement(engagement_id, resume: bool = False):
        key = str(engagement_id)
        dispatch_counts[key] = dispatch_counts.get(key, 0) + 1
        return SimpleNamespace(completed=True, should_abort=False)

    async def _fake_running_ids():
        # Reveal one more id per poll (a "wave"); once all are revealed, keep
        # reporting all of them as running (the lost-write storm).
        i = wave["i"]
        if i < n_total:
            available.append(ids[i])
            wave["i"] = i + 1
        return list(available)

    async def _fake_has_checkpoint(engagement_id):
        return False

    monkeypatch.setattr(worker_runtime, "run_engagement", _fake_run_engagement)
    monkeypatch.setattr(worker_runtime, "has_checkpoint", _fake_has_checkpoint)
    monkeypatch.setattr(worker_main, "_running_engagement_ids", _fake_running_ids)
    monkeypatch.setattr(worker_main, "_POLL_INTERVAL_S", 0.005)

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker_main._dispatch_loop(stop))

    # Let plenty of polls run so every id is revealed, completed, and the cap
    # is exceeded -> the oldest ids get evicted and (still reported running)
    # re-dispatched, proving the set didn't grow without bound.
    await asyncio.sleep(0.4)
    stop.set()
    await asyncio.wait_for(loop_task, timeout=2.0)

    # The earliest-finished ids exceeded the cap and were evicted, so they were
    # re-dispatched (>1). At least (n_total - cap) ids must have been
    # re-dispatched — that only happens if ``finished`` is bounded.
    redispatched = [k for k, c in dispatch_counts.items() if c > 1]
    assert len(redispatched) >= (n_total - cap), (
        f"expected >= {n_total - cap} ids evicted+re-dispatched (bounded set); "
        f"got {len(redispatched)}: {dispatch_counts}"
    )


# --------------------------------------------------------------------------- #
# async3-03 — pause/interrupt releases the per-engagement threshold lock AND   #
# the abort event (eviction is no longer gated on ``not interrupted``).       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pause_releases_threshold_lock_and_abort_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive ``run_engagement`` to the interrupt/pause path and assert both the
    threshold lock and the abort-event entry are evicted in its finally.

    We stub everything heavy (network, bundle, graph, checkpointer, abort
    consumer, world-state, terminal-status write) so only the lifecycle
    teardown runs. The graph returns a state with ``completed`` falsy ->
    ``interrupted=True`` -> the OLD code skipped eviction; the fixed code
    evicts unconditionally.
    """
    from orchestrator import abort_consumer, thresholds

    eid = uuid4()

    # Pre-populate the registries as if the run had touched them.
    thresholds._threshold_lock(eid)
    abort_consumer.request_abort(eid)
    assert str(eid) in thresholds._THRESHOLD_LOCKS
    assert str(eid) in abort_consumer._abort_events

    # --- Stub the runtime's heavy dependencies. ---
    async def _no_session_scope():  # pragma: no cover - not used directly
        raise AssertionError

    class _FakeSessionCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, _model, _pk):
            # Minimal engagement row stand-in for build_engagement_context.
            return SimpleNamespace(
                id=eid,
                llm_config_snapshot={},
            )

    def _fake_session_scope():
        return _FakeSessionCtx()

    monkeypatch.setattr("orchestrator.db.session_scope", _fake_session_scope)
    monkeypatch.setattr(
        "policy.engagement_dict.build_engagement_context",
        lambda row: {"engagement_id": str(eid)},
    )

    # Abort consumer: return a dummy cancellable task.
    async def _noop_consumer():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    def _fake_start_abort_consumer(engagement_id, *, runner):
        return asyncio.create_task(_noop_consumer())

    monkeypatch.setattr(
        "orchestrator.abort_consumer.start_abort_consumer",
        _fake_start_abort_consumer,
    )

    async def _fake_ensure_network(engagement_id):
        return f"net-{engagement_id}"

    async def _fake_teardown_network(engagement_id):
        return None

    monkeypatch.setattr(
        "sandbox.network.ensure_engagement_network", _fake_ensure_network
    )
    monkeypatch.setattr(
        "sandbox.network.teardown_engagement_network", _fake_teardown_network
    )

    # Build bundle returns a fake repo/policy with async close methods.
    class _Closeable:
        async def close(self):
            return None

        async def aclose(self):
            return None

    async def _fake_build_bundle(*a, **k):
        return object(), _Closeable(), _Closeable()

    monkeypatch.setattr(worker_runtime, "build_bundle", _fake_build_bundle)

    # Checkpointer disabled -> build_graph gets None; stub build_graph + the
    # graph so ainvoke returns a "paused" (interrupt) result.
    async def _fake_get_checkpointer():
        raise RuntimeError("checkpoint disabled in test")

    monkeypatch.setattr(
        "orchestrator.checkpoint.get_checkpointer", _fake_get_checkpointer
    )

    class _FakeGraph:
        async def ainvoke(self, initial, config=None):
            # Interrupt: a state with ``completed`` falsy.
            return SimpleNamespace(completed=False, should_abort=False)

    # ``build_graph`` is imported at module top into ``worker.runtime``'s
    # namespace, so patch the name bound there (patching the source module
    # wouldn't affect the already-bound reference).
    monkeypatch.setattr(
        worker_runtime, "build_graph", lambda bundle, checkpointer=None: _FakeGraph()
    )

    # Terminal-status write -> no-op (don't touch the DB).
    async def _fake_record_terminal_status(engagement_id, status):
        return None

    monkeypatch.setattr(
        "orchestrator.services.engagements.record_terminal_status",
        _fake_record_terminal_status,
    )

    # ``EngagementState`` construction inside run_engagement uses the real
    # class; the fresh-run branch builds one then calls graph.ainvoke. That's
    # fine — our fake graph ignores it.

    final = await worker_runtime.run_engagement(eid, resume=False)

    # The run paused (interrupt): completed falsy.
    assert final is not None and final.completed is False

    # async3-03: BOTH the threshold lock and the abort event were evicted on
    # the pause/interrupt path (previously skipped).
    assert str(eid) not in thresholds._THRESHOLD_LOCKS, (
        "threshold lock leaked on the pause/interrupt path"
    )
    assert str(eid) not in abort_consumer._abort_events, (
        "abort event leaked on the pause/interrupt path"
    )
