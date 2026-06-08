"""R3 #K4 Bug 2: the abort endpoint must actually kill in-flight work.

Background: ``POST /engagements/{id}/abort`` was stamping the row,
publishing the bus event, and writing the audit row — but the bus event
had no subscriber and ``SandboxRunner.terminate`` was never called.
Combined with the controller-only abort check (top of ``_step_agent``,
not inside the interactive loop), the documented "30s SIGTERM grace"
was actually "complete the current step then stop", which could be
many minutes for a recon step with ``interactive_llm_turn_cap=20``.

These tests pin down the K4 wire-up end-to-end:

  1. ``SandboxRunner.run`` registers its container into the shared
     in-flight registry on spawn and unregisters on exit (try/finally,
     so an exception path still cleans up).
  2. The abort consumer subscribes to ``engagement.abort_requested``;
     when one lands it terminates every in-flight container for that
     engagement in parallel with the documented 30s grace.
  3. Idempotency: a second abort publish for the same engagement is a
     no-op (the registry was drained by the first publish).
  4. Empty registry: an abort published with no in-flight work
     silently no-ops without errors.
  5. In-loop abort check: ``_check_abort_requested`` flips the ctx and
     ``_step_interactive`` returns ``ESCALATE_HUMAN`` with the
     ``aborted_mid_step`` note keyword on the next turn boundary.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.base import (
    StageContext,
    StepDecision,
    _check_abort_requested,
)
from orchestrator.abort_consumer import _terminate_all, run_abort_consumer
from sandbox.runner import (
    InFlightContainerRegistry,
    SandboxRunner,
    get_inflight_registry,
)
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# 1. SandboxRunner.run() registers + unregisters container on the registry.   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_runner_registers_container_during_run_and_unregisters_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry is the abort consumer's only handle on in-flight
    work. ``run()`` must register on spawn so the consumer can find
    the container, and unregister on exit so a re-spawn doesn't get
    stuck behind a stale entry."""
    eid = str(uuid4())
    registry = InFlightContainerRegistry()
    monkeypatch.setattr(
        "sandbox.runner.get_inflight_registry", lambda: registry
    )

    seen_during_run: set[str] = set()

    async def _fake_proc(*args: Any, **kwargs: Any) -> Any:
        # While we're "running" the tool, the registry should contain
        # exactly one entry for this engagement — the container name
        # ``run()`` just minted. Snapshot it here so we can prove the
        # registration window includes the actual work.
        engs = await registry.known_engagements()
        for e in engs:
            seen_during_run.update(
                await registry.snapshot_and_clear(e)
            )
            # Re-add so the finally clause sees something to discard.
            for c in seen_during_run:
                await registry.register(e, c)

        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"out", b""))
        proc.kill = MagicMock()
        proc.wait = AsyncMock(return_value=None)
        return proc

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _fake_proc
    )

    runner = SandboxRunner(engagement_id=eid)
    await runner.run(["echo", "hi"])

    # Registration was visible inside the run window.
    assert len(seen_during_run) == 1, (
        "Container should be registered while run() is executing"
    )
    name = next(iter(seen_during_run))
    assert name.startswith("serenade-sb-"), (
        "Container name follows the documented per-engagement pattern"
    )

    # And the finally clause cleared it.
    assert await registry.known_engagements() == set(), (
        "Container should be unregistered after run() returns"
    )


@pytest.mark.asyncio
async def test_runner_unregisters_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A docker subprocess that explodes must NOT leak a stale registry
    entry; the next abort would terminate a no-longer-running
    container, which would log a warning every time."""
    eid = str(uuid4())
    registry = InFlightContainerRegistry()
    monkeypatch.setattr(
        "sandbox.runner.get_inflight_registry", lambda: registry
    )

    async def _crashing_proc(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("docker daemon down")

    monkeypatch.setattr("asyncio.create_subprocess_exec", _crashing_proc)

    runner = SandboxRunner(engagement_id=eid)
    with pytest.raises(RuntimeError):
        await runner.run(["echo", "hi"])

    # The finally clause in run() must have cleared the registry.
    assert await registry.known_engagements() == set()


@pytest.mark.asyncio
async def test_runner_with_no_engagement_id_does_not_register() -> None:
    """A SandboxRunner constructed without an engagement_id (e.g. a
    cross-engagement health-check) must not pollute the registry —
    there's no engagement scope to attribute the container to and the
    abort consumer would never know to terminate it."""
    registry = InFlightContainerRegistry()
    # Direct register() call with engagement_id=None should be a no-op.
    await registry.register(None, "container-x")
    assert await registry.known_engagements() == set()


# --------------------------------------------------------------------------- #
# 2. Abort consumer: bus event triggers terminate() per in-flight container.  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_abort_consumer_terminates_every_inflight_container() -> None:
    """Two in-flight containers, one abort publish → terminate() called
    on BOTH (in parallel) with the documented 30s grace."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-A")
    await registry.register(str(eid), "container-B")

    terminate_calls: list[tuple[str, int]] = []

    class _Runner:
        async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
            terminate_calls.append((container_name, timeout_s))
            return True

    n = await _terminate_all(
        runner=_Runner(),  # type: ignore[arg-type]
        registry=registry,
        engagement_id=eid,
    )

    assert n == 2
    assert sorted(c for c, _ in terminate_calls) == ["container-A", "container-B"]
    # Per the K2 docstring: SIGTERM with 30s grace before SIGKILL.
    assert all(t == 30 for _, t in terminate_calls), (
        "terminate() must be called with timeout_s=30 (the documented grace)"
    )
    # Registry was drained.
    assert await registry.snapshot_and_clear(str(eid)) == set()


@pytest.mark.asyncio
async def test_abort_consumer_idempotent_on_double_publish() -> None:
    """A re-published abort (e.g. the operator double-clicked, or the
    bus replayed an old event) MUST NOT re-terminate already-killed
    containers. snapshot_and_clear is atomic so the second pass sees
    an empty set."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-X")

    terminate_calls: list[str] = []

    class _Runner:
        async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
            terminate_calls.append(container_name)
            return True

    # First publish — terminates the container.
    n1 = await _terminate_all(
        runner=_Runner(), registry=registry, engagement_id=eid  # type: ignore[arg-type]
    )
    assert n1 == 1

    # Second publish — registry is empty, must be a no-op.
    n2 = await _terminate_all(
        runner=_Runner(), registry=registry, engagement_id=eid  # type: ignore[arg-type]
    )
    assert n2 == 0
    # ``container-X`` was terminated exactly once across both publishes.
    assert terminate_calls.count("container-X") == 1


@pytest.mark.asyncio
async def test_abort_consumer_no_op_when_registry_empty() -> None:
    """A publish with no in-flight work (operator pressed the button
    just as the engagement landed at a clean checkpoint) is legitimate
    — log it and move on, never raise."""
    eid = uuid4()
    registry = InFlightContainerRegistry()

    class _Runner:
        async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
            raise AssertionError(
                "terminate() must NOT be called when there's nothing to kill"
            )

    n = await _terminate_all(
        runner=_Runner(), registry=registry, engagement_id=eid  # type: ignore[arg-type]
    )
    assert n == 0


@pytest.mark.asyncio
async def test_abort_consumer_subscribes_and_dispatches() -> None:
    """End-to-end: drive ``run_abort_consumer`` with a fake bus that
    yields one ``engagement.abort_requested`` event, then assert
    ``terminate()`` fired against the registered container."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-Z")

    # Fake bus that yields exactly one abort event and then stalls
    # (so the consumer doesn't exit prematurely — we cancel it).
    fired = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            yield {"kind": "engagement.abort_requested", "payload": {}}
            # Block forever after the one event so the cancel below is
            # what stops the loop (mirrors real Redis Streams behavior).
            fired.set()
            await asyncio.Event().wait()

    terminate_calls: list[str] = []

    class _Runner:
        async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
            terminate_calls.append(container_name)
            return True

    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=_Runner(),  # type: ignore[arg-type]
            bus=_FakeBus(),  # type: ignore[arg-type]
            registry=registry,
        )
    )
    # Wait until the consumer drained the bus event, then cancel.
    await asyncio.wait_for(fired.wait(), timeout=2.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert terminate_calls == ["container-Z"]


@pytest.mark.asyncio
async def test_abort_consumer_ignores_non_abort_events() -> None:
    """The bus carries many event kinds (stage.step.start, tool.run.*,
    etc). The consumer must filter to ``engagement.abort_requested``
    and ignore the rest, otherwise an innocent ``stage.step.start``
    would terminate the worker's own containers."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-Q")

    fired = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            yield {"kind": "stage.step.start", "payload": {}}
            yield {"kind": "agent.llm_turn", "payload": {}}
            fired.set()
            await asyncio.Event().wait()

    terminate_calls: list[str] = []

    class _Runner:
        async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
            terminate_calls.append(container_name)
            return True

    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=_Runner(),  # type: ignore[arg-type]
            bus=_FakeBus(),  # type: ignore[arg-type]
            registry=registry,
        )
    )
    await asyncio.wait_for(fired.wait(), timeout=2.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert terminate_calls == [], (
        "Non-abort events must not trigger terminate()"
    )
    # And the registry still holds the container — nothing killed it.
    assert "container-Q" in (
        await registry.snapshot_and_clear(str(eid))
    )


# --------------------------------------------------------------------------- #
# 3. In-loop abort check: ctx-level flag + DB-poll fallback.                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_check_abort_requested_fast_path_on_ctx_flag() -> None:
    """When ``ctx.should_abort`` is already True (the controller threaded
    it from state on a follow-up step entry), the helper must NOT touch
    the DB — it's the fast path."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
        should_abort=True,
        abort_reason="customer kill",
    )
    db_hit = False

    @asynccontextmanager
    async def _scope():
        nonlocal db_hit
        db_hit = True
        yield None

    # Even if the DB is unreachable, the fast path returns True.
    import orchestrator.db as _db
    _orig = _db.session_scope
    _db.session_scope = _scope  # type: ignore[assignment]
    try:
        assert await _check_abort_requested(ctx) is True
    finally:
        _db.session_scope = _orig
    assert db_hit is False, (
        "Fast path must skip the DB hit when ctx.should_abort is True"
    )


@pytest.mark.asyncio
async def test_check_abort_requested_polls_db_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ctx.should_abort is False but the engagement row has flipped
    to status='aborted' (an abort published mid-step), the helper must
    poll, hydrate the ctx, and return True."""
    eid = uuid4()
    ctx = StageContext(
        engagement_id=eid,
        engagement={},
        stage=Stage.RECON,
    )

    class _AbortedRow:
        status = "aborted"
        abort_reason = "polled-mid-step"

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return _AbortedRow()

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    monkeypatch.setattr("orchestrator.db.session_scope", _scope)

    assert await _check_abort_requested(ctx) is True
    # Helper hydrated the in-memory flag so the next call short-circuits.
    assert ctx.should_abort is True
    assert ctx.abort_reason == "polled-mid-step"


@pytest.mark.asyncio
async def test_check_abort_requested_returns_false_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row in status='running' returns False — the loop continues."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )

    class _RunningRow:
        status = "running"
        abort_reason = None

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return _RunningRow()

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    monkeypatch.setattr("orchestrator.db.session_scope", _scope)

    assert await _check_abort_requested(ctx) is False
    assert ctx.should_abort is False


@pytest.mark.asyncio
async def test_check_abort_requested_fails_open_on_db_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Postgres outage during the poll must NOT crash the agent loop
    — the next controller-step entry will retry. Helper logs and
    returns False."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )

    @asynccontextmanager
    async def _bad_scope():
        raise RuntimeError("postgres unreachable")
        yield None  # pragma: no cover

    monkeypatch.setattr("orchestrator.db.session_scope", _bad_scope)
    assert await _check_abort_requested(ctx) is False


# --------------------------------------------------------------------------- #
# 4. Interactive loop: aborted-mid-step short-circuit emits the marker note.  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_interactive_loop_exits_with_aborted_mid_step_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``ctx.should_abort=True`` flips between LLM turns, the next
    turn-boundary check inside ``_step_interactive`` must return
    ``ESCALATE_HUMAN`` with the ``aborted_mid_step`` marker in the
    note. Greppable by the report renderer + audit log."""
    from agents.base import BaseAgent
    from agents.recon import ReconAgent
    from llm.router import LLMResponse, ToolCall

    # Build a recon agent with mocked dependencies; we never want the
    # test to hit a real LLM or repo. We DO want a real ``_step_interactive``
    # so the in-loop check fires.
    repo = MagicMock()
    repo.upsert_node = AsyncMock()
    repo.upsert_edge = AsyncMock()
    repo.run_read_cypher = AsyncMock(return_value=[])
    repo.count = AsyncMock(return_value={})
    repo.get_nodes_by_kind = AsyncMock(return_value=[])

    router = MagicMock()

    # First turn returns a tool call so the loop doesn't exit on the
    # "no tool calls -> done" early-out. After that, the in-loop abort
    # check fires (we flip the flag below).
    call = ToolCall(name="nonexistent-tool", input={}, call_id="c1")
    router.complete = AsyncMock(return_value=LLMResponse(
        text="thinking", tool_calls=[call], structured=None,
        model="test", input_tokens=10, output_tokens=5,
        cached_input_tokens=0, cost_usd=0.001, latency_ms=1, raw={},
    ))

    agent = ReconAgent(
        router=router,
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )
    # Make _build_world_state_slice return something cheap and known.
    monkeypatch.setattr(
        ReconAgent, "_build_world_state_slice",
        AsyncMock(return_value={"hosts": []}),
    )
    # _handle_one_tool_call: bypass to a gated outcome so executed=False
    # and the loop hits the budget/abort check at the bottom.
    from agents.base import _ToolOutcome
    monkeypatch.setattr(
        BaseAgent, "_handle_one_tool_call",
        AsyncMock(return_value=_ToolOutcome(
            executed=False,
            nodes_written=0,
            feedback_text="unknown tool",
        )),
    )

    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
        should_abort=True,
        abort_reason="customer pressed kill",
    )

    result = await agent._step_interactive(ctx)
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert "aborted_mid_step" in result.notes, (
        f"Notes should carry the aborted_mid_step marker; got: {result.notes!r}"
    )
    assert "customer pressed kill" in result.notes


# --------------------------------------------------------------------------- #
# 5. Registry singleton: same instance across module-level callers.           #
# --------------------------------------------------------------------------- #


def test_inflight_registry_singleton_is_shared() -> None:
    """The runner's ``register`` and the consumer's ``terminate_all``
    must read from the SAME registry instance — otherwise the consumer
    sees an empty set and the kill switch silently fails."""
    a = get_inflight_registry()
    b = get_inflight_registry()
    assert a is b
