"""R3 #K5 Bug 1: close the abort-listener startup window.

Pre-fix, ``start_abort_consumer`` was spawned in ``run_engagement``
AFTER ``ensure_engagement_network`` and ``ensure_templates_fresh``. That
setup work can take seconds (docker network create, nuclei +
searchsploit DB sync). The subscriber subscribes with ``from_id="$"``
(NEW messages only), so any abort published in the gap was silently
dropped on the bus — only the controller's top-of-step DB poll caught
it on first step entry. That was a degraded-but-not-broken path; this
file pins the cleanup.

Fix (option A + C):
  - Option A: spawn the consumer BEFORE the setup work in
    ``worker/runtime.py``. Most of the gap evaporates.
  - Option C: inside ``run_abort_consumer``, before entering the
    subscribe loop, do a one-time DB poll for
    ``engagement.status == 'aborted'``. If true, fire
    ``_terminate_all`` immediately. Closes the residual ms-window
    between consumer spawn and first ``xread`` call.

These tests pin:
  1. Option C: when the engagement is ALREADY aborted at consumer
     startup, ``_terminate_all`` fires once before the subscribe loop
     touches Redis.
  2. Option C is idempotent: the subsequent bus delivery sees a
     drained registry (``snapshot_and_clear`` is atomic) and no-ops.
  3. Option A coverage: an abort published shortly after the consumer
     starts is still picked up by the bus subscriber (no regression).
  4. Option C is best-effort: a DB outage during the startup poll is
     logged and the subscribe loop still starts.
  5. Option A: ``run_engagement`` in ``worker/runtime.py`` starts the
     abort consumer BEFORE ``ensure_engagement_network``.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest

from orchestrator.abort_consumer import (
    _poll_engagement_status_aborted,
    run_abort_consumer,
)
from sandbox.runner import InFlightContainerRegistry


# --------------------------------------------------------------------------- #
# Shared fixtures                                                              #
# --------------------------------------------------------------------------- #


class _FakeRunner:
    def __init__(self) -> None:
        self.terminate_calls: list[tuple[str, int]] = []

    async def terminate(self, container_name: str, *, timeout_s: int = 30) -> bool:
        self.terminate_calls.append((container_name, timeout_s))
        return True


def _make_aborted_session_scope() -> Any:
    """Build a session_scope contextmanager that yields a session whose
    ``get`` returns an EngagementRow-like with ``status='aborted'``."""

    class _AbortedRow:
        status = "aborted"
        abort_reason = "kill-switch"

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return _AbortedRow()

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    return _scope


def _make_running_session_scope() -> Any:
    class _RunningRow:
        status = "running"
        abort_reason = None

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return _RunningRow()

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    return _scope


# --------------------------------------------------------------------------- #
# 1. Option C: pre-existing aborted status fires terminate at startup.        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_consumer_fires_terminate_when_engagement_already_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engagement row was flipped to status='aborted' BEFORE
    ``start_abort_consumer`` was called (i.e. the abort landed during
    the worker's setup window). When the consumer reaches the subscribe
    loop, option C's one-time DB poll must trigger ``_terminate_all``
    so the in-flight container gets killed.
    """
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-pre-abort")

    monkeypatch.setattr(
        "orchestrator.db.session_scope", _make_aborted_session_scope()
    )

    fired = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            # The startup poll fires BEFORE we get here. Set the marker
            # so the test knows the subscribe loop has begun, then
            # block until cancelled.
            fired.set()
            await asyncio.Event().wait()
            yield {"kind": "_unreachable", "payload": {}}

    runner = _FakeRunner()
    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=runner,  # type: ignore[arg-type]
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

    assert runner.terminate_calls == [("container-pre-abort", 30)], (
        "option C: pre-existing aborted status MUST trigger terminate "
        "with the documented 30s grace before the subscribe loop"
    )
    # Registry was drained by ``snapshot_and_clear``.
    assert await registry.snapshot_and_clear(str(eid)) == set()


@pytest.mark.asyncio
async def test_consumer_startup_poll_is_idempotent_with_bus_redelivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-braces: after option C fires terminate at startup, the
    eventual bus delivery of the same abort MUST be a no-op because
    ``snapshot_and_clear`` already drained the registry. Otherwise we'd
    double-terminate (and warn about already-dead containers)."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-once")

    monkeypatch.setattr(
        "orchestrator.db.session_scope", _make_aborted_session_scope()
    )

    delivered = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            # Simulate Redis Streams replaying the abort that originally
            # landed during the startup gap.
            yield {"kind": "engagement.abort_requested", "payload": {}}
            delivered.set()
            await asyncio.Event().wait()

    runner = _FakeRunner()
    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=runner,  # type: ignore[arg-type]
            bus=_FakeBus(),  # type: ignore[arg-type]
            registry=registry,
        )
    )
    await asyncio.wait_for(delivered.wait(), timeout=2.0)
    # Give the consumer a tick to process the event.
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Terminated EXACTLY ONCE despite both option-C and the bus event
    # firing — the registry was drained by the first pass.
    assert runner.terminate_calls == [("container-once", 30)], (
        "Option C + bus redelivery must be idempotent: terminate fires "
        "exactly once across both paths"
    )


# --------------------------------------------------------------------------- #
# 2. Option A coverage: aborts published AFTER startup still flow through.    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_consumer_catches_abort_published_shortly_after_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status was 'running' at startup → option C does nothing.
    A subsequent abort published on the bus (the normal path after
    option A moved the consumer start earlier) is still caught by the
    subscribe loop."""
    eid = uuid4()
    registry = InFlightContainerRegistry()
    await registry.register(str(eid), "container-late-abort")

    monkeypatch.setattr(
        "orchestrator.db.session_scope", _make_running_session_scope()
    )

    fired = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            # Simulate a small ms-window then an abort lands.
            yield {"kind": "engagement.abort_requested", "payload": {}}
            fired.set()
            await asyncio.Event().wait()

    runner = _FakeRunner()
    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=runner,  # type: ignore[arg-type]
            bus=_FakeBus(),  # type: ignore[arg-type]
            registry=registry,
        )
    )
    await asyncio.wait_for(fired.wait(), timeout=2.0)
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert runner.terminate_calls == [("container-late-abort", 30)], (
        "An abort published after consumer startup must still terminate "
        "the in-flight container via the bus subscriber"
    )


# --------------------------------------------------------------------------- #
# 3. Option C fails open on DB outage.                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_startup_poll_logs_and_returns_false_on_db_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Postgres outage during the startup poll must not crash the
    consumer — it logs, returns False, and the subscribe loop still
    enters. The controller's top-of-step DB poll is the longstanding
    safety net for aborts that landed in this case."""
    eid = uuid4()

    @asynccontextmanager
    async def _bad_scope():
        raise RuntimeError("postgres unreachable")
        yield None  # pragma: no cover

    monkeypatch.setattr("orchestrator.db.session_scope", _bad_scope)

    # The helper itself must return False on outage.
    assert await _poll_engagement_status_aborted(eid) is False


@pytest.mark.asyncio
async def test_consumer_proceeds_to_subscribe_when_startup_poll_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when the startup DB poll explodes, the consumer must still
    enter the subscribe loop (the bus is the primary kill-switch path
    once the engagement is live)."""
    eid = uuid4()
    registry = InFlightContainerRegistry()

    @asynccontextmanager
    async def _bad_scope():
        raise RuntimeError("postgres unreachable")
        yield None  # pragma: no cover

    monkeypatch.setattr("orchestrator.db.session_scope", _bad_scope)

    entered = asyncio.Event()

    class _FakeBus:
        async def subscribe(self, _engagement_id, *, from_id="$"):  # type: ignore[no-untyped-def]
            entered.set()
            await asyncio.Event().wait()
            yield {"kind": "_unreachable", "payload": {}}

    runner = _FakeRunner()
    task = asyncio.create_task(
        run_abort_consumer(
            eid,
            runner=runner,  # type: ignore[arg-type]
            bus=_FakeBus(),  # type: ignore[arg-type]
            registry=registry,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert runner.terminate_calls == [], (
        "DB outage during startup poll must not trigger terminate"
    )


@pytest.mark.asyncio
async def test_startup_poll_returns_false_for_running_engagement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: a 'running' engagement returns False from the helper
    (no spurious terminate on startup for healthy engagements)."""
    eid = uuid4()
    monkeypatch.setattr(
        "orchestrator.db.session_scope", _make_running_session_scope()
    )
    assert await _poll_engagement_status_aborted(eid) is False


# --------------------------------------------------------------------------- #
# 4. Source-level invariant: option A wired into worker/runtime.py.            #
# --------------------------------------------------------------------------- #


def test_runtime_starts_abort_consumer_before_setup_work() -> None:
    """Option A: in ``worker/runtime.py:run_engagement`` the call to
    ``start_abort_consumer`` must appear textually BEFORE the call to
    ``ensure_engagement_network`` and ``ensure_templates_fresh``. If a
    future refactor moves the consumer back AFTER the setup work, the
    startup gap reopens — this test catches that regression at import
    time."""
    from worker import runtime as _rt

    src = inspect.getsource(_rt.run_engagement)
    start_idx = src.find("start_abort_consumer(")
    network_idx = src.find("ensure_engagement_network(")
    templates_idx = src.find("ensure_templates_fresh(")
    assert start_idx != -1, "start_abort_consumer call not found"
    assert network_idx != -1, "ensure_engagement_network call not found"
    assert templates_idx != -1, "ensure_templates_fresh call not found"
    assert start_idx < network_idx, (
        "option A regression: start_abort_consumer must appear BEFORE "
        "ensure_engagement_network so the bus subscriber catches aborts "
        "published during network setup"
    )
    assert start_idx < templates_idx, (
        "option A regression: start_abort_consumer must appear BEFORE "
        "ensure_templates_fresh so the bus subscriber catches aborts "
        "published during the (slow) template refresh"
    )


@pytest.mark.asyncio
async def test_startup_poll_returns_true_for_aborted_engagement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lower-level helper test: when the row is status='aborted', the
    helper returns True regardless of consumer wiring."""
    eid = uuid4()
    monkeypatch.setattr(
        "orchestrator.db.session_scope", _make_aborted_session_scope()
    )
    assert await _poll_engagement_status_aborted(eid) is True
