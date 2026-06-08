"""R3 #M2: threshold-tripped warnings for solo-operator observability.

Covers:
- Token budget at 80% emits ``engagement.budget_warning`` once; the next
  step that's still over the threshold does NOT re-emit (debounced via
  ``warnings_fired``).
- Wallclock window past 80% elapsed emits the same kind with kind="wallclock".
- A rolling buffer of 10 critic verdicts where 6 are vetoes emits
  ``engagement.veto_rate_warning`` exactly once.
- New tool suppression during a stage emits ``engagement.suppression_spread``
  the first time the active suppressed-tool count rises; subsequent steps
  with the same count don't re-emit.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid4

import pytest

from orchestrator import thresholds
from orchestrator.stage_controller import EngagementState
from worldstate.schema import Stage


class _FakeBus:
    """In-memory event-bus stand-in. Captures every ``publish`` call so
    tests can assert on kind/payload + count."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish(self, *, engagement_id: UUID, kind: str, payload: dict) -> str:
        self.events.append({"engagement_id": engagement_id, "kind": kind, "payload": payload})
        return f"1-{len(self.events)}"


@pytest.fixture
def fake_bus(monkeypatch: pytest.MonkeyPatch) -> _FakeBus:
    bus = _FakeBus()
    monkeypatch.setattr(thresholds, "get_bus", lambda: bus)
    return bus


@pytest.fixture
def no_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the audit writer so tests don't try to open Postgres."""

    async def _noop(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(thresholds, "audit", _noop)


def _state(**overrides: Any) -> EngagementState:
    """Construct an EngagementState with sensible defaults for threshold tests."""
    base: dict[str, Any] = dict(
        engagement_id=uuid4(),
        engagement={
            "token_budget": 1_000_000,
            "window": {},
            "approval_policy": "manual",
        },
        current_stage=Stage.RECON,
    )
    base.update(overrides)
    return EngagementState(**base)


# --------------------------------------------------------------------------- #
# Token budget                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_token_budget_warning_fires_once(fake_bus: _FakeBus, no_audit: None) -> None:
    """Token budget 1M, current 800K -> next call past 800K emits;
    re-evaluation does NOT re-emit."""
    state = _state()
    state.total_tokens = 800_001  # 80.0001% — just over the 80% threshold

    await thresholds.check_thresholds(state)

    budget_events = [e for e in fake_bus.events if e["kind"] == "engagement.budget_warning"]
    assert len(budget_events) == 1
    assert budget_events[0]["payload"]["kind"] == "token"
    assert budget_events[0]["payload"]["usage"] >= 0.8
    assert "budget:token" in state.warnings_fired

    # Push tokens higher and re-check — debounced, no new event.
    state.total_tokens = 950_000
    await thresholds.check_thresholds(state)
    budget_events = [e for e in fake_bus.events if e["kind"] == "engagement.budget_warning"]
    assert len(budget_events) == 1  # still one


@pytest.mark.asyncio
async def test_token_budget_below_threshold_does_not_warn(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    state = _state()
    state.total_tokens = 500_000  # 50%
    await thresholds.check_thresholds(state)
    assert all(e["kind"] != "engagement.budget_warning" for e in fake_bus.events)
    assert "budget:token" not in state.warnings_fired


@pytest.mark.asyncio
async def test_token_budget_zero_is_unlimited(fake_bus: _FakeBus, no_audit: None) -> None:
    """token_budget=0 means unlimited (local backends); never warns."""
    state = _state(engagement={"token_budget": 0, "window": {}})
    state.total_tokens = 10_000_000_000
    await thresholds.check_thresholds(state)
    assert all(e["kind"] != "engagement.budget_warning" for e in fake_bus.events)


# --------------------------------------------------------------------------- #
# Wallclock                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wallclock_warning_fires_once(fake_bus: _FakeBus, no_audit: None) -> None:
    """80% of wallclock window elapsed -> warns with kind=wallclock."""
    now_ns = time.time_ns()
    # Window of 100 minutes; we're 81 minutes in
    one_min = 60_000_000_000
    start = now_ns - 81 * one_min
    end = start + 100 * one_min
    state = _state(
        engagement={
            "token_budget": 0,  # disable token branch so we isolate wallclock
            "window": {"start_ns": start, "end_ns": end},
        }
    )
    await thresholds.check_thresholds(state)

    wall_events = [
        e
        for e in fake_bus.events
        if e["kind"] == "engagement.budget_warning" and e["payload"]["kind"] == "wallclock"
    ]
    assert len(wall_events) == 1
    assert wall_events[0]["payload"]["usage"] >= 0.8
    assert "budget:wallclock" in state.warnings_fired

    # Re-check: debounced.
    await thresholds.check_thresholds(state)
    wall_events = [
        e
        for e in fake_bus.events
        if e["kind"] == "engagement.budget_warning" and e["payload"]["kind"] == "wallclock"
    ]
    assert len(wall_events) == 1


# --------------------------------------------------------------------------- #
# Critic veto rate                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_veto_rate_warning_emits_when_6_of_last_10(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    """6 vetoes out of the last 10 tool calls -> warn once."""
    state = _state(engagement={"token_budget": 0, "window": {}})
    # 6 True (veto), 4 False (approve)
    state.recent_critic_verdicts = [True] * 6 + [False] * 4

    await thresholds.check_thresholds(state)

    veto_events = [e for e in fake_bus.events if e["kind"] == "engagement.veto_rate_warning"]
    assert len(veto_events) == 1
    assert veto_events[0]["payload"]["vetoes"] == 6
    assert veto_events[0]["payload"]["rate"] == pytest.approx(0.6)
    assert "veto_rate" in state.warnings_fired

    # Re-check: debounced.
    await thresholds.check_thresholds(state)
    veto_events = [e for e in fake_bus.events if e["kind"] == "engagement.veto_rate_warning"]
    assert len(veto_events) == 1


@pytest.mark.asyncio
async def test_veto_rate_below_threshold_no_warning(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    """5/10 is exactly at the threshold — NOT over. No warning."""
    state = _state(engagement={"token_budget": 0, "window": {}})
    state.recent_critic_verdicts = [True] * 5 + [False] * 5
    await thresholds.check_thresholds(state)
    assert all(e["kind"] != "engagement.veto_rate_warning" for e in fake_bus.events)


@pytest.mark.asyncio
async def test_veto_rate_window_not_full_no_warning(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    """Even 100% veto rate on 5 calls doesn't trip — we wait for a full window."""
    state = _state(engagement={"token_budget": 0, "window": {}})
    state.recent_critic_verdicts = [True] * 5
    await thresholds.check_thresholds(state)
    assert all(e["kind"] != "engagement.veto_rate_warning" for e in fake_bus.events)


# --------------------------------------------------------------------------- #
# Suppression spread                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_suppression_spread_warning_on_new_mute(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    """First time a tool gets suppressed during a stage -> warn once."""
    state = _state(engagement={"token_budget": 0, "window": {}})
    state.suppressed_tools_stage = "recon"
    state.suppressed_tools = {"nuclei_scan": "worker_disabled"}

    await thresholds.check_thresholds(state)

    spread = [e for e in fake_bus.events if e["kind"] == "engagement.suppression_spread"]
    assert len(spread) == 1
    assert spread[0]["payload"]["stage"] == "recon"
    assert "nuclei_scan" in spread[0]["payload"]["added"]
    assert "suppression_spread:recon" in state.warnings_fired

    # Same count, same stage -> no second event.
    await thresholds.check_thresholds(state)
    spread = [e for e in fake_bus.events if e["kind"] == "engagement.suppression_spread"]
    assert len(spread) == 1


@pytest.mark.asyncio
async def test_suppression_spread_no_warning_when_count_unchanged(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    state = _state(engagement={"token_budget": 0, "window": {}})
    state.suppressed_tools_stage = "recon"
    state.suppressed_tools = {}
    state.last_known_suppression_stage_count = {"recon": 0}
    await thresholds.check_thresholds(state)
    assert all(e["kind"] != "engagement.suppression_spread" for e in fake_bus.events)


# --------------------------------------------------------------------------- #
# Audit-log writes                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_warnings_write_audit_log(
    fake_bus: _FakeBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every warning event also writes an audit-log row with the same
    action name + payload. This is the durable record."""
    audit_calls: list[dict[str, Any]] = []

    async def _capture(**kwargs: Any) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(thresholds, "audit", _capture)
    state = _state()
    state.total_tokens = 850_000  # 85% -> triggers token warning

    await thresholds.check_thresholds(state)

    assert len(audit_calls) == 1
    call = audit_calls[0]
    assert call["action"] == "engagement.budget_warning"
    assert call["actor"] == "threshold_monitor"
    assert call["payload"]["kind"] == "token"
    assert call["engagement_id"] == state.engagement_id


# --------------------------------------------------------------------------- #
# async-08: atomic check-then-act + bounded per-engagement lock              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_check_thresholds_emits_once(
    fake_bus: _FakeBus, no_audit: None
) -> None:
    """async-08: even if two ``check_thresholds`` run concurrently against
    the same state (a future fan-out), the per-engagement lock makes the
    check-then-``fired.add`` debounce atomic so the warning fires EXACTLY
    once — not twice from interleaved membership checks."""
    state = _state()
    state.total_tokens = 900_000  # over the 80% token threshold

    await asyncio.gather(
        thresholds.check_thresholds(state),
        thresholds.check_thresholds(state),
        thresholds.check_thresholds(state),
    )

    budget_events = [e for e in fake_bus.events if e["kind"] == "engagement.budget_warning"]
    assert len(budget_events) == 1, "debounce must hold under concurrent checks"


@pytest.mark.asyncio
async def test_release_engagement_lock_evicts(fake_bus: _FakeBus, no_audit: None) -> None:
    """async-03/async-08: the per-engagement threshold lock is created on
    first use and evicted by ``release_engagement_lock`` so the registry
    doesn't leak one asyncio.Lock per engagement on a long-lived worker."""
    state = _state()
    state.total_tokens = 900_000

    await thresholds.check_thresholds(state)
    key = str(state.engagement_id)
    assert key in thresholds._THRESHOLD_LOCKS

    thresholds.release_engagement_lock(state.engagement_id)
    assert key not in thresholds._THRESHOLD_LOCKS
    # Idempotent — a second release is a no-op, not an error.
    thresholds.release_engagement_lock(state.engagement_id)
