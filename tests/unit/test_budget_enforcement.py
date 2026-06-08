"""Tests for budget enforcement in the stage controller (Phase 16, Tier A #4)."""

from __future__ import annotations

import time
from uuid import uuid4

from orchestrator.stage_controller import EngagementState, _check_budgets
from worldstate.schema import Stage


def _state(*, token_budget: int = 1_000_000, end_offset_seconds: int = 3600) -> EngagementState:
    now_ns = time.time_ns()
    return EngagementState(
        engagement_id=uuid4(),
        engagement={
            "token_budget": token_budget,
            "window": {
                "start_ns": now_ns - 1_000_000_000,
                "end_ns": now_ns + end_offset_seconds * 1_000_000_000,
            },
        },
        current_stage=Stage.RECON,
    )


def test_under_budget_returns_none():
    s = _state()
    s.total_tokens = 100
    assert _check_budgets(s) is None


def test_token_budget_exceeded_returns_reason():
    s = _state(token_budget=500)
    s.total_tokens = 1000
    reason = _check_budgets(s)
    assert reason is not None
    assert "token_budget_exceeded" in reason


def test_wallclock_expired_returns_reason():
    s = _state(end_offset_seconds=-60)  # window ended a minute ago
    reason = _check_budgets(s)
    assert reason is not None
    assert "wallclock_window_exceeded" in reason


def test_missing_window_does_not_crash():
    """If the engagement has no window (unit-test-style construction),
    the wallclock check should silently skip."""
    s = EngagementState(
        engagement_id=uuid4(),
        engagement={"token_budget": 1_000_000},  # no `window`
        current_stage=Stage.RECON,
    )
    assert _check_budgets(s) is None


def test_zero_token_budget_disables_check():
    """Token budget of 0 should mean "no limit" rather than always-exceeded."""
    s = _state(token_budget=0)
    s.total_tokens = 1_000_000_000
    assert _check_budgets(s) is None
