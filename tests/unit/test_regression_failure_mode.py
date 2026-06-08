"""Tests for regression failure-mode classification.

Validates ``classify_regression_failure`` from ``src/agents/report.py``
produces the correct short tag for each council rationale pattern so
the report and EngagementSummary surface an operator-readable failure
mode rather than raw rationale strings.
"""

from __future__ import annotations

from agents.report import classify_regression_failure


def test_classify_approve_returns_none():
    assert classify_regression_failure("approve", "all good") is None


def test_classify_budget_exhausted():
    assert classify_regression_failure(
        "veto",
        "runaway-loop circuit breaker tripped: 50 regressions used >= 50 cap"
    ) == "budget_exhausted"


def test_classify_budget_keyword():
    assert classify_regression_failure(
        "veto", "regression budget exhausted"
    ) == "budget_exhausted"


def test_classify_novelty_replay():
    assert classify_regression_failure(
        "veto",
        "regression to recon/new_subnet_discovered with this exact delta "
        "(fp=abc123...) was already applied this engagement"
    ) == "novelty_replay"


def test_classify_replay_keyword():
    assert classify_regression_failure(
        "veto", "replay_loop detected"
    ) == "novelty_replay"


def test_classify_scope_violation():
    assert classify_regression_failure(
        "veto",
        "delta entity '8.8.8.0/24' is outside engagement CIDR scope"
    ) == "scope_violation"


def test_classify_excluded():
    assert classify_regression_failure(
        "veto",
        "delta entity '10.0.5.0/24' is in an excluded CIDR"
    ) == "scope_violation"


def test_classify_empty_delta():
    assert classify_regression_failure(
        "veto",
        "regression has empty delta — nothing for the re-entered stage to act on"
    ) == "empty_delta"


def test_classify_storm():
    assert classify_regression_failure(
        "escalate",
        "regression storm: 6 approved recon/new_subnet_discovered regressions "
        "within the last 10 minutes"
    ) == "regression_storm"


def test_classify_token_reserve():
    assert classify_regression_failure(
        "veto",
        "token reserve exhausted: used 900000 / 1000000 (90%); reserve floor is 15%"
    ) == "token_reserve_exhausted"


def test_classify_escalate_fallback():
    """Escalate with no recognisable rationale pattern → 'escalated'."""
    assert classify_regression_failure(
        "escalate",
        "world-state graph unavailable: connection refused"
    ) == "escalated"


def test_classify_other_veto():
    """Veto with unrecognised rationale → 'other'."""
    assert classify_regression_failure(
        "veto",
        "some novel future rationale we haven't categorised"
    ) == "other"
