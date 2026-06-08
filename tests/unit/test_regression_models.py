"""R2 #B Layer 2 — regression-row ORM + ledger helper tests.

Covers:
- ``RegressionRow`` ORM has the expected columns / types (drift check).
- ``record_regression`` inserts a row through ``session_scope`` and returns
  the new row's id so Phase C/D can patch the council verdict back onto it.
- ``record_regression`` is best-effort: a Postgres outage logs and returns
  ``None`` instead of breaking the agent loop. Same pattern as
  ``record_tool_run`` / ``record_evidence``.
- ``EngagementState`` exposes the new ``regressions_used`` counter that the
  BudgetCritic will read (Phase C).
- The module-level defaults ``REGRESSION_BUDGET_MAX_DEFAULT = 3`` and
  ``REGRESSION_TOKEN_RESERVE_PCT = 0.15`` match the spec.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from orchestrator import ledger
from orchestrator.models import (
    REGRESSION_BUDGET_MAX_DEFAULT,
    REGRESSION_TOKEN_RESERVE_PCT,
    RegressionRow,
)
from orchestrator.stage_controller import EngagementState
from worldstate.schema import Stage

# --------------------------------------------------------------------------- #
# Fake session scope (mirrors the pattern in test_r1_automation_fixes.py)     #
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Captures rows added via ``s.add(row)`` and stamps a fake id on flush
    so callers that depend on ``row.id`` after flush() get a value."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.flush_calls: int = 0

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        self.flush_calls += 1
        for row in self.added:
            if getattr(row, "id", None) is None:
                row.id = uuid4()


class _FakeSessionCtx:
    def __init__(self) -> None:
        self.session = _FakeSession()

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *_: Any) -> None:
        return None


# --------------------------------------------------------------------------- #
# Static ORM-shape tests (no DB needed)                                       #
# --------------------------------------------------------------------------- #


def test_regression_row_table_columns() -> None:
    """R2 #B §B.4.6: ``RegressionRow`` exposes every column the spec lists."""
    cols = {c.name for c in RegressionRow.__table__.columns}
    expected = {
        "id",
        "engagement_id",
        "requested_at",
        "target_stage",
        "trigger_kind",
        "delta_json",
        "evidence_json",
        "rationale",
        "council_decision",
        "council_rationale",
        "applied_at",
    }
    assert expected <= cols, f"missing columns: {expected - cols}"


def test_regression_row_council_fields_are_nullable() -> None:
    """The council patches its verdict onto the same row in Phase C, so
    ``council_decision`` / ``council_rationale`` MUST be nullable; the row
    is created before the council deliberates. ``applied_at`` is also
    nullable for the same reason (Phase D stamps it on commit)."""
    cols = {c.name: c for c in RegressionRow.__table__.columns}
    assert cols["council_decision"].nullable is True
    assert cols["council_rationale"].nullable is True
    assert cols["applied_at"].nullable is True
    # And these MUST be non-null at create time.
    assert cols["engagement_id"].nullable is False
    assert cols["target_stage"].nullable is False
    assert cols["trigger_kind"].nullable is False


def test_regression_row_engagement_id_is_indexed() -> None:
    """The UI's "regression history for engagement X" query is a hot path;
    the FK column must be indexed."""
    eng_col = RegressionRow.__table__.columns["engagement_id"]
    assert eng_col.index is True


def test_regression_budget_default_is_runaway_loop_circuit_breaker() -> None:
    """The module default is a runaway-loop circuit breaker, not a
    design constraint. Real engagements set ``regression_budget_max=0``
    (unlimited) via intake; the module constant is only consulted as a
    fallback when no engagement context is wired. The exact value is
    deliberately high so a healthy multi-segment engagement never
    bumps into it; loop safety lives in NoveltyCritic per-fingerprint
    replay rejection and ScopeCritic literal authorisation.
    """
    assert REGRESSION_BUDGET_MAX_DEFAULT >= 50


def test_regression_token_reserve_is_fifteen_percent() -> None:
    """B.4.5 reserves 15% of the engagement's token budget for regression
    LLM calls."""
    assert pytest.approx(0.15) == REGRESSION_TOKEN_RESERVE_PCT


def test_engagement_state_carries_regressions_used_counter() -> None:
    """B.4.5: the BudgetCritic (Phase C) reads ``regressions_used`` to
    enforce the cap. The field MUST default to 0 so old LangGraph
    checkpoints rehydrate cleanly post-deploy."""
    state = EngagementState(
        engagement_id=uuid4(),
        engagement={},
    )
    assert state.regressions_used == 0


# --------------------------------------------------------------------------- #
# record_regression — happy path                                              #
# --------------------------------------------------------------------------- #


async def test_record_regression_inserts_row_with_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_regression`` adds a ``RegressionRow`` with the exact field
    shape the spec requires."""
    ctx = _FakeSessionCtx()
    monkeypatch.setattr(ledger, "session_scope", lambda: ctx)

    eng_id = uuid4()
    row_id = await ledger.record_regression(
        engagement_id=eng_id,
        target_stage=Stage.RECON.value,
        trigger_kind="new_subnet_discovered",
        delta={"cidrs": ["10.0.5.0/24"]},
        evidence=["node-a", "node-b"],
        rationale="LATERAL credential implies reachability to 10.0.5.0/24",
    )

    assert row_id is not None
    assert len(ctx.session.added) == 1
    row = ctx.session.added[0]
    assert isinstance(row, RegressionRow)
    assert row.engagement_id == eng_id
    assert row.target_stage == "recon"
    assert row.trigger_kind == "new_subnet_discovered"
    assert row.delta_json == {"cidrs": ["10.0.5.0/24"]}
    assert row.evidence_json == ["node-a", "node-b"]
    assert "10.0.5.0/24" in row.rationale
    # Council / applied fields are unset at request time — Phase C/D fill them.
    assert row.council_decision is None
    assert row.council_rationale is None
    assert row.applied_at is None


async def test_record_regression_returns_row_id_so_council_can_patch_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase C will UPDATE the same row with the council verdict; the
    helper must surface the row id."""
    ctx = _FakeSessionCtx()
    monkeypatch.setattr(ledger, "session_scope", lambda: ctx)

    row_id = await ledger.record_regression(
        engagement_id=uuid4(),
        target_stage=Stage.ENUMERATION.value,
        trigger_kind="new_service_class_discovered",
        delta={"host_id": "h1", "ports": [445]},
        evidence=[],
        rationale="",
    )

    assert row_id is not None
    # And it MUST match the row that was added — same uuid object.
    assert ctx.session.added[0].id == row_id
    # Helper flushed so the id is real, not lazy.
    assert ctx.session.flush_calls == 1


# --------------------------------------------------------------------------- #
# record_regression — best-effort failure mode                                #
# --------------------------------------------------------------------------- #


async def test_record_regression_best_effort_returns_none_on_persist_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2 #B §B.4.6: ``record_regression`` mirrors ``record_tool_run`` —
    a Postgres outage logs and returns ``None`` rather than raising. The
    agent loop must not break on audit-row persistence failure; the event
    bus publish (Phase D) still happens so the UI surfaces the attempt."""

    class _Broken:
        async def __aenter__(self) -> Any:
            raise RuntimeError("postgres unavailable")

        async def __aexit__(self, *_: Any) -> None:
            return None

    # NB: lambda not inlinable — `session_scope()` is called by the helper
    # and MUST return a fresh ctx each call (matches the real session_scope
    # API). Inlining `_Broken()` here would share one instance across calls.
    monkeypatch.setattr(ledger, "session_scope", lambda: _Broken())  # noqa: PLW0108

    # Should NOT raise; should return None so callers can treat the audit
    # as "best effort, the run continues."
    row_id = await ledger.record_regression(
        engagement_id=uuid4(),
        target_stage=Stage.RECON.value,
        trigger_kind="new_subnet_discovered",
        delta={"cidrs": ["10.0.5.0/24"]},
        evidence=[],
        rationale="postgres down",
    )
    assert row_id is None
