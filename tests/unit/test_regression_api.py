"""R2 #E — regression API endpoint + EngagementRead surfaces.

Covers:
- ``GET /engagements/{id}/regressions`` returns ``RegressionRead`` rows
  in most-recent-first order.
- Council-pending rows (the in-flight window between the controller
  emitting the row and the council deciding) surface
  ``council_decision = None`` rather than vanishing.
- Council-vetoed and council-escalated rows ARE included (only
  ``applied_at`` differs vs APPROVED rows). The customer-facing report
  filters; the operator-facing API surfaces everything.
- ``EngagementRead`` exposes ``regressions_used`` (count of applied
  rows only) + ``regression_budget_max`` (3 by default).
- ``regressions_used`` reflects only ``applied_at IS NOT NULL`` rows —
  vetoed/escalated requests don't burn the budget.

The endpoints are exercised by calling the route functions directly
with a fake AsyncSession. The orchestrator's session_scope path is
the same as in unit tests for the ledger helpers; mocking the
SQLAlchemy session keeps the test independent of a real Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from contextlib import asynccontextmanager

import orchestrator.services.engagements as _svc_mod
from orchestrator.services.engagements import (
    EngagementRead,
    RegressionRead,
    _count_regressions_used,
    _row_to_read,
    get_engagement as _get_engagement_orig,
    list_engagements as _list_engagements_orig,
    list_regressions as _list_regressions_orig,
)
from orchestrator.services.errors import NotFoundError
from orchestrator.models import (
    REGRESSION_BUDGET_MAX_DEFAULT,
    EngagementRow,
    RegressionRow,
)

# --------------------------------------------------------------------------- #
# Compatibility wrappers (service functions no longer accept session=)        #
# --------------------------------------------------------------------------- #


async def get_engagement(engagement_id, session=None):
    if session is not None:
        @asynccontextmanager
        async def _scope():
            yield session
        orig = _svc_mod.session_scope
        _svc_mod.session_scope = _scope
        try:
            return await _get_engagement_orig(engagement_id)
        finally:
            _svc_mod.session_scope = orig
    return await _get_engagement_orig(engagement_id)


async def list_engagements(session=None):
    if session is not None:
        @asynccontextmanager
        async def _scope():
            yield session
        orig = _svc_mod.session_scope
        _svc_mod.session_scope = _scope
        try:
            return await _list_engagements_orig()
        finally:
            _svc_mod.session_scope = orig
    return await _list_engagements_orig()


async def list_regressions(engagement_id, session=None):
    if session is not None:
        @asynccontextmanager
        async def _scope():
            yield session
        orig = _svc_mod.session_scope
        _svc_mod.session_scope = _scope
        try:
            return await _list_regressions_orig(engagement_id)
        finally:
            _svc_mod.session_scope = orig
    return await _list_regressions_orig(engagement_id)


# --------------------------------------------------------------------------- #
# Fake SQLAlchemy session                                                     #
# --------------------------------------------------------------------------- #


class _FakeScalars:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return list(self._rows)


class _FakeResult:
    """Stand-in for sqlalchemy.Result.

    ``scalars()`` returns a ``_FakeScalars`` (which has ``.all()``);
    ``scalar()`` returns the first cell of the first row (used by the
    COUNT(*) query in ``_count_regressions_used``);
    ``all()`` returns the row tuples (used by the GROUP BY query in
    ``list_engagements``).
    """

    def __init__(self, rows: list[Any], *, scalar: Any = None) -> None:
        self._rows = rows
        self._scalar = scalar

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._rows)

    def scalar(self) -> Any:
        return self._scalar

    def all(self) -> list[Any]:
        return list(self._rows)


class _FakeSession:
    """Captures select() statements and returns scripted results.

    The endpoints under test issue four kinds of queries:
      1. ``select(EngagementRow).order_by(...)`` — list_engagements.
      2. ``session.get(EngagementRow, id)`` — get_engagement.
      3. ``select(func.count()).select_from(RegressionRow).where(...)``
         — _count_regressions_used.
      4. ``select(RegressionRow).where(...).order_by(...)`` —
         list_regressions, plus a GROUP BY variant in list_engagements.

    A test installs handlers via ``add_handler(predicate, result)``.
    The first matching predicate fires; otherwise the test fails loud
    rather than silently returning an empty result.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[Any, _FakeResult]] = []
        self._gets: dict[UUID, Any] = {}

    def install_get(self, engagement_id: UUID, row: Any) -> None:
        self._gets[engagement_id] = row

    def install_handler(self, predicate: Any, result: _FakeResult) -> None:
        self._handlers.append((predicate, result))

    async def get(self, _cls: Any, key: Any) -> Any:
        # Accept UUID or str for convenience; engagements API takes UUID.
        try:
            key_uuid = key if isinstance(key, UUID) else UUID(str(key))
        except Exception:
            key_uuid = key
        return self._gets.get(key_uuid)

    async def execute(self, stmt: Any) -> _FakeResult:
        compiled = str(stmt).lower()
        for predicate, result in self._handlers:
            if predicate(compiled, stmt):
                return result
        raise AssertionError(
            f"no matching handler installed for statement:\n{compiled[:400]}"
        )


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _engagement_row(name: str = "test") -> EngagementRow:
    now = datetime.now(UTC)
    row = EngagementRow(
        id=uuid4(),
        name=name,
        scope={"cidrs": ["10.0.0.0/24"], "domains": []},
        objectives=[],
        mode="guided",
        time_window_start=now,
        time_window_end=now + timedelta(hours=4),
        token_budget=500_000,
        wallclock_hours=4,
        authorization_artifact_sha256="x" * 64,
        status="draft",
        current_stage="scope_intake",
        created_by=None,
        llm_config_snapshot={},
        created_at=now,
        updated_at=now,
    )
    return row


def _regression_row(
    engagement_id: UUID,
    *,
    target_stage: str,
    trigger_kind: str,
    rationale: str = "",
    council_decision: str | None = None,
    council_rationale: str | None = None,
    applied_at: datetime | None = None,
    requested_at: datetime | None = None,
) -> RegressionRow:
    row = RegressionRow(
        id=uuid4(),
        engagement_id=engagement_id,
        requested_at=requested_at or datetime.now(UTC),
        target_stage=target_stage,
        trigger_kind=trigger_kind,
        delta_json={},
        evidence_json=[],
        rationale=rationale,
        council_decision=council_decision,
        council_rationale=council_rationale,
        applied_at=applied_at,
    )
    return row


# --------------------------------------------------------------------------- #
# EngagementRead surfaces                                                     #
# --------------------------------------------------------------------------- #


def test_row_to_read_defaults_regressions_used_to_zero() -> None:
    """When no regression count is supplied, the DTO defaults to 0
    (the common case: no regressions, no extra query). The
    ``regression_budget_max`` reads from the engagement row; new
    engagements default to ``0`` (unlimited)."""
    row = _engagement_row()
    read = _row_to_read(row)
    assert read.regressions_used == 0
    assert read.regression_budget_max == int(
        getattr(row, "regression_budget_max", 0) or 0
    )


def test_row_to_read_propagates_regression_budget_max_from_row() -> None:
    """When the row has an explicit cap, surface it through the DTO
    so the Overview tab can render '2 / N used'. The default for new
    engagements is ``0`` (unlimited) — operators set a positive cap
    only as a runaway-loop circuit breaker via the advanced intake
    form."""
    row = _engagement_row()
    row.regression_budget_max = 12
    read = _row_to_read(row, regressions_used=2)
    assert read.regressions_used == 2
    assert read.regression_budget_max == 12


def test_engagement_read_serialises_regressions_fields() -> None:
    """Pydantic round-trip: ensure the new fields are part of the
    serialised JSON shape the UI consumes."""
    row = _engagement_row()
    read = _row_to_read(row, regressions_used=1)
    dumped = read.model_dump(mode="json")
    assert dumped["regressions_used"] == 1
    assert "regression_budget_max" in dumped


# --------------------------------------------------------------------------- #
# _count_regressions_used                                                     #
# --------------------------------------------------------------------------- #


async def test_count_regressions_used_returns_scalar_count() -> None:
    """COUNT(*) over RegressionRow with applied_at IS NOT NULL."""
    session = _FakeSession()
    eng_id = uuid4()
    session.install_handler(
        lambda sql, _stmt: "count" in sql and "regressions" in sql,
        _FakeResult([], scalar=2),
    )
    n = await _count_regressions_used(session, eng_id)  # type: ignore[arg-type]
    assert n == 2


async def test_count_regressions_used_zero_when_query_fails() -> None:
    """Best-effort: a regressions-table outage MUST NOT 5xx the
    engagement detail page; the count silently drops to 0."""

    class _BrokenSession:
        async def execute(self, _stmt: Any) -> _FakeResult:
            raise RuntimeError("postgres down")

    n = await _count_regressions_used(_BrokenSession(), uuid4())  # type: ignore[arg-type]
    assert n == 0


# --------------------------------------------------------------------------- #
# GET /engagements/{id}                                                       #
# --------------------------------------------------------------------------- #


async def test_get_engagement_surfaces_regressions_used() -> None:
    """When the engagement has applied regressions, the
    ``EngagementRead`` returned by the single-get endpoint carries the
    count + cap so the Overview tab renders 'N / 3 used' off one call."""
    session = _FakeSession()
    eng = _engagement_row(name="with-regressions")
    session.install_get(eng.id, eng)
    # COUNT(*) returns 2 applied rows.
    session.install_handler(
        lambda sql, _stmt: "count" in sql and "regressions" in sql,
        _FakeResult([], scalar=2),
    )
    read = await get_engagement(eng.id, session=session)  # type: ignore[arg-type]
    assert isinstance(read, EngagementRead)
    assert read.regressions_used == 2
    # New engagements default to ``0`` (unlimited); the row's column
    # drives the DTO, not the module constant.
    assert read.regression_budget_max == int(
        getattr(eng, "regression_budget_max", 0) or 0
    )


async def test_get_engagement_404_when_missing() -> None:
    """The endpoint shape must still 404 when the engagement doesn't
    exist — the new regressions count code path must not mask that."""
    session = _FakeSession()
    eng_id = uuid4()
    # No install_get → session.get returns None.
    with pytest.raises(NotFoundError):
        await get_engagement(eng_id, session=session)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# GET /engagements (list)                                                      #
# --------------------------------------------------------------------------- #


async def test_list_engagements_batches_regression_counts() -> None:
    """One GROUP BY query covers every engagement, not N+1 round-trips.
    Engagements without applied regressions don't appear in the count
    result and default to 0 via dict.get()."""
    session = _FakeSession()
    e1 = _engagement_row(name="e1")
    e2 = _engagement_row(name="e2")
    e3 = _engagement_row(name="e3")
    # First select: the engagements themselves.
    session.install_handler(
        lambda sql, _stmt: (
            "select" in sql
            and "engagements" in sql
            and "regressions" not in sql
        ),
        _FakeResult([e1, e2, e3]),
    )
    # Second select: GROUP BY engagement_id, count over regressions.
    # Only e1 (3 applied) and e2 (1 applied) appear; e3 stays at 0.
    session.install_handler(
        lambda sql, _stmt: "regressions" in sql and "group by" in sql,
        _FakeResult([(e1.id, 3), (e2.id, 1)]),
    )
    reads = await list_engagements(session=session)  # type: ignore[arg-type]
    by_name = {r.name: r for r in reads}
    assert by_name["e1"].regressions_used == 3
    assert by_name["e2"].regressions_used == 1
    assert by_name["e3"].regressions_used == 0


async def test_list_engagements_count_query_failure_defaults_to_zero() -> None:
    """A regressions-table outage MUST NOT break the engagement list —
    every row falls back to 0 used."""

    class _PartialSession(_FakeSession):
        def __init__(self, engagements: list[EngagementRow]) -> None:
            super().__init__()
            self._engagements = engagements

        async def execute(self, stmt: Any) -> _FakeResult:  # type: ignore[override]
            sql = str(stmt).lower()
            if "regressions" in sql:
                raise RuntimeError("regressions table down")
            return _FakeResult(self._engagements)

    e1 = _engagement_row(name="e1")
    session = _PartialSession([e1])
    reads = await list_engagements(session=session)  # type: ignore[arg-type]
    assert len(reads) == 1
    assert reads[0].regressions_used == 0


# --------------------------------------------------------------------------- #
# GET /engagements/{id}/regressions                                            #
# --------------------------------------------------------------------------- #


async def test_list_regressions_returns_rows_most_recent_first() -> None:
    """The endpoint orders by ``requested_at desc`` so the operator
    sees the latest loop at the top of the panel."""
    session = _FakeSession()
    eng_id = uuid4()
    now = datetime.now(UTC)
    older = _regression_row(
        eng_id,
        target_stage="recon",
        trigger_kind="new_subnet_discovered",
        rationale="discovered 10.0.5.0/24",
        council_decision="approve",
        council_rationale="novelty + budget headroom",
        applied_at=now - timedelta(minutes=20),
        requested_at=now - timedelta(minutes=21),
    )
    newer = _regression_row(
        eng_id,
        target_stage="enumeration",
        trigger_kind="new_service_class_discovered",
        rationale="SMB port 445 opened on h1",
        council_decision="approve",
        council_rationale="approve",
        applied_at=now - timedelta(minutes=2),
        requested_at=now - timedelta(minutes=3),
    )
    # Endpoint ORDER BY desc → newer first.
    session.install_handler(
        lambda sql, _stmt: "regressions" in sql and "order by" in sql,
        _FakeResult([newer, older]),
    )
    out = await list_regressions(eng_id, session=session)  # type: ignore[arg-type]
    assert len(out) == 2
    assert all(isinstance(r, RegressionRead) for r in out)
    assert out[0].target_stage == "enumeration"
    assert out[1].target_stage == "recon"
    # Field round-trip.
    assert out[0].trigger_kind == "new_service_class_discovered"
    assert out[0].rationale == "SMB port 445 opened on h1"
    assert out[0].council_decision == "approve"
    assert out[0].applied_at is not None


async def test_list_regressions_surfaces_null_council_for_pending_rows() -> None:
    """A regression row exists in a window between
    ``record_regression`` (insert) and the council's
    ``update_regression_with_verdict`` patch. During that window the
    council_decision is None. The endpoint must surface it as null
    rather than dropping the row or coercing to a string — the UI
    distinguishes 'pending' from 'approve / veto / escalate'."""
    session = _FakeSession()
    eng_id = uuid4()
    pending = _regression_row(
        eng_id,
        target_stage="lateral",
        trigger_kind="new_subnet_discovered",
        rationale="awaiting council",
        council_decision=None,
        council_rationale=None,
        applied_at=None,
    )
    session.install_handler(
        lambda sql, _stmt: "regressions" in sql and "order by" in sql,
        _FakeResult([pending]),
    )
    out = await list_regressions(eng_id, session=session)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0].council_decision is None
    assert out[0].council_rationale is None
    assert out[0].applied_at is None


async def test_list_regressions_includes_vetoed_and_escalated() -> None:
    """The operator-facing API surfaces every regression row — applied,
    vetoed, AND escalated. ``applied_at`` is None on the non-APPROVE
    branches; the operator sees 'the agent tried, the council blocked'.
    """
    session = _FakeSession()
    eng_id = uuid4()
    vetoed = _regression_row(
        eng_id,
        target_stage="exploit",
        trigger_kind="new_subnet_discovered",
        rationale="agent wanted to re-exploit",
        council_decision="veto",
        council_rationale="budget exhausted",
        applied_at=None,
    )
    escalated = _regression_row(
        eng_id,
        target_stage="loot",
        trigger_kind="creds_found_for_new_realm",
        rationale="agent wanted another loop",
        council_decision="escalate",
        council_rationale="needs human review",
        applied_at=None,
    )
    session.install_handler(
        lambda sql, _stmt: "regressions" in sql and "order by" in sql,
        _FakeResult([vetoed, escalated]),
    )
    out = await list_regressions(eng_id, session=session)  # type: ignore[arg-type]
    assert {r.council_decision for r in out} == {"veto", "escalate"}
    assert all(r.applied_at is None for r in out)


async def test_list_regressions_empty_for_engagement_with_no_loops() -> None:
    """Common case: an engagement with no regressions returns []."""
    session = _FakeSession()
    eng_id = uuid4()
    session.install_handler(
        lambda sql, _stmt: "regressions" in sql and "order by" in sql,
        _FakeResult([]),
    )
    out = await list_regressions(eng_id, session=session)  # type: ignore[arg-type]
    assert out == []
