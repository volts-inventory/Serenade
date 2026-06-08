"""R3 #M2.5: ``EngagementSummary.warnings_fired`` projects audit-log rows
back into the ``EngagementState.warnings_fired`` debounce-key shape so the
Overview tab's red-glow tile treatment can land off a single SQL pull.

Covers:
- Empty list when no threshold-warning audit rows exist.
- ``budget:token`` / ``budget:wallclock`` projected from the
  ``engagement.budget_warning`` action and its ``kind`` payload field.
- ``veto_rate`` projected from ``engagement.veto_rate_warning``.
- ``suppression_spread:<stage>`` projected from
  ``engagement.suppression_spread`` and its ``stage`` payload field.
- A degraded audit-log query (raises) returns ``[]`` and leaves the
  other counters intact — same defensive stance as the rest of the
  summary endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from contextlib import asynccontextmanager

import orchestrator.services.engagements as _svc_mod
from orchestrator.services.engagements import (
    EngagementSummary,
    get_engagement_summary,
)
from orchestrator.models import AuditLogRow, EngagementRow


class _SummaryTestCtx:
    def __init__(self, session):
        self._session = session
        self._orig = None

    def __enter__(self):
        self._orig = _svc_mod.session_scope

        @asynccontextmanager
        async def _scope():
            yield self._session
        _svc_mod.session_scope = _scope
        return self

    def __exit__(self, *_):
        _svc_mod.session_scope = self._orig


# Re-implement the lightweight fakes locally rather than importing from the
# sibling endpoint test file — this keeps the fixture surface explicit and
# avoids cross-file fixture coupling.


class _FakeScalar:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value

    def scalars(self):  # noqa: ANN201 — match SQLAlchemy duck shape
        class _S:
            def all(_self):
                return []

        return _S()

    def all(self):  # noqa: ANN201
        return []


class _FakeRowsResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self):  # noqa: ANN201
        class _S:
            def __init__(_self, r):
                _self._r = r

            def all(_self):
                return list(_self._r)

        return _S(self._rows)

    def all(self):  # noqa: ANN201
        return list(self._rows)

    def scalar(self):  # noqa: ANN201
        return None


class _FakeGroupByResult:
    def __init__(self, tuples: list[tuple[str, str, int]]) -> None:
        self._tuples = tuples

    def all(self):  # noqa: ANN201
        return list(self._tuples)

    def scalars(self):  # noqa: ANN201
        class _S:
            def all(_self):
                return []

        return _S()

    def scalar(self):  # noqa: ANN201
        return None


class _SeededSession:
    """Stand-in for AsyncSession with a per-query dispatcher.

    Returns warning rows on the ``audit_log`` query whose SQL fragment
    matches ``from audit_log`` AND mentions ``engagement.budget_warning``
    (the warnings query uses an ``IN`` clause). The legacy audit-log
    code path (used by ``_suppressed_tools_for`` before R3 #J2 Bug 2 made
    ``suppressed_tools`` the source of truth) still routes here and gets
    an empty list for back-compat.
    """

    def __init__(
        self,
        *,
        engagement: EngagementRow | None,
        warning_rows: list[Any],
    ) -> None:
        self.engagement = engagement
        self.warning_rows = warning_rows

    async def get(self, _cls: Any, _key: Any) -> Any:
        return self.engagement

    async def execute(self, stmt: Any) -> Any:
        sql = str(stmt).lower()
        if "from tool_runs" in sql:
            return _FakeScalar(0)
        if "from approvals" in sql:
            return _FakeGroupByResult([])
        if "from suppressed_tools" in sql:
            return _FakeRowsResult([])
        if "from audit_log" in sql:
            # The warning-query uses ``AuditLogRow.action.in_(...)``;
            # SQLAlchemy renders that as ``... IN (...)`` while the
            # legacy suppressed-tools audit-log path (now empty since
            # R3 #J2 Bug 2 moved the source of truth to
            # ``suppressed_tools``) doesn't carry an IN clause. Use the
            # IN-clause marker as the discriminator.
            if " in (" in sql.lower():
                return _FakeRowsResult(
                    [(r.action, r.payload) for r in self.warning_rows]
                )
            return _FakeRowsResult([])
        if "from llm_calls" in sql and "sum" in sql:
            return _FakeScalar(0.0)
        if "from regressions" in sql:
            return _FakeScalar(0)
        return _FakeRowsResult([])


def _make_engagement_row() -> EngagementRow:
    now = datetime.now(UTC)
    return EngagementRow(
        id=uuid4(),
        name="threshold-warning-fixture",
        scope={"cidrs": ["10.0.0.0/24"], "domains": []},
        objectives=[],
        mode="guided",
        time_window_start=now - timedelta(hours=2),
        time_window_end=now + timedelta(hours=2),
        token_budget=1_000_000,
        wallclock_hours=4,
        authorization_artifact_sha256="x" * 64,
        status="running",
        current_stage="exploit",
        created_by=None,
        llm_config_snapshot={},
        created_at=now - timedelta(hours=2),
        updated_at=now,
    )


# --------------------------------------------------------------------------- #
# warnings_fired projection                                                   #
# --------------------------------------------------------------------------- #


def test_engagement_summary_default_warnings_empty() -> None:
    """A fresh summary defaults to an empty list — matches the UI's healthy
    state (no red glow on any tile)."""
    s = EngagementSummary()
    assert s.warnings_fired == []


@pytest.mark.asyncio
async def test_summary_warnings_empty_when_no_audit_rows() -> None:
    """No audit-log warning rows → ``warnings_fired`` stays empty."""
    eng = _make_engagement_row()
    session = _SeededSession(engagement=eng, warning_rows=[])
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == []


@pytest.mark.asyncio
async def test_summary_warnings_projects_token_budget() -> None:
    """``engagement.budget_warning`` with kind='token' projects to
    ``budget:token``."""
    eng = _make_engagement_row()
    rows = [
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.budget_warning",
            payload={"kind": "token", "usage": 0.82, "used": 820_000, "budget": 1_000_000},
        ),
    ]
    session = _SeededSession(engagement=eng, warning_rows=rows)
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == ["budget:token"]


@pytest.mark.asyncio
async def test_summary_warnings_projects_wallclock_budget() -> None:
    """``engagement.budget_warning`` with kind='wallclock' projects to
    ``budget:wallclock`` — distinct from the token-budget key so the UI
    can glow either the Spend or the Wallclock tile independently."""
    eng = _make_engagement_row()
    rows = [
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.budget_warning",
            payload={
                "kind": "wallclock",
                "usage": 0.82,
                "elapsed_ns": 6 * int(3.6e12),
                "window_ns": 8 * int(3.6e12),
            },
        ),
    ]
    session = _SeededSession(engagement=eng, warning_rows=rows)
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == ["budget:wallclock"]


@pytest.mark.asyncio
async def test_summary_warnings_projects_veto_rate() -> None:
    """``engagement.veto_rate_warning`` projects to ``veto_rate``."""
    eng = _make_engagement_row()
    rows = [
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.veto_rate_warning",
            payload={"rate": 0.6, "window": 10, "vetoes": 6},
        ),
    ]
    session = _SeededSession(engagement=eng, warning_rows=rows)
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == ["veto_rate"]


@pytest.mark.asyncio
async def test_summary_warnings_projects_suppression_spread_per_stage() -> None:
    """``engagement.suppression_spread`` projects to
    ``suppression_spread:<stage>`` — one debounce key per stage so a
    pipeline that hits the warning in two different stages surfaces both."""
    eng = _make_engagement_row()
    rows = [
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.suppression_spread",
            payload={"stage": "exploit", "added": ["nuclei"], "count": 1},
        ),
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.suppression_spread",
            payload={"stage": "lateral", "added": ["mimikatz"], "count": 1},
        ),
    ]
    session = _SeededSession(engagement=eng, warning_rows=rows)
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    # Sorted output — stable JSON ordering for the UI.
    assert out.warnings_fired == [
        "suppression_spread:exploit",
        "suppression_spread:lateral",
    ]


@pytest.mark.asyncio
async def test_summary_warnings_dedupes_and_sorts_all_kinds() -> None:
    """All four debounce-key shapes can land on the same engagement;
    duplicates are deduped (the audit-log captures every fire, but the
    threshold module debounces so duplicates are unexpected — still, the
    projection must collapse them defensively) and the output is sorted."""
    eng = _make_engagement_row()
    rows = [
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.budget_warning",
            payload={"kind": "token", "usage": 0.82},
        ),
        # Duplicate budget:token — must collapse.
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.budget_warning",
            payload={"kind": "token", "usage": 0.92},
        ),
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.budget_warning",
            payload={"kind": "wallclock", "usage": 0.82},
        ),
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.veto_rate_warning",
            payload={"rate": 0.7, "window": 10},
        ),
        AuditLogRow(
            engagement_id=eng.id,
            actor="threshold_monitor",
            action="engagement.suppression_spread",
            payload={"stage": "exploit", "count": 3},
        ),
    ]
    session = _SeededSession(engagement=eng, warning_rows=rows)
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == [
        "budget:token",
        "budget:wallclock",
        "suppression_spread:exploit",
        "veto_rate",
    ]


@pytest.mark.asyncio
async def test_summary_warnings_falls_back_to_empty_on_audit_outage() -> None:
    """A raising audit-log query must NOT 5xx the digest — the field
    quietly falls back to ``[]`` and the rest of the summary surfaces."""
    eng = _make_engagement_row()

    class _BoomOnWarnings(_SeededSession):
        async def execute(self, stmt: Any) -> Any:
            sql = str(stmt).lower()
            if "from audit_log" in sql and " in (" in sql:
                raise RuntimeError("audit_log index corrupt")
            return await super().execute(stmt)

    session = _BoomOnWarnings(engagement=eng, warning_rows=[])
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.warnings_fired == []
    # Other counters surface fine (zeroes from the seeded session).
    assert out.tool_calls_total == 0
