"""R3 #H3 Bug 2: GET /engagements/{id}/summary aggregates decision digest.

Covers:
- A seeded engagement with 10 tool runs (3 auto-approved, 2 critic-vetoed,
  1 human-approved, 4 default-pending), 1 suppressed tool, 1 applied
  regression returns the matching counters on the summary endpoint.
- A missing engagement still 404s through the endpoint.
- A partial Postgres outage doesn't 5xx — individual counters silently
  fall back to 0 so the operator sees a degraded digest, not an empty
  page.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from contextlib import asynccontextmanager

import orchestrator.services.engagements as _svc_mod
from orchestrator.services.engagements import (
    EngagementSummary,
    get_engagement_summary,
)
from orchestrator.services.errors import NotFoundError


def _patch_session(session):
    """Context manager that monkeypatches session_scope for a test."""
    @asynccontextmanager
    async def _scope():
        yield session
    return _scope


class _SummaryTestCtx:
    """Helper to monkey-patch session_scope around get_engagement_summary calls."""

    def __init__(self, session):
        self._session = session
        self._orig = None

    def __enter__(self):
        self._orig = _svc_mod.session_scope
        _svc_mod.session_scope = _patch_session(self._session)
        return self

    def __exit__(self, *_):
        _svc_mod.session_scope = self._orig
from orchestrator.models import (
    EngagementRow,
    SuppressedToolRow,
)


class _FakeScalar:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value

    def scalars(self):
        class _S:
            def all(_self):
                return []

        return _S()

    def all(self):
        return []


class _FakeRowsResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self):
        class _S:
            def __init__(_self, r):
                _self._r = r

            def all(_self):
                return list(_self._r)

        return _S(self._rows)

    def all(self):
        return list(self._rows)

    def scalar(self):
        return None


class _FakeGroupByResult:
    """For the GROUP BY query on approvals."""

    def __init__(self, tuples: list[tuple[str, str, int]]) -> None:
        self._tuples = tuples

    def all(self):
        return list(self._tuples)

    def scalars(self):
        class _S:
            def all(_self):
                return []

        return _S()

    def scalar(self):
        return None


class _SeededSession:
    """Stand-in for AsyncSession that drives ``get_engagement_summary``.

    The endpoint issues queries in this order:
      1. ``session.get(EngagementRow, engagement_id)`` — existence check.
      2. COUNT on ToolRunRow.
      3. GROUP BY on ApprovalRow.
      4. SELECT on AuditLogRow (via _suppressed_tools_for).
      5. SUM on LLMCallRow.
      6. COUNT on RegressionRow (via _count_regressions_used).

    Each test seeds the per-query response.
    """

    def __init__(
        self,
        *,
        engagement: EngagementRow | None,
        n_tools: int,
        approval_tuples: list[tuple[str, str, int]],
        suppressed_rows: list[Any],
        llm_cost: float,
        regressions_used: int,
    ) -> None:
        self.engagement = engagement
        self.n_tools = n_tools
        self.approval_tuples = approval_tuples
        # R3 #J2 Bug 2: ``suppressed_rows`` are now ``SuppressedToolRow``
        # entries from the dedicated table; legacy ``AuditLogRow``
        # fixtures are still accepted (the dispatcher returns them on the
        # audit-log query path for back-compat) but are no longer the
        # source of truth the API reads.
        self.suppressed_rows = suppressed_rows
        self.llm_cost = llm_cost
        self.regressions_used = regressions_used
        self._call_idx = 0

    async def get(self, _cls: Any, key: Any) -> Any:
        return self.engagement

    async def execute(self, stmt: Any) -> Any:
        sql = str(stmt).lower()
        # Order-independent dispatch based on SQL fragments.
        if "from tool_runs" in sql:
            return _FakeScalar(self.n_tools)
        if "from approvals" in sql and "group by" in sql:
            return _FakeGroupByResult(self.approval_tuples)
        if "from suppressed_tools" in sql:
            # R3 #J2 Bug 2: dedicated source of truth.
            return _FakeRowsResult(self.suppressed_rows)
        if "from audit_log" in sql:
            # Legacy path (kept for tests that haven't migrated). Returns
            # empty so the new table is the only thing that counts.
            return _FakeRowsResult([])
        if "from llm_calls" in sql and "sum" in sql:
            return _FakeScalar(self.llm_cost)
        if "from regressions" in sql:
            # This is the COUNT on regressions.
            return _FakeScalar(self.regressions_used)
        # Unknown: produce an empty result rather than blowing up.
        return _FakeRowsResult([])


def _make_engagement_row() -> EngagementRow:
    now = datetime.now(UTC)
    return EngagementRow(
        id=uuid4(),
        name="seeded",
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


@pytest.mark.asyncio
async def test_summary_aggregates_seeded_counts() -> None:
    """Brief-mandated seed: 10 tool runs (3 auto-approved, 2 critic-vetoed,
    1 human-approved, 4 default), 1 suppressed tool, 1 applied regression."""
    eng = _make_engagement_row()
    session = _SeededSession(
        engagement=eng,
        n_tools=10,
        approval_tuples=[
            ("auto-policy", "approve", 3),
            ("critic", "reject", 2),
            ("alice", "approve", 1),
        ],
        suppressed_rows=[
            SuppressedToolRow(
                id=uuid4(),
                engagement_id=eng.id,
                tool_name="mimikatz_via_worker",
                reason="worker_disabled",
                stage="lateral",
            ),
        ],
        llm_cost=12.34,
        regressions_used=1,
    )

    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)

    assert isinstance(out, EngagementSummary)
    assert out.tool_calls_total == 10
    assert out.tool_calls_auto_approved == 3
    assert out.tool_calls_critic_vetoed == 2
    # R3 #I3 Bug 1: the gated-human counter is renamed; "ungated" is the
    # arithmetic remainder (10 - 3 - 2 - 1 = 4).
    assert out.tool_calls_human_approved_gated == 1
    assert out.tool_calls_ungated == 4
    assert out.suppressed_tools_count == 1
    assert out.llm_cost_usd == 12.34
    # Wallclock should be ~120 minutes (set up 2h ago). Allow a wide
    # tolerance — the seed datetime is "now" minus 2h, but clock skew
    # between the test setup and the endpoint call is in seconds.
    assert 100.0 < out.wallclock_minutes_used < 140.0
    assert out.regressions_used == 1
    assert out.tool_cost_usd == 0.0  # placeholder until R4 ships tool pricing


@pytest.mark.asyncio
async def test_summary_404_when_engagement_missing() -> None:
    """No engagement row → 404."""
    session = _SeededSession(
        engagement=None,
        n_tools=0,
        approval_tuples=[],
        suppressed_rows=[],
        llm_cost=0.0,
        regressions_used=0,
    )
    with _SummaryTestCtx(session), pytest.raises(NotFoundError):
        await get_engagement_summary(uuid4())


@pytest.mark.asyncio
async def test_summary_resilient_to_partial_outage() -> None:
    """An individual counter failure must NOT 5xx the operator out. The
    counter silently falls to 0 and the other counters still surface.

    We simulate this by raising ``RuntimeError`` for the tool_runs query
    only; everything else returns canned data.
    """
    eng = _make_engagement_row()

    class _PartialSession(_SeededSession):
        async def execute(self, stmt: Any) -> Any:
            sql = str(stmt).lower()
            if "from tool_runs" in sql:
                raise RuntimeError("tool_runs index corrupt")
            return await super().execute(stmt)

    session = _PartialSession(
        engagement=eng,
        n_tools=999,  # ignored — the query raises
        approval_tuples=[("auto-policy", "approve", 5)],
        suppressed_rows=[],
        llm_cost=2.0,
        regressions_used=0,
    )
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    # tool_calls_total fell back to 0; other counters survived.
    assert out.tool_calls_total == 0
    assert out.tool_calls_auto_approved == 5
    assert out.llm_cost_usd == 2.0


def test_engagement_summary_default_shape() -> None:
    """All counters default to 0; cost fields default to 0.0. The UI
    relies on this so it can render placeholders without null-checks."""
    s = EngagementSummary()
    assert s.tool_calls_total == 0
    assert s.tool_calls_auto_approved == 0
    assert s.tool_calls_critic_vetoed == 0
    # R3 #I3 Bug 1: the gated counter replaces the old over-counting field.
    assert s.tool_calls_human_approved_gated == 0
    assert s.tool_calls_ungated == 0
    assert s.suppressed_tools_count == 0
    assert s.llm_cost_usd == 0.0
    assert s.tool_cost_usd == 0.0
    assert s.wallclock_minutes_used == 0.0
    assert s.regressions_used == 0
    assert s.findings_count == 0


# --------------------------------------------------------------------------- #
# R3 #I3 Bug 1 — ungated-counter coverage                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_summary_brief_seed_3_auto_2_veto_1_human_4_ungated() -> None:
    """R3 #I3 Bug 1: brief-mandated seed for the renamed counter shape.

    10 tool runs total:
      - 3 auto-approved (approval rows with decided_by='auto-policy')
      - 2 critic-vetoed (approval rows with decided_by='critic', decision='reject')
      - 1 human-approved gated (approval row with decided_by='alice')
      - 4 ungated (no approval row at all — derived as ``total -
        auto - veto - human_gated``).

    Asserts each of the four counters independently.
    """
    eng = _make_engagement_row()
    session = _SeededSession(
        engagement=eng,
        n_tools=10,
        approval_tuples=[
            ("auto-policy", "approve", 3),
            ("critic", "reject", 2),
            ("alice", "approve", 1),
        ],
        suppressed_rows=[],
        llm_cost=0.0,
        regressions_used=0,
    )
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.tool_calls_total == 10
    assert out.tool_calls_auto_approved == 3
    assert out.tool_calls_critic_vetoed == 2
    assert out.tool_calls_human_approved_gated == 1
    assert out.tool_calls_ungated == 4


@pytest.mark.asyncio
async def test_summary_ungated_floor_at_zero() -> None:
    """R3 #I3 Bug 1: when approval-row counts somehow exceed tool_runs
    (eg. stale rows from a deleted run), ``tool_calls_ungated`` must NOT
    go negative — clamp at 0 so the UI never renders nonsense."""
    eng = _make_engagement_row()
    session = _SeededSession(
        engagement=eng,
        n_tools=2,  # fewer tool_runs than approvals
        approval_tuples=[
            ("auto-policy", "approve", 5),  # 5 > 2
        ],
        suppressed_rows=[],
        llm_cost=0.0,
        regressions_used=0,
    )
    with _SummaryTestCtx(session):
        out = await get_engagement_summary(eng.id)
    assert out.tool_calls_ungated == 0


# --------------------------------------------------------------------------- #
# R3 #I3 Bug 2 — findings_count via Neo4j repo                                #
# --------------------------------------------------------------------------- #


class _FakeFindingsRepo:
    """Stand-in for Neo4jRepo. ``count(engagement_id)`` returns a fixed
    label-count dict; the ``coverage`` helper picks ``"Finding"`` from
    that map."""

    def __init__(self, n_findings: int) -> None:
        self._n = n_findings

    async def count(self, _engagement_id: UUID) -> dict[str, int]:
        return {"Finding": self._n, "Host": 0, "Service": 0}


@pytest.mark.asyncio
async def test_summary_findings_count_reads_from_neo4j() -> None:
    """R3 #I3 Bug 2: seed 5 Finding nodes via the fake repo; assert that
    ``findings_count == 5`` instead of the old hard-coded 0."""
    eng = _make_engagement_row()
    session = _SeededSession(
        engagement=eng,
        n_tools=0,
        approval_tuples=[],
        suppressed_rows=[],
        llm_cost=0.0,
        regressions_used=0,
    )
    repo = _FakeFindingsRepo(n_findings=5)
    import orchestrator.services.findings as _findings_mod
    orig_repo = _findings_mod.get_repo
    _findings_mod.get_repo = lambda: repo  # type: ignore[assignment]
    try:
        with _SummaryTestCtx(session):
            out = await get_engagement_summary(eng.id)
    finally:
        _findings_mod.get_repo = orig_repo
    assert out.findings_count == 5


@pytest.mark.asyncio
async def test_summary_findings_count_falls_back_to_zero_when_neo4j_down() -> None:
    """R3 #I3 Bug 2: a raising repo (eg. Neo4j unreachable) must NOT 5xx
    the digest — the counter quietly falls back to 0."""

    class _BoomRepo:
        async def count(self, _engagement_id: UUID) -> dict[str, int]:
            raise RuntimeError("neo4j down")

    eng = _make_engagement_row()
    session = _SeededSession(
        engagement=eng,
        n_tools=0,
        approval_tuples=[],
        suppressed_rows=[],
        llm_cost=0.0,
        regressions_used=0,
    )
    import orchestrator.services.findings as _findings_mod
    orig_repo = _findings_mod.get_repo
    _findings_mod.get_repo = lambda: _BoomRepo()  # type: ignore[assignment]
    try:
        with _SummaryTestCtx(session):
            out = await get_engagement_summary(eng.id)
    finally:
        _findings_mod.get_repo = orig_repo
    assert out.findings_count == 0
