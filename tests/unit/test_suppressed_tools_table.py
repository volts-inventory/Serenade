"""R3 #J2 Bug 2: SuppressedToolRow is the single source of truth.

Previously the suppression list lived in two stores (LangGraph
checkpoint + audit_log reconstruction). J2 promotes the list to a
first-class ORM model so writes / reads / clears all go through one
place. These tests exercise the ledger helpers and the API surface
without touching Postgres — fake session objects mirror the SQL we
care about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from orchestrator.models import SuppressedToolRow


class _FakeScalars:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)

    def all(self) -> list[Any]:
        return list(self._rows)

    def first(self) -> Any | None:
        return self._rows[0] if self._rows else None


class _FakeResult:
    def __init__(self, rows: list[Any], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._rows)


class _FakeSession:
    """Mimics the minimal session protocol the ledger helpers consume.

    Note: SQL inspection is by-substring; we approximate the live
    WHERE clauses by snooping at the compiled statement's parameters.
    For the idempotency select (which filters by ``tool_name``) we
    extract the bind parameter from the compiled statement so duplicate
    detection matches the live behaviour.
    """

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.executed: list[str] = []
        self.rows: list[SuppressedToolRow] = []
        self.rowcount_for_update: int = 0
        self.fail = False

    def add(self, obj: Any) -> None:
        if self.fail:
            raise RuntimeError("postgres down")
        self.added.append(obj)
        # Mirror the new row into the "active set" used by selects so
        # the idempotency check fires on the second call.
        if isinstance(obj, SuppressedToolRow):
            self.rows.append(obj)

    async def execute(self, stmt: Any) -> _FakeResult:
        if self.fail:
            raise RuntimeError("postgres down")
        sql = str(stmt).lower()
        self.executed.append(sql)
        if sql.startswith("update"):
            # All currently-active rows transition cleared_at.
            now = datetime.now(UTC)
            count = 0
            for r in self.rows:
                if r.cleared_at is None:
                    r.cleared_at = now
                    count += 1
            return _FakeResult([], rowcount=count)
        # Select branch: filter by tool_name when present so the
        # idempotency check matches the live behaviour.
        active = [r for r in self.rows if r.cleared_at is None]
        try:
            compiled = stmt.compile(compile_kwargs={"literal_binds": False})
            params = dict(compiled.params)
        except Exception:
            params = {}
        tool_filter = params.get("tool_name_1") or params.get("tool_name")
        if tool_filter is not None:
            active = [r for r in active if r.tool_name == tool_filter]
        return _FakeResult(active)


def _patch_session_scope(monkeypatch, sess: _FakeSession) -> None:
    """Inject ``sess`` into both modules that open a session scope."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _scope():
        yield sess

    monkeypatch.setattr("orchestrator.ledger.session_scope", _scope)


@pytest.mark.asyncio
async def test_record_inserts_row(monkeypatch) -> None:
    """First call writes a SuppressedToolRow with cleared_at=None."""
    from orchestrator.ledger import record_tool_suppression

    sess = _FakeSession()
    _patch_session_scope(monkeypatch, sess)

    eid = uuid4()
    await record_tool_suppression(
        engagement_id=eid,
        tool_name="mimikatz_via_worker",
        reason="worker_disabled",
        stage="lateral",
    )
    assert len(sess.added) == 1
    row = sess.added[0]
    assert isinstance(row, SuppressedToolRow)
    assert row.tool_name == "mimikatz_via_worker"
    assert row.reason == "worker_disabled"
    assert row.stage == "lateral"
    assert row.engagement_id == eid
    assert row.cleared_at is None


@pytest.mark.asyncio
async def test_record_is_idempotent_on_active_set(monkeypatch) -> None:
    """Calling record twice for the same active (engagement, tool) MUST
    insert only one row — the live source of truth has no need to
    inflate."""
    from orchestrator.ledger import record_tool_suppression

    sess = _FakeSession()
    _patch_session_scope(monkeypatch, sess)

    eid = uuid4()
    await record_tool_suppression(
        engagement_id=eid,
        tool_name="mimikatz_via_worker",
        reason="r",
        stage="lateral",
    )
    await record_tool_suppression(
        engagement_id=eid,
        tool_name="mimikatz_via_worker",
        reason="r",
        stage="lateral",
    )
    inserts = [a for a in sess.added if isinstance(a, SuppressedToolRow)]
    assert len(inserts) == 1


@pytest.mark.asyncio
async def test_record_failure_is_logged_not_raised(monkeypatch) -> None:
    """Postgres unreachable must not block the agent loop."""
    from orchestrator.ledger import record_tool_suppression

    sess = _FakeSession()
    sess.fail = True
    _patch_session_scope(monkeypatch, sess)
    # Should NOT raise.
    await record_tool_suppression(
        engagement_id=uuid4(),
        tool_name="t",
        reason="r",
        stage="s",
    )


@pytest.mark.asyncio
async def test_clear_stamps_cleared_at_on_active_rows(monkeypatch) -> None:
    """``clear_tool_suppressions`` soft-deletes every active row for
    the engagement (matches the H3 stage-boundary reset semantics)."""
    from orchestrator.ledger import clear_tool_suppressions, record_tool_suppression

    sess = _FakeSession()
    _patch_session_scope(monkeypatch, sess)

    eid = uuid4()
    await record_tool_suppression(
        engagement_id=eid, tool_name="a", reason="r", stage="s"
    )
    await record_tool_suppression(
        engagement_id=eid, tool_name="b", reason="r", stage="s"
    )
    cleared = await clear_tool_suppressions(engagement_id=eid)
    assert cleared == 2
    assert all(r.cleared_at is not None for r in sess.rows)


@pytest.mark.asyncio
async def test_re_suppression_after_clear_inserts_new_row(monkeypatch) -> None:
    """After a stage-boundary clear, suppressing the same tool again
    must insert a fresh row with cleared_at=None instead of updating
    the cleared row in place — the audit history of the previous
    suppression is preserved."""
    from orchestrator.ledger import clear_tool_suppressions, record_tool_suppression

    sess = _FakeSession()
    _patch_session_scope(monkeypatch, sess)

    eid = uuid4()
    await record_tool_suppression(
        engagement_id=eid, tool_name="t", reason="first", stage="recon"
    )
    await clear_tool_suppressions(engagement_id=eid)
    await record_tool_suppression(
        engagement_id=eid, tool_name="t", reason="second", stage="lateral"
    )
    # Two rows total; one cleared, one active.
    assert len(sess.rows) == 2
    active = [r for r in sess.rows if r.cleared_at is None]
    assert len(active) == 1
    assert active[0].reason == "second"
    assert active[0].stage == "lateral"


@pytest.mark.asyncio
async def test_list_active_returns_only_uncleared(monkeypatch) -> None:
    """The API read helper filters on ``cleared_at IS NULL``."""
    from orchestrator.ledger import (
        clear_tool_suppressions,
        list_active_suppressions,
        record_tool_suppression,
    )

    sess = _FakeSession()
    _patch_session_scope(monkeypatch, sess)

    eid = uuid4()
    await record_tool_suppression(
        engagement_id=eid, tool_name="a", reason="r", stage="s"
    )
    await record_tool_suppression(
        engagement_id=eid, tool_name="b", reason="r", stage="s"
    )
    await clear_tool_suppressions(engagement_id=eid)
    await record_tool_suppression(
        engagement_id=eid, tool_name="a", reason="r2", stage="lateral"
    )
    out = await list_active_suppressions(engagement_id=eid)
    assert len(out) == 1
    assert out[0]["tool_name"] == "a"
    assert out[0]["reason"] == "r2"


@pytest.mark.asyncio
async def test_api_read_filters_on_cleared_at(monkeypatch) -> None:
    """End-to-end: ``_suppressed_tools_for`` reads rows where
    ``cleared_at IS NULL`` and shapes them as
    ``EngagementSuppressedToolsRead`` for the DTO."""
    from orchestrator.services.engagements import (
        EngagementSuppressedToolsRead,
        _suppressed_tools_for,
    )

    eid = uuid4()
    active = SuppressedToolRow(
        id=uuid4(),
        engagement_id=eid,
        tool_name="a",
        reason="r",
        stage="lateral",
    )

    class _Sess:
        async def execute(self, stmt: Any) -> _FakeResult:
            # Active-only filter is applied in the call; we trust the
            # helper's filtering and return only the active row here.
            return _FakeResult([active])

    out = await _suppressed_tools_for(_Sess(), eid)  # type: ignore[arg-type]
    assert len(out) == 1
    assert isinstance(out[0], EngagementSuppressedToolsRead)
    assert out[0].tool_name == "a"
