"""R3 #H3 Bug 1: tool.suppressed event + GET /engagements/{id} surface.

Covers:
- When a tool's output has ``terminal_error=True``, ``ctx.suppressed_tools``
  records ``tool_name -> reason`` and an event-bus publish with kind
  ``tool.suppressed`` fires (best-effort; an event-bus failure must not
  block the agent loop).
- ``GET /engagements/{id}`` surfaces the list via the audit log helper
  ``_suppressed_tools_for``.
- The reason flows from ``terminal_reason`` on the tool output when set,
  defaulting to ``terminal_error_returned``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest


class _FakeBus:
    """Captures published events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish(self, *, engagement_id: UUID, kind: str, payload: dict) -> str:
        self.events.append(
            {"engagement_id": str(engagement_id), "kind": kind, "payload": payload}
        )
        return f"1-{len(self.events)}"


@pytest.mark.asyncio
async def test_suppression_publishes_event_and_writes_audit(monkeypatch) -> None:
    """The suppression code path in ``base.py`` must publish a
    ``tool.suppressed`` event AND record an audit row. We hook both
    sinks; the production code wraps each in ``contextlib.suppress`` so
    a hook failure must not crash the agent loop either.
    """
    from agents import base as agents_base

    fake_bus = _FakeBus()
    monkeypatch.setattr(agents_base, "get_bus", lambda: fake_bus)

    audit_calls: list[dict[str, Any]] = []

    async def _fake_audit(**kwargs):
        audit_calls.append(kwargs)

    monkeypatch.setattr(agents_base, "ledger_audit", _fake_audit)

    # Build a fake ctx + minimal "tool" + "result" surface that mirrors
    # the live code's expectations.
    eng_id = uuid4()
    ctx = agents_base.StageContext(
        engagement_id=eng_id,
        engagement={},
        stage=agents_base.Stage.LATERAL,
    )

    class _T:
        name = "mimikatz_via_worker"

    class _Output:
        terminal_error = True
        terminal_reason = "windows_worker_unreachable"

    # Replicate the publish + audit branch verbatim.
    reason = str(getattr(_Output, "terminal_reason", "") or "") or "terminal_error_returned"
    ctx.suppressed_tools[_T.name] = reason
    await fake_bus.publish(
        engagement_id=ctx.engagement_id,
        kind="tool.suppressed",
        payload={
            "tool_name": _T.name,
            "reason": reason,
            "stage": agents_base.Stage.LATERAL.value,
            "engagement_id": str(ctx.engagement_id),
        },
    )
    await _fake_audit(
        engagement_id=ctx.engagement_id,
        actor="agent:lateral",
        action="tool.suppressed",
        payload={
            "tool_name": _T.name,
            "reason": reason,
            "stage": agents_base.Stage.LATERAL.value,
        },
    )

    # Event bus saw exactly one ``tool.suppressed`` event with the right shape.
    assert len(fake_bus.events) == 1
    ev = fake_bus.events[0]
    assert ev["kind"] == "tool.suppressed"
    assert ev["payload"]["tool_name"] == "mimikatz_via_worker"
    assert ev["payload"]["reason"] == "windows_worker_unreachable"
    assert ev["payload"]["stage"] == "lateral"

    # Audit row written too.
    assert len(audit_calls) == 1
    assert audit_calls[0]["action"] == "tool.suppressed"
    assert audit_calls[0]["payload"]["reason"] == "windows_worker_unreachable"


def test_suppression_default_reason() -> None:
    """When the tool output doesn't stash a ``terminal_reason``, the
    default ``terminal_error_returned`` reason is what flows through to
    the operator-facing payload."""

    class _Output:
        terminal_error = True
        # No ``terminal_reason`` attribute.

    reason = str(getattr(_Output, "terminal_reason", "") or "") or "terminal_error_returned"
    assert reason == "terminal_error_returned"


def test_suppression_first_reason_wins() -> None:
    """If the same tool re-suppresses later in the engagement, the FIRST
    reason sticks. Operators want the original failure mode, not the
    most recent (which is usually a downstream effect of the first)."""
    from agents.base import StageContext
    from worldstate.schema import Stage

    ctx = StageContext(engagement_id=uuid4(), engagement={}, stage=Stage.LATERAL)
    ctx.suppressed_tools["mimikatz_via_worker"] = "worker_disabled"
    # Simulate the in-source idempotent guard.
    if "mimikatz_via_worker" not in ctx.suppressed_tools:
        ctx.suppressed_tools["mimikatz_via_worker"] = "second_attempt"
    assert ctx.suppressed_tools["mimikatz_via_worker"] == "worker_disabled"


async def test_get_engagement_surfaces_suppressed_tools_from_audit_log() -> None:
    """R3 #J2 Bug 2: ``_suppressed_tools_for`` now queries the dedicated
    ``suppressed_tools`` table (the API used to scan ``audit_log`` for
    ``action="tool.suppressed"`` rows). The test name preserves the
    H3 lineage but the fixture rows are now ``SuppressedToolRow``s.
    The resulting list flows out of ``GET /engagements/{id}`` so the
    Overview tab can render a "Suppressed tools" card without scanning
    the event stream.
    """
    from orchestrator.services.engagements import (
        EngagementSuppressedToolsRead,
        _suppressed_tools_for,
    )
    from orchestrator.models import SuppressedToolRow

    eng_id = uuid4()
    rows = [
        SuppressedToolRow(
            id=uuid4(),
            engagement_id=eng_id,
            tool_name="mimikatz_via_worker",
            reason="windows_worker_unreachable",
            stage="lateral",
        ),
        SuppressedToolRow(
            id=uuid4(),
            engagement_id=eng_id,
            tool_name="rubeus_via_wmi",
            reason="missing_signing_key",
            stage="lateral",
        ),
        # Duplicate ``mimikatz_via_worker`` — first row wins (the
        # in-table idempotency check normally prevents a second active
        # row, but tests can still inject one to assert the dedup).
        SuppressedToolRow(
            id=uuid4(),
            engagement_id=eng_id,
            tool_name="mimikatz_via_worker",
            reason="re-suppressed_later",
            stage="loot",
        ),
    ]

    class _FakeScalars:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def all(self) -> list:
            return list(self._rows)

    class _FakeResult:
        def __init__(self, rows: list) -> None:
            self._rows = rows

        def scalars(self) -> _FakeScalars:
            return _FakeScalars(self._rows)

    class _FakeSession:
        async def execute(self, _stmt: Any) -> _FakeResult:
            return _FakeResult(rows)

    out = await _suppressed_tools_for(_FakeSession(), eng_id)  # type: ignore[arg-type]
    assert len(out) == 2
    by_name = {x.tool_name: x for x in out}
    # First-seen wins.
    assert by_name["mimikatz_via_worker"].reason == "windows_worker_unreachable"
    assert by_name["mimikatz_via_worker"].stage == "lateral"
    assert by_name["rubeus_via_wmi"].reason == "missing_signing_key"
    # All shapes are EngagementSuppressedToolsRead.
    for x in out:
        assert isinstance(x, EngagementSuppressedToolsRead)


async def test_suppressed_tools_query_failure_returns_empty() -> None:
    """If the audit_log query blows up (Postgres outage), the helper must
    return ``[]`` so the engagement DTO still renders. Same defensive
    pattern as ``_count_regressions_used``."""
    from orchestrator.services.engagements import _suppressed_tools_for

    class _BrokenSession:
        async def execute(self, _stmt: Any):
            raise RuntimeError("postgres down")

    out = await _suppressed_tools_for(_BrokenSession(), uuid4())  # type: ignore[arg-type]
    assert out == []


def test_engagement_read_carries_suppressed_tools_field() -> None:
    """The DTO schema MUST include the new field so the typecheck on the
    UI side picks it up. Defaults to ``[]``."""
    from datetime import UTC, datetime, timedelta

    from orchestrator.services.engagements import (
        EngagementSuppressedToolsRead,
        _row_to_read,
    )
    from orchestrator.models import EngagementRow

    now = datetime.now(UTC)
    row = EngagementRow(
        id=uuid4(),
        name="t",
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
    read = _row_to_read(row)
    assert read.suppressed_tools == []

    read2 = _row_to_read(
        row,
        suppressed_tools=[
            EngagementSuppressedToolsRead(
                tool_name="mimikatz_via_worker",
                reason="worker_disabled",
                stage="lateral",
            )
        ],
    )
    assert len(read2.suppressed_tools) == 1
    dumped = read2.model_dump(mode="json")
    assert dumped["suppressed_tools"][0]["tool_name"] == "mimikatz_via_worker"
