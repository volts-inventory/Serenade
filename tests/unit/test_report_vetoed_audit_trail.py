"""R3 #H3 Bug 3: report renderer surfaces critic identity + audit_row_id
on vetoed findings, AND a per-critic breakdown on council regressions.

Covers:
- A fixture ``DraftReport`` with one vetoed finding (carrying
  ``audit_row_id``, ``critic_name``, ``critic_confidence``, ``tool_run_ids``)
  renders all four fields in the markdown output.
- A fixture ``RegressionAppendix`` carrying ``member_critiques`` renders
  a per-critic table (one row per critic) in the markdown output.
- Both fields default to empty so engagements with no veto / no
  council-decided regression still render cleanly.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from agents.report import DraftFinding, DraftReport, RegressionAppendix
from reports.renderer import render_markdown


def test_vetoed_finding_renders_audit_trail() -> None:
    """The markdown renderer must surface every audit-trail field a
    customer auditor needs to pivot from "rejected finding X" back to
    the audit_log row + raw tool runs."""
    audit_row_id = str(uuid4())
    tr1, tr2 = str(uuid4()), str(uuid4())

    finding = DraftFinding(
        title="SQLi in /api/login",
        description="Stacked-query SQLi observed via sqlmap; rejected as duplicate.",
        severity="high",
        critic_verdict="veto",
        critic_concern="duplicate of finding-12 (same parameter, same payload class)",
        audit_row_id=audit_row_id,
        critic_name="ReportCritic",
        critic_confidence=0.92,
        tool_run_ids=[tr1, tr2],
    )

    draft = DraftReport(
        engagement_id=uuid4(),
        title="Test Engagement",
        executive_summary="-",
        coverage={"hosts": 1},
        vetoed_findings=[finding],
    )

    md = render_markdown(draft)
    assert audit_row_id in md, "audit_row_id must appear in the rendered markdown"
    assert "ReportCritic" in md, "critic_name must appear"
    assert "0.92" in md, "critic_confidence must appear"
    assert tr1 in md and tr2 in md, "tool_run_ids must appear"
    # The rationale text still flows through.
    assert "duplicate of finding-12" in md


def test_regression_council_breakdown_renders_per_critic() -> None:
    """``RegressionAppendix.member_critiques`` must render as a per-critic
    table — one row per critic with the verdict / confidence / concern.
    """
    breakdown = [
        {
            "critic_name": "NoveltyCritic",
            "verdict": "approve",
            "confidence": 0.9,
            "primary_concern": "delta covers a previously-unseen CIDR",
        },
        {
            "critic_name": "BudgetCritic",
            "verdict": "approve",
            "confidence": 0.7,
            "primary_concern": "regressions_used=1, under cap",
        },
        {
            "critic_name": "ScopeCritic",
            "verdict": "veto",
            "confidence": 0.85,
            "primary_concern": "dissenting on scope risk",
        },
    ]
    r = RegressionAppendix(
        target_stage="lateral",
        trigger_kind="novelty",
        rationale="found a new domain controller in the post-exploit slice",
        council_decision="approve",
        council_rationale="2 of 3 approve",
        requested_at="2025-01-01T00:00:00Z",
        member_critiques=breakdown,
    )
    draft = DraftReport(
        engagement_id=uuid4(),
        title="Test Engagement",
        executive_summary="-",
        coverage={"hosts": 1},
        regressions=[r],
    )

    md = render_markdown(draft)
    assert "NoveltyCritic" in md
    assert "BudgetCritic" in md
    assert "ScopeCritic" in md
    # Each critic's verdict appears.
    assert "approve" in md
    assert "veto" in md
    # The concerns appear.
    assert "delta covers a previously-unseen CIDR" in md
    assert "dissenting on scope risk" in md


def test_empty_member_critiques_does_not_render_breakdown() -> None:
    """When ``member_critiques`` is empty (legacy / single-critic mode)
    we should not render an empty per-critic table — the regression row
    itself still appears."""
    r = RegressionAppendix(
        target_stage="exploit",
        trigger_kind="evidence_gap",
        rationale="missing evidence on the SQLi finding",
        council_decision="approve",
        council_rationale="single-critic mode",
        requested_at="2025-01-01T00:00:00Z",
        member_critiques=[],
    )
    draft = DraftReport(
        engagement_id=uuid4(),
        title="Test Engagement",
        executive_summary="-",
        coverage={"hosts": 1},
        regressions=[r],
    )
    md = render_markdown(draft)
    # Regression row itself surfaces.
    assert "exploit" in md
    assert "evidence_gap" in md
    # The "council breakdown" heading must NOT appear when there are no critics.
    assert "council breakdown" not in md


def test_finding_without_veto_does_not_force_audit_fields() -> None:
    """An accepted finding must still render cleanly; the new audit
    fields default to None / [] so the renderer's ``{% if %}`` gates
    keep them invisible."""
    finding = DraftFinding(
        title="Accepted finding",
        description="ok",
        severity="low",
        critic_verdict="approve",
    )
    draft = DraftReport(
        engagement_id=uuid4(),
        title="t",
        executive_summary="-",
        coverage={"hosts": 1},
        findings=[finding],
    )
    md = render_markdown(draft)
    assert "Accepted finding" in md
    # No audit-row label leaks into accepted findings (we never set it).
    assert "**Audit row**" not in md


# --------------------------------------------------------------------------- #
# R3 #I3 Bug 3 — audit_row_id matches the actual inserted AuditLogRow.id      #
# --------------------------------------------------------------------------- #
#
# The prior H3 implementation minted a fresh uuid4 for the finding stamp
# AND called ``ledger.audit()`` (which inserted an AuditLogRow with its
# OWN SQLA-default uuid). The two ids diverged. An auditor pivoting from
# the finding's audit_row_id back to the ledger via "SELECT * FROM
# audit_log WHERE id = $audit_row_id" found nothing.
#
# Fix: mint a single uuid here, insert AuditLogRow(id=<that-uuid>, ...)
# directly via session_scope, and stamp the same uuid onto the finding.
# This test drives ``ReportAgent._review_findings`` against a fake
# ReportCritic that issues a VETO, captures the in-flight AuditLogRow
# from a stubbed ``session_scope``, and asserts the captured row's id
# equals the finding's audit_row_id.


@pytest.mark.asyncio
async def test_vetoed_finding_audit_row_id_matches_inserted_audit_log_row(
    monkeypatch, tmp_path
) -> None:
    from unittest.mock import AsyncMock, MagicMock
    from uuid import UUID
    from contextlib import asynccontextmanager


    # Lazy import — keeps the rest of the test file importable when the
    # full agent dep graph isn't available.
    from agents.report import ReportAgent
    from agents.critic import Critique, Verdict
    from orchestrator import crypto
    from orchestrator import db as _db_module
    from orchestrator.settings import get_settings

    # The veto path now routes through ``ledger.append_audit_row`` which
    # computes the HMAC chain — keep the master key in a tmp dir.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()

    captured_rows: list[Any] = []

    class _EmptyResult:
        def scalars(self):  # noqa: ANN001, ANN201
            class _S:
                def first(self):  # noqa: ANN001, ANN201
                    return None

            return _S()

    class _CaptureSession:
        def add(self, row: Any) -> None:
            captured_rows.append(row)

        async def execute(self, *_a, **_k):  # noqa: ANN002, ANN003, ANN201
            # No prior rows → genesis (seq=1, prev_hash=None).
            return _EmptyResult()

        async def flush(self) -> None:
            return None

        def begin_nested(self):  # noqa: ANN201
            # SAVEPOINT used by ledger.append_audit_row; a no-op CM here.
            class _Savepoint:
                async def __aenter__(self_inner):  # noqa: N805, ANN001, ANN204
                    return None

                async def __aexit__(self_inner, *exc):  # noqa: N805, ANN001, ANN204
                    return False

            return _Savepoint()

    @asynccontextmanager
    async def _fake_session_scope():
        yield _CaptureSession()

    # Patch BOTH the canonical export and the rebound name inside the
    # report module — the production code does ``from orchestrator.db
    # import session_scope`` inside the try-block, so we monkey-patch
    # the module-level attribute that the import resolves to.
    orig = _db_module.session_scope
    _db_module.session_scope = _fake_session_scope  # type: ignore[assignment]
    try:
        # Build a minimal ReportAgent with a stubbed ReportCritic that
        # always VETOs.
        critic = MagicMock()
        critic.review = AsyncMock(
            return_value=Critique(
                verdict=Verdict.VETO,
                confidence=0.92,
                primary_concern="duplicate of existing finding",
                critic_name="ReportCritic",
            )
        )

        agent = ReportAgent(
            router=MagicMock(),
            repo=MagicMock(),
            registry=MagicMock(),
            policy=MagicMock(),
            sandbox=MagicMock(),
            report_critic=critic,
        )

        finding = DraftFinding(
            title="vetoed finding",
            description="-",
            severity="high",
            evidence_refs=[
                {"tool_run_id": str(uuid4())},
                {"tool_run_id": str(uuid4())},
            ],
        )
        engagement_id = uuid4()
        draft = DraftReport(
            engagement_id=engagement_id,
            title="t",
            executive_summary="-",
            coverage={"hosts": 1},
            findings=[finding],
        )

        ctx = MagicMock()
        ctx.engagement_id = engagement_id

        await agent._review_findings(ctx, draft)

        # Critique fired → veto → finding moved to vetoed bucket.
        assert finding in draft.vetoed_findings
        assert finding.critic_verdict == "veto"
        assert finding.audit_row_id is not None
        # Exactly one AuditLogRow was inserted.
        assert len(captured_rows) == 1
        inserted = captured_rows[0]
        # The inserted row's id matches the stamp on the finding — the
        # whole point of Bug 3. Compare via UUID() to normalise str/UUID.
        assert UUID(finding.audit_row_id) == inserted.id
        # The inserted row also carries the right engagement_id, action,
        # and the same audit_row_id inside its payload.
        assert inserted.engagement_id == engagement_id
        assert inserted.action == "report.finding_vetoed"
        assert inserted.payload["audit_row_id"] == finding.audit_row_id
        assert inserted.actor == "critic:ReportCritic"
        # Genesis row of this engagement's chain: seq=1, no prev_hash, a
        # populated row_hash.
        assert inserted.seq == 1
        assert inserted.prev_hash is None
        assert inserted.row_hash
    finally:
        _db_module.session_scope = orig
        get_settings.cache_clear()
        crypto.reset_cache()


@pytest.mark.asyncio
async def test_vetoed_finding_audit_row_id_unset_when_insert_fails() -> None:
    """R3 #I3 Bug 3: when the audit-row insert fails (Postgres unreachable,
    schema mismatch, etc.) the finding must NOT carry a dangling
    audit_row_id pointer. Leave it as None so the renderer's ``{% if %}``
    gate hides the field."""
    from unittest.mock import AsyncMock, MagicMock
    from contextlib import asynccontextmanager

    from agents.report import ReportAgent
    from agents.critic import Critique, Verdict
    from orchestrator import db as _db_module

    @asynccontextmanager
    async def _boom_session_scope():
        yield None
        raise RuntimeError("postgres unreachable")

    orig = _db_module.session_scope
    _db_module.session_scope = _boom_session_scope  # type: ignore[assignment]
    try:
        critic = MagicMock()
        critic.review = AsyncMock(
            return_value=Critique(
                verdict=Verdict.VETO,
                confidence=0.5,
                primary_concern="x",
                critic_name="ReportCritic",
            )
        )
        agent = ReportAgent(
            router=MagicMock(),
            repo=MagicMock(),
            registry=MagicMock(),
            policy=MagicMock(),
            sandbox=MagicMock(),
            report_critic=critic,
        )
        finding = DraftFinding(
            title="vetoed",
            description="-",
            severity="high",
        )
        engagement_id = uuid4()
        draft = DraftReport(
            engagement_id=engagement_id,
            title="t",
            executive_summary="-",
            coverage={"hosts": 1},
            findings=[finding],
        )
        ctx = MagicMock()
        ctx.engagement_id = engagement_id

        await agent._review_findings(ctx, draft)

        # Veto still recorded on the finding (verdict + bucket) — only
        # the cross-system pointer is suppressed when the insert fails.
        assert finding.critic_verdict == "veto"
        assert finding in draft.vetoed_findings
        assert finding.audit_row_id is None
    finally:
        _db_module.session_scope = orig
