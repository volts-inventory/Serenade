"""R3 #I2 Bug 1 — ReportAgent emits ``StepDecision.REGRESS_TO`` when a
finding's evidence is missing.

Before this fix the report-stage playbook *claimed* to drive a
``REGRESS_TO`` for evidence-less findings but ``ReportAgent`` had no
``_maybe_build_regression_request`` and the stage was terminal in
``_next_stage`` — the playbook lied. This locks the producer half in.

Tests:
- Finding whose ``tool_run_ids`` reference tool_runs that no longer
  exist → REGRESS_TO with the finding's id in the delta.
- Finding with empty evidence → REGRESS_TO with the finding's id in
  the delta.
- All findings have valid evidence → no REGRESS_TO (ADVANCE path).
- ``regressions_used == regression_budget_max`` → no REGRESS_TO even
  when a real evidence gap exists (producer-side budget cap).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agents.base import StageContext, StepDecision
from agents.regression import RegressionRequest, RegressionTriggerKind
from agents.report import ReportAgent
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _stub_repo(findings: list[dict[str, Any]] | None = None) -> MagicMock:
    """Build a Neo4jRepo stub that answers the report-stage queries.

    ``findings`` is a flat list — the stub re-buckets it by severity to
    match ``worldstate.queries.findings_by_severity``'s contract.
    """
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 1,
        "Service": 0,
        "WebApp": 0,
        "Endpoint": 0,
        "Finding": len(findings or []),
        "ExploitCandidate": 0,
        "Session": 0,
    })

    async def get_nodes_by_kind(eid: UUID, kind: str) -> list[dict[str, Any]]:
        # findings_by_severity reads Finding nodes; everything else empty.
        if kind == "Finding":
            return [dict(f) for f in (findings or [])]
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)
    return repo


def _make_report_agent(repo: MagicMock) -> ReportAgent:
    return ReportAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


def _ctx(
    *,
    regressions_used: int = 0,
    regression_budget_max: int = 3,
) -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={
            "approval_policy": "manual",
            "scope": {"cidrs": ["10.0.0.0/16"], "domains": ["corp.example.com"]},
        },
        stage=Stage.REPORT,
        regressions_used=regressions_used,
        regression_budget_max=regression_budget_max,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_finding_with_bad_tool_run_ids_triggers_regress_to() -> None:
    """A finding whose ``tool_run_ids`` all reference tool_runs that
    Postgres can't find → ``REGRESS_TO`` with the finding's id in the
    delta. The originating stage (default ENUMERATION here) is the
    target.
    """
    finding_id = str(uuid4())
    bad_tr1 = uuid4()
    bad_tr2 = uuid4()
    findings = [
        {
            "id": finding_id,
            "title": "Stale finding with vanished tool_runs",
            "description": "tool_runs cited here were rolled back",
            "severity": "high",
            "evidence": [
                {"tool_run_id": str(bad_tr1)},
                {"tool_run_id": str(bad_tr2)},
            ],
            "category": "web",
        }
    ]
    repo = _stub_repo(findings=findings)
    agent = _make_report_agent(repo)
    ctx = _ctx()

    # Mock validate_evidence_refs to claim ALL the cited tool_run_ids
    # are missing — i.e. they exist as UUID-strings on the finding but
    # do not exist as rows in Postgres tool_runs for this engagement.
    async def _all_missing(eid: UUID, ids: list[UUID]) -> list[UUID]:
        return list(ids)

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_all_missing,
    ):
        decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert req.delta == {"finding_id": finding_id, "regenerate_evidence": True}
    assert finding_id in req.evidence
    # Trigger kind is the structural shape; the actual mechanics live
    # in the council. Smoke-check that one of the enumerated kinds was
    # picked (i.e. the request validates).
    assert isinstance(req.trigger_kind, RegressionTriggerKind)
    # Category "web" → ENUMERATION (the web_enum agent's stage).
    assert req.target_stage == Stage.ENUMERATION


@pytest.mark.asyncio
async def test_finding_with_empty_evidence_triggers_regress_to() -> None:
    """A finding with no evidence at all (``evidence_refs == []``) also
    drives a ``REGRESS_TO`` — the persona forbids shipping evidence-less
    claims and the controller can re-enter the originating stage to
    back the claim with a real tool run.
    """
    finding_id = str(uuid4())
    findings = [
        {
            "id": finding_id,
            "title": "Claim with no backing",
            "description": "agent surfaced a finding but never attached evidence",
            "severity": "medium",
            "evidence": [],
            "category": "exploit",
        }
    ]
    repo = _stub_repo(findings=findings)
    agent = _make_report_agent(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert req.delta == {"finding_id": finding_id, "regenerate_evidence": True}
    assert finding_id in req.evidence
    # Category "exploit" → EXPLOIT stage.
    assert req.target_stage == Stage.EXPLOIT


@pytest.mark.asyncio
async def test_all_findings_evidence_valid_does_not_trigger_regress_to() -> None:
    """When every finding has at least one evidence_ref AND every cited
    tool_run validates against Postgres, there is no gap — the agent
    must NOT emit ``REGRESS_TO``.
    """
    good_tr = uuid4()
    findings = [
        {
            "id": str(uuid4()),
            "title": "Backed finding",
            "description": "every cited tool_run exists in Postgres",
            "severity": "low",
            "evidence": [{"tool_run_id": str(good_tr)}],
            "category": "misconfiguration",
        }
    ]
    repo = _stub_repo(findings=findings)
    agent = _make_report_agent(repo)
    ctx = _ctx()

    async def _none_missing(eid: UUID, ids: list[UUID]) -> list[UUID]:
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_budget_exhausted_suppresses_regress_to() -> None:
    """``regressions_used >= regression_budget_max`` short-circuits the
    producer-side cap — the council's BudgetCritic would veto anyway and
    we don't want known-doomed requests polluting the audit log.
    """
    finding_id = str(uuid4())
    findings = [
        {
            "id": finding_id,
            "title": "Would normally regress",
            "description": "evidence list is empty",
            "severity": "high",
            "evidence": [],
            "category": "web",
        }
    ]
    repo = _stub_repo(findings=findings)
    agent = _make_report_agent(repo)
    ctx = _ctx(regressions_used=3, regression_budget_max=3)

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_finding_origin_stage_field_overrides_category_default() -> None:
    """When a finding has its own ``origin_stage`` field set, that wins
    over the category-based heuristic — the persistence layer knows
    better than a string match."""
    finding_id = str(uuid4())
    findings = [
        {
            "id": finding_id,
            "title": "Loot-stage finding with web-shaped category",
            "description": "agents can disagree with the heuristic",
            "severity": "medium",
            "evidence": [],
            "category": "web",  # heuristic would say ENUMERATION
            "origin_stage": "loot",  # but the finding itself says LOOT
        }
    ]
    repo = _stub_repo(findings=findings)
    agent = _make_report_agent(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert req.target_stage == Stage.LOOT


@pytest.mark.asyncio
async def test_no_findings_does_not_trigger_regress_to() -> None:
    """The "no significant findings" path must not regress — there is
    nothing whose evidence is missing because there is nothing at all.
    """
    repo = _stub_repo(findings=[])
    agent = _make_report_agent(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None
