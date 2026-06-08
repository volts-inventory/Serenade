"""R3 #H2 Bug 1 — LateralAgent emits ``StepDecision.REGRESS_TO recon``
when a captured credential references a subnet/domain outside the
current recon coverage but inside engagement scope.

Before this fix the regression mechanism (``StepDecision.REGRESS_TO``,
``RegressionRequest``, ``StageRegressionCouncil``) was end-to-end
plumbed but no agent ever emitted it. This locks the producer half in.

Tests:
- Happy path: in-scope IP outside coverage + a validated cred → emits
  ``REGRESS_TO`` with the implied ``/24`` in the delta.
- Out-of-scope IP → no regression (scope gate).
- Budget exhausted → no regression (producer-side cap check).
- Two credentials mention the same uncovered subnet → one regression
  request with a deduplicated delta.
- No validated cred yet → no regression (progress gate).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import StageContext, StepDecision
from agents.lateral import LateralAgent
from agents.regression import RegressionRequest, RegressionTriggerKind
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _stub_repo(
    creds: list[dict[str, Any]] | None = None,
    hosts: list[dict[str, Any]] | None = None,
    domains: list[dict[str, Any]] | None = None,
    creds_with_hosts: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a Neo4jRepo stub that returns the supplied lists for the
    queries LateralAgent makes. ``creds_with_hosts`` mirrors what
    ``credentials_with_known_hosts`` returns (validated creds count).
    """
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": len(hosts or []),
        "Service": 0,
        "WebApp": 0,
        "Endpoint": 0,
        "Finding": 0,
        "ExploitCandidate": 0,
        "Session": 0,
    })

    async def get_nodes_by_kind(eid: UUID, kind: str) -> list[dict[str, Any]]:
        if kind == "Credential":
            return list(creds or [])
        if kind == "Host":
            return list(hosts or [])
        if kind == "Domain":
            return list(domains or [])
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)

    async def _succ(node_id, edge_type):
        if edge_type != "HAS_SESSION_ON":
            return []
        nid = str(node_id)
        for cwh in (creds_with_hosts or []):
            if str(cwh.get("credential", {}).get("id")) == nid:
                return list(cwh.get("hosts", []))
        return []

    repo.get_successors_by_edge_type = AsyncMock(side_effect=_succ)
    return repo


def _make_lateral(repo: MagicMock) -> LateralAgent:
    return LateralAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


def _ctx(
    *,
    scope: dict[str, Any] | None = None,
    regressions_used: int = 0,
    regression_budget_max: int = 3,
) -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={
            "approval_policy": "manual",
            "scope": scope or {"cidrs": ["10.0.0.0/16"], "domains": ["corp.example.com"]},
        },
        stage=Stage.LATERAL,
        regressions_used=regressions_used,
        regression_budget_max=regression_budget_max,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_credential_mentions_uncovered_in_scope_subnet_triggers_regress_to_recon() -> None:
    """In-scope IP (10.0.5.5) outside recon coverage (only 10.0.1.0/24
    enumerated) plus a validated cred → REGRESS_TO with delta /24."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql",
            "domain": "corp.example.com",
            "source": "DB account for 10.0.5.5",
            "valid": True,
        }
    ]
    hosts = [
        {"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"},
        {"id": str(uuid4()), "ip": "10.0.1.20", "os_family": "linux"},
    ]
    # One validated cred so the progress gate passes.
    creds_with_hosts = [{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=creds_with_hosts)
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert req.target_stage == Stage.RECON
    assert req.trigger_kind == RegressionTriggerKind.NEW_SUBNET_DISCOVERED
    assert req.delta == {"cidrs": ["10.0.5.0/24"]}
    assert creds[0]["id"] in req.evidence


@pytest.mark.asyncio
async def test_out_of_scope_credential_target_does_not_trigger_regress_to() -> None:
    """A cred that references an IP outside engagement scope must NOT
    drive a regression — the scope gate filters it."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc",
            "source": "backup at 192.168.250.5",  # outside 10.0.0.0/16
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    # Untested cred + targetable host so the default CONTINUE path applies.
    untested = [
        {"id": str(uuid4()), "username": "u", "valid": None, "source": "x"}
    ]
    repo = _stub_repo(
        creds=creds + untested,
        hosts=hosts,
        creds_with_hosts=[{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}],
    )
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_budget_exhausted_suppresses_regress_to() -> None:
    """regressions_used >= regression_budget_max → no REGRESS_TO emitted,
    even if the credential genuinely implies a missing subnet. The
    council would veto via BudgetCritic; the producer-side cap stops
    the no-op request from polluting the audit log."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql",
            "source": "DB account for 10.0.5.5",
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    repo = _stub_repo(
        creds=creds,
        hosts=hosts,
        creds_with_hosts=[{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}],
    )
    agent = _make_lateral(repo)
    ctx = _ctx(regressions_used=3, regression_budget_max=3)

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_two_credentials_same_subnet_produce_one_deduplicated_regression() -> None:
    """Two creds referencing 10.0.5.5 + 10.0.5.10 collapse to a single
    /24 delta — one REGRESS_TO with the deduplicated subnet."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql-1",
            "source": "DB at 10.0.5.5",
            "valid": True,
        },
        {
            "id": str(uuid4()),
            "username": "svc-sql-2",
            "source": "secondary at 10.0.5.10",
            "valid": True,
        },
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    repo = _stub_repo(
        creds=creds,
        hosts=hosts,
        creds_with_hosts=[{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}],
    )
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    # Both creds derived the same /24; delta should carry it exactly once.
    assert req.delta == {"cidrs": ["10.0.5.0/24"]}
    # Both cred ids should be on the evidence list — they both motivated
    # the request.
    assert set(req.evidence) == {creds[0]["id"], creds[1]["id"]}


@pytest.mark.asyncio
async def test_no_progress_yet_suppresses_regression() -> None:
    """If lateral hasn't validated a single cred this engagement, the
    regression is premature — let lateral try its existing creds first.
    """
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql",
            "source": "DB account for 10.0.5.5",
            "valid": None,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    # creds_with_hosts is empty -> validated_credentials_count = 0
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=[])
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_in_scope_domain_in_credential_triggers_regression_with_domain_delta() -> None:
    """A cred whose source mentions an in-scope-but-uncovered FQDN
    should yield a delta with the FQDN in ``domains``."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc",
            "source": "credentials harvested from db.corp.example.com",
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    repo = _stub_repo(
        creds=creds,
        hosts=hosts,
        creds_with_hosts=[{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}],
    )
    agent = _make_lateral(repo)
    ctx = _ctx(
        scope={"cidrs": ["10.0.0.0/16"], "domains": ["corp.example.com"]},
    )

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert "domains" in req.delta
    assert "db.corp.example.com" in req.delta["domains"]
