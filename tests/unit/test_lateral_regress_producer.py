"""R3 #I1 Bug 4 + Bug 5 — fixes to the LateralAgent regression producer.

Bug 4: ``_scan_credentials_for_pivot_targets`` previously scanned both
``source`` and the FULL ``username`` blob. A UPN like
``svc-sql@db.corp.example.com`` has its HOME domain (the part after the
``@``) on the right-hand side — that is AD context, not a discovered
pivot target. The R3 #H2 docstring already excluded the ``domain`` field
for exactly this reason; including the username suffix re-introduced
the bug. The fix: split at ``@`` and discard the right-hand side.

Bug 5: ``_maybe_build_regression_request`` required
``validated_credentials_count >= 1`` before emitting REGRESS_TO. That
refused the fresh-phished-cred case: an engagement that opens with a
single dump-style credential (``source="dumped from DB at 10.0.5.5"``)
has zero validated creds YET, but exactly that case is the one the
operator wanted to surface 10.0.5.5 to recon. The fix: relax to "≥1
validated cred OR ≥1 cred with a clear pivot signal AND turn>=2".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import StageContext, StepDecision
from agents.lateral import LateralAgent
from agents.regression import RegressionRequest
from worldstate.schema import Stage


def _stub_repo(
    creds: list[dict[str, Any]] | None = None,
    hosts: list[dict[str, Any]] | None = None,
    domains: list[dict[str, Any]] | None = None,
    creds_with_hosts: list[dict[str, Any]] | None = None,
) -> MagicMock:
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


def _ctx(*, turn: int = 1, scope: dict[str, Any] | None = None) -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={
            "approval_policy": "manual",
            "scope": scope or {
                "cidrs": ["10.0.0.0/16"],
                "domains": ["corp.example.com"],
            },
        },
        stage=Stage.LATERAL,
        turn=turn,
        regressions_used=0,
        regression_budget_max=3,
    )


# --------------------------------------------------------------------------- #
# Bug 4: username UPN suffix must NOT be treated as a pivot target.            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_username_upn_suffix_is_not_a_pivot_target() -> None:
    """A credential whose username is a UPN like
    ``svc-sql@db.corp.example.com`` and whose ``source`` is empty must
    NOT trigger a regression. The right-hand side of the UPN is the
    cred's home realm (AD context), not a discovered host.

    Even though the FQDN is in scope and uncovered, it is the cred's
    own home — surfacing it for recon would be wrong (it's where the
    cred AUTHENTICATES, not a discovered target).
    """
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql@db.corp.example.com",
            "domain": "corp.example.com",
            "source": "",
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    # Validated cred (cred is `valid=True` + has session on host).
    creds_with_hosts = [{"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=creds_with_hosts)
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_username_lhs_ip_still_scanned() -> None:
    """We only discard the RIGHT-hand side of the UPN. If by some chance
    an IP literal appears in the LEFT-hand side (well, usernames don't
    usually carry IPs but the scanner shouldn't blindly drop the whole
    username), the scanner still picks up IPs from the lhs."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "10.0.5.5",  # lhs is the whole thing (no @)
            "source": "",
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    creds_with_hosts = [
        {"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}
    ]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=creds_with_hosts)
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    # The IP is in scope and uncovered → REGRESS_TO fires.
    assert decision == StepDecision.REGRESS_TO
    assert isinstance(ctx.pending_regression_request, RegressionRequest)
    assert ctx.pending_regression_request.delta == {"cidrs": ["10.0.5.0/24"]}


# --------------------------------------------------------------------------- #
# Bug 5: fresh-phished cred path (zero validated creds yet, but pivot signal). #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fresh_phished_cred_triggers_regression_after_first_turn() -> None:
    """0 validated creds + 1 cred carrying a clear pivot signal +
    turn>=2 → REGRESS_TO fires (the relaxed branch (b) gate)."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "phished_admin",
            "source": "dumped from DB at 10.0.5.5",
            "valid": None,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    # ZERO creds with hosts — validated_credentials_count = 0.
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=[])
    agent = _make_lateral(repo)
    # turn=2: lateral has already had its first chance to validate.
    ctx = _ctx(turn=2)

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    assert req.delta == {"cidrs": ["10.0.5.0/24"]}
    # Rationale must indicate the fresh-phished branch fired.
    assert "0 validated" in req.rationale or "fresh" in req.rationale.lower()


@pytest.mark.asyncio
async def test_fresh_phished_cred_suppressed_on_turn_one() -> None:
    """0 validated creds + pivot signal but turn==1: lateral hasn't had
    a chance to validate yet, so we hold off — let lateral try first."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "phished_admin",
            "source": "dumped from DB at 10.0.5.5",
            "valid": None,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=[])
    agent = _make_lateral(repo)
    ctx = _ctx(turn=1)

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None


@pytest.mark.asyncio
async def test_validated_branch_rationale_mentions_validated_count() -> None:
    """The validated-cred branch (a) must surface a different rationale
    string than the fresh-phished branch (b) so the council reviewing
    the request can tell them apart."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "svc-sql",
            "source": "DB account for 10.0.5.5",
            "valid": True,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    creds_with_hosts = [
        {"credential": {"id": creds[0]["id"]}, "hosts": [hosts[0]]}
    ]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=creds_with_hosts)
    agent = _make_lateral(repo)
    ctx = _ctx()

    decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    req = ctx.pending_regression_request
    assert isinstance(req, RegressionRequest)
    # The validated branch rationale should mention validated count > 0.
    assert "validated" in req.rationale.lower()
    # Should NOT carry the fresh-phished marker.
    assert "fresh" not in req.rationale.lower()


@pytest.mark.asyncio
async def test_no_validated_no_signal_no_regression() -> None:
    """Both branches fail (0 validated + no usable pivot signal) → no
    regression even at turn=2."""
    creds = [
        {
            "id": str(uuid4()),
            "username": "phished",
            "source": "captured locally",  # no IP/FQDN
            "valid": None,
        }
    ]
    hosts = [{"id": str(uuid4()), "ip": "10.0.1.10", "os_family": "linux"}]
    repo = _stub_repo(creds=creds, hosts=hosts, creds_with_hosts=[])
    agent = _make_lateral(repo)
    ctx = _ctx(turn=2)

    decision = await agent._decide_advance(ctx)

    assert decision != StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is None
