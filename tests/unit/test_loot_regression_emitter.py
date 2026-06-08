"""Tests for LootAgent's CREDENTIALS_IMPLY_PIVOT regression emitter.

Verifies the structural wiring:
  - ``_maybe_build_regression_request`` returns None when budget is exhausted.
  - ``_decide_advance`` calls the regression check and returns REGRESS_TO
    when the method returns a request.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agents.base import StageContext, StepDecision
from agents.loot import LootAgent
from agents.regression import RegressionRequest, RegressionTriggerKind
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _stub_repo(
    *,
    session_count: int = 1,
    looted_session_ids: set[str] | None = None,
) -> MagicMock:
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 1,
        "Service": 0,
        "WebApp": 0,
        "Endpoint": 0,
        "Finding": 0,
        "ExploitCandidate": 0,
        "Session": session_count,
    })
    sessions = [
        {"id": str(uuid4()), "host_id": str(uuid4()), "user": f"user-{i}"}
        for i in range(session_count)
    ]

    async def _get_nodes(_eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Session":
            return sessions
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=_get_nodes)

    # _sessions_looted uses run_cypher
    looted = looted_session_ids or set()

    async def _run_cypher(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if "AFFECTS" in cypher:
            return [{"sid": sid} for sid in looted]
        return []

    repo.run_cypher = AsyncMock(side_effect=_run_cypher)
    return repo


def _make_loot(repo: MagicMock) -> LootAgent:
    return LootAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


def _ctx(
    *,
    turn: int = 1,
    regressions_used: int = 0,
    regression_budget_max: int = 3,
) -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.LOOT,
        turn=turn,
        regressions_used=regressions_used,
        regression_budget_max=regression_budget_max,
    )


# --------------------------------------------------------------------------- #
# _maybe_build_regression_request: budget exhausted -> None                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_regression_returns_none_when_budget_exhausted() -> None:
    """When regressions_used >= regression_budget_max, the producer-side
    budget check returns None immediately."""
    repo = _stub_repo()
    agent = _make_loot(repo)
    ctx = _ctx(regressions_used=3, regression_budget_max=3)
    # Provide a slice with data that would otherwise trigger the check.
    ctx.last_world_state_slice = {
        "discovered_credentials": [
            {
                "cred_id": str(uuid4()),
                "host_id": str(uuid4()),
                "service_class_hint": "mssql",
                "host_services": [],
            }
        ],
    }
    result = await agent._maybe_build_regression_request(ctx)
    assert result is None


# --------------------------------------------------------------------------- #
# _maybe_build_regression_request: returns None when slice has no data         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_regression_returns_none_when_no_discovered_credentials() -> None:
    """When the slice doesn't contain discovered_credentials, no
    regression fires (conservative fallback)."""
    repo = _stub_repo()
    agent = _make_loot(repo)
    ctx = _ctx()
    ctx.last_world_state_slice = {"coverage": {}, "sessions": 1, "turn": 1}
    result = await agent._maybe_build_regression_request(ctx)
    assert result is None


# --------------------------------------------------------------------------- #
# _decide_advance: returns REGRESS_TO when regression method fires             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_decide_advance_returns_regress_to_when_regression_fires() -> None:
    """Mock ``_maybe_build_regression_request`` to return a request and
    verify that ``_decide_advance`` returns REGRESS_TO and populates
    ``ctx.pending_regression_request``."""
    repo = _stub_repo(session_count=1)
    agent = _make_loot(repo)
    ctx = _ctx()

    fake_request = RegressionRequest(
        target_stage=Stage.ENUMERATION,
        trigger_kind=RegressionTriggerKind.CREDENTIALS_IMPLY_PIVOT,
        delta={"host_id": "host-1", "service_class_hint": "mssql"},
        evidence=["cred-1", "host-1"],
        rationale="loot: test regression",
    )
    with patch.object(
        agent,
        "_maybe_build_regression_request",
        new_callable=AsyncMock,
        return_value=fake_request,
    ):
        decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.REGRESS_TO
    assert ctx.pending_regression_request is fake_request


# --------------------------------------------------------------------------- #
# _decide_advance: proceeds normally when no regression fires                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_decide_advance_continues_when_no_regression() -> None:
    """When ``_maybe_build_regression_request`` returns None, the normal
    state-driven advance logic takes over."""
    repo = _stub_repo(session_count=1)
    agent = _make_loot(repo)
    ctx = _ctx()

    with patch.object(
        agent,
        "_maybe_build_regression_request",
        new_callable=AsyncMock,
        return_value=None,
    ):
        decision = await agent._decide_advance(ctx)

    assert decision == StepDecision.CONTINUE
    assert ctx.pending_regression_request is None
