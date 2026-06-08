"""Delta-only world-state slices on regression re-entry for the recon agent.

When the stage controller re-enters recon with a regression delta (e.g.
``{"cidrs": ["10.0.5.0/24"]}``), the agent's ``_build_world_state_slice``
must narrow to ONLY the delta CIDRs — not the full engagement scope.
Without a delta the full scope is used as before.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.base import StageContext
from agents.recon import ReconAgent
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _stub_repo(hosts: list[dict[str, Any]]) -> MagicMock:
    """Return a mock repo whose ``get_nodes_by_kind`` yields *hosts*."""
    repo = MagicMock()
    repo.get_nodes_by_kind = AsyncMock(return_value=hosts)
    repo.count = AsyncMock(return_value={
        "Host": len(hosts),
        "Service": 0,
        "WebApp": 0,
        "Endpoint": 0,
        "Finding": 0,
        "ExploitCandidate": 0,
        "Session": 0,
    })
    repo.run_cypher = AsyncMock(return_value=[])
    repo.run_read_cypher = AsyncMock(return_value=[])
    return repo


def _make_agent(hosts: list[dict[str, Any]]) -> ReconAgent:
    repo = _stub_repo(hosts)
    return ReconAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


HOSTS = [
    {"ip": "10.0.5.1", "os_family": "linux", "first_seen": "2025-01-01"},
    {"ip": "10.0.5.2", "os_family": "linux", "first_seen": "2025-01-01"},
    {"ip": "10.0.1.10", "os_family": "windows", "first_seen": "2025-01-01"},
    {"ip": "10.0.1.20", "os_family": "windows", "first_seen": "2025-01-01"},
]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recon_slice_narrows_to_delta_cidrs() -> None:
    """With ``regression_delta={"cidrs": ["10.0.5.0/24"]}``, only hosts
    in that CIDR appear and scope is narrowed to the delta CIDRs."""
    agent = _make_agent(HOSTS)
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"scope": {"cidrs": ["10.0.0.0/16"]}},
        stage=Stage.RECON,
        regression_delta={"cidrs": ["10.0.5.0/24"]},
        regression_origin_stage="lateral",
    )

    slice_ = await agent._build_world_state_slice(ctx)

    # Only 10.0.5.x hosts should be present
    ips = [h["ip"] for h in slice_["known_hosts"]]
    assert set(ips) == {"10.0.5.1", "10.0.5.2"}

    # Scope should be narrowed to the delta CIDRs, not the full engagement scope
    assert slice_["scope"] == {"cidrs": ["10.0.5.0/24"]}

    # Regression context should be present
    assert "regression_context" in slice_
    assert slice_["regression_context"]["delta"] == {"cidrs": ["10.0.5.0/24"]}
    assert slice_["regression_context"]["origin_stage"] == "lateral"


@pytest.mark.asyncio
async def test_recon_slice_full_scope_without_delta() -> None:
    """Without a regression delta, the full engagement scope and all hosts
    appear in the slice."""
    agent = _make_agent(HOSTS)
    full_scope = {"cidrs": ["10.0.0.0/16"]}
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"scope": full_scope},
        stage=Stage.RECON,
        regression_delta=None,
    )

    slice_ = await agent._build_world_state_slice(ctx)

    # All hosts should be present
    ips = [h["ip"] for h in slice_["known_hosts"]]
    assert set(ips) == {"10.0.5.1", "10.0.5.2", "10.0.1.10", "10.0.1.20"}

    # Scope should be the full engagement scope
    assert slice_["scope"] == full_scope

    # No regression context
    assert "regression_context" not in slice_
