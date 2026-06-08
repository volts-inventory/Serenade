"""R3 #H2 Bug 3 — world-state slices must be capped past ~25 entries so
the planner prompt stays inside prompt-cache-friendly bounds.

Before this fix, web_enum / net_enum / vuln_triage / privesc / loot /
lateral fed unbounded lists of hosts / candidates / sessions into the
slice dict the agent JSON-dumped into every planner prompt. At 150
hosts the dump crossed 100KB and torpedoed per-turn cost.

Tests:
- For each agent, build a world-state with 150 hosts (or 150 candidates
  / 150 sessions, whichever the agent's slice consumes), call
  ``_build_world_state_slice``, and assert the returned list is capped
  to ≤25 entries plus a ``_truncated`` sentinel with ``_total=150``.
- The ``_truncate_slice`` helper itself round-trips short lists
  unchanged (length-preserving when no truncation needed).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.base import _SLICE_CAP_DEFAULT, _truncate_slice
from agents.lateral import LateralAgent
from agents.loot import LootAgent
from agents.net_enum import NetEnumAgent
from agents.privesc import PrivEscAgent
from agents.vuln_triage import VulnTriageAgent
from agents.web_enum import WebEnumAgent
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# helper assertions                                                            #
# --------------------------------------------------------------------------- #


def _assert_truncated_list(out: list[Any], *, expected_total: int) -> None:
    """Assert the slice list is capped to <=25 entries plus a sentinel
    pointing at the original total."""
    assert isinstance(out, list)
    assert len(out) == _SLICE_CAP_DEFAULT + 1, (
        f"expected {_SLICE_CAP_DEFAULT}+1 entries (cap + sentinel), "
        f"got {len(out)}"
    )
    sentinel = out[-1]
    assert isinstance(sentinel, dict)
    assert "_truncated" in sentinel
    assert sentinel["_total"] == expected_total


# --------------------------------------------------------------------------- #
# pure helper test                                                             #
# --------------------------------------------------------------------------- #


def test_truncate_slice_short_input_unchanged() -> None:
    items = [{"id": i} for i in range(5)]
    out = _truncate_slice(items, kind_label="Host")
    assert out == items  # no sentinel appended


def test_truncate_slice_appends_sentinel_when_over_cap() -> None:
    items = [{"id": i} for i in range(_SLICE_CAP_DEFAULT + 50)]
    out = _truncate_slice(items, kind_label="Host")
    assert len(out) == _SLICE_CAP_DEFAULT + 1
    assert out[-1]["_total"] == _SLICE_CAP_DEFAULT + 50
    assert "Host" in out[-1]["_truncated"]


# --------------------------------------------------------------------------- #
# per-agent slice-cap fixtures                                                 #
# --------------------------------------------------------------------------- #


def _shared(repo: MagicMock) -> dict[str, Any]:
    return {
        "router": MagicMock(),
        "repo": repo,
        "registry": MagicMock(),
        "policy": MagicMock(),
        "sandbox": MagicMock(),
    }


def _repo_with_hosts(n: int) -> MagicMock:
    """Build a repo stub that returns n Host nodes for any query that
    fans out per-host (hosts_with_open_services, get_nodes_by_kind)."""
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": n, "Service": n, "WebApp": 0, "Endpoint": 0,
        "Finding": 0, "ExploitCandidate": 0, "Session": 0,
    })
    hosts = [
        {"id": str(uuid4()), "ip": f"10.0.{i // 256}.{i % 256}",
         "os_family": "windows"}
        for i in range(n)
    ]

    async def get_nodes_by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Host":
            return list(hosts)
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)

    # hosts_with_open_services follows RUNS_ON predecessor edges per host;
    # give every host one AD service so net_enum's ad_candidates list is
    # exactly n long.
    repo.get_predecessors_by_edge_type = AsyncMock(
        side_effect=lambda node_id, edge_type: [{"port": 445, "name": "smb"}]
    )
    return repo


def _ctx(stage: Stage = Stage.ENUMERATION) -> Any:
    from agents.base import StageContext

    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=stage,
    )


# --------------------------------------------------------------------------- #
# Per-agent assertions                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_net_enum_caps_ad_candidates_at_25() -> None:
    repo = _repo_with_hosts(150)
    agent = NetEnumAgent(**_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.ENUMERATION))
    _assert_truncated_list(slice_["ad_candidates"], expected_total=150)


@pytest.mark.asyncio
async def test_vuln_triage_caps_hosts_with_versions_at_25() -> None:
    """Build 150 hosts each with one versioned service; assert the
    hosts_with_versions list is capped to 25 + sentinel."""
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 150, "Service": 150, "WebApp": 0, "Endpoint": 0,
        "Finding": 0, "ExploitCandidate": 0, "Session": 0,
    })
    hosts = [
        {"id": str(uuid4()), "ip": f"10.0.{i // 256}.{i % 256}"}
        for i in range(150)
    ]

    async def get_nodes_by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Host":
            return list(hosts)
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)
    # Each host has one versioned service via RUNS_ON predecessors.
    repo.get_predecessors_by_edge_type = AsyncMock(
        side_effect=lambda node_id, edge_type: [
            {"port": 22, "name": "ssh", "product": "OpenSSH", "version": "8.0"}
        ]
    )

    agent = VulnTriageAgent(triage_council=None, **_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.VULN_TRIAGE))
    _assert_truncated_list(slice_["hosts_with_versions"], expected_total=150)


@pytest.mark.asyncio
async def test_lateral_caps_targetable_hosts_at_lockout_safe_cap() -> None:
    """R3 #J2 Bug 8: LateralAgent passes the explicit
    ``LOCKOUT_SAFE_SLICE_CAP`` (20) so the spray-protection bound is
    not coupled to the agent-wide ``_SLICE_CAP_DEFAULT``."""
    repo = _repo_with_hosts(150)
    agent = LateralAgent(**_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.LATERAL))
    targetable = slice_["targetable_hosts"]
    # cap = 20, plus sentinel = 21 entries.
    assert isinstance(targetable, list)
    assert len(targetable) == LateralAgent.LOCKOUT_SAFE_SLICE_CAP + 1, (
        f"expected {LateralAgent.LOCKOUT_SAFE_SLICE_CAP}+1 entries "
        f"(cap + sentinel), got {len(targetable)}"
    )
    sentinel = targetable[-1]
    assert isinstance(sentinel, dict)
    assert sentinel.get("_total") == 150


@pytest.mark.asyncio
async def test_privesc_caps_unprivileged_unscanned_at_25() -> None:
    """Build 150 unprivileged shell sessions; the slice list is capped."""
    sessions = [
        {"id": str(uuid4()), "host_id": str(uuid4()),
         "user": f"u{i}", "privileged": False, "type": "shell"}
        for i in range(150)
    ]
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 0, "Service": 0, "WebApp": 0, "Endpoint": 0,
        "Finding": 0, "ExploitCandidate": 0, "Session": 150,
    })

    async def get_nodes_by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Session":
            return list(sessions)
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)
    repo.run_cypher = AsyncMock(return_value=[])
    repo.run_read_cypher = AsyncMock(return_value=[])

    agent = PrivEscAgent(**_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.PRIVESC))
    _assert_truncated_list(slice_["unprivileged_unscanned"], expected_total=150)


@pytest.mark.asyncio
async def test_loot_caps_unlooted_sessions_at_25() -> None:
    """Build 150 unlooted sessions; the slice list is capped."""
    sessions = [
        {"id": str(uuid4()), "host_id": str(uuid4()), "user": f"u{i}"}
        for i in range(150)
    ]
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 0, "Service": 0, "WebApp": 0, "Endpoint": 0,
        "Finding": 0, "ExploitCandidate": 0, "Session": 150,
    })

    async def get_nodes_by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Session":
            return list(sessions)
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)
    repo.run_cypher = AsyncMock(return_value=[])
    repo.run_read_cypher = AsyncMock(return_value=[])

    agent = LootAgent(**_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.LOOT))
    _assert_truncated_list(slice_["unlooted_sessions"], expected_total=150)


@pytest.mark.asyncio
async def test_web_enum_caps_webapps_at_25() -> None:
    """webapps_summary fans out per webapp; build 150 of them and assert
    the slice list is capped."""
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 0, "Service": 0, "WebApp": 150, "Endpoint": 0,
        "Finding": 0, "ExploitCandidate": 0, "Session": 0,
    })
    webapps = [
        {"id": str(uuid4()), "base_url": f"http://w{i}", "title": "t",
         "technologies": []}
        for i in range(150)
    ]

    async def get_nodes_by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "WebApp":
            return list(webapps)
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=get_nodes_by_kind)
    # No endpoints per webapp.
    repo.get_nodes_by_kind_field = AsyncMock(return_value=[])

    agent = WebEnumAgent(**_shared(repo))
    slice_ = await agent._build_world_state_slice(_ctx(Stage.ENUMERATION))
    _assert_truncated_list(slice_["webapps"], expected_total=150)
