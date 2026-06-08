"""R3 #K5 Bug 2: stub Service path must emit graph edges.

Pre-fix, when httpx hit a port that nmap hadn't surfaced (e.g. 8443
outside the default top-1000), recon synthesized a stub Service node so
the WebApp.service_id FK resolved, but:
  - No ``RUNS_ON`` edge from the stub Service to its Host.
  - No ``OF_WEBAPP`` edge from the WebApp to the (stub) Service.

The WebApp.service_id FK was set, but edge-based traversal
(``WebApp -OF_WEBAPP-> Service -RUNS_ON-> Host``) was broken until a
later nmap pass enriched the Service. Report rendering walks edges,
not just FKs, so the affected-host pivot nulled out for stub-only
findings.

Fix:
  - Emit ``RUNS_ON`` (Service → Host) for the stub Service, matching
    the nmap success-path convention.
  - Emit ``OF_WEBAPP`` (WebApp → Service) unconditionally (was guarded
    by ``if svc_id is not None``).

These tests pin:
  1. Stub path: WebApp persisted, stub Service persisted, both edges
     emitted with the right kinds + endpoints.
  2. Resolved path (no regression): WebApp persisted, no stub Service,
     ``OF_WEBAPP`` edge to the REAL Service.
  3. Determinism survives: a later nmap pass that surfaces the same
     port collapses on the same deterministic Service id; edges remain
     intact (MERGE-on-id no-op).
  4. The edge endpoints use the deterministic ids (so a re-parse
     produces the same edge identity, not duplicates).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import StageContext
from agents.recon import ReconAgent
from tools.httpx_ import HttpxOutput, HttpxResult
from tools.nmap_ import NmapHost, NmapOutput, NmapPort
from worldstate.schema import (
    Edge,
    EdgeKind,
    Host,
    Service,
    Stage,
    WebApp,
    deterministic_node_id,
    natural_key_for_host,
    natural_key_for_service,
)


def _agent(*, services: list[dict] | None = None, hosts: list[dict] | None = None) -> ReconAgent:
    repo = MagicMock()
    repo.upsert_node = AsyncMock()
    repo.upsert_edge = AsyncMock()
    repo.count = AsyncMock(return_value={})

    _services = services or []
    _hosts = hosts or []

    async def _get_nodes_by_kind(_eid: Any, kind: str) -> list[dict]:
        if kind == "Service":
            return _services
        if kind == "Host":
            return _hosts
        return []

    repo.get_nodes_by_kind = AsyncMock(side_effect=_get_nodes_by_kind)
    return ReconAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


def _ctx(engagement_id: UUID) -> StageContext:
    return StageContext(
        engagement_id=engagement_id,
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )


def _httpx_8443() -> HttpxOutput:
    """An httpx hit on a port nmap's top-1000 wouldn't surface."""
    return HttpxOutput(
        command="httpx -u https://target.local:8443",
        probes=[
            HttpxResult(
                url="https://target.local:8443/",
                host="target.local",
                port=8443,
                scheme="https",
                status_code=200,
                title="Hidden Console",
                technologies=["nginx"],
            )
        ],
    )


def _edges(agent: ReconAgent) -> list[Edge]:
    """Pull the Edge objects passed to ``repo.upsert_edge``."""
    return [
        call.args[0]
        for call in agent.repo.upsert_edge.call_args_list
        if call.args and isinstance(call.args[0], Edge)
    ]


# --------------------------------------------------------------------------- #
# 1. Stub path: both edges fire alongside the stub Service node.              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stub_service_path_emits_runs_on_edge_to_host() -> None:
    """When the Service lookup misses, recon synthesizes a stub Service
    AND must wire it to its Host with ``RUNS_ON`` (matches the nmap
    success-path convention) so graph traversal works immediately."""
    eid = uuid4()
    agent = _agent()
    # Default ``run_read_cypher`` returns [] → stub path engaged.
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )

    stub_service = next(n for n in out if isinstance(n, Service))
    expected_host_id = deterministic_node_id(
        "Host", eid, natural_key_for_host(eid, "target.local")
    )
    runs_on = [
        e
        for e in _edges(agent)
        if e.kind == EdgeKind.RUNS_ON
        and e.src_id == stub_service.id
        and e.dst_id == expected_host_id
    ]
    assert len(runs_on) == 1, (
        f"expected exactly one Service -RUNS_ON-> Host edge from the "
        f"stub Service; got edges: {_edges(agent)!r}"
    )
    # And it's scoped to the engagement.
    assert runs_on[0].engagement_id == eid


@pytest.mark.asyncio
async def test_stub_service_path_emits_of_webapp_edge_from_webapp_to_service() -> None:
    """The WebApp must link to the (stub) Service via ``OF_WEBAPP`` so
    the edge-based traversal that the report renderer walks (WebApp →
    Service → Host) is intact from the moment the stub is persisted."""
    eid = uuid4()
    agent = _agent()
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )

    webapp = next(n for n in out if isinstance(n, WebApp))
    stub_service = next(n for n in out if isinstance(n, Service))
    of_webapp = [
        e
        for e in _edges(agent)
        if e.kind == EdgeKind.OF_WEBAPP
        and e.src_id == webapp.id
        and e.dst_id == stub_service.id
    ]
    assert len(of_webapp) == 1, (
        f"expected exactly one WebApp -OF_WEBAPP-> Service edge on the "
        f"stub path; got edges: {_edges(agent)!r}"
    )
    assert of_webapp[0].engagement_id == eid


@pytest.mark.asyncio
async def test_stub_service_edges_use_deterministic_ids() -> None:
    """The edge endpoints must use the deterministic Host + Service ids
    so a re-parse produces the same edge identity (MERGE-on-id no-op),
    not a duplicate edge with the same endpoints."""
    eid = uuid4()
    agent = _agent()
    await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )

    expected_host_id = deterministic_node_id(
        "Host", eid, natural_key_for_host(eid, "target.local")
    )
    expected_service_id = deterministic_node_id(
        "Service",
        eid,
        natural_key_for_service(expected_host_id, 8443, "tcp"),
    )

    edges = _edges(agent)
    runs_on = next(e for e in edges if e.kind == EdgeKind.RUNS_ON)
    assert runs_on.src_id == expected_service_id
    assert runs_on.dst_id == expected_host_id

    of_webapp = next(e for e in edges if e.kind == EdgeKind.OF_WEBAPP)
    assert of_webapp.dst_id == expected_service_id


# --------------------------------------------------------------------------- #
# 2. Resolved path: no stub, OF_WEBAPP edge still fires (no regression).      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolved_service_path_emits_of_webapp_edge_to_real_service() -> None:
    """When nmap already created the Service, no stub is emitted but
    the WebApp -OF_WEBAPP-> Service edge MUST still fire (this was the
    pre-fix behavior -- pin it so the cleanup doesn't regress it)."""
    eid = uuid4()
    real_svc_id = UUID("55555555-5555-5555-5555-555555555555")
    host_id = UUID("66666666-6666-6666-6666-666666666666")

    agent = _agent(
        services=[{"id": str(real_svc_id), "port": 8443, "host_id": str(host_id), "engagement_id": str(eid)}],
        hosts=[{"id": str(host_id), "ip": "target.local", "engagement_id": str(eid)}],
    )
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )

    # No stub Service emitted on the resolved path.
    assert [n for n in out if isinstance(n, Service)] == [], (
        "Real Service already in the graph -- no stub should be emitted"
    )
    webapp = next(n for n in out if isinstance(n, WebApp))
    of_webapp = [
        e
        for e in _edges(agent)
        if e.kind == EdgeKind.OF_WEBAPP
        and e.src_id == webapp.id
        and e.dst_id == real_svc_id
    ]
    assert len(of_webapp) == 1, (
        "OF_WEBAPP edge to the REAL Service must still fire on the "
        "resolved (non-stub) path"
    )


# --------------------------------------------------------------------------- #
# 3. Determinism across a later nmap pass.                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_later_nmap_pass_collapses_on_same_service_id_and_edges_persist() -> None:
    """Scenario: httpx hits 8443 first → stub Service + edges emitted.
    Later, nmap discovers 8443 → the deterministic Service id collides
    (host+port+proto-keyed), so the upsert enriches the SAME row in
    place. The original stub-path edges remain valid (same endpoints)
    and the nmap pass adds its own RUNS_ON edge with identical
    endpoints (MERGE-on-id no-op in Neo4j)."""
    eid = uuid4()
    agent = _agent()

    # First pass: httpx (stub path engaged).
    out_httpx = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )
    stub_service = next(n for n in out_httpx if isinstance(n, Service))

    # Second pass: nmap discovers 8443. New Service id MUST match the
    # stub's deterministic id (same host+port+proto natural key).
    nmap_out = NmapOutput(
        command="nmap -p 8443 target.local",
        hosts=[
            NmapHost(
                ip="target.local",
                state="up",
                hostnames=["target.local"],
                ports=[
                    NmapPort(
                        port=8443,
                        proto="tcp",
                        state="open",
                        name="https",
                        product="nginx",
                    )
                ],
            )
        ],
    )
    out_nmap = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, nmap_out, tool_run_id=uuid4()
    )
    nmap_service = next(n for n in out_nmap if isinstance(n, Service))
    assert nmap_service.id == stub_service.id, (
        "deterministic Service id MUST collide across stub + nmap "
        "passes so MERGE-on-id enriches the row in place"
    )

    # The collected edges include BOTH the stub-path RUNS_ON (from the
    # httpx parse) and the nmap-path RUNS_ON. They have identical
    # endpoints → Neo4j MERGE collapses to a single edge in practice;
    # at the Edge-object level we just verify both fired with matching
    # endpoints and the OF_WEBAPP edge is still present.
    edges = _edges(agent)
    expected_host_id = deterministic_node_id(
        "Host", eid, natural_key_for_host(eid, "target.local")
    )
    runs_on_with_correct_endpoints = [
        e
        for e in edges
        if e.kind == EdgeKind.RUNS_ON
        and e.src_id == stub_service.id
        and e.dst_id == expected_host_id
    ]
    assert len(runs_on_with_correct_endpoints) >= 1, (
        "RUNS_ON edge with stub-Service-id → Host must persist after a "
        "later nmap pass (deterministic endpoints survive enrichment)"
    )
    of_webapp_edges = [
        e
        for e in edges
        if e.kind == EdgeKind.OF_WEBAPP and e.dst_id == stub_service.id
    ]
    assert len(of_webapp_edges) >= 1, (
        "OF_WEBAPP edge from the httpx pass survives — points at the "
        "same deterministic Service id that nmap later enriched"
    )


# --------------------------------------------------------------------------- #
# 4. Multi-probe sanity: each httpx probe gets its own pair of edges.         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multiple_stub_probes_each_get_their_own_edges() -> None:
    """Two httpx hits on different stub ports (8080 + 8443) must each
    get their own RUNS_ON + OF_WEBAPP edges. Catches a regression where
    a loop scope leaked a single ``host_id_for_stub`` across probes."""
    eid = uuid4()
    agent = _agent()
    out = await agent._parse_and_persist(
        _ctx(eid),
        MagicMock(),
        {},
        HttpxOutput(
            command="httpx",
            probes=[
                HttpxResult(
                    url="http://target.local:8080/",
                    host="target.local",
                    port=8080,
                    scheme="http",
                    status_code=200,
                ),
                HttpxResult(
                    url="https://target.local:8443/",
                    host="target.local",
                    port=8443,
                    scheme="https",
                    status_code=200,
                ),
            ],
        ),
        tool_run_id=uuid4(),
    )

    services = [n for n in out if isinstance(n, Service)]
    webapps = [n for n in out if isinstance(n, WebApp)]
    assert len(services) == 2 and len(webapps) == 2

    edges = _edges(agent)
    runs_on = [e for e in edges if e.kind == EdgeKind.RUNS_ON]
    of_webapp = [e for e in edges if e.kind == EdgeKind.OF_WEBAPP]
    # Each probe contributes one RUNS_ON + one OF_WEBAPP.
    assert len(runs_on) == 2
    assert len(of_webapp) == 2
    # And each Service id is referenced by both an OF_WEBAPP dst and a
    # RUNS_ON src (no edge dangling on a different id).
    runs_on_srcs = {e.src_id for e in runs_on}
    of_webapp_dsts = {e.dst_id for e in of_webapp}
    assert runs_on_srcs == of_webapp_dsts == {s.id for s in services}


# --------------------------------------------------------------------------- #
# 5. No Host node leakage: stub path doesn't materialize a Host node here.    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stub_path_does_not_emit_host_node_only_edge_endpoints() -> None:
    """The Host node is created by the nmap parser, not the httpx
    parser — the stub path only references the deterministic Host id
    on its edges. Pin that contract so a future "create the Host too"
    refactor is a deliberate choice, not a silent expansion."""
    eid = uuid4()
    agent = _agent()
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _httpx_8443(), tool_run_id=uuid4()
    )
    assert [n for n in out if isinstance(n, Host)] == [], (
        "stub path must not emit a Host node — Host creation is the "
        "nmap parser's job; the stub path only cites a deterministic "
        "Host id on its RUNS_ON edge"
    )
