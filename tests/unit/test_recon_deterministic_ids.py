"""R3 #K1 Bug 2: deterministic UUIDs for Host / Service / WebApp.

Pre-fix, ``ReconAgent._parse_and_persist`` minted ``uuid4()`` for every
Host / Service / WebApp on every parse pass. The RECON interrupt-resume
path therefore wrote duplicate Host nodes (different uuids, same ``ip``)
on every re-parse — Neo4j's MERGE-on-id then could not dedupe across
runs, leaving the engagement graph with N copies of every asset after
N resumes. The report's affected-assets pivot then double-counted.

The fix mirrors the J2 deterministic-id pattern (Endpoint, Finding,
Credential, ExploitCandidate, AttackPath): derive the node id from
``uuid5(_SERENADE_NS, f"{kind}:{engagement_id}:{natural_key}")`` so a
re-parse of the same tool output collapses on identity.

These tests lock in the natural-key invariants for the three new
helpers and the end-to-end "re-parse on resume produces no duplicates"
property for ReconAgent's nmap + httpx parsers.
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
    Host,
    Service,
    Stage,
    WebApp,
    deterministic_node_id,
    natural_key_for_host,
    natural_key_for_service,
    natural_key_for_webapp,
)


# --------------------------------------------------------------------------- #
# Pure natural-key helper unit tests                                          #
# --------------------------------------------------------------------------- #


def test_host_natural_key_same_engagement_same_ip_collides() -> None:
    """The whole point of the helper: same engagement + same ip yields
    the same key (deduplication of resume-driven re-parses)."""
    eid = uuid4()
    a = natural_key_for_host(eid, "10.0.0.1")
    b = natural_key_for_host(eid, "10.0.0.1")
    assert a == b


def test_host_natural_key_different_engagements_diverge() -> None:
    """Two engagements with overlapping IPs (RFC1918 collision across
    customers, lab fixture reuse) MUST get distinct keys so host
    identity does not leak across engagements."""
    a = natural_key_for_host(uuid4(), "10.0.0.1")
    b = natural_key_for_host(uuid4(), "10.0.0.1")
    assert a != b


def test_service_natural_key_protocol_case_insensitive() -> None:
    """Two parsers emitting "TCP" vs "tcp" for the same Service must
    collide — the helper normalizes protocol case."""
    host_id = UUID("11111111-1111-1111-1111-111111111111")
    a = natural_key_for_service(host_id, 443, "TCP")
    b = natural_key_for_service(host_id, 443, "tcp")
    assert a == b


def test_service_natural_key_different_ports_diverge() -> None:
    host_id = UUID("11111111-1111-1111-1111-111111111111")
    a = natural_key_for_service(host_id, 80, "tcp")
    b = natural_key_for_service(host_id, 443, "tcp")
    assert a != b


def test_webapp_natural_key_normalizes_trailing_slash_and_case() -> None:
    """``https://h/`` and ``https://h`` and ``HTTPS://H`` collide so
    re-probing the same target writes the same WebApp identity."""
    sid = UUID("22222222-2222-2222-2222-222222222222")
    a = natural_key_for_webapp(sid, "https://example.com/")
    b = natural_key_for_webapp(sid, "https://example.com")
    c = natural_key_for_webapp(sid, "HTTPS://EXAMPLE.COM")
    assert a == b == c


# --------------------------------------------------------------------------- #
# Integration: ReconAgent._parse_and_persist for nmap output                  #
# --------------------------------------------------------------------------- #


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


def _one_host_nmap() -> NmapOutput:
    return NmapOutput(
        command="nmap -sV 10.0.0.1",
        hosts=[
            NmapHost(
                ip="10.0.0.1",
                state="up",
                hostnames=["target.local"],
                os_family="linux",
                ports=[
                    NmapPort(port=443, proto="tcp", state="open", name="https"),
                    NmapPort(port=22, proto="tcp", state="open", name="ssh"),
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_same_engagement_same_ip_yields_same_host_id_across_parses() -> None:
    """Re-parse the same nmap output twice — the second pass MUST
    produce the same Host id (MERGE-on-id no-op)."""
    eid = uuid4()
    agent = _agent()
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    hosts_a = [n for n in out_a if isinstance(n, Host)]
    hosts_b = [n for n in out_b if isinstance(n, Host)]
    assert len(hosts_a) == 1 and len(hosts_b) == 1
    assert hosts_a[0].id == hosts_b[0].id
    # And the helper agrees on the value.
    expected = deterministic_node_id(
        "Host", eid, natural_key_for_host(eid, "10.0.0.1")
    )
    assert hosts_a[0].id == expected


@pytest.mark.asyncio
async def test_same_host_port_proto_yields_same_service_id() -> None:
    """Service identity is (host, port, protocol). Re-parsing yields
    the same Service id for the same triple."""
    eid = uuid4()
    agent = _agent()
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    svcs_a = {(s.port, s.proto): s.id for s in out_a if isinstance(s, Service)}
    svcs_b = {(s.port, s.proto): s.id for s in out_b if isinstance(s, Service)}
    assert svcs_a == svcs_b  # identical mapping across parses
    # Both ports represented.
    assert (443, "tcp") in svcs_a
    assert (22, "tcp") in svcs_a


@pytest.mark.asyncio
async def test_different_engagements_yield_different_host_ids_same_ip() -> None:
    """Two engagements scanning the same IP MUST get distinct Host
    ids — node identity does not leak across engagements."""
    eid_a = uuid4()
    eid_b = uuid4()
    agent = _agent()
    out_a = await agent._parse_and_persist(
        _ctx(eid_a), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid_b), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    host_a = next(n for n in out_a if isinstance(n, Host))
    host_b = next(n for n in out_b if isinstance(n, Host))
    assert host_a.id != host_b.id


@pytest.mark.asyncio
async def test_reparse_on_resume_emits_no_duplicate_host_ids() -> None:
    """End-to-end: simulate an interrupt + resume by calling
    ``_parse_and_persist`` twice with the same tool output. The set
    of Host ids across both passes must be the same (no duplicates)."""
    eid = uuid4()
    agent = _agent()
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_host_nmap(), tool_run_id=uuid4()
    )
    ids_a = {n.id for n in out_a}
    ids_b = {n.id for n in out_b}
    assert ids_a == ids_b


# --------------------------------------------------------------------------- #
# Integration: ReconAgent._parse_and_persist for httpx output                 #
# --------------------------------------------------------------------------- #


def _one_webapp_httpx() -> HttpxOutput:
    return HttpxOutput(
        command="httpx -u https://target.local",
        probes=[
            HttpxResult(
                url="https://target.local/",
                host="target.local",
                port=443,
                scheme="https",
                status_code=200,
                title="Welcome",
                technologies=["nginx"],
            )
        ],
    )


@pytest.mark.asyncio
async def test_reparse_httpx_yields_same_webapp_id() -> None:
    """A re-probe of the same httpx target — even when the Service
    lookup misses (the deterministic-fallback path) — MUST produce
    the same WebApp id on the second parse."""
    eid = uuid4()
    agent = _agent()
    # ``run_read_cypher`` returns [] (the default), forcing the
    # deterministic-fallback service id path in recon.
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    wa_a = next(n for n in out_a if isinstance(n, WebApp))
    wa_b = next(n for n in out_b if isinstance(n, WebApp))
    assert wa_a.id == wa_b.id
    # service_id (the deterministic fallback) is also stable.
    assert wa_a.service_id == wa_b.service_id


@pytest.mark.asyncio
async def test_reparse_httpx_with_resolved_service_yields_same_webapp_id() -> None:
    """When the Service IS in the graph, the WebApp id is keyed on
    that real service id -- and is still deterministic across parses."""
    eid = uuid4()
    real_svc_id = UUID("33333333-3333-3333-3333-333333333333")
    host_id = UUID("44444444-4444-4444-4444-444444444444")

    agent = _agent(
        services=[{"id": str(real_svc_id), "port": 443, "host_id": str(host_id), "engagement_id": str(eid)}],
        hosts=[{"id": str(host_id), "ip": "target.local", "engagement_id": str(eid)}],
    )
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    wa_a = next(n for n in out_a if isinstance(n, WebApp))
    wa_b = next(n for n in out_b if isinstance(n, WebApp))
    assert wa_a.id == wa_b.id
    assert wa_a.service_id == real_svc_id
    # Expected id matches the helper output.
    expected = deterministic_node_id(
        "WebApp",
        eid,
        natural_key_for_webapp(real_svc_id, "https://target.local/"),
    )
    assert wa_a.id == expected


# --------------------------------------------------------------------------- #
# nmap-XML importer parity                                                    #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# R3 #K4 Bug 3: WebApp.service_id stays resolvable on the synthesized        #
# fallback. Pre-fix: when nmap missed the port but httpx hit it, recon       #
# synthesized a deterministic service_id and used it on the WebApp, but       #
# never created the matching Service node — the WebApp's foreign key         #
# dangled and the report's WebApp → Service → Host pivot nulled out.         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_httpx_fallback_emits_stub_service_node() -> None:
    """When the Service lookup misses (nmap top-1000 missed 8443 but
    httpx hit it), recon must emit a STUB Service node alongside the
    WebApp so the WebApp.service_id resolves to a real graph row, not
    a dangling UUID."""
    eid = uuid4()
    agent = _agent()
    # ``run_read_cypher`` returns [] (default) → fallback path engaged.
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    webapps = [n for n in out if isinstance(n, WebApp)]
    services = [n for n in out if isinstance(n, Service)]
    assert len(webapps) == 1, "expected exactly one WebApp"
    assert len(services) == 1, (
        "expected a stub Service node alongside the WebApp so the "
        "WebApp.service_id resolves to a real graph row (Bug 3)"
    )
    # The stub's id must equal the WebApp.service_id (otherwise the
    # foreign key is still dangling).
    assert services[0].id == webapps[0].service_id
    # And the stub's natural key matches (host, port, tcp) per the
    # fallback's synthesis rule — re-probing the same target on a later
    # parse must collapse on the SAME id (deterministic).
    assert services[0].port == 443
    assert services[0].proto == "tcp"


@pytest.mark.asyncio
async def test_httpx_with_real_service_does_not_emit_stub() -> None:
    """When the Service lookup HITS (nmap already created it), no stub
    is emitted -- we don't want two Service nodes for the same port."""
    eid = uuid4()
    real_svc_id = UUID("44444444-4444-4444-4444-444444444444")
    host_id = UUID("55555555-5555-5555-5555-555555555555")

    agent = _agent(
        services=[{"id": str(real_svc_id), "port": 443, "host_id": str(host_id), "engagement_id": str(eid)}],
        hosts=[{"id": str(host_id), "ip": "target.local", "engagement_id": str(eid)}],
    )
    out = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    services = [n for n in out if isinstance(n, Service)]
    assert services == [], (
        "Real Service already in the graph -- no stub should be emitted"
    )
    webapp = next(n for n in out if isinstance(n, WebApp))
    assert webapp.service_id == real_svc_id


@pytest.mark.asyncio
async def test_httpx_fallback_stub_service_is_deterministic_across_reparses() -> None:
    """Re-probing the same WebApp must produce the SAME stub Service id
    (so a future nmap pass that touches the port can enrich the row in
    place via MERGE-on-id, not create a duplicate)."""
    eid = uuid4()
    agent = _agent()
    out_a = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    out_b = await agent._parse_and_persist(
        _ctx(eid), MagicMock(), {}, _one_webapp_httpx(), tool_run_id=uuid4()
    )
    svc_a = next(n for n in out_a if isinstance(n, Service))
    svc_b = next(n for n in out_b if isinstance(n, Service))
    assert svc_a.id == svc_b.id
    # And the host_id is also deterministic (same engagement, same host).
    assert svc_a.host_id == svc_b.host_id


def test_nmap_xml_importer_uses_deterministic_host_ids() -> None:
    """The nmap-XML importer (orchestrator/importers/nmap_xml.py) is a
    second parse site for Host / Service. R3 #K1 brings it into parity
    with ReconAgent so a re-import of the same XML doesn't duplicate."""
    from orchestrator.importers.nmap_xml import parse_nmap_xml

    xml = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <status state="up"/>
    <address addr="10.0.0.1" addrtype="ipv4"/>
    <hostnames><hostname name="t.local"/></hostnames>
    <ports><port protocol="tcp" portid="22"><state state="open"/><service name="ssh"/></port></ports>
  </host>
</nmaprun>"""
    eid = uuid4()
    tr_id = uuid4()
    nodes_a, _ = parse_nmap_xml(xml, engagement_id=eid, tool_run_id=tr_id)
    nodes_b, _ = parse_nmap_xml(xml, engagement_id=eid, tool_run_id=tr_id)
    host_a = next(n for n in nodes_a if isinstance(n, Host))
    host_b = next(n for n in nodes_b if isinstance(n, Host))
    assert host_a.id == host_b.id
    svc_a = next(n for n in nodes_a if isinstance(n, Service))
    svc_b = next(n for n in nodes_b if isinstance(n, Service))
    assert svc_a.id == svc_b.id
