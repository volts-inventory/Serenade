"""ReconAgent — discovers live hosts and enumerates services.

Tools: nmap_host_discovery, nmap_service_scan, httpx_probe.
Writes: Host, Service, WebApp nodes; RUNS_ON, HOSTS edges.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision
from llm.prompts.base import PERSONA_RECON
from tools.base import Tool
from tools.httpx_ import HttpxOutput
from tools.nmap_ import NmapOutput
from worldstate.queries import coverage
from worldstate.schema import (
    Edge,
    EdgeKind,
    Host,
    Service,
    Stage,
    WebApp,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_host,
    natural_key_for_service,
    natural_key_for_webapp,
)

log = structlog.get_logger(__name__)


def _ip_in_any_delta_cidr(ip_str: str, cidrs: list[str]) -> bool:
    """Return True if *ip_str* falls within any of the given CIDRs.

    Handles IPv4-mapped IPv6 addresses (``::ffff:10.0.5.1``) by
    extracting the embedded IPv4 address before comparison, so hosts
    stored with mapped addresses still match IPv4 delta CIDRs.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except (ValueError, TypeError):
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    for c in cidrs:
        try:
            net = ipaddress.ip_network(c, strict=False)
            if addr.version == net.version and addr in net:
                return True
        except (ValueError, TypeError):
            continue
    return False


class ReconAgent(BaseAgent):
    stage = Stage.RECON
    system_persona = PERSONA_RECON
    # Recon does broad enumeration across many tools — host discovery,
    # service-version scans, HTTP probing. Needs more turns than the
    # 15-default so it can branch across all open-port categories without
    # tripping the cap on the first long sweep.
    max_loop_turns = 20

    # Zero-progress watchdog threshold. After this many consecutive
    # turns with zero hosts in the world-state graph, the agent escalates
    # to a human instead of looping until `max_turns`. Faster signal on
    # firewalled scopes / no-tool-call hallucinations.
    ZERO_HOST_WATCHDOG_THRESHOLD = 3
    # "service coverage" threshold — we treat the stage as
    # converged when services-per-host reaches this fraction of hosts.
    # 0.9 lets one slow host straggle without holding the whole stage
    # back; 1.0 would deadlock on an unresponsive box.
    SERVICE_COVERAGE_ADVANCE_RATIO = 0.9

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        hosts = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Host")

        if ctx.regression_delta is not None:
            # Delta-scoped re-entry: narrow to ONLY the delta CIDRs plus
            # tight upstream context so the agent doesn't re-scan the
            # entire engagement scope.
            delta_cidrs = ctx.regression_delta.get("cidrs") or []
            filtered = [
                h for h in hosts
                if _ip_in_any_delta_cidr(h.get("ip", ""), delta_cidrs)
            ]
            host_summary = [
                {"ip": h.get("ip"), "os_family": h.get("os_family"), "first_seen": h.get("first_seen")}
                for h in filtered[:25]
            ]
            return {
                "coverage": cov,
                "scope": {"cidrs": delta_cidrs},
                "known_hosts": host_summary,
                "turn": ctx.turn,
                "max_turns": self.budget.max_turns,
                "regression_context": {
                    "delta": ctx.regression_delta,
                    "origin_stage": ctx.regression_origin_stage,
                },
            }

        # Full-scope branch (no active regression delta).
        # Trim each host to a compact summary
        host_summary = [
            {"ip": h.get("ip"), "os_family": h.get("os_family"), "first_seen": h.get("first_seen")}
            for h in hosts[:25]
        ]
        return {
            "coverage": cov,
            "scope": ctx.engagement.get("scope", {}),
            "known_hosts": host_summary,
            "turn": ctx.turn,
            "max_turns": self.budget.max_turns,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior network-discovery specialist's per-turn workflow.

        Encodes the mental model from PERSONA_RECON as a branching
        decision tree the planner can execute one tool call at a time.
        The branch chosen depends on what's already in the world-state
        graph (host count, service coverage) — recon escalates from
        cheap-broad discovery to expensive-deep service scans only
        after the cheap pass produced live targets.
        """
        cov = slice_.get("coverage", {})
        hosts = int(cov.get("hosts", 0))
        services = int(cov.get("services", 0))
        webapps = int(cov.get("webapps", 0))
        scope = slice_.get("scope", {})

        # Stage 1: zero hosts — cheap-broad discovery first.
        if hosts == 0:
            return (
                f"PLAYBOOK [recon · stage 1 of 3: discovery]. State: hosts={hosts}, "
                f"services={services}, webapps={webapps}; scope={scope!r}. "
                "STEP: run nmap_host_discovery against the scope CIDRs to find live "
                "targets — cheapest broad signal before anything else. Prefer ports "
                "discovery first (top-1000), service-version scans come AFTER. If "
                "scope is a single /32 you may skip discovery and go straight to "
                "service-scan, but state that intent. "
                "REFUSE: brute-force, exploit attempts, anything destructive — that "
                "is exploit-stage work. NEVER assume a service from a port number. "
                "EXIT: as soon as one Host is in graph this turn ends and stage 2 "
                "begins on the next turn."
            )

        # Stage 2: hosts known, services lagging — focused service-version sweep.
        if services < hosts:
            return (
                f"PLAYBOOK [recon · stage 2 of 3: service-version]. State: hosts={hosts}, "
                f"services={services}, webapps={webapps}. STEP: run nmap_service_scan "
                "against the known hosts with top-1000 ports, scripts=default, "
                "timing=3. One scan per host (or batched per /28) — do not retry the "
                "same scan with louder flags hoping for different output; an empty "
                "result is a result. "
                "REFUSE: aggressive timing, full-port (1-65535) without operator "
                "ask, version brute-force on closed ports. "
                "EXIT: stage 3 begins when services-per-host >= "
                f"{self.SERVICE_COVERAGE_ADVANCE_RATIO:.0%} or after three turns "
                "with zero new services."
            )

        # Stage 3: services known — HTTP fingerprint so WebEnum has a base to work from.
        return (
            f"PLAYBOOK [recon · stage 3 of 3: web fingerprint]. State: hosts={hosts}, "
            f"services={services}, webapps={webapps}. STEP: probe HTTP(S) services "
            "with httpx_probe to populate WebApp nodes the enumeration stage will "
            "consume. One probe per discovered web service. After this turn there "
            "is nothing else for recon to do — emit a final-state message with no "
            "tool call to terminate the loop. "
            "REFUSE: directory bruteforce, vulnerability scans, parameter fuzzing — "
            "all of that is WebEnum's job. ESCALATE: any host whose fingerprint "
            "matches a honeypot pattern (canary banners, improbably-broad port set)."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        cov = await coverage(self.repo, ctx.engagement_id)
        # Real advance criterion — services-per-host has crossed
        # the coverage ratio (most hosts have at least their open ports
        # enumerated). Captures "we have substantive ground truth"; a
        # simpler "at least one host with one service" gate would advance
        # too eagerly on a partial first-host result.
        hosts = cov["hosts"]
        services = cov["services"]
        if hosts >= 1 and services >= max(1, int(self.SERVICE_COVERAGE_ADVANCE_RATIO * hosts)):
            # Real progress resets the watchdog. We're advancing, so
            # clear it as a courtesy for any downstream introspection.
            ctx.consecutive_zero_host_turns = 0
            return StepDecision.ADVANCE
        # Zero-progress watchdog. If we've completed multiple
        # consecutive turns with zero hosts discovered, escalate now
        # rather than burning the rest of `max_turns`. Common causes:
        # firewall drops nmap, scope is empty/misparsed, LLM emitted
        # zero tool calls and is hallucinating "thorough recon".
        if cov["hosts"] == 0:
            ctx.consecutive_zero_host_turns += 1
            if ctx.consecutive_zero_host_turns >= self.ZERO_HOST_WATCHDOG_THRESHOLD:
                log.warning(
                    "recon.zero_host_watchdog_escalate",
                    turn=ctx.turn,
                    consecutive=ctx.consecutive_zero_host_turns,
                )
                return StepDecision.ESCALATE_HUMAN
        else:
            # Hosts exist but services are missing; that's partial progress,
            # not a zero-host stall. Reset so a later collapse to zero (e.g.
            # graph wipe) restarts the counter cleanly.
            ctx.consecutive_zero_host_turns = 0
        if ctx.turn >= self.budget.max_turns:
            return StepDecision.ESCALATE_HUMAN
        return StepDecision.CONTINUE

    async def _parse_and_persist(
        self,
        ctx: StageContext,
        tool: Tool[Any, Any],
        tool_input: dict[str, Any],
        tool_output: BaseModel,
        tool_run_id: UUID,
    ) -> list[WorldStateNode]:
        nodes: list[WorldStateNode] = []
        if isinstance(tool_output, NmapOutput):
            host_id_by_ip: dict[str, UUID] = {}
            for h in tool_output.hosts:
                if h.state != "up":
                    continue
                # Deterministic Host id keyed on (engagement, ip): re-parsing
                # the same nmap output on resume lands on the same id so
                # MERGE-on-id collapses the duplicate write to a no-op.
                host = Host(
                    id=deterministic_node_id(
                        "Host",
                        ctx.engagement_id,
                        natural_key_for_host(ctx.engagement_id, h.ip),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    ip=h.ip,
                    hostnames=h.hostnames,
                    os_family=h.os_family,
                    os_detail=h.os_detail,
                    mac=h.mac,
                )
                nodes.append(host)
                host_id_by_ip[h.ip] = host.id
                for p in h.ports:
                    if p.state != "open":
                        continue
                    # Deterministic Service id keyed on
                    # (host, port, protocol). Same rationale as Host.
                    svc = Service(
                        id=deterministic_node_id(
                            "Service",
                            ctx.engagement_id,
                            natural_key_for_service(host.id, p.port, p.proto),
                        ),
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                        host_id=host.id,
                        port=p.port,
                        proto=p.proto,
                        name=p.name,
                        product=p.product,
                        version=p.version,
                        banner=p.extrainfo,
                        cpes=p.cpe,
                    )
                    nodes.append(svc)
                    # RUNS_ON edge
                    await self.repo.upsert_edge(
                        Edge(
                            kind=EdgeKind.RUNS_ON,
                            src_id=svc.id,
                            dst_id=host.id,
                            engagement_id=ctx.engagement_id,
                            provenance=[tool_run_id],
                        )
                    )
        elif isinstance(tool_output, HttpxOutput):
            # Resolve Service by (host_ip, port, "tcp")
            # so the WebApp.service_id is real, not a dangling UUID.
            service_by_host_port = await _index_services_by_host_port(self.repo, ctx.engagement_id)
            for probe in tool_output.probes:
                svc_id = service_by_host_port.get((probe.host, probe.port))
                # Deterministic WebApp id keyed on
                # (service, base_url). When the Service can't be
                # resolved we synthesize a STABLE fallback service id
                # from (engagement, host, port) so re-probing the same
                # URL still collides on the same WebApp identity — the
                # old ``uuid4()`` fallback minted a new id every call
                # and re-introduced exactly the duplication this
                # refactor closes.
                resolved_service_id = svc_id or deterministic_node_id(
                    "Service",
                    ctx.engagement_id,
                    natural_key_for_service(
                        deterministic_node_id(
                            "Host",
                            ctx.engagement_id,
                            natural_key_for_host(ctx.engagement_id, probe.host),
                        ),
                        probe.port,
                        "tcp",
                    ),
                )
                # When the Service lookup misses (nmap's top-1000 didn't cover
                # this port but httpx hit it via a tech-specific probe set,
                # e.g. 8443/8080/8888), emit a stub Service node so the
                # WebApp's foreign key resolves to an actual graph row. The
                # stub is intentionally minimal — ``name`` / ``product`` /
                # ``version`` are nullable in the schema. A later nmap pass
                # that touches this port collapses on the SAME id
                # (deterministic on host+port+proto) and the upsert enriches
                # the row in place.
                if svc_id is None:
                    host_id_for_stub = deterministic_node_id(
                        "Host",
                        ctx.engagement_id,
                        natural_key_for_host(ctx.engagement_id, probe.host),
                    )
                    stub_service = Service(
                        id=resolved_service_id,
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                        host_id=host_id_for_stub,
                        port=probe.port,
                        proto="tcp",
                        name=None,
                        product=None,
                        version=None,
                        banner=None,
                    )
                    nodes.append(stub_service)
                    # Emit the Service → Host RUNS_ON edge alongside the stub
                    # Service so edge-based Host ↔ Service traversal works
                    # immediately (without it, only the WebApp's FK would
                    # point at the stub Service and the edge walk would
                    # break until a later nmap pass enriched the node).
                    # Matches the nmap success-path convention at line
                    # 234-243 in this file and the nmap-XML importer.
                    await self.repo.upsert_edge(
                        Edge(
                            kind=EdgeKind.RUNS_ON,
                            src_id=resolved_service_id,
                            dst_id=host_id_for_stub,
                            engagement_id=ctx.engagement_id,
                            provenance=[tool_run_id],
                        )
                    )
                webapp = WebApp(
                    id=deterministic_node_id(
                        "WebApp",
                        ctx.engagement_id,
                        natural_key_for_webapp(resolved_service_id, probe.url),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    service_id=resolved_service_id,
                    base_url=probe.url,
                    vhost=probe.host,
                    technologies=probe.technologies,
                    title=probe.title,
                )
                nodes.append(webapp)
                # WebApp → Service OF_WEBAPP edge MUST fire on both branches
                # (resolved or stub) so the edge-based traversal the report
                # renderer walks (WebApp -OF_WEBAPP-> Service -RUNS_ON->
                # Host) stays intact even before a later nmap pass enriches
                # the Service. Use ``resolved_service_id`` (the stub id when
                # we synthesized one; the real id when the lookup hit).
                await self.repo.upsert_edge(
                    Edge(
                        kind=EdgeKind.OF_WEBAPP,
                        src_id=webapp.id,
                        dst_id=resolved_service_id,
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                    )
                )
        return nodes


async def _index_services_by_host_port(repo, engagement_id) -> dict[tuple[str, int], UUID]:
    """Build a (host_ip, port) -> Service.id index by joining Service to Host.

    Lets ReconAgent's WebApp creation cite the real Service.id rather
    than a placeholder UUID.
    """
    services = await repo.get_nodes_by_kind(engagement_id, "Service")
    hosts = await repo.get_nodes_by_kind(engagement_id, "Host")
    # Build host_id -> ip lookup
    host_ip_by_id: dict[str, str] = {}
    for h in hosts:
        hid = h.get("id")
        hip = h.get("ip")
        if hid and hip:
            host_ip_by_id[str(hid)] = str(hip)
    index: dict[tuple[str, int], UUID] = {}
    for s in services:
        sid = s.get("id")
        port = s.get("port")
        host_id = s.get("host_id")
        if sid and port is not None and host_id:
            host_ip = host_ip_by_id.get(str(host_id))
            if host_ip:
                try:
                    index[(host_ip, int(port))] = UUID(str(sid))
                except (ValueError, TypeError):
                    continue
    return index
