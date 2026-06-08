"""nmap XML importer.

Reuses `tools/nmap_.py::_parse_nmap_xml` so the parser stays one piece of
code. Outputs are converted to world-state nodes (Host + Service) for the
target engagement with `provenance` pointing at the synthetic
`import:nmap_xml` tool run.
"""

from __future__ import annotations

from uuid import UUID

from tools.nmap_ import _parse_nmap_xml
from worldstate.schema import (
    Edge,
    EdgeKind,
    Host,
    Service,
    deterministic_node_id,
    natural_key_for_host,
    natural_key_for_service,
)


def parse_nmap_xml(
    xml_text: str, *, engagement_id: UUID, tool_run_id: UUID
) -> tuple[list[Host | Service], list[Edge]]:
    """Convert an nmap XML payload into world-state nodes + edges.

    Host / Service ids are deterministic — a re-import
    of the same XML collapses on identity (MERGE-on-id no-op) instead
    of writing duplicates with fresh uuid4 ids.
    """
    nodes: list[Host | Service] = []
    edges: list[Edge] = []
    for parsed_host in _parse_nmap_xml(xml_text):
        if parsed_host.state != "up":
            continue
        host = Host(
            id=deterministic_node_id(
                "Host",
                engagement_id,
                natural_key_for_host(engagement_id, parsed_host.ip),
            ),
            engagement_id=engagement_id,
            provenance=[tool_run_id],
            ip=parsed_host.ip,
            hostnames=parsed_host.hostnames,
            os_family=parsed_host.os_family,
            os_detail=parsed_host.os_detail,
            mac=parsed_host.mac,
        )
        nodes.append(host)
        for p in parsed_host.ports:
            if p.state != "open":
                continue
            svc = Service(
                id=deterministic_node_id(
                    "Service",
                    engagement_id,
                    natural_key_for_service(host.id, p.port, p.proto),
                ),
                engagement_id=engagement_id,
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
            edges.append(
                Edge(
                    kind=EdgeKind.RUNS_ON,
                    src_id=svc.id,
                    dst_id=host.id,
                    engagement_id=engagement_id,
                    provenance=[tool_run_id],
                )
            )
    return nodes, edges
