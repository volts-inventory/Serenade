"""Scan file import — plain async functions.

Extracted from ``orchestrator.api.imports``.  Takes raw bytes instead
of FastAPI ``UploadFile``.
"""

from __future__ import annotations

import ipaddress as _ipaddress
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse as _urlparse
from uuid import UUID, uuid4

from orchestrator.db import session_scope
from orchestrator.importers.nmap_xml import parse_nmap_xml
from orchestrator.importers.nuclei_jsonl import parse_nuclei_jsonl_text
from orchestrator.ledger import append_audit_row
from orchestrator.models import EngagementRow, ToolRunRow
from orchestrator.services.errors import ValidationError
from orchestrator.services.findings import get_repo

ImportKind = Literal["nmap_xml", "nuclei_jsonl"]


async def import_scan(
    engagement_id: UUID,
    kind: ImportKind,
    data: bytes,
    filename: str = "",
    principal: str = "local",
) -> dict[str, Any]:
    """Import a prior scan into the world-state graph.

    Takes raw bytes (the TUI reads the file).
    """
    repo = get_repo()

    if kind not in ("nmap_xml", "nuclei_jsonl"):
        raise ValidationError(f"unsupported import kind: {kind}")

    if len(data) > 50 * 1024 * 1024:
        raise ValidationError("file too large (50 MB max)")
    text = data.decode("utf-8", errors="replace")

    tool_run_id = uuid4()
    started = datetime.now(UTC)

    # db-03: previously this opened THREE independent ``session_scope()``
    # transactions — ``record_tool_run`` (its own scope), the engagement read
    # (a SECOND scope opened while no shared session existed), and ``audit``
    # (a THIRD scope). There was no atomic boundary and, when called under an
    # ambient session, a nested scope opened a competing SQLite connection
    # that could self-lock. We now open exactly ONE session at a time (never
    # nested) and thread the engagement read + ``ToolRunRow`` write into the
    # first, then append the completion audit row through the same-session
    # ``append_audit_row`` variant in a second bounded transaction.
    #
    # The tool-run row is committed FIRST, before the in-memory repo upserts:
    # ``postgres_provenance_validator`` (fail-CLOSED) verifies each node's
    # cited ``tool_run_id`` against the ``tool_runs`` table through its OWN
    # connection, so the dispatch row must be durable before ``upsert_node``
    # runs or every imported node would be rejected as unverifiable.
    async with session_scope() as s:
        _eng = await s.get(EngagementRow, engagement_id)
        scope_cidrs = (_eng.scope or {}).get("cidrs", []) if _eng else []
        s.add(
            ToolRunRow(
                id=tool_run_id,
                engagement_id=engagement_id,
                tool=f"import:{kind}",
                input_json={"filename": filename, "size_bytes": len(data)},
                sandbox_id="import",
                started_at=started,
                ended_at=None,
                exit_code=0,
                stdout_blob_id=None,
                stderr_blob_id=None,
            )
        )

    if kind == "nmap_xml":
        nodes, edges = parse_nmap_xml(text, engagement_id=engagement_id, tool_run_id=tool_run_id)
        in_scope_host_ids, dropped = _filter_hosts_to_scope(nodes, scope_cidrs)
        nodes = [n for n in nodes if n.kind != "Host" or n.id in in_scope_host_ids]
        nodes = [n for n in nodes if n.kind != "Service" or n.host_id in in_scope_host_ids]
        edges = [e for e in edges if e.src_id in in_scope_host_ids or e.dst_id in in_scope_host_ids]
        for n in nodes:
            await repo.upsert_node(n)
        for e in edges:
            await repo.upsert_edge(e)
        counts = {
            "hosts": sum(1 for n in nodes if n.kind == "Host"),
            "services": sum(1 for n in nodes if n.kind == "Service"),
            "dropped_out_of_scope_hosts": dropped,
        }
    elif kind == "nuclei_jsonl":
        findings = parse_nuclei_jsonl_text(text, engagement_id=engagement_id, tool_run_id=tool_run_id)
        kept_findings, dropped_findings = _filter_findings_to_scope(findings, scope_cidrs)
        for f in kept_findings:
            await repo.upsert_node(f)
        counts = {"findings": len(kept_findings), "dropped_out_of_scope_findings": dropped_findings}
    else:
        raise ValidationError(f"unknown import kind: {kind}")

    async with session_scope() as s:
        await append_audit_row(
            s,
            engagement_id=engagement_id,
            actor=principal,
            action=f"import.{kind}.completed",
            payload={"filename": filename, "counts": counts, "tool_run_id": str(tool_run_id)},
        )
    return {"tool_run_id": str(tool_run_id), "kind": kind, "imported": counts}


# --------------------------------------------------------------------------- #
# Scope filters                                                               #
# --------------------------------------------------------------------------- #


def _ip_in_scope(ip_str: str, scope_cidrs: list[str]) -> bool:
    try:
        addr = _ipaddress.ip_address(ip_str)
    except (ValueError, TypeError):
        return False
    for cidr in scope_cidrs or []:
        try:
            if addr in _ipaddress.ip_network(cidr, strict=False):
                return True
        except (ValueError, TypeError):
            continue
    return False


def _filter_hosts_to_scope(nodes: list, scope_cidrs: list[str]) -> tuple[set, int]:
    if not scope_cidrs:
        return {n.id for n in nodes if n.kind == "Host"}, 0
    keep: set = set()
    dropped = 0
    for n in nodes:
        if n.kind != "Host":
            continue
        if _ip_in_scope(getattr(n, "ip", ""), scope_cidrs):
            keep.add(n.id)
        else:
            dropped += 1
    return keep, dropped


def _filter_findings_to_scope(findings: list, scope_cidrs: list[str]) -> tuple[list, int]:
    if not scope_cidrs:
        return findings, 0
    kept: list = []
    dropped = 0
    for f in findings:
        refs = getattr(f, "references", []) or []
        host: str | None = None
        for r in refs:
            try:
                host = _urlparse(str(r)).hostname
                if host:
                    break
            except Exception:
                continue
        if (host and _ip_in_scope(host, scope_cidrs)) or host is None:
            kept.append(f)
        else:
            dropped += 1
    return kept, dropped
