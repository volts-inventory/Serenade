"""Common world-state queries used by agents to compute their context slice.

Agents do NOT free-form query the graph; they call these typed helpers so the
query surface is bounded and auditable.

All queries operate on the in-process networkx graph via InMemoryGraphRepo.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from worldstate.neo4j_repo import CREDENTIAL_SECRET_FIELDS, InMemoryGraphRepo

# Placeholder substituted for credential secret values before they can reach
# an LLM prompt or world-state slice.
REDACTED_PLACEHOLDER = "[ENCRYPTED]"


def redact_credential_secrets(node: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a Credential node with secret fields redacted.

    Each non-None secret field is replaced with ``[ENCRYPTED]``; None stays
    None. Non-secret fields (username/domain/source/valid/...) are untouched.
    """
    redacted = dict(node)
    for field in CREDENTIAL_SECRET_FIELDS:
        if redacted.get(field) is not None:
            redacted[field] = REDACTED_PLACEHOLDER
    return redacted


async def hosts_with_open_services(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> list[dict[str, Any]]:
    """Find Host nodes and follow RUNS_ON edges to collect their Services."""
    hosts = await repo.get_nodes_by_kind(engagement_id, "Host")
    result = []
    for h in hosts:
        host_id = h.get("id")
        if not host_id:
            continue
        # Services point to Hosts via RUNS_ON edges (Service -RUNS_ON-> Host).
        services = await repo.get_predecessors_by_edge_type(
            UUID(str(host_id)), "RUNS_ON"
        )
        result.append({"host": h, "services": services})
    # Sort by number of services descending.
    result.sort(key=lambda r: len(r["services"]), reverse=True)
    return result


async def webapps_summary(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> list[dict[str, Any]]:
    """Find WebApp nodes and count their Endpoints."""
    webapps = await repo.get_nodes_by_kind(engagement_id, "WebApp")
    result = []
    for w in webapps:
        webapp_id = w.get("id")
        if not webapp_id:
            continue
        # Count endpoints that reference this webapp via webapp_id field.
        endpoints = await repo.get_nodes_by_kind_field(
            engagement_id, "Endpoint", "webapp_id", str(webapp_id)
        )
        result.append({"webapp": w, "endpoints": len(endpoints)})
    # Sort by endpoint count descending.
    result.sort(key=lambda r: r["endpoints"], reverse=True)
    return result


async def findings_by_severity(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> dict[str, list[dict[str, Any]]]:
    findings = await repo.get_nodes_by_kind(engagement_id, "Finding")
    # Sort by cvss_v3_1 descending (None treated as 0).
    findings.sort(
        key=lambda f: float(f.get("cvss_v3_1") or 0), reverse=True
    )
    buckets: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        sev = f.get("severity", "info")
        buckets.setdefault(sev, []).append(f)
    return buckets


async def credentials_with_known_hosts(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> list[dict[str, Any]]:
    """Find Credential nodes with HAS_SESSION_ON edges to Hosts."""
    creds = await repo.get_nodes_by_kind(engagement_id, "Credential")
    result = []
    for c in creds:
        cred_id = c.get("id")
        if not cred_id:
            continue
        hosts = await repo.get_successors_by_edge_type(
            UUID(str(cred_id)), "HAS_SESSION_ON"
        )
        if hosts:
            result.append(
                {"credential": redact_credential_secrets(c), "hosts": hosts}
            )
    return result


async def attack_paths(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> list[dict[str, Any]]:
    paths = await repo.get_nodes_by_kind(engagement_id, "AttackPath")
    paths.sort(key=lambda p: float(p.get("cost", 1.0)))
    return paths


async def unmatched_candidates(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> list[dict[str, Any]]:
    """ExploitCandidates with no corresponding ExploitAttempt, strongest first."""
    candidates = await repo.get_nodes_by_kind(engagement_id, "ExploitCandidate")
    attempts = await repo.get_nodes_by_kind(engagement_id, "ExploitAttempt")
    attempted = {str(a.get("candidate_id")) for a in attempts if a.get("candidate_id")}
    out = [c for c in candidates if str(c.get("id")) not in attempted]
    out.sort(key=lambda c: float(c.get("match_strength", 0.0) or 0.0), reverse=True)
    return out


async def coverage(
    repo: InMemoryGraphRepo, engagement_id: UUID
) -> dict[str, Any]:
    """Coarse coverage signals; used by stage controller for rule-based transitions."""
    counts = await repo.count(engagement_id)
    return {
        "hosts": counts.get("Host", 0),
        "services": counts.get("Service", 0),
        "webapps": counts.get("WebApp", 0),
        "endpoints": counts.get("Endpoint", 0),
        "findings": counts.get("Finding", 0),
        "candidates": counts.get("ExploitCandidate", 0),
        "sessions": counts.get("Session", 0),
    }
