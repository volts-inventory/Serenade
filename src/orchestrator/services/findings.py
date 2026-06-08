"""Read-only finding/world-state queries — plain async functions.

Extracted from ``orchestrator.api.findings``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from orchestrator.provenance import postgres_provenance_validator
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.queries import (
    coverage,
    findings_by_severity,
    hosts_with_open_services,
    webapps_summary,
)

_repo_singleton: InMemoryGraphRepo | None = None


def get_repo() -> InMemoryGraphRepo:
    global _repo_singleton
    if _repo_singleton is None:
        _repo_singleton = InMemoryGraphRepo(
            provenance_validator=postgres_provenance_validator,
        )
    return _repo_singleton


async def get_coverage(engagement_id: UUID) -> dict[str, Any]:
    return await coverage(get_repo(), engagement_id)


async def get_hosts(engagement_id: UUID) -> list[dict[str, Any]]:
    return await hosts_with_open_services(get_repo(), engagement_id)


async def get_webapps(engagement_id: UUID) -> list[dict[str, Any]]:
    return await webapps_summary(get_repo(), engagement_id)


async def get_findings(engagement_id: UUID) -> dict[str, Any]:
    return await findings_by_severity(get_repo(), engagement_id)
