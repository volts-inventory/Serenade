"""In-memory-graph reimplementations of the post-compromise → vuln_triage
regression queries (ported off Neo4j Cypher).

Covers:
- PrivEscAgent._high_findings_without_candidates
- LateralAgent._recently_validated_on_known_hosts
- LateralAgent._hosts_with_authed_candidates
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agents.lateral import LateralAgent
from agents.privesc import PrivEscAgent
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import (
    Credential,
    Edge,
    EdgeKind,
    ExploitCandidate,
    Finding,
    Host,
    Severity,
)


def _deps(repo: InMemoryGraphRepo) -> dict:
    return {
        "router": MagicMock(),
        "repo": repo,
        "registry": MagicMock(),
        "policy": MagicMock(),
        "sandbox": MagicMock(),
    }


async def _finding(repo, eid, title, severity=Severity.HIGH):
    f = Finding(
        engagement_id=eid,
        provenance=[uuid4()],
        title=title,
        description="d",
        severity=severity,
    )
    await repo.upsert_node(f)
    return f


async def _candidate_for(repo, eid, finding_id):
    c = ExploitCandidate(
        engagement_id=eid,
        provenance=[uuid4()],
        finding_id=finding_id,
        source="exploitdb",
        source_ref="EDB-1",
        match_strength=0.9,
        risk_class="exploit",
    )
    await repo.upsert_node(c)
    return c


@pytest.mark.asyncio
async def test_privesc_linpeas_finding_without_candidate_is_returned() -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    f = await _finding(repo, eid, "linpeas SUID abuse", Severity.HIGH)
    host = Host(engagement_id=eid, provenance=[uuid4()], ip="10.0.0.5")
    await repo.upsert_node(host)
    await repo.upsert_edge(
        Edge(kind=EdgeKind.AFFECTS, src_id=f.id, dst_id=host.id, engagement_id=eid, provenance=[uuid4()])
    )
    agent = PrivEscAgent(**_deps(repo))

    out = await agent._high_findings_without_candidates(eid)
    assert out == [(str(f.id), str(host.id))]


@pytest.mark.asyncio
async def test_privesc_finding_with_candidate_is_excluded() -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    f = await _finding(repo, eid, "linpeas kernel exploit", Severity.CRITICAL)
    await _candidate_for(repo, eid, f.id)
    agent = PrivEscAgent(**_deps(repo))

    assert await agent._high_findings_without_candidates(eid) == []


@pytest.mark.asyncio
async def test_privesc_ignores_non_linpeas_and_low_severity() -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    await _finding(repo, eid, "nuclei xss", Severity.HIGH)  # not linpeas
    await _finding(repo, eid, "linpeas writable cron", Severity.MEDIUM)  # too low
    agent = PrivEscAgent(**_deps(repo))

    assert await agent._high_findings_without_candidates(eid) == []


@pytest.mark.asyncio
async def test_lateral_validated_cred_on_host_is_returned() -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    host = Host(engagement_id=eid, provenance=[uuid4()], ip="10.0.0.9")
    cred = Credential(
        engagement_id=eid, provenance=[uuid4()], username="root", source="secretsdump", valid=True
    )
    await repo.upsert_node(host)
    await repo.upsert_node(cred)
    await repo.upsert_edge(
        Edge(kind=EdgeKind.HAS_SESSION_ON, src_id=cred.id, dst_id=host.id, engagement_id=eid, provenance=[uuid4()])
    )
    agent = LateralAgent(**_deps(repo))

    assert await agent._recently_validated_on_known_hosts(eid) == [
        (str(host.id), str(cred.id))
    ]


@pytest.mark.asyncio
async def test_lateral_invalid_cred_excluded() -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    host = Host(engagement_id=eid, provenance=[uuid4()], ip="10.0.0.9")
    cred = Credential(
        engagement_id=eid, provenance=[uuid4()], username="root", source="default_creds", valid=False
    )
    await repo.upsert_node(host)
    await repo.upsert_node(cred)
    await repo.upsert_edge(
        Edge(kind=EdgeKind.HAS_SESSION_ON, src_id=cred.id, dst_id=host.id, engagement_id=eid, provenance=[uuid4()])
    )
    agent = LateralAgent(**_deps(repo))

    assert await agent._recently_validated_on_known_hosts(eid) == []


@pytest.mark.asyncio
async def test_lateral_host_already_triaged_is_filtered() -> None:
    """A validated cred on a host that already has a candidate-backed finding
    affecting it is considered already triaged and excluded."""
    eid = uuid4()
    repo = InMemoryGraphRepo()
    host = Host(engagement_id=eid, provenance=[uuid4()], ip="10.0.0.9")
    cred = Credential(
        engagement_id=eid, provenance=[uuid4()], username="root", source="secretsdump", valid=True
    )
    f = await _finding(repo, eid, "smb signing disabled", Severity.HIGH)
    await _candidate_for(repo, eid, f.id)
    await repo.upsert_node(host)
    await repo.upsert_node(cred)
    await repo.upsert_edge(
        Edge(kind=EdgeKind.HAS_SESSION_ON, src_id=cred.id, dst_id=host.id, engagement_id=eid, provenance=[uuid4()])
    )
    await repo.upsert_edge(
        Edge(kind=EdgeKind.AFFECTS, src_id=f.id, dst_id=host.id, engagement_id=eid, provenance=[uuid4()])
    )
    agent = LateralAgent(**_deps(repo))

    assert await agent._hosts_with_authed_candidates(eid) == {str(host.id)}
    # ...so the host is filtered out of the regression producer.
    assert await agent._recently_validated_on_known_hosts(eid) == []
