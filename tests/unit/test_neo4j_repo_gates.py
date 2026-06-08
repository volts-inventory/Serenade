"""Tests for the world-state graph repo gates (provenance, cypher removal)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from worldstate.neo4j_repo import InMemoryGraphRepo, Neo4jRepo, ProvenanceRequiredError
from worldstate.schema import Edge, EdgeKind


def test_neo4j_repo_alias():
    """Neo4jRepo is a backwards-compatible alias for InMemoryGraphRepo."""
    assert Neo4jRepo is InMemoryGraphRepo


def test_run_read_cypher_raises_not_implemented():
    """run_read_cypher is no longer supported and raises NotImplementedError."""
    repo = InMemoryGraphRepo()
    with pytest.raises(NotImplementedError, match="no longer supported"):
        asyncio.run(repo.run_read_cypher("MATCH (n) RETURN n"))


def test_run_cypher_raises_not_implemented():
    """run_cypher (alias) also raises NotImplementedError."""
    repo = InMemoryGraphRepo()
    with pytest.raises(NotImplementedError, match="no longer supported"):
        asyncio.run(repo.run_cypher("MATCH (n) CREATE (m) RETURN n"))


@pytest.mark.asyncio
async def test_upsert_edge_requires_provenance():
    """Edge without provenance must be refused."""
    repo = InMemoryGraphRepo()
    bad_edge = Edge(
        kind=EdgeKind.RUNS_ON,
        src_id=uuid4(),
        dst_id=uuid4(),
        engagement_id=uuid4(),
        # provenance defaults to []
    )
    with pytest.raises(ProvenanceRequiredError):
        await repo.upsert_edge(bad_edge)
