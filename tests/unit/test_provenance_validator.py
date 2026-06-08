"""Tests for the provenance validator hook on InMemoryGraphRepo."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from worldstate.neo4j_repo import InMemoryGraphRepo, ProvenanceRequiredError
from worldstate.schema import Host


@pytest.mark.asyncio
async def test_upsert_node_with_validator_rejects_unknown_provenance():
    eid = uuid4()

    async def validator(engagement_id: UUID, tool_run_ids: list[UUID]) -> bool:
        # Always reject
        return False

    repo = InMemoryGraphRepo(provenance_validator=validator)

    h = Host(engagement_id=eid, ip="10.0.0.5", provenance=[uuid4()])
    with pytest.raises(ProvenanceRequiredError, match="unknown tool_run_ids"):
        await repo.upsert_node(h)


@pytest.mark.asyncio
async def test_upsert_node_with_validator_accepts_known_provenance():
    eid = uuid4()
    tr = uuid4()
    seen: list[tuple[UUID, list[UUID]]] = []

    async def validator(engagement_id: UUID, tool_run_ids: list[UUID]) -> bool:
        seen.append((engagement_id, tool_run_ids))
        return True

    repo = InMemoryGraphRepo(provenance_validator=validator)

    h = Host(engagement_id=eid, ip="10.0.0.5", provenance=[tr])
    await repo.upsert_node(h)
    assert seen == [(eid, [tr])]
    # Verify the node was actually inserted.
    nodes = await repo.get_nodes_by_kind(eid, "Host")
    assert len(nodes) == 1
    assert nodes[0]["ip"] == "10.0.0.5"


@pytest.mark.asyncio
async def test_no_validator_falls_back_to_presence_check():
    """If no validator is injected, only the format-level check runs."""
    repo = InMemoryGraphRepo()

    eid = uuid4()
    # Empty provenance still rejected
    h = Host(engagement_id=eid, ip="10.0.0.5", provenance=[])
    with pytest.raises(ProvenanceRequiredError, match="without provenance"):
        await repo.upsert_node(h)
