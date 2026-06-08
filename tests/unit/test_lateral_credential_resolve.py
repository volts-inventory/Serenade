"""LateralAgent._resolve_credential_id tests.

The resolver uses the secondary index (get_nodes_by_kind_field) for O(1)
lookup by username, then filters by domain if specified.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agents.lateral import LateralAgent
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import Credential


def _make_lateral(repo: InMemoryGraphRepo) -> LateralAgent:
    return LateralAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


@pytest.mark.asyncio
async def test_resolve_with_domain_returns_matching_credential() -> None:
    """AD cred path: a cred carrying a ``domain`` must resolve to its id."""
    eid = uuid4()
    cred_id = uuid4()
    repo = InMemoryGraphRepo()
    cred = Credential(
        id=cred_id,
        engagement_id=eid,
        provenance=[uuid4()],
        username="admin",
        domain="corp.example.com",
        source="kerberoast",
    )
    await repo.upsert_node(cred)
    agent = _make_lateral(repo)

    result = await agent._resolve_credential_id(
        eid, username="admin", domain="corp.example.com"
    )
    assert result == cred_id


@pytest.mark.asyncio
async def test_resolve_without_domain_returns_first_match() -> None:
    """Local-user path: no domain filter, returns first match."""
    eid = uuid4()
    cred_id = uuid4()
    repo = InMemoryGraphRepo()
    cred = Credential(
        id=cred_id,
        engagement_id=eid,
        provenance=[uuid4()],
        username="local",
        source="smb_null",
    )
    await repo.upsert_node(cred)
    agent = _make_lateral(repo)

    result = await agent._resolve_credential_id(eid, username="local", domain=None)
    assert result == cred_id


@pytest.mark.asyncio
async def test_resolve_returns_none_for_empty_username() -> None:
    """Empty / falsy username short-circuits before hitting the repo."""
    repo = InMemoryGraphRepo()
    agent = _make_lateral(repo)
    result = await agent._resolve_credential_id(uuid4(), username=None, domain="x")
    assert result is None


@pytest.mark.asyncio
async def test_resolve_returns_none_when_no_rows() -> None:
    """Real not-found: cred isn't in the graph. Returns None, no raise."""
    repo = InMemoryGraphRepo()
    agent = _make_lateral(repo)
    result = await agent._resolve_credential_id(
        uuid4(), username="ghost", domain=None
    )
    assert result is None


@pytest.mark.asyncio
async def test_resolve_domain_mismatch_returns_none() -> None:
    """Credential exists but with a different domain -- should not match."""
    eid = uuid4()
    repo = InMemoryGraphRepo()
    cred = Credential(
        id=uuid4(),
        engagement_id=eid,
        provenance=[uuid4()],
        username="admin",
        domain="other.domain.com",
        source="kerberoast",
    )
    await repo.upsert_node(cred)
    agent = _make_lateral(repo)

    result = await agent._resolve_credential_id(
        eid, username="admin", domain="corp.example.com"
    )
    assert result is None
