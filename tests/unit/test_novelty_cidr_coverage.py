"""Regression test for ws-07: NoveltyCritic CIDR coverage check.

An already-fully-enumerated CIDR delta must be classified as ``existing``
(not novel) so the already-known VETO path can fire, closing the supply side
of the regression loop.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from agents.regression_council import (
    _cidr_fully_covered,
    _classify_delta_entities,
)
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import Host


def test_cidr_fully_covered_pure_helper():
    # /30 -> usable hosts .1 and .2 (RFC: 4 addrs, 2 usable).
    covered = {"10.0.0.1", "10.0.0.2"}
    assert _cidr_fully_covered("10.0.0.0/30", covered) is True
    # Missing one host -> not covered.
    assert _cidr_fully_covered("10.0.0.0/30", {"10.0.0.1"}) is False
    # Empty graph -> not covered.
    assert _cidr_fully_covered("10.0.0.0/30", set()) is False


def test_cidr_too_large_treated_as_novel():
    # A /8 is too large to enumerate cheaply -> treated as novel (not covered).
    assert _cidr_fully_covered("10.0.0.0/8", {"10.0.0.1"}) is False


def test_malformed_cidr_treated_as_novel():
    assert _cidr_fully_covered("not-a-cidr", {"10.0.0.1"}) is False


def test_single_host_cidr():
    assert _cidr_fully_covered("10.0.0.5/32", {"10.0.0.5"}) is True
    assert _cidr_fully_covered("10.0.0.5/32", set()) is False


@pytest.mark.asyncio
async def test_classify_marks_covered_cidr_as_existing():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    # Populate the two usable hosts of 10.0.0.0/30.
    for ip in ("10.0.0.1", "10.0.0.2"):
        await repo.upsert_node(
            Host(engagement_id=eid, ip=ip, provenance=[uuid4()])
        )

    novel, existing = await _classify_delta_entities(
        repo, eid, {"cidrs": ["10.0.0.0/30"]}
    )
    assert novel == 0
    assert existing == 1


@pytest.mark.asyncio
async def test_classify_marks_uncovered_cidr_as_novel():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    # Only one of the two usable hosts present.
    await repo.upsert_node(Host(engagement_id=eid, ip="10.0.0.1", provenance=[uuid4()]))

    novel, existing = await _classify_delta_entities(
        repo, eid, {"cidrs": ["10.0.0.0/30"]}
    )
    assert novel == 1
    assert existing == 0


@pytest.mark.asyncio
async def test_classify_empty_graph_cidr_is_novel():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    novel, existing = await _classify_delta_entities(
        repo, eid, {"cidrs": ["192.168.1.0/30"]}
    )
    assert novel == 1
    assert existing == 0
