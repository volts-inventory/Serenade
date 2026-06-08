"""Regression tests for ws-03 (provenance fails CLOSED on DB error) and
ws-08 (EvidenceRef.excerpt verified against its blob)."""

from __future__ import annotations

from uuid import uuid4

import pytest

import orchestrator.provenance as provenance
from orchestrator.provenance import (
    postgres_provenance_validator,
    verify_evidence_excerpts,
    verify_excerpt_against_blob,
)
from worldstate.neo4j_repo import InMemoryGraphRepo, ProvenanceRequiredError
from worldstate.schema import EvidenceRef, Host


# --------------------------------------------------------------------------- #
# ws-03: provenance validator fails CLOSED on a DB error                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_validator_returns_false_on_db_error(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    # session_scope() is entered as an async context manager; raising on call
    # simulates the DB being unreachable.
    monkeypatch.setattr(provenance, "session_scope", boom)

    ok = await postgres_provenance_validator(uuid4(), [uuid4()])
    assert ok is False


@pytest.mark.asyncio
async def test_validator_returns_false_on_empty_input():
    assert await postgres_provenance_validator(uuid4(), []) is False


@pytest.mark.asyncio
async def test_upsert_rejects_node_when_provenance_unverifiable(monkeypatch):
    """During a validator outage the node is rejected, not admitted."""

    async def failing_validator(engagement_id, tool_run_ids):
        # Mirror the fail-closed contract of the real validator.
        return False

    repo = InMemoryGraphRepo(provenance_validator=failing_validator)
    h = Host(engagement_id=uuid4(), ip="10.0.0.9", provenance=[uuid4()])
    with pytest.raises(ProvenanceRequiredError):
        await repo.upsert_node(h)


# --------------------------------------------------------------------------- #
# ws-08: EvidenceRef.excerpt is verified against the referenced blob          #
# --------------------------------------------------------------------------- #


def test_excerpt_present_in_blob_verifies():
    blob = b"nmap scan report: 22/tcp open ssh OpenSSH 8.9"
    assert verify_excerpt_against_blob("22/tcp open ssh", blob) is True


def test_fabricated_excerpt_fails_verification():
    blob = b"nmap scan report: 22/tcp open ssh OpenSSH 8.9"
    assert verify_excerpt_against_blob("CVE-2099-9999 root shell obtained", blob) is False


def test_excerpt_at_byte_offset():
    blob = b"0123456789HELLO0123456789"
    # Exact offset matches.
    assert verify_excerpt_against_blob("HELLO", blob, byte_offset=10) is True


def test_excerpt_offset_off_by_a_bit_still_verifies():
    """ws3-02: a correct excerpt cited at a slightly-off offset (e.g. the start
    of its containing line/record) must NOT be flagged. The verifier falls back
    to a whole-blob substring search when the offset window misses."""
    blob = b"0123456789HELLO0123456789"
    # Offset 0 points before the excerpt; the excerpt is still genuinely in the
    # blob, so it must verify (tolerant), not be flagged as fabricated.
    assert verify_excerpt_against_blob("HELLO", blob, byte_offset=0) is True


def test_excerpt_offset_with_genuinely_fabricated_quote_fails():
    """Tolerant offset fallback must still catch a quote that isn't in the blob
    at all."""
    blob = b"0123456789HELLO0123456789"
    assert verify_excerpt_against_blob("GOODBYE", blob, byte_offset=0) is False


def test_empty_excerpt_is_trivially_verified():
    assert verify_excerpt_against_blob(None, None) is True
    assert verify_excerpt_against_blob("", b"anything") is True


def test_missing_blob_is_kept_not_failed_closed():
    """ws3-01: with no blob there is no positive evidence of tampering, so the
    excerpt is KEPT (return True). Fail-closed here would strip all real
    evidence (no production producer attaches a blob_id)."""
    assert verify_excerpt_against_blob("something", None) is True


@pytest.mark.asyncio
async def test_verify_evidence_excerpts_reports_bad_indices():
    refs = [
        EvidenceRef(tool_run_id=uuid4(), excerpt="real line"),
        EvidenceRef(tool_run_id=uuid4(), excerpt="fabricated quote"),
        EvidenceRef(tool_run_id=uuid4(), excerpt=None),  # no excerpt -> ok
    ]
    blob = b"this is the real line from the tool output"

    async def fetcher(_ref):
        return blob

    bad = await verify_evidence_excerpts(refs, fetcher)
    assert bad == [1]
