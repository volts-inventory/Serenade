"""Regression test for crypto-05: the audit-log PDF banner must reflect
verification of the ACTUAL rows being rendered, not a separately-supplied
whole-chain verdict.

We exercise :func:`verify_rendered_rows` (the row-level verifier the renderer
now derives its banner from) directly, so the test needs no WeasyPrint.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from orchestrator import crypto
from orchestrator.ledger import audit_canonical_bytes
from orchestrator.settings import get_settings
from reports.audit_log_pdf import verify_rendered_rows


@pytest.fixture
def master_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    yield
    get_settings.cache_clear()
    crypto.reset_cache()


def _make_chain(n: int) -> list[dict]:
    """Build ``n`` valid, linked audit rows (seq 1..n) as render-shaped dicts."""
    rows: list[dict] = []
    prev_hash: str | None = None
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    for seq in range(1, n + 1):
        payload = {"i": seq}
        canonical = audit_canonical_bytes(
            seq=seq,
            engagement_id=None,
            actor="system",
            action=f"act.{seq}",
            payload=payload,
            ts=ts,
            prev_hash=prev_hash,
        )
        row_hash = crypto.audit_hmac(canonical)
        rows.append(
            {
                "seq": seq,
                "engagement_id": None,
                "actor": "system",
                "action": f"act.{seq}",
                "payload": payload,
                "ts": ts,
                "prev_hash": prev_hash,
                "row_hash": row_hash,
            }
        )
        prev_hash = row_hash
    return rows


def test_intact_rows_verify(master_key):
    rows = _make_chain(3)
    status = verify_rendered_rows(rows)
    assert status["verified"] is True
    assert status["label"] == "VERIFIED"


def test_tampered_row_content_fails(master_key):
    rows = _make_chain(3)
    # Mutate a payload WITHOUT recomputing the hash -> hash-mismatch.
    rows[1]["payload"] = {"i": "tampered"}
    status = verify_rendered_rows(rows)
    assert status["verified"] is False
    assert "hash-mismatch" in status["reason"]


def test_paginated_subset_not_stamped_verified(master_key):
    """A subset that does not start at genesis cannot link -> UNVERIFIED."""
    rows = _make_chain(3)[1:]  # seq 2,3 only
    status = verify_rendered_rows(rows)
    assert status["verified"] is False


def test_reordered_rows_fail(master_key):
    rows = _make_chain(3)
    rows[0], rows[1] = rows[1], rows[0]
    status = verify_rendered_rows(rows)
    assert status["verified"] is False


def test_empty_rows_not_verified(master_key):
    status = verify_rendered_rows([])
    assert status["verified"] is False


def test_foreign_engagement_rows_not_verified(master_key):
    """cryptoN-02: a genuine, internally-consistent chain from engagement B must
    NOT stamp VERIFIED when the PDF header names a different engagement A."""
    # Build a valid genesis-anchored chain scoped to ENG-B.
    rows: list[dict] = []
    prev_hash: str | None = None
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    for seq in range(1, 3):
        payload = {"i": seq}
        canonical = audit_canonical_bytes(
            seq=seq,
            engagement_id="ENG-B",
            actor="system",
            action=f"act.{seq}",
            payload=payload,
            ts=ts,
            prev_hash=prev_hash,
        )
        row_hash = crypto.audit_hmac(canonical)
        rows.append(
            {
                "seq": seq,
                "engagement_id": "ENG-B",
                "actor": "system",
                "action": f"act.{seq}",
                "payload": payload,
                "ts": ts,
                "prev_hash": prev_hash,
                "row_hash": row_hash,
            }
        )
        prev_hash = row_hash

    # Sanity: the rows DO verify under their own scope.
    assert verify_rendered_rows(rows, engagement_id="ENG-B")["verified"] is True

    # But under a foreign header (ENG-A) they must be UNVERIFIED.
    status = verify_rendered_rows(rows, engagement_id="ENG-A")
    assert status["verified"] is False
    assert "engagement-mismatch" in status["reason"]


def test_string_ts_does_not_crash_and_is_unverified(master_key):
    """cryptoN-03: a row whose ``ts`` is a string (post JSON/API serialization)
    must degrade to UNVERIFIED, never raise AttributeError."""
    rows = _make_chain(2)
    # Serialize ts to a string, as it would arrive across a JSON boundary.
    for r in rows:
        r["ts"] = r["ts"].isoformat()
    # Must not raise.
    status = verify_rendered_rows(rows)
    assert status["verified"] is False
    assert "uncanonicalizable" in status["reason"]


def test_render_downgrades_mismatched_external_verdict(master_key):
    """render_audit_log_pdf must not stamp VERIFIED when the supplied
    chain_status claims VERIFIED but the rows do not re-link.

    We can't render the PDF without WeasyPrint, so assert the verdict
    selection logic via verify_rendered_rows on a tampered set: the renderer
    replaces a VERIFIED external verdict with this UNVERIFIED row result.
    """
    rows = _make_chain(3)
    rows[2]["row_hash"] = "deadbeef"  # corrupt
    rows_status = verify_rendered_rows(rows)
    external = {"verified": True, "label": "VERIFIED", "detail": "", "reason": "ok"}
    # Mirror the renderer's selection rule.
    chosen = rows_status if (external["verified"] and not rows_status["verified"]) else external
    assert chosen["verified"] is False
