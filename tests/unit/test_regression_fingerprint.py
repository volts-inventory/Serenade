"""Tests for the delta-fingerprint mechanism.

Validates the canonical fingerprint used by NoveltyCritic's replay-loop
guard. The fingerprint is the sha256 of the canonical-JSON delta — two
deltas that are semantically the same (same CIDRs in a different order,
same host IDs with different dict-key ordering, a non-strict CIDR vs its
canonical form) MUST produce the same fingerprint. Two deltas that
differ in any substantive way MUST produce different fingerprints.

Section anchors: R2 #4.1 — 4.5 (§B.4.3.delta_keying).
"""

from __future__ import annotations

from agents.regression_council import fingerprint_delta


def test_order_independent_cidr_lists():
    """``["10.0.6.0/24", "10.0.5.0/24"]`` and
    ``["10.0.5.0/24", "10.0.6.0/24"]`` MUST collide."""
    a = fingerprint_delta({"cidrs": ["10.0.6.0/24", "10.0.5.0/24"]})
    b = fingerprint_delta({"cidrs": ["10.0.5.0/24", "10.0.6.0/24"]})
    assert a == b


def test_cidr_canonicalisation():
    """``10.0.5.1/24`` and ``10.0.5.0/24`` are the same network
    (``ipaddress.ip_network(strict=False)`` normalises both to
    ``10.0.5.0/24``)."""
    a = fingerprint_delta({"cidrs": ["10.0.5.1/24"]})
    b = fingerprint_delta({"cidrs": ["10.0.5.0/24"]})
    assert a == b


def test_different_cidrs_have_different_fingerprints():
    """``10.0.5.0/24`` and ``10.0.6.0/24`` are genuinely different networks."""
    a = fingerprint_delta({"cidrs": ["10.0.5.0/24"]})
    b = fingerprint_delta({"cidrs": ["10.0.6.0/24"]})
    assert a != b


def test_overlapping_but_unequal_cidr_lists_differ():
    """``["10.0.5.0/24", "10.0.6.0/24"]`` and ``["10.0.5.0/24"]`` are
    overlapping (the first is a superset) but NOT identical — they hash
    differently. This is by design: each request brings distinct surface
    to recon, and subset-rejection is left to a future optimisation."""
    a = fingerprint_delta({"cidrs": ["10.0.5.0/24", "10.0.6.0/24"]})
    b = fingerprint_delta({"cidrs": ["10.0.5.0/24"]})
    assert a != b


def test_dict_key_order_independence():
    """``{"host_id": "x", "ports": [445]}`` and
    ``{"ports": [445], "host_id": "x"}`` MUST collide.
    (canonicalisation sorts dict keys.)"""
    a = fingerprint_delta({"host_id": "abc-123", "ports": [445, 3389]})
    b = fingerprint_delta({"ports": [3389, 445], "host_id": "abc-123"})
    assert a == b


def test_domain_case_insensitivity():
    """Domains are lowercased before hashing."""
    a = fingerprint_delta({"domains": ["API.Target.COM"]})
    b = fingerprint_delta({"domains": ["api.target.com"]})
    assert a == b


def test_hostname_url_stripping():
    """``https://api.target.com:8443/health`` and ``api.target.com``
    map to the same hostname after scheme/port/path stripping."""
    a = fingerprint_delta({"urls": ["https://api.target.com:8443/health"]})
    b = fingerprint_delta({"urls": ["api.target.com"]})
    assert a == b


def test_list_deduplication():
    """Duplicate entries are collapsed before hashing."""
    a = fingerprint_delta({"cidrs": ["10.0.5.0/24", "10.0.5.0/24"]})
    b = fingerprint_delta({"cidrs": ["10.0.5.0/24"]})
    assert a == b


def test_empty_delta_is_deterministic():
    """An empty dict always hashes the same."""
    a = fingerprint_delta({})
    b = fingerprint_delta({})
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_nested_dict_canonicalisation():
    """Deeply nested structures are recursed."""
    a = fingerprint_delta({"config": {"b": 2, "a": 1}})
    b = fingerprint_delta({"config": {"a": 1, "b": 2}})
    assert a == b


def test_three_uncovered_subnets_one_engagement():
    """Canonical "multi-segment enterprise" scenario.

    Three sequential CIDR discoveries from lateral on one engagement
    must produce three distinct fingerprints. The NoveltyCritic should
    APPROVE each because the fingerprints differ, even though all three
    share the same ``(RECON, NEW_SUBNET_DISCOVERED)`` pair.
    """
    fps = [
        fingerprint_delta({"cidrs": [f"10.0.{i}.0/24"]})
        for i in (5, 6, 7)
    ]
    assert len(set(fps)) == 3  # all unique
