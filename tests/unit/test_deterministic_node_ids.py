"""R3 #J2 Bug 1: deterministic UUIDs for duplicate-risk node kinds.

H2 added enumeration-substage checkpoints to avoid re-parse duplicates,
but the actual root cause is that agents minted fresh ``uuid4()`` ids
for every parse pass — Neo4j MERGE-on-id then could not dedupe across
runs. J2 fixes the cause: every duplicate-risk kind (Endpoint, Finding,
Credential, ExploitCandidate, AttackPath) now derives its id from
``uuid5(_SERENADE_NS, f"{kind}:{engagement_id}:{natural_key}")`` so
re-parse / resume / retry collapses on identity.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from worldstate.schema import (
    _SERENADE_NS,
    deterministic_node_id,
    natural_key_for_attack_path,
    natural_key_for_credential,
    natural_key_for_endpoint,
    natural_key_for_exploit_candidate,
    natural_key_for_finding,
)


def test_same_inputs_produce_same_uuid() -> None:
    """Determinism is the whole point — two calls with the same
    (kind, engagement, natural_key) MUST return the same UUID."""
    eid = uuid4()
    nk = natural_key_for_endpoint(
        webapp_id=UUID("11111111-1111-1111-1111-111111111111"),
        method="GET",
        path="/admin",
    )
    a = deterministic_node_id("Endpoint", eid, nk)
    b = deterministic_node_id("Endpoint", eid, nk)
    assert a == b
    assert isinstance(a, UUID)


def test_different_engagements_yield_different_ids() -> None:
    """Two engagements with the same natural key MUST get distinct ids
    so node identity doesn't leak across runs."""
    eid_a = uuid4()
    eid_b = uuid4()
    nk = "GET|/admin"
    a = deterministic_node_id("Endpoint", eid_a, nk)
    b = deterministic_node_id("Endpoint", eid_b, nk)
    assert a != b


def test_different_kinds_yield_different_ids() -> None:
    """An Endpoint and a Finding sharing the same natural_key text MUST
    NOT collide — the kind prefix is part of the hash input."""
    eid = uuid4()
    nk = "shared_natural_key"
    a = deterministic_node_id("Endpoint", eid, nk)
    b = deterministic_node_id("Finding", eid, nk)
    assert a != b


def test_endpoint_natural_key_includes_method_path_webapp() -> None:
    """Endpoints are identified by (webapp, method, path) — varying
    any of the three must produce a distinct natural key."""
    wid = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    base = natural_key_for_endpoint(webapp_id=wid, method="GET", path="/x")
    method_changed = natural_key_for_endpoint(webapp_id=wid, method="POST", path="/x")
    path_changed = natural_key_for_endpoint(webapp_id=wid, method="GET", path="/y")
    webapp_changed = natural_key_for_endpoint(webapp_id=None, method="GET", path="/x")
    assert len({base, method_changed, path_changed, webapp_changed}) == 4


def test_finding_natural_key_hashes_title_for_compactness() -> None:
    """The full title is hashed so URL-bound noise doesn't blow up the
    natural-key string. Two distinct titles MUST yield distinct keys."""
    hid = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    a = natural_key_for_finding(category="cve-2024-1", affected_node_id=hid, title="t1")
    b = natural_key_for_finding(category="cve-2024-1", affected_node_id=hid, title="t2")
    assert a != b
    # Hashed component is 16 hex chars; total key is bounded.
    assert len(a.split("|")[-1]) == 16


def test_credential_natural_key_separates_by_source_host() -> None:
    """The same (username, domain) extracted from two different hosts
    must produce different natural keys — lateral validation needs each
    cred's evidence chain kept distinct per host."""
    host_a = UUID("cccccccc-cccc-cccc-cccc-aaaaaaaaaaaa")
    host_b = UUID("cccccccc-cccc-cccc-cccc-bbbbbbbbbbbb")
    a = natural_key_for_credential(
        username="alice", domain="corp", source_host_id=host_a
    )
    b = natural_key_for_credential(
        username="alice", domain="corp", source_host_id=host_b
    )
    assert a != b


def test_credential_natural_key_collapses_without_source_host() -> None:
    """A credential without a source-host (e.g. fresh netexec auth)
    collapses across re-parses on (username, domain) — the second
    parse becomes a MERGE no-op."""
    a = natural_key_for_credential(
        username="alice", domain="corp", source_host_id=None
    )
    b = natural_key_for_credential(
        username="alice", domain="corp", source_host_id=None
    )
    assert a == b


def test_exploit_candidate_natural_key_dedupes_searchsploit_reruns() -> None:
    """Two ``searchsploit`` lookups for the same finding/CVE must
    collide on the natural key so a re-run is idempotent."""
    fid = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    a = natural_key_for_exploit_candidate(
        finding_id=fid, source="exploitdb", source_ref="EDB-12345"
    )
    b = natural_key_for_exploit_candidate(
        finding_id=fid, source="exploitdb", source_ref="EDB-12345"
    )
    assert a == b


def test_attack_path_natural_key_respects_step_order() -> None:
    """A path A→B→C and a path A→C→B for the same objective are
    distinct — order matters for attack-path identity."""
    a, b, c = uuid4(), uuid4(), uuid4()
    k1 = natural_key_for_attack_path(objective="dom_admin", step_node_ids=[a, b, c])
    k2 = natural_key_for_attack_path(objective="dom_admin", step_node_ids=[a, c, b])
    assert k1 != k2


def test_namespace_is_zero_uuid() -> None:
    """The published namespace is the zero UUID so a future ops script
    can reproduce expected ids without consulting Serenade source."""
    assert _SERENADE_NS == UUID("00000000-0000-0000-0000-000000000000")


def test_empty_kind_raises() -> None:
    """An empty ``kind`` is a programmer error — refuse it so a future
    typo doesn't silently produce wrong ids."""
    with pytest.raises(ValueError):
        deterministic_node_id("", uuid4(), "natural_key")


def test_reparse_yields_same_ids_for_nuclei_finding() -> None:
    """End-to-end smoke: the importer's deterministic id is stable
    across two invocations on the same JSONL — what the bug brief
    described as ``re-parse on resume → same uuid (no duplicate)``."""
    from orchestrator.importers.nuclei_jsonl import parse_nuclei_jsonl_text

    jsonl = (
        '{"template-id":"cve-2024-1","info":{"name":"X","severity":"medium",'
        '"description":"d"},"matched-at":"https://h/v","classification":{}}\n'
    )
    eid = uuid4()
    tr_id = uuid4()
    a = parse_nuclei_jsonl_text(jsonl, engagement_id=eid, tool_run_id=tr_id)
    b = parse_nuclei_jsonl_text(jsonl, engagement_id=eid, tool_run_id=tr_id)
    assert len(a) == 1 and len(b) == 1
    assert a[0].id == b[0].id


def test_reparse_yields_same_ids_across_engagements_are_distinct() -> None:
    """Same JSONL parsed under two engagements yields two distinct
    Finding ids — node identity must not leak across engagements."""
    from orchestrator.importers.nuclei_jsonl import parse_nuclei_jsonl_text

    jsonl = (
        '{"template-id":"cve-2024-1","info":{"name":"X","severity":"medium",'
        '"description":"d"},"matched-at":"https://h/v","classification":{}}\n'
    )
    eid_a = uuid4()
    eid_b = uuid4()
    tr_id = uuid4()
    a = parse_nuclei_jsonl_text(jsonl, engagement_id=eid_a, tool_run_id=tr_id)
    b = parse_nuclei_jsonl_text(jsonl, engagement_id=eid_b, tool_run_id=tr_id)
    assert a[0].id != b[0].id
