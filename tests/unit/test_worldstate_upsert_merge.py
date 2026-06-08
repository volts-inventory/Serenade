"""Regression tests for ws-01 (additive/monotonic upsert merge) and
ws-04 (Credential redaction at the read boundary)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import (
    Credential,
    Host,
    Service,
    deterministic_node_id,
    natural_key_for_host,
    natural_key_for_service,
)


# --------------------------------------------------------------------------- #
# ws-01: a later low-detail re-parse must NOT wipe enriched fields            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stub_reparse_does_not_wipe_enriched_host_fields():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    nid = deterministic_node_id("Host", eid, natural_key_for_host(eid, "10.0.0.5"))

    rich = Host(
        id=nid,
        engagement_id=eid,
        ip="10.0.0.5",
        hostnames=["dc01.corp.local"],
        os_family="windows",
        os_detail="Windows Server 2019",
        confidence=0.9,
        provenance=[uuid4()],
    )
    await repo.upsert_node(rich)

    # Bare re-parse: same deterministic id, fewer fields, lower confidence.
    stub = Host(
        id=nid,
        engagement_id=eid,
        ip="10.0.0.5",
        confidence=0.5,
        provenance=[uuid4()],
    )
    await repo.upsert_node(stub)

    nodes = await repo.get_nodes_by_kind(eid, "Host")
    assert len(nodes) == 1
    h = nodes[0]
    # Enriched fields survive the stub re-parse.
    assert h["hostnames"] == ["dc01.corp.local"]
    assert h["os_family"] == "windows"
    assert h["os_detail"] == "Windows Server 2019"
    # Confidence is monotonic: never demoted.
    assert h["confidence"] == 0.9
    # Provenance accumulates (union of both parses).
    assert len(h["provenance"]) == 2


@pytest.mark.asyncio
async def test_service_stub_after_full_keeps_name_product_version():
    """The exact recon flow: nmap writes a full Service, httpx re-upserts a
    stub on the same deterministic id."""
    repo = InMemoryGraphRepo()
    eid = uuid4()
    host_id = uuid4()
    sid = deterministic_node_id(
        "Service", eid, natural_key_for_service(host_id, 443, "tcp")
    )

    full = Service(
        id=sid,
        engagement_id=eid,
        host_id=host_id,
        port=443,
        proto="tcp",
        name="https",
        product="Apache httpd",
        version="2.4.52",
        cpes=["cpe:/a:apache:http_server:2.4.52"],
        provenance=[uuid4()],
    )
    await repo.upsert_node(full)

    stub = Service(
        id=sid,
        engagement_id=eid,
        host_id=host_id,
        port=443,
        proto="tcp",
        # name/product/version all None on the stub.
        provenance=[uuid4()],
    )
    await repo.upsert_node(stub)

    svc = (await repo.get_nodes_by_kind(eid, "Service"))[0]
    assert svc["name"] == "https"
    assert svc["product"] == "Apache httpd"
    assert svc["version"] == "2.4.52"
    assert svc["cpes"] == ["cpe:/a:apache:http_server:2.4.52"]


@pytest.mark.asyncio
async def test_reparse_can_still_enrich_missing_fields():
    """Merge is additive: a second parse fills in fields the first lacked."""
    repo = InMemoryGraphRepo()
    eid = uuid4()
    nid = deterministic_node_id("Host", eid, natural_key_for_host(eid, "10.0.0.6"))

    # First parse has low confidence and no os/hostnames.
    first = Host(
        id=nid, engagement_id=eid, ip="10.0.0.6", confidence=0.4, provenance=[uuid4()]
    )
    await repo.upsert_node(first)

    second = Host(
        id=nid,
        engagement_id=eid,
        ip="10.0.0.6",
        os_family="linux",
        hostnames=["web01"],
        confidence=0.8,
        provenance=[uuid4()],
    )
    await repo.upsert_node(second)

    h = (await repo.get_nodes_by_kind(eid, "Host"))[0]
    assert h["os_family"] == "linux"
    assert h["hostnames"] == ["web01"]
    # Confidence advances upward (0.4 -> 0.8).
    assert h["confidence"] == 0.8


@pytest.mark.asyncio
async def test_list_fields_union_across_reparses():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    nid = deterministic_node_id("Host", eid, natural_key_for_host(eid, "10.0.0.7"))

    await repo.upsert_node(
        Host(id=nid, engagement_id=eid, ip="10.0.0.7", hostnames=["a"], provenance=[uuid4()])
    )
    await repo.upsert_node(
        Host(id=nid, engagement_id=eid, ip="10.0.0.7", hostnames=["b"], provenance=[uuid4()])
    )
    h = (await repo.get_nodes_by_kind(eid, "Host"))[0]
    assert set(h["hostnames"]) == {"a", "b"}


# --------------------------------------------------------------------------- #
# wsN-01: an *observed* boolean (default False) downgrade must apply, but an   #
# *absent* (unobserved) default must not clobber a populated value            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_observed_false_boolean_downgrades_prior_true():
    """A parse that genuinely observed ``Service.tls=False`` must be able to
    correct a prior ``tls=True`` back down. Because ``tls``'s schema default
    is ``False``, the sentinel rule alone would mask this; the ``observed``
    set (pydantic ``model_fields_set``) marks the field authoritative."""
    repo = InMemoryGraphRepo()
    eid = uuid4()
    host_id = uuid4()
    sid = deterministic_node_id(
        "Service", eid, natural_key_for_service(host_id, 8443, "tcp")
    )

    # First parse observed tls=True (explicitly passed -> in model_fields_set).
    await repo.upsert_node(
        Service(
            id=sid,
            engagement_id=eid,
            host_id=host_id,
            port=8443,
            proto="tcp",
            tls=True,
            provenance=[uuid4()],
        )
    )
    svc = (await repo.get_nodes_by_kind(eid, "Service"))[0]
    assert svc["tls"] is True

    # Second parse explicitly observed tls=False -> must downgrade.
    await repo.upsert_node(
        Service(
            id=sid,
            engagement_id=eid,
            host_id=host_id,
            port=8443,
            proto="tcp",
            tls=False,
            provenance=[uuid4()],
        )
    )
    svc = (await repo.get_nodes_by_kind(eid, "Service"))[0]
    assert svc["tls"] is False


@pytest.mark.asyncio
async def test_unobserved_default_boolean_does_not_clobber_prior_true():
    """A stub re-parse that NEVER set ``tls`` (so it carries the unobserved
    default ``False``) must NOT demote a prior observed ``tls=True``. This is
    the ws-01 clobber guard the wsN-01 fix must not reintroduce."""
    repo = InMemoryGraphRepo()
    eid = uuid4()
    host_id = uuid4()
    sid = deterministic_node_id(
        "Service", eid, natural_key_for_service(host_id, 9443, "tcp")
    )

    await repo.upsert_node(
        Service(
            id=sid,
            engagement_id=eid,
            host_id=host_id,
            port=9443,
            proto="tcp",
            tls=True,
            provenance=[uuid4()],
        )
    )

    # Stub never mentions tls -> it is NOT in model_fields_set, so the merge
    # treats its default False as a no-information stub and keeps True.
    await repo.upsert_node(
        Service(
            id=sid,
            engagement_id=eid,
            host_id=host_id,
            port=9443,
            proto="tcp",
            provenance=[uuid4()],
        )
    )
    svc = (await repo.get_nodes_by_kind(eid, "Service"))[0]
    assert svc["tls"] is True


# --------------------------------------------------------------------------- #
# ws-04: get_nodes_by_kind("Credential") is redacted by default              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_nodes_by_kind_credential_redacts_by_default():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    cred = Credential(
        engagement_id=eid,
        username="alice",
        password="hunter2",
        nt_hash="aabbccddeeff00112233445566778899",
        source="secretsdump",
        provenance=[uuid4()],
    )
    await repo.upsert_node(cred)

    creds = await repo.get_nodes_by_kind(eid, "Credential")
    assert len(creds) == 1
    c = creds[0]
    # Non-secret fields preserved.
    assert c["username"] == "alice"
    # Secret fields redacted.
    assert c["password"] == "[ENCRYPTED]"
    assert c["nt_hash"] == "[ENCRYPTED]"
    assert "hunter2" not in str(c)


@pytest.mark.asyncio
async def test_explicit_unredacted_accessor_returns_plaintext():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    cred = Credential(
        engagement_id=eid,
        username="bob",
        password="s3cr3t",
        source="smb",
        provenance=[uuid4()],
    )
    await repo.upsert_node(cred)

    # The encryption/execution path explicitly opts in.
    via_flag = await repo.get_nodes_by_kind(eid, "Credential", include_secrets=True)
    assert via_flag[0]["password"] == "s3cr3t"

    via_accessor = await repo.get_credentials_unredacted(eid)
    assert via_accessor[0]["password"] == "s3cr3t"


@pytest.mark.asyncio
async def test_non_credential_kinds_unaffected_by_redaction():
    repo = InMemoryGraphRepo()
    eid = uuid4()
    await repo.upsert_node(
        Host(engagement_id=eid, ip="10.0.0.8", provenance=[uuid4()])
    )
    hosts = await repo.get_nodes_by_kind(eid, "Host")
    assert hosts[0]["ip"] == "10.0.0.8"
