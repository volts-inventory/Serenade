"""World-state snapshot persistence across engagement pause/resume.

Proves the ``world_state_snapshots`` store + the in-place ``load_json``
rehydrate:

  * ``save_world_state`` then ``load_world_state`` round-trips the exact
    ``graph_json``.
  * A serialized graph holds Credential secret fields Fernet-encrypted (no
    plaintext on disk); a FRESH repo + ``load_json`` restores the plaintext
    and the hosts/services.
  * ``load_world_state`` for an unknown engagement returns ``None``;
    ``load_json`` replaces prior contents and rebuilds the secondary indices.

Master key + SQLite DB are rooted in a tmp dir (mirroring
``test_audit_chain.py``) so no real ``~/.serenade`` state is touched and
``create_all`` builds the table from the ORM metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from orchestrator import crypto
from orchestrator import db as _db
from orchestrator.crypto import _FERNET_PREFIX
from orchestrator.db import create_all, session_scope
from orchestrator.models import EngagementRow
from orchestrator.settings import get_settings
from orchestrator.world_state_store import load_world_state, save_world_state
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import Credential, Finding, Host, Severity


@pytest.fixture
async def store_db(monkeypatch, tmp_path):
    """Master key + SQLite DB rooted in a tmp dir, schema created."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    _db._engine = None
    _db._sessionmaker = None
    _db.init_engine(get_settings())
    await create_all()
    yield tmp_path
    if _db._engine is not None:
        await _db._engine.dispose()
    _db._engine = None
    _db._sessionmaker = None
    get_settings.cache_clear()
    crypto.reset_cache()


async def _make_engagement_with_id(eid_uuid) -> str:
    """Insert a minimal engagement row (FK parent) for a given id."""
    eid = str(eid_uuid)
    now = datetime.now(UTC)
    async with session_scope() as s:
        s.add(
            EngagementRow(
                id=eid,
                name="t",
                time_window_start=now,
                time_window_end=now + timedelta(hours=8),
            )
        )
    return eid


async def _make_engagement() -> str:
    """Insert a minimal engagement row (FK parent) and return its id."""
    return await _make_engagement_with_id(uuid4())


@pytest.mark.asyncio
async def test_save_then_load_roundtrips_exact_json(store_db) -> None:
    eid = await _make_engagement()
    payload = '{"nodes": [], "links": [], "directed": true}'

    await save_world_state(eid, payload)
    got = await load_world_state(eid)

    assert got == payload


@pytest.mark.asyncio
async def test_load_unknown_engagement_returns_none(store_db) -> None:
    assert await load_world_state(str(uuid4())) is None


@pytest.mark.asyncio
async def test_upsert_overwrites_prior_snapshot(store_db) -> None:
    eid = await _make_engagement()
    await save_world_state(eid, '{"v": 1}')
    await save_world_state(eid, '{"v": 2}')
    assert await load_world_state(eid) == '{"v": 2}'


@pytest.mark.asyncio
async def test_credential_secret_encrypted_at_rest_and_restored(store_db) -> None:
    eid_uuid = uuid4()
    eid = str(eid_uuid)
    # The snapshot row FKs to engagements.id; create the parent.
    now = datetime.now(UTC)
    async with session_scope() as s:
        s.add(
            EngagementRow(
                id=eid,
                name="t",
                time_window_start=now,
                time_window_end=now + timedelta(hours=8),
            )
        )

    tool_run_id = uuid4()
    repo = InMemoryGraphRepo()
    host = Host(
        engagement_id=eid_uuid,
        provenance=[tool_run_id],
        ip="10.0.0.5",
    )
    cred = Credential(
        engagement_id=eid_uuid,
        provenance=[tool_run_id],
        username="admin",
        password="hunter2-secret",
        source="default_creds",
    )
    await repo.upsert_node(host)
    await repo.upsert_node(cred)

    graph_json = repo.to_json()
    await save_world_state(eid, graph_json)

    # The stored snapshot must hold NO plaintext password and a Fernet token.
    stored = await load_world_state(eid)
    assert stored is not None
    assert "hunter2-secret" not in stored
    assert _FERNET_PREFIX in stored

    # A FRESH repo rehydrated via load_json exposes the Host and the
    # decrypted credential plaintext.
    fresh = InMemoryGraphRepo()
    fresh.load_json(stored)

    hosts = await fresh.get_nodes_by_kind(eid_uuid, "Host")
    assert len(hosts) == 1
    assert hosts[0]["ip"] == "10.0.0.5"

    # ws-04: the default Credential accessor REDACTS secrets so a caller can't
    # accidentally get plaintext; the explicit unredacted accessor (the
    # encryption/persistence path) restores the real decrypted material.
    redacted = await fresh.get_nodes_by_kind(eid_uuid, "Credential")
    assert len(redacted) == 1
    assert redacted[0]["password"] == "[ENCRYPTED]"

    creds = await fresh.get_credentials_unredacted(eid_uuid)
    assert len(creds) == 1
    assert creds[0]["password"] == "hunter2-secret"


@pytest.mark.asyncio
async def test_whole_snapshot_encrypted_at_rest_incl_non_credential_fields(store_db) -> None:
    # Secrets routinely land in non-Credential fields (Finding description /
    # evidence excerpts). The WHOLE snapshot blob must be ciphertext at rest,
    # so such secrets are never written in cleartext.
    import sqlite3

    eid_uuid = uuid4()
    eid = await _make_engagement_with_id(eid_uuid)
    tool_run_id = uuid4()
    secret = "LEAKED-PW-IN-FINDING-9f3c"

    repo = InMemoryGraphRepo()
    await repo.upsert_node(
        Finding(
            engagement_id=eid_uuid,
            provenance=[tool_run_id],
            title="secretsdump",
            description=f"recovered admin password {secret} from SAM",
            severity=Severity.HIGH,
        )
    )
    await save_world_state(eid, repo.to_json())

    # Read the RAW column directly (bypass the store's decrypt).
    db_path = store_db / "serenade.db"
    con = sqlite3.connect(str(db_path))
    try:
        raw = con.execute(
            "SELECT graph_json FROM world_state_snapshots WHERE engagement_id = ?",
            (eid,),
        ).fetchone()[0]
    finally:
        con.close()

    assert secret not in raw, "secret in a non-Credential field leaked to disk in cleartext"
    assert raw.startswith(_FERNET_PREFIX), "whole snapshot blob is not encrypted"

    # And it round-trips back.
    restored = await load_world_state(eid)
    assert restored is not None and secret in restored


@pytest.mark.asyncio
async def test_wrong_master_key_load_fails_safe(store_db, monkeypatch) -> None:
    # A snapshot written under key A cannot be read after the master key is
    # rotated/lost: load_world_state returns None (resume starts empty) rather
    # than returning ciphertext.
    eid = await _make_engagement()
    await save_world_state(eid, '{"nodes": [], "links": [], "directed": true}')
    assert await load_world_state(eid) is not None

    # Rotate the master key: delete it and clear the Fernet cache.
    crypto.master_key_path().unlink()
    crypto.reset_cache()

    assert await load_world_state(eid) is None


@pytest.mark.asyncio
async def test_load_json_replaces_contents_and_rebuilds_indices(store_db) -> None:
    eid = uuid4()
    other_eid = uuid4()
    tool_run_id = uuid4()

    # Source repo: one host on the snapshot engagement.
    src = InMemoryGraphRepo()
    await src.upsert_node(
        Host(engagement_id=eid, provenance=[tool_run_id], ip="192.168.1.10")
    )
    snapshot = src.to_json()

    # Target repo pre-loaded with a DIFFERENT host (stale prior contents).
    target = InMemoryGraphRepo()
    await target.upsert_node(
        Host(engagement_id=other_eid, provenance=[tool_run_id], ip="172.16.0.1")
    )

    target.load_json(snapshot)

    # Prior contents are gone; only the snapshot's host remains.
    stale = await target.get_nodes_by_kind(other_eid, "Host")
    assert stale == []

    # The indexed-field query works after load (indices were rebuilt).
    by_ip = await target.get_nodes_by_kind_field(eid, "Host", "ip", "192.168.1.10")
    assert len(by_ip) == 1
    assert by_ip[0]["ip"] == "192.168.1.10"
