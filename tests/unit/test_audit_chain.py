"""At-rest tamper-evidence for the audit log (tracking issue #112, P0 #2).

Proves the per-engagement HMAC hash chain:

  * Each row carries a per-chain monotonic ``seq`` (genesis = 1), a
    ``prev_hash`` linking to the predecessor's ``row_hash`` (NULL only
    for genesis), and a master-key-HMAC ``row_hash`` over the canonical
    serialization (which includes ``prev_hash`` — that's the chain).
  * ``verify_audit_chain`` recomputes the chain and distinguishes
    edited rows (hash-mismatch), inserted/reordered rows (broken-link),
    deleted rows (seq-gap), and duplicate seqs (seq-dup).
  * The ``engagement_id=None`` global chain behaves like a per-engagement
    one under the ``__global__`` sentinel scope.

The master key is pointed at a tmp dir via ``DATA_DIR`` (mirroring the
fixture in ``test_credential_protection.py``) and the SQLite engine is
rebuilt against that dir so no real ``~/.serenade`` state is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from orchestrator import crypto, ledger
from orchestrator import db as _db
from orchestrator.db import create_all, session_scope
from orchestrator.models import AuditLogRow, EngagementRow
from orchestrator.settings import get_settings


async def _make_engagement() -> str:
    """Insert a minimal engagement row and return its id.

    The audit_log FK to engagements means per-engagement chains need a
    real parent row; the global (engagement_id=None) chain does not."""
    eid = str(uuid4())
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


@pytest.fixture
async def audit_db(monkeypatch, tmp_path):
    """Master key + SQLite DB rooted in a tmp dir, schema created."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    # Rebuild the engine against the tmp data_dir.
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


async def _rows_for(engagement_id):
    scope_is_none = not engagement_id
    async with session_scope() as s:
        from sqlalchemy import select

        stmt = select(AuditLogRow)
        if scope_is_none:
            stmt = stmt.where(AuditLogRow.engagement_id.is_(None))
        else:
            stmt = stmt.where(AuditLogRow.engagement_id == engagement_id)
        stmt = stmt.order_by(AuditLogRow.seq.asc())
        return list((await s.execute(stmt)).scalars().all())


@pytest.mark.asyncio
async def test_genesis_and_chaining(audit_db) -> None:
    eid = await _make_engagement()
    for i in range(3):
        await ledger.audit(
            engagement_id=eid, actor="system", action=f"act.{i}", payload={"i": i}
        )

    rows = await _rows_for(eid)
    assert [r.seq for r in rows] == [1, 2, 3]

    genesis = rows[0]
    assert genesis.seq == 1
    assert genesis.prev_hash is None
    assert genesis.row_hash  # non-empty HMAC

    # Each subsequent row links to its predecessor's row_hash.
    for prev, cur in zip(rows, rows[1:]):
        assert cur.prev_hash == prev.row_hash
        assert cur.row_hash != prev.row_hash


@pytest.mark.asyncio
async def test_verify_untampered_chain_ok(audit_db) -> None:
    eid = await _make_engagement()
    for i in range(5):
        await ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is True
    assert result.first_bad_seq is None
    assert result.reason == "ok"


@pytest.mark.asyncio
async def test_tampered_payload_flagged_as_hash_mismatch(audit_db) -> None:
    eid = await _make_engagement()
    for i in range(5):
        await ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})

    # Edit a middle row's payload directly in the DB (bypassing the ledger).
    async with session_scope() as s:
        await s.execute(
            text(
                "UPDATE audit_log SET payload = :p "
                "WHERE engagement_id = :e AND seq = :seq"
            ),
            {"p": '{"i": 999}', "e": eid, "seq": 3},
        )

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert result.reason == "hash-mismatch"


@pytest.mark.asyncio
async def test_deleted_middle_row_flagged_as_seq_gap(audit_db) -> None:
    eid = await _make_engagement()
    for i in range(5):
        await ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})

    async with session_scope() as s:
        await s.execute(
            text("DELETE FROM audit_log WHERE engagement_id = :e AND seq = :seq"),
            {"e": eid, "seq": 3},
        )

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert result.reason == "seq-gap"


@pytest.mark.asyncio
async def test_broken_link_flagged(audit_db) -> None:
    """Swapping a row's prev_hash to a bogus value (the shape of an
    inserted-or-reordered row) is caught as a broken link before the
    hash-mismatch check matters."""
    eid = await _make_engagement()
    for i in range(4):
        await ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})

    async with session_scope() as s:
        await s.execute(
            text(
                "UPDATE audit_log SET prev_hash = :p "
                "WHERE engagement_id = :e AND seq = :seq"
            ),
            {"p": "0" * 64, "e": eid, "seq": 3},
        )

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert result.reason == "broken-link"


@pytest.mark.asyncio
async def test_sequential_appends_contiguous(audit_db) -> None:
    """Sequential appends produce a contiguous 1..N chain that verifies."""
    eid = await _make_engagement()
    for i in range(8):
        await ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})

    rows = await _rows_for(eid)
    seqs = [r.seq for r in rows]
    assert seqs == list(range(1, 9))
    assert len(seqs) == len(set(seqs))  # no dup seq slots

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is True, result


@pytest.mark.asyncio
async def test_concurrent_appends_no_fork(audit_db) -> None:
    """Concurrent appends never fork the chain: whatever survives SQLite's
    single-writer lock contention is still a single contiguous,
    dup-free chain that verifies. (On Postgres ``with_for_update`` serialises
    the tail read; on SQLite the writer lock + the UNIQUE(engagement_id, seq)
    backstop give the same no-fork guarantee.)"""
    import asyncio

    eid = await _make_engagement()
    await asyncio.gather(
        *(
            ledger.audit(engagement_id=eid, action=f"act.{i}", payload={"i": i})
            for i in range(8)
        )
    )

    rows = await _rows_for(eid)
    seqs = [r.seq for r in rows]
    # No fork / no duplicate seq slot, and contiguous from 1 over whatever
    # landed (lock contention may drop a best-effort write, but never
    # corrupts the chain).
    assert seqs == list(range(1, len(seqs) + 1))
    assert len(seqs) == len(set(seqs))
    assert len(rows) >= 1

    result = await ledger.verify_audit_chain(eid)
    assert result.ok is True, result


@pytest.mark.asyncio
async def test_unique_seq_constraint_blocks_fork(audit_db) -> None:
    """The UNIQUE(chain_scope, seq) constraint is the race backstop: a
    hand-rolled second row claiming an existing seq slot is rejected.

    The constraint now lives on the non-null ``chain_scope`` column (db-02);
    the hand-rolled row must mirror the real row's ``chain_scope`` (the
    engagement id) for the collision to fire, exactly as ``append_audit_row``
    sets it."""
    eid = await _make_engagement()
    await ledger.audit(engagement_id=eid, action="act.0", payload={})

    with pytest.raises(Exception):
        async with session_scope() as s:
            s.add(
                AuditLogRow(
                    engagement_id=eid,
                    chain_scope=eid,
                    actor="system",
                    action="dup",
                    payload={},
                    seq=1,
                    prev_hash=None,
                    row_hash="x" * 64,
                )
            )


@pytest.mark.asyncio
async def test_seq_collision_retry_preserves_caller_writes(audit_db, monkeypatch) -> None:
    """A seq collision inside ``append_audit_row`` must roll back ONLY the
    failed INSERT (via SAVEPOINT), never the caller's other pending writes on
    a shared session. Regression for the shared-session ``s.rollback()`` bug.
    """
    from sqlalchemy import select

    eid = await _make_engagement()
    await ledger.audit(engagement_id=eid, action="a1", payload={})
    await ledger.audit(engagement_id=eid, action="a2", payload={})  # chain now at seq 2

    # Force the first append attempt to read a STALE tail (seq=1) so it
    # computes seq=2 and collides with the committed seq=2 row; the retry
    # reads the real tail (seq=2) and lands at seq=3.
    real_read_tail = ledger._read_chain_tail
    calls = {"n": 0}

    async def stale_first(s, engagement_id, *, for_update):
        calls["n"] += 1
        if calls["n"] == 1:
            stmt = select(AuditLogRow).where(
                AuditLogRow.engagement_id == str(engagement_id),
                AuditLogRow.seq == 1,
            )
            return (await s.execute(stmt)).scalars().first()
        return await real_read_tail(s, engagement_id, for_update=for_update)

    monkeypatch.setattr(ledger, "_read_chain_tail", stale_first)

    biz_eid = str(uuid4())
    now = datetime.now(UTC)
    async with session_scope() as s:
        # Caller's business write on the SHARED session, flushed before the
        # audit append (mirrors services/engagements._audit).
        s.add(
            EngagementRow(
                id=biz_eid,
                name="biz",
                time_window_start=now,
                time_window_end=now + timedelta(hours=1),
            )
        )
        await s.flush()
        row = await ledger.append_audit_row(
            s, engagement_id=eid, actor="system", action="a3", payload={}
        )
        assert row.seq == 3  # collided at 2, retried to 3

    assert calls["n"] >= 2, "expected a collision + retry"

    # The caller's business write must have survived the collision rollback.
    async with session_scope() as s:
        got = await s.get(EngagementRow, biz_eid)
    assert got is not None, "caller's pending write was discarded by the audit retry"

    assert (await ledger.verify_audit_chain(eid)).ok is True


@pytest.mark.asyncio
async def test_global_chain_engagement_none(audit_db) -> None:
    """Rows with engagement_id=None share one global chain and verify."""
    for i in range(4):
        await ledger.audit(engagement_id=None, action=f"global.{i}", payload={"i": i})

    rows = await _rows_for(None)
    assert [r.seq for r in rows] == [1, 2, 3, 4]
    assert rows[0].prev_hash is None
    assert all(r.engagement_id is None for r in rows)

    result = await ledger.verify_audit_chain(None)
    assert result.ok is True
    assert result.reason == "ok"

    # A real-engagement chain is independent of the global one (its own
    # seq counter starts at 1).
    eid = await _make_engagement()
    await ledger.audit(engagement_id=eid, action="eng.0", payload={})
    eng_rows = await _rows_for(eid)
    assert [r.seq for r in eng_rows] == [1]
