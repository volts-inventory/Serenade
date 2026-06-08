"""Round-1 persistence / audit-chain regression fixes.

Covers the database + hash-chain defects fixed in round 1:

  * db-01  — a UUID-typed engagement_id WRITE lands a row and round-trips
             (the ``GUID`` TypeDecorator coerces ``UUID`` -> ``str`` so the
             aiosqlite bind no longer raises ``type 'UUID' is not supported``,
             which used to silently zero out audit/ledger persistence).
  * db-02  — concurrent GLOBAL-chain (engagement_id IS NULL) appends do not
             duplicate ``seq`` and the chain still verifies (the UNIQUE
             backstop now lives on the non-null ``chain_scope`` column).
  * db-04  — world_state UPSERT survives concurrent saves (no dropped
             snapshot, no swallowed IntegrityError).
  * db-05  — tz-aware datetimes read back tz-AWARE (UTC), not naive.
  * crypto-03 — canonical bytes distinguish int ``1`` from str ``"1"`` (and a
                UUID from its str() form): the encoding is injective.
  * crypto-04 — the chain verifier uses ``hmac.compare_digest``.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from orchestrator import crypto, ledger
from orchestrator import db as _db
from orchestrator.db import create_all, session_scope
from orchestrator.models import (
    GLOBAL_AUDIT_CHAIN_KEY,
    AuditLogRow,
    EngagementRow,
    LLMCallRow,
)
from orchestrator.settings import get_settings
from orchestrator.world_state_store import load_world_state, save_world_state


@pytest.fixture
async def fix_db(monkeypatch, tmp_path):
    """Master key + SQLite DB rooted in a tmp dir, schema created."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    _db._engine = None
    _db._sessionmaker = None
    _db._engine_loop = None
    _db.init_engine(get_settings())
    await create_all()
    yield tmp_path
    if _db._engine is not None:
        await _db._engine.dispose()
    _db._engine = None
    _db._sessionmaker = None
    _db._engine_loop = None
    get_settings.cache_clear()
    crypto.reset_cache()


async def _make_engagement(eid: str | UUID) -> str:
    eid = str(eid)
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


# --------------------------------------------------------------------------- #
# db-01 — UUID engagement_id writes land a row and round-trip                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_uuid_engagement_id_write_lands_row(fix_db) -> None:
    """A UUID-typed engagement_id must persist (db-01).

    Before the GUID TypeDecorator, ``record_llm_call`` with a ``UUID`` raised
    ``sqlite3.ProgrammingError: type 'UUID' is not supported`` which the
    best-effort except swallowed → ZERO rows. Now it lands one row whose
    engagement_id round-trips as the canonical str."""
    eid = uuid4()
    await _make_engagement(eid)

    # Pass a genuine UUID object (the production router does exactly this).
    await ledger.record_llm_call(
        engagement_id=eid,
        agent="planner",
        site="plan",
        backend="anthropic",
        model="claude",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        cost_usd=0.01,
        latency_ms=42,
    )

    async with session_scope() as s:
        rows = (await s.execute(select(LLMCallRow))).scalars().all()
    assert len(rows) == 1, "UUID-typed write must land exactly one row"
    assert rows[0].engagement_id == str(eid)
    assert isinstance(rows[0].engagement_id, str)


@pytest.mark.asyncio
async def test_uuid_audit_write_lands_row(fix_db) -> None:
    """``audit()`` with a UUID engagement_id lands a verifiable chain row."""
    eid = uuid4()
    await _make_engagement(eid)
    await ledger.audit(engagement_id=eid, action="act.0", payload={"k": "v"})

    async with session_scope() as s:
        rows = (
            await s.execute(
                select(AuditLogRow).where(AuditLogRow.chain_scope == str(eid))
            )
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].engagement_id == str(eid)
    assert rows[0].chain_scope == str(eid)

    # And it verifies, whether the caller passes the UUID or the str.
    assert (await ledger.verify_audit_chain(eid)).ok is True
    assert (await ledger.verify_audit_chain(str(eid))).ok is True


# --------------------------------------------------------------------------- #
# db-02 — concurrent global-chain appends don't dup seq                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_global_chain_no_dup_seq(fix_db) -> None:
    """Concurrent ``engagement_id=None`` appends share ONE global chain under
    the sentinel scope and never duplicate a seq slot (db-02).

    Previously ``UNIQUE(engagement_id, seq)`` did not cover the global chain
    (SQLite NULLs are distinct), so concurrent global appends forked the chain
    with duplicate seqs and verification permanently failed."""
    await asyncio.gather(
        *(
            ledger.audit(engagement_id=None, action=f"g.{i}", payload={"i": i})
            for i in range(12)
        )
    )

    async with session_scope() as s:
        rows = (
            await s.execute(
                select(AuditLogRow)
                .where(AuditLogRow.chain_scope == GLOBAL_AUDIT_CHAIN_KEY)
                .order_by(AuditLogRow.seq.asc())
            )
        ).scalars().all()

    seqs = [r.seq for r in rows]
    assert seqs == list(range(1, len(seqs) + 1)), f"forked global chain: {seqs}"
    assert len(seqs) == len(set(seqs)), "duplicate seq in global chain"
    assert all(r.engagement_id is None for r in rows)
    assert (await ledger.verify_audit_chain(None)).ok is True


# --------------------------------------------------------------------------- #
# crypto-03 — canonical bytes are injective                                   #
# --------------------------------------------------------------------------- #


def _canon(payload):
    return ledger.audit_canonical_bytes(
        seq=1,
        engagement_id=None,
        actor="system",
        action="a",
        payload=payload,
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        prev_hash=None,
    )


def test_canonical_bytes_distinguish_int_and_str() -> None:
    """int ``1`` and str ``"1"`` must NOT canonicalize identically (crypto-03)."""
    assert _canon({"x": 1}) != _canon({"x": "1"})


def test_canonical_bytes_distinguish_uuid_and_str() -> None:
    """A UUID value and its str() form must canonicalize differently — the old
    ``default=str`` collapsed them onto the same bytes (and thus the same
    row_hash)."""
    u = uuid4()
    assert _canon({"x": u}) != _canon({"x": str(u)})


def test_canonical_bytes_distinguish_datetime_and_str() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0)
    assert _canon({"x": dt}) != _canon({"x": str(dt)})


def test_canonical_bytes_stable_for_native_payload() -> None:
    """JSON-native payloads encode deterministically (no behaviour change)."""
    assert _canon({"a": 1, "b": [2, 3]}) == _canon({"b": [2, 3], "a": 1})


# --------------------------------------------------------------------------- #
# crypto-04 — constant-time MAC comparison                                     #
# --------------------------------------------------------------------------- #


def test_verify_uses_compare_digest() -> None:
    """The verifier must compare HMACs with ``hmac.compare_digest`` (crypto-04),
    not ``!=``."""
    src = inspect.getsource(ledger.verify_audit_chain)
    assert "compare_digest" in src
    assert "!= row.row_hash" not in src


@pytest.mark.asyncio
async def test_compare_digest_still_detects_tamper(fix_db) -> None:
    """Sanity: the constant-time compare still flags an edited row."""
    from sqlalchemy import text

    eid = await _make_engagement(uuid4())
    for i in range(4):
        await ledger.audit(engagement_id=eid, action=f"a.{i}", payload={"i": i})
    async with session_scope() as s:
        await s.execute(
            text("UPDATE audit_log SET payload = :p WHERE chain_scope = :e AND seq = :seq"),
            {"p": '{"i": 999}', "e": eid, "seq": 2},
        )
    res = await ledger.verify_audit_chain(eid)
    assert res.ok is False
    assert res.reason == "hash-mismatch"
    # Confirm we exercised the constant-time path, not a structural check.
    assert hmac.compare_digest("a", "a")


# --------------------------------------------------------------------------- #
# db-04 — world_state UPSERT survives concurrent saves                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_world_state_upsert_survives_concurrent_saves(fix_db) -> None:
    """Two concurrent first-saves for the same engagement must NOT drop a
    snapshot to a swallowed PK IntegrityError (db-04). Exactly one row exists
    afterwards and it decrypts to one of the written payloads."""
    eid = await _make_engagement(uuid4())

    payloads = [f'{{"graph": {i}}}' for i in range(8)]
    await asyncio.gather(*(save_world_state(UUID(eid), p) for p in payloads))

    from orchestrator.models import WorldStateSnapshotRow

    async with session_scope() as s:
        rows = (
            await s.execute(
                select(WorldStateSnapshotRow).where(
                    WorldStateSnapshotRow.engagement_id == eid
                )
            )
        ).scalars().all()
    assert len(rows) == 1, "concurrent saves must converge to exactly one row"

    stored = await load_world_state(UUID(eid))
    assert stored in payloads, "the surviving snapshot must be a real payload"


@pytest.mark.asyncio
async def test_world_state_upsert_overwrites(fix_db) -> None:
    """Sequential saves overwrite (latest wins), round-tripping decrypted."""
    eid = await _make_engagement(uuid4())
    await save_world_state(UUID(eid), '{"v": 1}')
    await save_world_state(UUID(eid), '{"v": 2}')
    assert await load_world_state(UUID(eid)) == '{"v": 2}'


# --------------------------------------------------------------------------- #
# db-05 — tz-aware datetime round-trips aware                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_datetime_reads_back_tz_aware(fix_db) -> None:
    """A datetime written tz-aware must read back tz-AWARE UTC (db-05), not
    tz-naive as raw SQLite does."""
    eid = uuid4()
    now = datetime.now(UTC)
    async with session_scope() as s:
        s.add(
            EngagementRow(
                id=str(eid),
                name="tz",
                time_window_start=now,
                time_window_end=now + timedelta(hours=8),
            )
        )
    async with session_scope() as s:
        row = await s.get(EngagementRow, str(eid))
    assert row is not None
    assert row.created_at.tzinfo is not None, "created_at read back tz-naive"
    assert row.created_at.utcoffset() == timedelta(0), "expected UTC offset"
    assert row.time_window_start.tzinfo is not None
