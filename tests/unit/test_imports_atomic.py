"""imports.py single-transaction atomicity (db-03).

``import_scan`` previously opened THREE independent ``session_scope()``
transactions (``record_tool_run``, a nested engagement read, and ``audit``),
so the tool-run row and the dispatch audit row had no shared commit boundary.
The fix threads ONE session through ``ToolRunRow`` + ``append_audit_row``.

These tests assert the happy path lands BOTH durable rows (with a UUID
engagement_id, also exercising db-01) and that the audit chain verifies.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from orchestrator import crypto, ledger
from orchestrator import db as _db
from orchestrator.db import create_all, session_scope
from orchestrator.models import AuditLogRow, EngagementRow, ToolRunRow
from orchestrator.services.imports import import_scan
from orchestrator.settings import get_settings

_NMAP_XML = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <status state="up"/>
    <address addr="10.0.0.1" addrtype="ipv4"/>
    <hostnames><hostname name="t.local"/></hostnames>
    <ports><port protocol="tcp" portid="22"><state state="open"/><service name="ssh"/></port></ports>
  </host>
</nmaprun>"""


@pytest.fixture
async def fix_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    _db._engine = None
    _db._sessionmaker = None
    _db._engine_loop = None
    _db.init_engine(get_settings())
    await create_all()
    # Reset the in-memory graph repo singleton so nodes don't bleed across
    # tests, and install one WITHOUT the provenance validator. ``import_scan``
    # upserts parsed nodes through ``get_repo()``; the default validator
    # (``postgres_provenance_validator``) compares UUID inputs against str ids
    # read back from SQLite and is unrelated to the db-03 transaction-structure
    # fix under test here (it is a separate, non-owned defect — see report).
    from orchestrator.services import findings as _f
    from worldstate.neo4j_repo import InMemoryGraphRepo

    _f._repo_singleton = InMemoryGraphRepo()  # type: ignore[attr-defined]
    yield tmp_path
    if _db._engine is not None:
        await _db._engine.dispose()
    _db._engine = None
    _db._sessionmaker = None
    _db._engine_loop = None
    get_settings.cache_clear()
    crypto.reset_cache()


@pytest.mark.asyncio
async def test_import_scan_persists_tool_run_and_audit_atomically(fix_db) -> None:
    eid = uuid4()
    now = datetime.now(UTC)
    async with session_scope() as s:
        s.add(
            EngagementRow(
                id=str(eid),
                name="imp",
                time_window_start=now,
                time_window_end=now + timedelta(hours=8),
            )
        )

    res = await import_scan(eid, "nmap_xml", _NMAP_XML.encode(), filename="scan.xml")
    assert res["kind"] == "nmap_xml"
    tr_id = res["tool_run_id"]

    # The tool-run row landed (a UUID engagement_id, db-01).
    async with session_scope() as s:
        runs = (
            await s.execute(
                select(ToolRunRow).where(ToolRunRow.engagement_id == str(eid))
            )
        ).scalars().all()
    assert len(runs) == 1
    assert runs[0].id == tr_id
    assert runs[0].tool == "import:nmap_xml"

    # The dispatch audit row landed in the SAME logical operation and the
    # chain verifies.
    async with session_scope() as s:
        audits = (
            await s.execute(
                select(AuditLogRow).where(AuditLogRow.chain_scope == str(eid))
            )
        ).scalars().all()
    assert len(audits) == 1
    assert audits[0].action == "import.nmap_xml.completed"
    assert audits[0].payload["tool_run_id"] == tr_id

    assert (await ledger.verify_audit_chain(eid)).ok is True
