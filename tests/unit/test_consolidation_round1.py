"""Round-1 consolidation fixes: cross-file items the per-domain fixers
flagged but could not own.

1. provenance validator UUID<->str comparison (exposed once the db-01 GUID
   fix made tool_runs actually persist).
2. ws-05 producer wiring: distinct searchsploit queries that share an EDB-ID
   must yield distinct ExploitCandidate ids.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

import orchestrator.crypto as crypto
import orchestrator.db as _db
from orchestrator.db import create_all, session_scope
from datetime import UTC, datetime, timedelta

from orchestrator.models import EngagementRow, ToolRunRow
from orchestrator.provenance import postgres_provenance_validator
from orchestrator.settings import get_settings
from worldstate.schema import (
    deterministic_node_id,
    natural_key_for_exploit_candidate,
)


@pytest.fixture
async def fix_db(monkeypatch, tmp_path):
    """SQLite DB rooted in a tmp dir, schema created (mirrors persistence tests)."""
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


def test_exploit_candidate_distinct_per_query_same_edb_id():
    """Two distinct service queries sharing one EDB-ID must not collide."""
    eng = uuid4()
    edb_id = "50064"
    t1 = deterministic_node_id("ServiceQuery", eng, "vsftpd 2.3.4")
    t2 = deterministic_node_id("ServiceQuery", eng, "proftpd 1.3.5")

    id1 = deterministic_node_id(
        "ExploitCandidate",
        eng,
        natural_key_for_exploit_candidate(
            finding_id=None, source="exploitdb", source_ref=edb_id, target_node_id=t1
        ),
    )
    id2 = deterministic_node_id(
        "ExploitCandidate",
        eng,
        natural_key_for_exploit_candidate(
            finding_id=None, source="exploitdb", source_ref=edb_id, target_node_id=t2
        ),
    )
    assert id1 != id2  # distinct services stay distinct


def test_exploit_candidate_same_query_dedups():
    """Re-running the same query must produce the same id (idempotent)."""
    eng = uuid4()
    t = deterministic_node_id("ServiceQuery", eng, "vsftpd 2.3.4")
    mk = lambda: deterministic_node_id(  # noqa: E731
        "ExploitCandidate",
        eng,
        natural_key_for_exploit_candidate(
            finding_id=None, source="exploitdb", source_ref="50064", target_node_id=t
        ),
    )
    assert mk() == mk()


@pytest.mark.asyncio
async def test_provenance_validator_accepts_real_uuid_toolrun(fix_db):
    """A UUID-typed tool_run that genuinely exists must validate True.

    Regression for the UUID(caller) vs str(GUID read-back) set-difference bug:
    before the fix the validator rejected every node because
    set(UUIDs) - set(strs) is never empty.
    """
    eng = uuid4()
    run_id = uuid4()
    now = datetime.now(UTC)
    async with session_scope() as s:
        s.add(
            EngagementRow(
                id=str(eng),
                name="t",
                time_window_start=now,
                time_window_end=now + timedelta(hours=8),
            )
        )
    async with session_scope() as s:
        s.add(ToolRunRow(id=run_id, engagement_id=eng, tool="nmap", exit_code=0))

    # Pass UUID objects (as real callers do); GUID reads back as str internally.
    assert await postgres_provenance_validator(eng, [run_id]) is True
    # A fabricated id must still be rejected (fail-closed).
    assert await postgres_provenance_validator(eng, [run_id, uuid4()]) is False
