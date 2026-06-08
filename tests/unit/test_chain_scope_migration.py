"""Migration 0005 — audit_log ``chain_scope`` column (db-02).

Verifies the new revision applies on top of the full chain (0001..0005)
against a fresh SQLite DB, that ``alembic head`` is linear at 0005, and that
the resulting ``audit_log`` schema matches the ORM: a NON-NULL ``chain_scope``
column with the unique backstop on ``(chain_scope, seq)`` (not the old
``(engagement_id, seq)``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect

_VERSIONS = Path(__file__).resolve().parents[2] / "infra" / "alembic" / "versions"
_ORDER = [
    "0001_initial_schema",
    "0002_audit_log_hash_chain",
    "0003_engagement_destructive_opt_in",
    "0004_world_state_snapshots",
    "0005_audit_log_chain_scope",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _VERSIONS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_revision_linkage_linear() -> None:
    """0005 chains off 0004 — the head stays linear."""
    m4 = _load("0004_world_state_snapshots")
    m5 = _load("0005_audit_log_chain_scope")
    assert m5.revision == "0005"
    assert m5.down_revision == "0004"
    assert m4.revision == "0004"


def _apply_all(conn) -> None:
    mc = MigrationContext.configure(conn)
    ops = Operations(mc)
    import alembic.op as _global_op  # noqa: PLC0415

    _global_op._proxy = ops  # type: ignore[attr-defined]
    for name in _ORDER:
        _load(name).upgrade()


def test_0005_applies_and_matches_orm(monkeypatch, tmp_path) -> None:
    """Run 0001..0005 and assert the resulting audit_log schema."""
    # The 0002 backfill touches the master key only when rows exist; a fresh
    # DB has none, but point DATA_DIR at a tmp dir to be safe.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    engine = create_engine(f"sqlite:///{tmp_path}/m.db", future=True)
    with engine.begin() as conn:
        _apply_all(conn)

    with engine.connect() as conn:
        insp = inspect(conn)
        cols = {c["name"]: c for c in insp.get_columns("audit_log")}
        uqs = {
            uq["name"]: tuple(uq["column_names"])
            for uq in insp.get_unique_constraints("audit_log")
        }

    assert "chain_scope" in cols, "chain_scope column missing after 0005"
    assert cols["chain_scope"]["nullable"] is False, "chain_scope must be NOT NULL"

    assert ("chain_scope", "seq") in uqs.values(), (
        f"UNIQUE(chain_scope, seq) missing; have {uqs}"
    )
    assert ("engagement_id", "seq") not in uqs.values(), (
        "old UNIQUE(engagement_id, seq) should have been dropped"
    )
    engine.dispose()
