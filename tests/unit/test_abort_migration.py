"""Squashed-migration test: verify the consolidated 0001_initial_schema
migration creates all tables with the expected columns.

Previously this tested the incremental 0012 migration that added abort
columns. With the Postgres-to-SQLite migration, all individual migrations
were squashed into a single 0001_initial_schema.py. This test now verifies
the squashed migration creates the engagements table with the abort columns
included (abort_requested_at, abort_reason, aborted_by).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import create_engine, inspect

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "alembic"
    / "versions"
    / "0001_initial_schema.py"
)


def _load_migration_module():
    """Import the migration file as a module without going through the
    alembic CLI."""
    spec = importlib.util.spec_from_file_location(
        "_initial_schema_0001", _MIGRATION_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_file_exists() -> None:
    """The squashed migration file MUST be present."""
    assert _MIGRATION_PATH.exists(), (
        f"Missing migration file at {_MIGRATION_PATH}."
    )


def test_migration_revision_id() -> None:
    mod = _load_migration_module()
    assert mod.revision == "0001"
    assert mod.down_revision is None


def test_squashed_migration_applies_cleanly_against_sqlite() -> None:
    """End-to-end: run the squashed migration against an in-memory SQLite
    and verify all 9 tables are created with the expected columns."""
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    mod = _load_migration_module()
    engine = create_engine("sqlite:///:memory:", future=True)

    with engine.begin() as conn:
        mc = MigrationContext.configure(conn)
        ops = Operations(mc)
        import alembic.op as _global_op  # noqa: PLC0415
        _global_op._proxy = ops  # type: ignore[attr-defined]
        mod.upgrade()

    with engine.connect() as conn:
        insp = inspect(conn)
        tables = set(insp.get_table_names())

    expected_tables = {
        "engagements",
        "tool_runs",
        "evidence",
        "llm_calls",
        "audit_log",
        "approvals",
        "suppressed_tools",
        "regressions",
        "reflection_memory",
    }
    assert expected_tables <= tables, (
        f"Missing tables: {expected_tables - tables}"
    )

    # Verify the abort columns are present (the original test's focus).
    with engine.connect() as conn:
        insp = inspect(conn)
        cols = {c["name"]: c for c in insp.get_columns("engagements")}

    for name in ("abort_requested_at", "abort_reason", "aborted_by"):
        assert name in cols, f"Abort column {name!r} missing from engagements"
        assert cols[name]["nullable"] is True, (
            f"Column {name!r} should be nullable=True"
        )

    # Verify downgrade drops all tables.
    with engine.begin() as conn:
        mc = MigrationContext.configure(conn)
        ops = Operations(mc)
        import alembic.op as _global_op  # noqa: PLC0415
        _global_op._proxy = ops  # type: ignore[attr-defined]
        mod.downgrade()

    with engine.connect() as conn:
        insp = inspect(conn)
        remaining = set(insp.get_table_names())

    for t in expected_tables:
        assert t not in remaining, f"downgrade() left table {t!r} behind"

    engine.dispose()
