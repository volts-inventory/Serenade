"""Audit-log global-chain race fix: non-null ``chain_scope`` column.

Finding db-02. The hash-chain race backstop was ``UNIQUE(engagement_id,
seq)``, but SQLite (and ANSI SQL) treat NULLs as DISTINCT in a unique index,
so two GLOBAL-chain rows (``engagement_id IS NULL``) could claim the same
``seq`` slot without colliding — silently forking the global chain so it never
verifies again.

This migration adds a NON-NULL ``chain_scope`` column that mirrors
``engagement_id`` for per-engagement chains and carries the ``__global__``
sentinel for global rows, backfills it, then moves the unique constraint from
``(engagement_id, seq)`` to ``(chain_scope, seq)`` so the backstop actually
covers the global chain. ``engagement_id`` stays nullable (the FK and the
``engagement_id IS NULL`` read semantics are preserved; the hashed canonical
bytes still fold in ``engagement_id`` = NULL for global rows, so existing rows
verify unchanged).

Idempotent: re-running after a partial upgrade is a no-op (the column /
constraint existence is checked first).

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Mirror of orchestrator.models.GLOBAL_AUDIT_CHAIN_KEY — duplicated so the
# migration has no runtime dependency on the application package layout.
_GLOBAL_AUDIT_CHAIN_KEY = "__global__"


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("audit_log")}

    # 1. Add ``chain_scope`` nullable so the backfill can populate it before
    #    we require it.
    if "chain_scope" not in cols:
        with op.batch_alter_table("audit_log") as batch:
            batch.add_column(sa.Column("chain_scope", sa.String(36), nullable=True))

    # 2. Backfill: per-engagement chains mirror ``engagement_id``; global rows
    #    (engagement_id IS NULL) get the sentinel.
    audit = sa.table(
        "audit_log",
        sa.column("engagement_id", sa.String),
        sa.column("chain_scope", sa.String),
    )
    bind.execute(
        sa.update(audit)
        .where(audit.c.engagement_id.isnot(None))
        .values(chain_scope=audit.c.engagement_id)
    )
    bind.execute(
        sa.update(audit)
        .where(audit.c.engagement_id.is_(None))
        .values(chain_scope=_GLOBAL_AUDIT_CHAIN_KEY)
    )

    # 3. Move the unique backstop from (engagement_id, seq) to
    #    (chain_scope, seq); add the lookup index; tighten to NOT NULL.
    existing_uqs = {uq["name"] for uq in insp.get_unique_constraints("audit_log")}
    existing_idx = {ix["name"] for ix in insp.get_indexes("audit_log")}
    with op.batch_alter_table("audit_log") as batch:
        batch.alter_column(
            "chain_scope",
            existing_type=sa.String(36),
            nullable=False,
            server_default=_GLOBAL_AUDIT_CHAIN_KEY,
        )
        if "uq_audit_log_chain_seq" in existing_uqs:
            batch.drop_constraint("uq_audit_log_chain_seq", type_="unique")
        batch.create_unique_constraint(
            "uq_audit_log_chain_seq", ["chain_scope", "seq"]
        )
        if "ix_audit_log_chain_scope" not in existing_idx:
            batch.create_index("ix_audit_log_chain_scope", ["chain_scope"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_uqs = {uq["name"] for uq in insp.get_unique_constraints("audit_log")}
    existing_idx = {ix["name"] for ix in insp.get_indexes("audit_log")}
    with op.batch_alter_table("audit_log") as batch:
        if "ix_audit_log_chain_scope" in existing_idx:
            batch.drop_index("ix_audit_log_chain_scope")
        if "uq_audit_log_chain_seq" in existing_uqs:
            batch.drop_constraint("uq_audit_log_chain_seq", type_="unique")
        batch.create_unique_constraint(
            "uq_audit_log_chain_seq", ["engagement_id", "seq"]
        )
        batch.drop_column("chain_scope")
