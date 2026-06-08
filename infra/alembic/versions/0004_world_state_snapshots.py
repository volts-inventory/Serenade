"""Durable world-state snapshot store.

Persist the per-engagement world-state graph across engagement pause/resume.
Adds the ``world_state_snapshots`` table: one row per engagement (PK =
``engagement_id``, FK to ``engagements.id``), holding the node-link
serialization (``graph_json``) produced by ``InMemoryGraphRepo.to_json()``.
Credential secret fields inside ``graph_json`` are Fernet-encrypted at rest
(the same crypto path used for the audit-log HMAC chain), so no plaintext
secret reaches disk.

The worker UPSERTs this row after every agent step that wrote nodes and
rehydrates the repo in place on resume. The snapshot write and the LangGraph
checkpoint are separate (non-transactional) writes — deterministic ``uuid5``
node ids make a slightly-stale snapshot + agent re-run re-converge.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "world_state_snapshots" in insp.get_table_names():
        return
    op.create_table(
        "world_state_snapshots",
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("graph_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("world_state_snapshots")
