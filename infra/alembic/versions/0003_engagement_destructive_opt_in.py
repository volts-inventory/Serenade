"""Engagement destructive-PoC opt-in flag.

Tracking issue #112, P2 #3. Adds a fail-closed ``destructive_opt_in`` boolean
column to ``engagements`` (default False). Read by ``policy.engagement_dict.
build_engagement_context`` and consumed by ``agents.poc_runtime.PoCDispatcher.
can_auto_run`` to gate destructive curated PoCs (``destructive_without_optin``).

This is the curated-PoC path ONLY — the policy gate (``policy/scope.py``) DENYs
destructive *tools* unconditionally regardless of this flag.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_cols = {c["name"] for c in insp.get_columns("engagements")}
    if "destructive_opt_in" not in existing_cols:
        with op.batch_alter_table("engagements") as batch:
            batch.add_column(
                sa.Column(
                    "destructive_opt_in",
                    sa.Boolean(),
                    nullable=False,
                    server_default="0",
                )
            )


def downgrade() -> None:
    with op.batch_alter_table("engagements") as batch:
        batch.drop_column("destructive_opt_in")
