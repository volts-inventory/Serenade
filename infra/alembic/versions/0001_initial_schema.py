"""Squashed initial schema — SQLite-compatible.

All 9 application tables created with SQLite-compatible types:
- UUID columns use String(36) (SQLite has no native UUID type)
- JSONB columns use JSON (SQLite stores as TEXT, SQLAlchemy handles ser/de)
- server_default uses string literals compatible with SQLite

This replaces the original Postgres migration chain (0001..0014).

Revision ID: 0001
Revises:
Create Date: 2026-05-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -- engagements --
    op.create_table(
        "engagements",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("scope", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("objectives", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("mode", sa.String(32), nullable=False, server_default="guided"),
        sa.Column("time_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("time_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("token_budget", sa.Integer, nullable=False, server_default="2000000"),
        sa.Column("wallclock_hours", sa.Integer, nullable=False, server_default="8"),
        sa.Column(
            "authorization_artifact_sha256", sa.String(64), nullable=False, server_default=""
        ),
        sa.Column("per_tool_token_budget", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="draft"),
        sa.Column("current_stage", sa.String(32), nullable=False, server_default="scope_intake"),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("llm_config_snapshot", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("abort_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("abort_reason", sa.Text(), nullable=True),
        sa.Column("aborted_by", sa.String(100), nullable=True),
        sa.Column(
            "regression_budget_max", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    # -- tool_runs --
    op.create_table(
        "tool_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool", sa.String(100), nullable=False),
        sa.Column("input_json", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("sandbox_id", sa.String(100), nullable=False, server_default="dev"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_code", sa.Integer, nullable=True),
        sa.Column("stdout_blob_id", sa.String(36), nullable=True),
        sa.Column("stderr_blob_id", sa.String(36), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )
    op.create_index("ix_tool_runs_engagement_id", "tool_runs", ["engagement_id"])

    # -- evidence --
    op.create_table(
        "evidence",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(100), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("blob_uri", sa.String(500), nullable=False),
        sa.Column(
            "content_type",
            sa.String(100),
            nullable=False,
            server_default="application/octet-stream",
        ),
        sa.Column("redacted", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_evidence_engagement_id", "evidence", ["engagement_id"])
    op.create_index("ix_evidence_sha256", "evidence", ["sha256"])

    # -- llm_calls --
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent", sa.String(100), nullable=False),
        sa.Column("site", sa.String(32), nullable=False),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cached_input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_llm_calls_engagement_id", "llm_calls", ["engagement_id"])

    # -- audit_log --
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("actor", sa.String(100), nullable=False, server_default="system"),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_audit_log_engagement_id", "audit_log", ["engagement_id"])

    # -- approvals --
    op.create_table(
        "approvals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("requested_payload", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("decision", sa.String(50), nullable=True),
        sa.Column("decided_by", sa.String(100), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("updated_by", sa.String(100), nullable=False, server_default="bootstrap"),
    )
    op.create_index("ix_approvals_engagement_id", "approvals", ["engagement_id"])

    # -- suppressed_tools --
    op.create_table(
        "suppressed_tools",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("reason", sa.String(200), nullable=False, server_default="unknown"),
        sa.Column("stage", sa.String(32), nullable=False, server_default="unknown"),
        sa.Column(
            "suppressed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_suppressed_tools_engagement_id", "suppressed_tools", ["engagement_id"]
    )
    op.create_index(
        "ix_suppressed_tools_tool_name", "suppressed_tools", ["tool_name"]
    )

    # -- regressions --
    op.create_table(
        "regressions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("target_stage", sa.String(32), nullable=False),
        sa.Column("trigger_kind", sa.String(64), nullable=False),
        sa.Column("delta_json", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("evidence_json", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("rationale", sa.Text(), nullable=False, server_default=""),
        sa.Column("delta_fingerprint", sa.String(64), nullable=True),
        sa.Column("council_decision", sa.String(32), nullable=True),
        sa.Column("council_rationale", sa.Text(), nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_regressions_engagement_id", "regressions", ["engagement_id"]
    )
    op.create_index(
        "ix_regressions_engagement_fingerprint",
        "regressions",
        ["engagement_id", "target_stage", "trigger_kind", "delta_fingerprint"],
    )

    # -- reflection_memory --
    op.create_table(
        "reflection_memory",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "engagement_id",
            sa.String(36),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(50), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        "ix_reflection_memory_engagement", "reflection_memory", ["engagement_id", "stage"]
    )


def downgrade() -> None:
    op.drop_table("reflection_memory")
    op.drop_table("regressions")
    op.drop_table("suppressed_tools")
    op.drop_table("approvals")
    op.drop_table("audit_log")
    op.drop_table("llm_calls")
    op.drop_table("evidence")
    op.drop_table("tool_runs")
    op.drop_table("engagements")
