"""Audit-log at-rest tamper-evidence: per-engagement HMAC hash chain.

Tracking issue #112, P0 #2. Adds the three chain columns to ``audit_log``
and a ``UNIQUE(engagement_id, seq)`` constraint, then backfills existing
rows per chain (ordered by ``(ts, id)``) with ``seq`` 1..N, computing
``prev_hash`` / ``row_hash`` forward off the master key.

Chains are scoped by ``engagement_id``; rows with ``engagement_id IS NULL``
form one shared chain under the ``__global__`` sentinel scope. The
backfill is idempotent: a row that already carries a ``row_hash`` is left
untouched, so re-running ``alembic upgrade`` after a partial run does not
re-hash (and thus re-link) already-chained rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-31
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Mirror of orchestrator.models.GLOBAL_AUDIT_CHAIN_KEY — duplicated here so
# the migration has no runtime dependency on the application package layout.
_GLOBAL_AUDIT_CHAIN_KEY = "__global__"


def _audit_hmac(canonical: bytes) -> str:
    """HMAC-SHA256 over the master key, matching ``crypto.audit_hmac``."""
    import hmac
    from hashlib import sha256

    from orchestrator.crypto import load_or_create_key

    return hmac.new(load_or_create_key(), canonical, sha256).hexdigest()


def _canonical_bytes(
    *, seq: int, engagement_id, actor, action, payload, ts, prev_hash
) -> bytes:
    """Identical canonical form to ``ledger.audit_canonical_bytes`` so the
    backfilled hashes verify against the live verifier."""
    from datetime import datetime, timezone

    if isinstance(ts, datetime) and ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    ts_iso = ts.isoformat() if isinstance(ts, datetime) else str(ts)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            pass
    return json.dumps(
        {
            "seq": seq,
            "engagement_id": str(engagement_id) if engagement_id else None,
            "actor": actor,
            "action": action,
            "payload": payload,
            "ts": ts_iso,
            "prev_hash": prev_hash or "",
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_cols = {c["name"] for c in insp.get_columns("audit_log")}

    # 1. Add the three columns. ``seq``/``row_hash`` land nullable first so
    #    the backfill can populate them before we (logically) require them.
    with op.batch_alter_table("audit_log") as batch:
        if "seq" not in existing_cols:
            batch.add_column(sa.Column("seq", sa.Integer(), nullable=True))
        if "prev_hash" not in existing_cols:
            batch.add_column(sa.Column("prev_hash", sa.String(64), nullable=True))
        if "row_hash" not in existing_cols:
            batch.add_column(sa.Column("row_hash", sa.String(64), nullable=True))

    # 2. Backfill each chain forward. Group rows by chain scope, order by
    #    (ts, id), assign seq 1..N, link prev_hash/row_hash.
    audit = sa.table(
        "audit_log",
        sa.column("id", sa.String),
        sa.column("engagement_id", sa.String),
        sa.column("actor", sa.String),
        sa.column("action", sa.String),
        sa.column("payload", sa.JSON),
        sa.column("ts", sa.DateTime),
        sa.column("seq", sa.Integer),
        sa.column("prev_hash", sa.String),
        sa.column("row_hash", sa.String),
    )
    rows = list(
        bind.execute(
            sa.select(
                audit.c.id,
                audit.c.engagement_id,
                audit.c.actor,
                audit.c.action,
                audit.c.payload,
                audit.c.ts,
                audit.c.row_hash,
            ).order_by(audit.c.ts, audit.c.id)
        )
    )

    # Bucket by chain scope (None -> global sentinel).
    chains: dict[str, list] = {}
    for r in rows:
        scope = r.engagement_id if r.engagement_id else _GLOBAL_AUDIT_CHAIN_KEY
        chains.setdefault(scope, []).append(r)

    for _scope, chain_rows in chains.items():
        seq = 0
        prev_hash: str | None = None
        for r in chain_rows:
            seq += 1
            # Idempotent: an already-chained row keeps its hash and just
            # advances the running prev_hash pointer.
            if r.row_hash:
                prev_hash = r.row_hash
                continue
            canonical = _canonical_bytes(
                seq=seq,
                engagement_id=r.engagement_id,
                actor=r.actor,
                action=r.action,
                payload=r.payload,
                ts=r.ts,
                prev_hash=prev_hash,
            )
            row_hash = _audit_hmac(canonical)
            bind.execute(
                sa.update(audit)
                .where(audit.c.id == r.id)
                .values(seq=seq, prev_hash=prev_hash, row_hash=row_hash)
            )
            prev_hash = row_hash

    # 3. Now tighten the columns to NOT NULL and add the unique constraint.
    with op.batch_alter_table("audit_log") as batch:
        batch.alter_column("seq", existing_type=sa.Integer(), nullable=False)
        batch.alter_column("row_hash", existing_type=sa.String(64), nullable=False)
        batch.create_unique_constraint(
            "uq_audit_log_chain_seq", ["engagement_id", "seq"]
        )


def downgrade() -> None:
    with op.batch_alter_table("audit_log") as batch:
        batch.drop_constraint("uq_audit_log_chain_seq", type_="unique")
        batch.drop_column("row_hash")
        batch.drop_column("prev_hash")
        batch.drop_column("seq")
