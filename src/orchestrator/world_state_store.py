"""Durable world-state snapshot persistence helpers.

The world-state graph lives in memory on ``AgentBundle.repo`` during a run.
These helpers snapshot it to (and rehydrate it from) the
``world_state_snapshots`` table so discovered hosts / services / findings /
credentials survive an engagement pause/resume.

``save_world_state`` is best-effort (mirroring ``ledger.audit``): a
persistence failure logs a warning and returns so the agent loop is never
blocked. One row per engagement (UPSERT) — the latest snapshot wins.

Atomicity: the snapshot write and the LangGraph checkpoint are SEPARATE
writes, not one transaction. World-state node ids are deterministic
``uuid5``, so a slightly-stale snapshot plus an agent re-run re-converge onto
the same nodes — eventual consistency, not corruption. We deliberately do NOT
make the two writes one transaction.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from orchestrator import crypto
from orchestrator.db import session_scope
from orchestrator.models import WorldStateSnapshotRow, _now

log = structlog.get_logger(__name__)


async def save_world_state(engagement_id: UUID, graph_json: str) -> None:
    """UPSERT the world-state snapshot for an engagement.

    The ENTIRE serialized graph is Fernet-encrypted before it touches disk,
    not just credential fields -- secrets routinely land in non-Credential
    nodes (Finding evidence/description, Session fields, raw tool-output
    excerpts), so the whole blob is the encryption boundary. Best-effort: a
    persistence failure logs ``world_state.persist_failed`` and returns so
    the agent loop is never blocked, mirroring ``ledger.audit``.
    """
    try:
        blob = crypto.encrypt(graph_json)
        async with session_scope() as s:
            # Real dialect UPSERT (db-04). The previous read-then-insert had a
            # race: two concurrent saves for the same engagement could both
            # read ``None`` and both INSERT the same PK; the loser's commit
            # raised IntegrityError that the best-effort except swallowed,
            # silently DROPPING that snapshot. ON CONFLICT DO UPDATE makes the
            # write idempotent and race-free — the latest blob always wins.
            updated_at = _now()
            stmt = sqlite_insert(WorldStateSnapshotRow).values(
                engagement_id=str(engagement_id),
                graph_json=blob,
                updated_at=updated_at,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[WorldStateSnapshotRow.engagement_id],
                set_={"graph_json": blob, "updated_at": updated_at},
            )
            await s.execute(stmt)
    except Exception as e:
        log.warning(
            "world_state.persist_failed",
            engagement_id=str(engagement_id),
            err=str(e),
        )


async def load_world_state(engagement_id: UUID) -> str | None:
    """Return the decrypted ``graph_json`` for an engagement, or ``None``.

    Decrypts the whole-blob ciphertext written by :func:`save_world_state`.
    If the blob can't be decrypted (master key rotated/lost -- it still looks
    like a Fernet token after the decrypt attempt), this logs
    ``world_state.decrypt_failed`` and returns ``None`` so resume fails SAFE
    (starts from an empty graph and re-derives) instead of feeding ciphertext
    to tools. A legacy plaintext snapshot round-trips unchanged. Best-effort:
    a load failure also logs and returns ``None``.
    """
    try:
        async with session_scope() as s:
            stmt = select(WorldStateSnapshotRow.graph_json).where(
                WorldStateSnapshotRow.engagement_id == str(engagement_id)
            )
            blob = (await s.execute(stmt)).scalars().first()
        if blob is None:
            return None
        decrypted = crypto.decrypt(blob)
        if crypto.is_encrypted(decrypted):
            log.warning(
                "world_state.decrypt_failed",
                engagement_id=str(engagement_id),
                reason="snapshot still encrypted after decrypt (master key mismatch?)",
            )
            return None
        return decrypted
    except Exception as e:
        log.warning(
            "world_state.load_failed",
            engagement_id=str(engagement_id),
            err=str(e),
        )
        return None
