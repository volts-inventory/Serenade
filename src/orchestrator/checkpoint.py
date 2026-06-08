"""LangGraph checkpointer — SQLite-backed durable state.

Replaces the Postgres-backed ``AsyncPostgresSaver`` with a lightweight
SQLite implementation so engagements survive process restarts and
``interrupt()`` can durably pause for operator approval.

Thread-id convention: ``engagement:{engagement_id}``. This makes resume
trivial — operator approves an approval, worker re-invokes the graph
for the same thread_id, and LangGraph rehydrates from the checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite
import structlog
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from orchestrator.settings import Settings, get_settings

log = structlog.get_logger(__name__)


def _json_serial(obj: Any) -> Any:
    """Custom JSON serializer for types not natively supported."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SQLiteCheckpointer(BaseCheckpointSaver):
    """Async SQLite-backed checkpoint saver for LangGraph.

    Stores checkpoints as JSON in a ``checkpoints`` table within the same
    SQLite database used by the rest of the application. Implements the
    async context manager protocol expected by LangGraph.
    """

    def __init__(self, db_path: str | Path) -> None:
        super().__init__()
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> SQLiteCheckpointer:
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                state_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
            """
        )
        await self._conn.commit()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "SQLiteCheckpointer not initialized — use as async context manager"
        return self._conn

    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Load the latest checkpoint for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            cursor = await self.conn.execute(
                """
                SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                       state_json, metadata_json
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_id),
            )
        else:
            cursor = await self.conn.execute(
                """
                SELECT thread_id, checkpoint_id, parent_checkpoint_id,
                       state_json, metadata_json
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (thread_id,),
            )

        row = await cursor.fetchone()
        if row is None:
            return None

        _, ckpt_id, parent_id, state_json, metadata_json = row
        checkpoint = json.loads(state_json)
        metadata = json.loads(metadata_json)

        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": ckpt_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    async def alist(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints, most recent first."""
        query = "SELECT thread_id, checkpoint_id, parent_checkpoint_id, state_json, metadata_json FROM checkpoints"
        params: list[Any] = []
        conditions: list[str] = []

        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                conditions.append("thread_id = ?")
                params.append(thread_id)

        if before and "configurable" in before:
            before_id = before["configurable"].get("checkpoint_id")
            if before_id:
                conditions.append("checkpoint_id < ?")
                params.append(before_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()

        for row in rows:
            tid, ckpt_id, parent_id, state_json, metadata_json = row
            checkpoint = json.loads(state_json)
            metadata = json.loads(metadata_json)

            parent_config = None
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_id": parent_id,
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_id": ckpt_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        state_json = json.dumps(checkpoint, default=_json_serial)
        metadata_json = json.dumps(
            metadata if isinstance(metadata, dict) else {},
            default=_json_serial,
        )
        now = datetime.now(UTC).isoformat()

        await self.conn.execute(
            """
            INSERT INTO checkpoints
                (thread_id, checkpoint_id, parent_checkpoint_id, state_json, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (thread_id, checkpoint_id)
            DO UPDATE SET
                state_json = excluded.state_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (thread_id, checkpoint_id, parent_id, state_json, metadata_json, now),
        )
        await self.conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes (used by LangGraph for interrupt support).

        For simplicity we store writes as a JSON blob keyed by
        (thread_id, checkpoint_id, task_id).
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        # Ensure the pending_writes table exists
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                writes_json TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_id, task_id)
            )
            """
        )

        writes_json = json.dumps(writes, default=_json_serial)
        await self.conn.execute(
            """
            INSERT INTO checkpoint_writes
                (thread_id, checkpoint_id, task_id, writes_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (thread_id, checkpoint_id, task_id)
            DO UPDATE SET writes_json = excluded.writes_json
            """,
            (thread_id, checkpoint_id, task_id, writes_json),
        )
        await self.conn.commit()

    # Synchronous methods — LangGraph may call these in some code paths.
    # Raise NotImplementedError to force async usage.
    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        raise NotImplementedError("Use aget_tuple() — SQLiteCheckpointer is async-only")

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Any:
        raise NotImplementedError("Use alist() — SQLiteCheckpointer is async-only")

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Use aput() — SQLiteCheckpointer is async-only")

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        raise NotImplementedError("Use aput_writes() — SQLiteCheckpointer is async-only")


# ---------------------------------------------------------------------------
# Singleton factory + thread-id helper
# ---------------------------------------------------------------------------

_saver: SQLiteCheckpointer | None = None


async def get_checkpointer(settings: Settings | None = None) -> SQLiteCheckpointer:
    """Lazy singleton. First call creates the checkpoints table."""
    global _saver
    if _saver is not None:
        return _saver

    s = settings or get_settings()
    data_dir = Path(s.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "serenade.db"

    saver = SQLiteCheckpointer(db_path)
    await saver.__aenter__()
    _saver = saver
    return saver


def thread_id_for(engagement_id: UUID) -> str:
    return f"engagement:{engagement_id}"
