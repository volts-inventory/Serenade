"""Async SQLAlchemy engine + session factory — SQLite backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from orchestrator.settings import Settings, get_settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
# The event loop the current ``_engine`` was created on. The async SQLite
# engine's connection pool binds its aiosqlite worker threads / asyncio
# primitives to whatever loop first touched it; reusing it from a DIFFERENT
# loop (e.g. a second ``asyncio.run(...)``, a test that builds a fresh loop,
# or the API loop vs. the worker loop) risks "bound to a different event loop"
# errors or lost wakeups (finding async-04). We track the owning loop and
# transparently rebuild the engine when the running loop changes.
_engine_loop: asyncio.AbstractEventLoop | None = None


def _sqlite_dsn(settings: Settings) -> str:
    """Build the ``sqlite+aiosqlite`` DSN from settings."""
    data_dir = Path(settings.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "serenade.db"
    return f"sqlite+aiosqlite:///{db_path}"


def _current_loop() -> asyncio.AbstractEventLoop | None:
    """The running loop, or ``None`` if called outside a loop (e.g. setup)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _dispose_stale_engine() -> None:
    """Dispose the previous engine before rebinding (asyncN-02).

    The loop-rebind path (``_rebind_if_loop_changed``) reassigns the ``_engine``
    global; without disposing the old engine first its aiosqlite background
    thread + connection pool linger until GC, leaking a worker thread per
    rebind. ``AsyncEngine.dispose()`` is a coroutine and the *old* engine's
    loop is typically no longer running here, so we dispose synchronously via
    the underlying sync engine, which closes the pooled DBAPI connections (and
    thus the aiosqlite threads) without needing the old event loop. Best-effort
    so a dispose hiccup never blocks the rebind."""
    global _engine
    old = _engine
    if old is None:
        return
    try:
        old.sync_engine.dispose()
    except Exception:
        pass


def init_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine, _sessionmaker, _engine_loop
    s = settings or get_settings()
    dsn = _sqlite_dsn(s)
    _dispose_stale_engine()
    _engine = create_async_engine(dsn, future=True)
    _engine_loop = _current_loop()

    # Enable WAL mode on every raw DBAPI connection so concurrent reads
    # (TUI polling) don't block writes.
    @event.listens_for(_engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        # SQLite serialises writers (one at a time). Wait up to 5s for the
        # write lock instead of failing immediately with "database is
        # locked" so concurrent audit-chain appends queue behind each other
        # rather than dropping their best-effort write.
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def _rebind_if_loop_changed() -> None:
    """Rebuild the engine/sessionmaker if the running loop differs from the
    one the current engine was bound to (async-04).

    No-op when there is no running loop (synchronous setup), when no engine
    exists yet, or when the engine was created outside a loop (``_engine_loop
    is None``) — in those cases the first real ``async with`` adopts the
    running loop. We only rebind when both loops are known AND differ, which
    is the genuine cross-loop hazard. ``init_engine`` disposes the stale
    engine (``_dispose_stale_engine``) before rebinding so its aiosqlite
    thread/pool is released immediately rather than on GC (asyncN-02)."""
    if _engine is None:
        return
    running = _current_loop()
    if running is None or _engine_loop is None:
        return
    if running is not _engine_loop:
        # Drop the stale engine and rebuild against the running loop.
        init_engine()


def get_engine() -> AsyncEngine:
    if _engine is None:
        init_engine()
    else:
        _rebind_if_loop_changed()
    assert _engine is not None
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        init_engine()
    else:
        _rebind_if_loop_changed()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    sm = get_sessionmaker()
    async with sm() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def create_all() -> None:
    """Create all tables in the SQLite database.

    Used for initial setup and the dev ``make seed`` path. For production
    schema management use ``alembic upgrade head``.
    """
    from orchestrator import models  # noqa: F401 — import to register tables

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
