"""Per-engagement sandbox network management.

Per-engagement Docker network with full UUID name (no 8-char collision risk)
and an asyncio lock around inspect+create so concurrent engagement starts
don't race.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import structlog

log = structlog.get_logger(__name__)


# Per-name asyncio lock so concurrent ensure_engagement_network() calls for
# the same engagement serialize correctly.
#
# async-03/10: this used to be a `defaultdict(asyncio.Lock)` that was NEVER
# evicted — one Lock leaked per distinct engagement network for the worker's
# lifetime (unbounded growth). It is now a plain dict whose entries are popped
# in `teardown_engagement_network` once the network is gone, and the lock is
# created lazily by `_lock_for` (so the Lock binds to the running loop at
# first use rather than to whatever loop happened to import this module).
_name_locks: dict[str, asyncio.Lock] = {}


def _lock_for(name: str) -> asyncio.Lock:
    """Loop-safe lazy lock creation. The Lock is created on first use so it
    binds to the currently-running event loop, not the import-time loop."""
    lock = _name_locks.get(name)
    if lock is None:
        lock = asyncio.Lock()
        _name_locks[name] = lock
    return lock


def engagement_network_name(engagement_id: UUID) -> str:
    """Full-UUID-based name. Docker network names support letters, digits,
    underscores, periods, and dashes; UUIDs match. ~36 chars total.
    """
    return f"serenade-sb-{engagement_id}"


async def ensure_engagement_network(engagement_id: UUID, *, internal: bool = True) -> str:
    """Idempotently create a per-engagement Docker network. Returns its name.

    `internal=True` (the new default) marks the network
    as Docker-internal — containers cannot reach any host outside the network
    by default. The proxy-logger (mitmproxy) is dual-homed onto both the
    engagement network and the orchestrator's default network, and becomes
    the ONLY egress path. Tools route via `HTTP_PROXY` env (set by
    `SandboxRunner`); raw-TCP tools (nmap, smbclient) reach only in-scope
    targets via the proxy or via additional egress containers.

    `internal=False` retains the legacy behavior (bridge with default
    Docker SNAT to the host) — kept as an escape hatch for dev
    environments where the proxy-routing wiring isn't set up.

    Uses an asyncio lock to make inspect+create atomic across concurrent
    callers for the same engagement. If `docker network create` reports
    "already exists" (TOCTOU between two processes), treat as success.
    """
    name = engagement_network_name(engagement_id)
    async with _lock_for(name):
        # Inspect first
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "network",
            "inspect",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await inspect.wait() == 0:
            return name
        # Create
        create_args = ["docker", "network", "create", "--driver", "bridge"]
        if internal:
            # L3 isolation: no default route to the host; only in-network
            # containers can be reached.
            create_args.append("--internal")
        create_args.append(name)
        create = await asyncio.create_subprocess_exec(
            *create_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await create.communicate()
        if create.returncode == 0:
            log.info("sandbox.network.created", name=name)
            return name
        err_text = stderr.decode("utf-8", errors="replace")
        # An "already exists" race is success.
        if "already exists" in err_text.lower():
            log.info("sandbox.network.exists_after_race", name=name)
            return name
        log.error("sandbox.network.create_failed", name=name, err=err_text)
        # Per-engagement isolation must be all-or-nothing — refuse to fall
        # back to a shared bridge silently.
        raise RuntimeError(f"could not create per-engagement network {name}: {err_text}")


async def teardown_engagement_network(engagement_id: UUID) -> None:
    """Best-effort: remove the per-engagement network when the engagement ends."""
    name = engagement_network_name(engagement_id)
    lock = _lock_for(name)
    async with lock:
        rm = await asyncio.create_subprocess_exec(
            "docker",
            "network",
            "rm",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await rm.wait()
        log.info("sandbox.network.removed", name=name)
    # async-03/10: evict the per-name lock now the network is gone so the dict
    # cannot grow without bound. Only drop the SAME lock object we just held —
    # if a concurrent ensure_engagement_network() raced and re-created the
    # entry, leave the newcomer in place. Pop happens AFTER releasing the lock
    # so no waiter is stranded on an orphaned object.
    if _name_locks.get(name) is lock and not lock.locked():
        _name_locks.pop(name, None)
