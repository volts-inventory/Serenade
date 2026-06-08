"""Per-engagement mitmproxy lifecycle.

This module spawns a dedicated mitmproxy container per engagement on
the engagement's Docker network, with a
per-engagement logs volume.

Lifecycle:
- `ensure_engagement_proxy(engagement_id)` — idempotent. Spawns if missing.
- `teardown_engagement_proxy(engagement_id)` — best-effort; called when the
  engagement completes or is aborted.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import structlog

from sandbox.network import engagement_network_name

log = structlog.get_logger(__name__)


_PROXY_IMAGE = "mitmproxy/mitmproxy:11.0.0"
_PROXY_PORT = 8080

# async-03/10: previously a `defaultdict(asyncio.Lock)` that was never evicted
# (one Lock leaked per distinct proxy name for the worker's lifetime). Now a
# plain dict with lazy, loop-safe creation (`_lock_for`) and eviction in
# `teardown_engagement_proxy`.
_locks: dict[str, asyncio.Lock] = {}


def _lock_for(name: str) -> asyncio.Lock:
    """Loop-safe lazy lock creation: binds the Lock to the running loop at
    first use rather than the import-time loop."""
    lock = _locks.get(name)
    if lock is None:
        lock = asyncio.Lock()
        _locks[name] = lock
    return lock


def engagement_proxy_name(engagement_id: UUID) -> str:
    return f"serenade-proxy-{engagement_id}"


def engagement_proxy_logs_volume(engagement_id: UUID) -> str:
    return f"serenade-proxy-logs-{engagement_id}"


def _build_proxy_run_argv(
    *,
    name: str,
    network: str,
    volume: str,
    engagement_id: UUID,
    scope_hosts: list[str] | None,
) -> list[str]:
    """Construct the ``docker run`` argv for the per-engagement proxy.

    Factored out so the constructed argv (hardening, addons, scope env) is
    unit-testable without Docker. When ``scope_hosts`` is provided, the proxy
    ALSO loads the scope-filter addon (sandbox-12b defence-in-depth): the
    trusted, server-side allow-list is injected via ``SERENADE_SCOPE_HOSTS``
    and out-of-scope HTTP egress is dropped at the proxy. When it is absent the
    proxy keeps its original single-addon behaviour (interface preserved).
    """
    argv = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--network",
        network,
        # The proxy ALSO needs to reach the orchestrator's default
        # network for outbound (real targets). Connect to the second
        # network in a follow-up step.
        "-v",
        f"{volume}:/logs",
        "-e",
        f"ENGAGEMENT_ID={engagement_id}",
    ]
    addons = ["-s", "/scripts/log_evidence.py"]
    if scope_hosts:
        # Trusted (server-side) scope allow-list — newline-separated so the
        # addon parses bare hostnames and CIDRs. NOT client-controlled.
        argv += ["-e", "SERENADE_SCOPE_HOSTS=" + "\n".join(scope_hosts)]
        addons += ["-s", "/scripts/scope_filter.py"]
    argv += [
        _PROXY_IMAGE,
        "mitmdump",
        "--mode",
        "regular@8080",
        *addons,
    ]
    return argv


async def ensure_engagement_proxy(
    engagement_id: UUID, *, scope_hosts: list[str] | None = None
) -> str:
    """Idempotently spawn a per-engagement mitmproxy container.

    Returns the container name. The proxy is on the engagement's
    network at port 8080; tools use `HTTP_PROXY=http://{name}:8080`.

    ``scope_hosts`` (optional): an allow-list of bare hostnames / CIDRs for the
    engagement. When supplied, the proxy enforces scope a SECOND time at the
    HTTP layer (sandbox-12b) by loading ``scope_filter.py`` and receiving the
    list via the trusted ``SERENADE_SCOPE_HOSTS`` env var. When omitted, the
    proxy behaves exactly as before (logging addon only); scope still rests on
    the extract_targets/policy gate. NOTE (defence-in-depth follow-up): the
    caller — the worker engagement bootstrap — must read the engagement's scope
    from the worldstate/engagement row and pass it here to activate the second
    layer; wiring that read lives in a non-owned file (see report).
    """
    name = engagement_proxy_name(engagement_id)
    network = engagement_network_name(engagement_id)
    volume = engagement_proxy_logs_volume(engagement_id)

    async with _lock_for(name):
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "container",
            "inspect",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await inspect.wait() == 0:
            return name

        if not scope_hosts:
            log.warning(
                "proxy.scope_filter_disabled",
                container=name,
                detail=(
                    "no scope_hosts passed; proxy runs logging addon only. "
                    "HTTP egress scope rests on the extract_targets gate "
                    "(sandbox-12b defence-in-depth follow-up)."
                ),
            )

        create = await asyncio.create_subprocess_exec(
            *_build_proxy_run_argv(
                name=name,
                network=network,
                volume=volume,
                engagement_id=engagement_id,
                scope_hosts=scope_hosts,
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _out, err = await create.communicate()
        if create.returncode != 0:
            err_text = err.decode("utf-8", errors="replace")
            log.error(
                "proxy.spawn_failed",
                engagement_id=str(engagement_id),
                err=err_text,
            )
            raise RuntimeError(f"failed to spawn engagement proxy: {err_text}")

        # Dual-home: attach to the orchestrator's default network so the
        # proxy can reach real targets outside the --internal engagement net.
        connect = await asyncio.create_subprocess_exec(
            "docker",
            "network",
            "connect",
            "serenade_default",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err2 = await connect.communicate()
        if connect.returncode != 0:
            log.warning(
                "proxy.dual_home_failed",
                container=name,
                err=err2.decode("utf-8", errors="replace"),
            )
        log.info("proxy.spawned", container=name, engagement_id=str(engagement_id))
        return name


async def teardown_engagement_proxy(engagement_id: UUID) -> None:
    name = engagement_proxy_name(engagement_id)
    lock = _lock_for(name)
    async with lock:
        rm = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await rm.wait()
        log.info("proxy.removed", container=name)
    # async-03/10: evict the per-name lock now the proxy is gone so the dict is
    # bounded. Only drop the same object we held and only if no waiter raced in.
    if _locks.get(name) is lock and not lock.locked():
        _locks.pop(name, None)
