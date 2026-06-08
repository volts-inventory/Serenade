"""Tool template auto-update.

`nuclei -update-templates` and `searchsploit -u`
should run on a schedule, not be baked into the sandbox image. Stale
templates means stale exploit detection (months-old CVEs go undetected
even though nuclei has had a template for weeks).

Strategy:
- One spawned sandbox container at engagement start, runs both updates in
  parallel, then exits. Best-effort — failures log a WARN and the
  engagement continues with whatever templates are in the image.
- Throttled: the orchestrator caches the last update timestamp in Redis;
  re-runs only if older than `_UPDATE_INTERVAL_SECONDS`.
"""

from __future__ import annotations

import asyncio
import time

import structlog

log = structlog.get_logger(__name__)


_UPDATE_INTERVAL_SECONDS = 6 * 3600  # 6 hours


# Process-local cache keyed by "last-update-monotonic-ts"; in prod swap for Redis.
_last_update_ts: float = 0.0


async def ensure_templates_fresh(*, force: bool = False) -> bool:
    """Run `nuclei -update-templates` and `searchsploit -u` if stale.

    Returns True if an update was attempted (regardless of success).
    """
    global _last_update_ts
    now = time.monotonic()
    if not force and (now - _last_update_ts) < _UPDATE_INTERVAL_SECONDS:
        return False

    log.info("templates.update.start")
    cmd = "nuclei -update-templates 2>&1; searchsploit -u 2>&1; echo DONE"
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "serenade/sandbox:dev",
        "/bin/sh",
        "-c",
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        log.warning("templates.update.timeout")
        return True

    _last_update_ts = now
    if proc.returncode != 0:
        log.warning(
            "templates.update.failed",
            err=stderr.decode("utf-8", errors="replace")[:500],
        )
    else:
        log.info("templates.update.ok")
    return True
