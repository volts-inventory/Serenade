"""Windows worker dispatcher (orchestrator side).

Pushes a job onto Redis list `serenade:windows_worker:jobs` and BRPOPs
the per-job result key. The Windows worker code (PowerShell agent) lives
under `windows_worker/`; this module is the Python half.

Disabled by default; flip `SERENADE_WINDOWS_WORKER_ENABLED=1` and set
`SERENADE_WINDOWS_WORKER_AUTH_TOKEN` to a long random value to turn on.

Wire format: see `windows_worker/serenade_windows_job.schema.json`.

Hardening:
- `assembly_b64` jobs carry an HMAC-SHA256 signature over the bytes
  (`assembly_sig`), keyed on `SERENADE_WINDOWS_WORKER_SIGNING_KEY`. The
  worker rejects assemblies whose signature doesn't verify. Without the
  signing key, assembly dispatch is refused (not a silent passthrough).
- `heartbeat_age_seconds()` is checked pre-dispatch via
  `assert_worker_ready()`; a stale worker returns a fast error instead of
  burning the engagement's wallclock on BRPOP timeouts.
- Heartbeat payload includes `isolation` (one of `hyperv-snapshot`,
  `firecracker`, `bare-metal`, `none`); the orchestrator refuses
  high-risk jobs (mimikatz, assembly) to workers self-reporting `none`
  unless `SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION=1`.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import structlog

log = structlog.get_logger(__name__)

JOB_QUEUE_DEFAULT = "serenade:windows_worker:jobs"
RESULT_QUEUE_DEFAULT = "serenade:windows_worker:results"
HEARTBEAT_KEY_DEFAULT = "serenade:windows_worker:heartbeat"
HEARTBEAT_MAX_AGE_SECONDS = 60  # matches worker's TTL on the heartbeat key

_HIGH_RISK_KINDS = {"mimikatz", "assembly"}
_VALID_ISOLATION_MODES = {"hyperv-snapshot", "firecracker", "bare-metal", "none"}


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val else default


def is_enabled() -> bool:
    return _env("SERENADE_WINDOWS_WORKER_ENABLED", "0") == "1"


@dataclass
class WindowsJobResult:
    job_id: str
    exit_code: int
    stdout: str
    stderr: str
    started_at: str
    ended_at: str
    timed_out: bool = False


class WindowsJobQueueDisabled(RuntimeError):
    """Raised when a caller tries to submit a job without the feature on."""


class WindowsJobAuthMissing(RuntimeError):
    """Raised when SERENADE_WINDOWS_WORKER_AUTH_TOKEN isn't configured."""


class WindowsWorkerNotReady(RuntimeError):
    """Heartbeat is stale or worker hasn't checked in yet; dispatch refused."""


class WindowsWorkerSigningKeyMissing(RuntimeError):
    """SERENADE_WINDOWS_WORKER_SIGNING_KEY not set; assembly dispatch refused."""


class WindowsWorkerUnsafeIsolation(RuntimeError):
    """Worker self-reports isolation=none for a high-risk job and the unsafe
    opt-in is not set."""


@dataclass
class WindowsWorkerHealth:
    enabled: bool
    auth_token_configured: bool
    signing_key_configured: bool
    heartbeat_age_seconds: float | None
    isolation_mode: str | None
    healthy: bool
    note: str


def _signing_key() -> bytes | None:
    val = _env("SERENADE_WINDOWS_WORKER_SIGNING_KEY")
    return val.encode("utf-8") if val and len(val) >= 16 else None


def sign_assembly(assembly_bytes: bytes) -> str:
    """Return hex HMAC-SHA256 over the assembly bytes.

    Raises WindowsWorkerSigningKeyMissing if the key isn't configured.
    Worker MUST refuse assembly jobs whose `assembly_sig` doesn't
    verify; that check lives in
    `windows_worker/agent/Run-Assembly.ps1`.
    """
    key = _signing_key()
    if key is None:
        raise WindowsWorkerSigningKeyMissing(
            "SERENADE_WINDOWS_WORKER_SIGNING_KEY must be set (>=16 chars) "
            "to dispatch assembly jobs."
        )
    return hmac.new(key, assembly_bytes, sha256).hexdigest()


class WindowsJobQueue:
    """Submit Windows-side jobs and await results via Redis."""

    def __init__(self, redis_url: str | None = None) -> None:
        if not is_enabled():
            raise WindowsJobQueueDisabled(
                "SERENADE_WINDOWS_WORKER_ENABLED is not '1'; refusing to dispatch."
            )
        self.auth_token = _env("SERENADE_WINDOWS_WORKER_AUTH_TOKEN")
        if not self.auth_token or len(self.auth_token) < 16:
            raise WindowsJobAuthMissing(
                "SERENADE_WINDOWS_WORKER_AUTH_TOKEN must be set to a string >=16 chars."
            )
        self.redis_url = redis_url or _env("REDIS_URL", "redis://redis:6379/0")
        self.job_queue = _env("SERENADE_WINDOWS_WORKER_QUEUE", JOB_QUEUE_DEFAULT)
        self.result_queue = _env("SERENADE_WINDOWS_WORKER_RESULT_QUEUE", RESULT_QUEUE_DEFAULT)
        self.heartbeat_key = _env("SERENADE_WINDOWS_WORKER_HEARTBEAT_KEY", HEARTBEAT_KEY_DEFAULT)
        self.default_timeout = int(_env("SERENADE_WINDOWS_WORKER_JOB_TIMEOUT_SECONDS", "300"))

    async def _redis(self):
        try:
            from redis.asyncio import Redis  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'redis' package is required for Windows worker support. "
                "Install it with: pip install serenade[windows]"
            ) from exc

        return Redis.from_url(self.redis_url, decode_responses=True)

    async def heartbeat_payload(self) -> dict | None:
        """Return the parsed heartbeat payload, or None if no heartbeat.

        Worker writes JSON like `{"ts": "...", "isolation": "hyperv-snapshot"}`
        to the heartbeat key. For backwards compatibility we accept a bare ISO
        timestamp string (legacy format) and treat isolation as "unknown".
        """
        r = await self._redis()
        try:
            raw = await r.get(self.heartbeat_key)
            if not raw:
                return None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "ts" in parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"ts": raw, "isolation": "unknown"}
        finally:
            await r.aclose()

    async def heartbeat_age_seconds(self) -> float | None:
        """Return seconds since the worker last refreshed its heartbeat key,
        or None if the worker has never written one (or it expired)."""
        payload = await self.heartbeat_payload()
        if payload is None:
            return None
        iso = payload.get("ts")
        if not iso:
            return None
        try:
            ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (datetime.now(UTC) - ts).total_seconds()

    async def isolation_mode(self) -> str | None:
        payload = await self.heartbeat_payload()
        return payload.get("isolation") if payload else None

    async def assert_worker_ready(self, kind: str) -> None:
        """Pre-flight: refuse dispatch if the worker is stale or unsafe.

        Raises WindowsWorkerNotReady if heartbeat is missing or too old.
        Raises WindowsWorkerUnsafeIsolation if `kind` is high-risk and the
        worker reports `isolation: none` without the explicit unsafe opt-in.
        """
        age = await self.heartbeat_age_seconds()
        if age is None or age > HEARTBEAT_MAX_AGE_SECONDS:
            raise WindowsWorkerNotReady(
                f"Windows worker heartbeat is stale "
                f"({'never' if age is None else f'{age:.1f}s'}); "
                f"refusing to dispatch {kind} job."
            )
        if kind in _HIGH_RISK_KINDS:
            iso = await self.isolation_mode()
            if iso == "none" and _env("SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION") != "1":
                raise WindowsWorkerUnsafeIsolation(
                    f"Worker reports isolation=none; refusing high-risk job "
                    f"(kind={kind}). Set SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION=1 "
                    "to override (solo-pentest / lab only)."
                )

    async def submit_mimikatz(
        self,
        engagement_id: UUID,
        args: list[str],
        timeout_seconds: int | None = None,
    ) -> WindowsJobResult:
        return await self._submit(
            engagement_id=engagement_id,
            kind="mimikatz",
            payload={"args": args},
            timeout_seconds=timeout_seconds,
        )

    async def submit_assembly(
        self,
        engagement_id: UUID,
        assembly_bytes: bytes,
        args: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> WindowsJobResult:
        # HMAC over the assembly bytes. The worker MUST verify this before
        # invoking the entry point — otherwise anyone with Redis access (or the
        # auth_token via the job payload) could push arbitrary .NET code.
        sig = sign_assembly(assembly_bytes)
        return await self._submit(
            engagement_id=engagement_id,
            kind="assembly",
            payload={
                "assembly_b64": base64.b64encode(assembly_bytes).decode("ascii"),
                "assembly_sig": sig,
                "args": args or [],
            },
            timeout_seconds=timeout_seconds,
        )

    async def submit_shell(
        self,
        engagement_id: UUID,
        exe: str,
        args: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> WindowsJobResult:
        return await self._submit(
            engagement_id=engagement_id,
            kind="shell",
            payload={"exe": exe, "args": args or []},
            timeout_seconds=timeout_seconds,
        )

    async def _submit(
        self,
        engagement_id: UUID,
        kind: str,
        payload: dict,
        timeout_seconds: int | None,
    ) -> WindowsJobResult:
        # Pre-flight check. Refuse to BRPOP for 300s on a dead worker;
        # refuse high-risk dispatch to a worker without isolation.
        await self.assert_worker_ready(kind)
        job_id = str(uuid4())
        timeout = timeout_seconds or self.default_timeout
        job = {
            "job_id": job_id,
            "engagement_id": str(engagement_id),
            "kind": kind,
            "submitted_at": datetime.now(UTC).isoformat(),
            "auth_token": self.auth_token,
            "timeout_seconds": timeout,
            **payload,
        }
        result_key = f"{self.result_queue}:{job_id}"
        r = await self._redis()
        try:
            log.info(
                "windows_worker.submit",
                job_id=job_id,
                kind=kind,
                engagement_id=str(engagement_id),
                timeout=timeout,
            )
            await r.rpush(self.job_queue, json.dumps(job))
            # BRPOP with the job's own timeout + a small grace, so the
            # orchestrator never hangs longer than the operator asked for.
            grace = 10
            popped = await r.brpop([result_key], timeout=timeout + grace)
            if popped is None:
                log.warning("windows_worker.timeout", job_id=job_id, timeout=timeout)
                return WindowsJobResult(
                    job_id=job_id,
                    exit_code=-99,
                    stdout="",
                    stderr=f"orchestrator-side timeout after {timeout + grace}s",
                    started_at=job["submitted_at"],
                    ended_at=datetime.now(UTC).isoformat(),
                    timed_out=True,
                )
            _key, raw = popped
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as e:
                log.error("windows_worker.bad_result_json", job_id=job_id, err=str(e))
                return WindowsJobResult(
                    job_id=job_id,
                    exit_code=-98,
                    stdout="",
                    stderr=f"worker returned non-JSON result: {e}",
                    started_at=job["submitted_at"],
                    ended_at=datetime.now(UTC).isoformat(),
                )
            stdout = ""
            stderr = ""
            try:
                if result.get("stdout_b64"):
                    stdout = base64.b64decode(result["stdout_b64"]).decode("utf-8", errors="replace")
                if result.get("stderr_b64"):
                    stderr = base64.b64decode(result["stderr_b64"]).decode("utf-8", errors="replace")
            except (ValueError, TypeError) as e:
                stderr = f"<decode failed: {e}>\n{stderr}"
            return WindowsJobResult(
                job_id=job_id,
                exit_code=int(result.get("exit_code", -97)),
                stdout=stdout,
                stderr=stderr,
                started_at=result.get("started_at", job["submitted_at"]),
                ended_at=result.get("ended_at", datetime.now(UTC).isoformat()),
            )
        finally:
            await r.aclose()
