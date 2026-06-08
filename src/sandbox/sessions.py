"""Session execution runtime.

A `Session` (worldstate node) represents an interactive channel into a
compromised target — SSH shell, WinRM, SMB named pipe, MSF meterpreter,
etc. The `SessionManager` owns the lifecycle:

  open(host, type, credentials) -> Session
  exec(session_id, command) -> ExecResult
  close(session_id)

Sessions live as long-running sandbox containers (one per session). The
container's stdin/stdout becomes the channel; `exec` writes a command +
delimiter and reads back to the delimiter.

Currently supports the `ssh` channel type — `ssh user@host` inside a
sandbox container, kept alive with `-o ServerAliveInterval=30`.
WinRM/SMB/Meterpreter channels follow the same pattern; not yet
implemented.

All exec output is captured to MinIO as evidence. Commands run by a
session inherit the session's RiskClass.POST_EXPLOIT, so the OPA gate
and (eventually) PreLateralCritic fire on every exec.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import structlog

log = structlog.get_logger(__name__)


# Session containers are long-lived `sleep infinity` shells we `docker exec`
# into. Apply the same hardening as the per-call sandbox runner: drop all
# Docker default caps (re-adding NET_RAW for ssh/raw tooling),
# no-new-privileges, and pid/memory ceilings so a session can't fork-bomb or
# memory-bomb the host. See sandbox/runner.py::_HARDENING_FLAGS.
_SESSION_HARDENING_FLAGS: list[str] = [
    "--cap-drop=ALL",
    "--cap-add=NET_RAW",
    "--security-opt=no-new-privileges",
    "--pids-limit=512",
    "--memory=2g",
    "--memory-swap=2g",
]


def _build_session_run_argv(
    *,
    container_name: str,
    env_args: list[str],
    network: str,
    image: str,
) -> list[str]:
    """Construct the session container ``docker run`` argv.

    Factored out so tests can assert — without Docker — that a session ALWAYS
    attaches an explicit ``--network`` (never the default bridge, sandbox-10)
    and carries the hardening flags. ``network`` must be a non-empty string;
    the caller guarantees it via ``network or engagement_network_name(...)``.
    """
    if not network:
        # Defence-in-depth: never let a session container fall onto the default
        # bridge. The caller already derives a per-engagement network, but guard
        # here too so no future code path can spawn an unnetworked session.
        raise ValueError("session container must have an explicit network")
    return [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        *env_args,
        "--network",
        network,
        *_SESSION_HARDENING_FLAGS,
        image,
        "sleep",
        "infinity",
    ]


@dataclass
class SessionHandle:
    """In-process bookkeeping for an open session."""

    id: UUID
    container_name: str
    type: Literal["ssh", "winrm", "smb", "msf"]
    target_host: str
    engagement_id: UUID
    opened_at: datetime
    proc: asyncio.subprocess.Process | None = None
    transcript: list[str] = field(default_factory=list)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: str


class SessionManager:
    """Process-local registry of open sessions. One per worker.

    Production: this becomes a Redis-backed registry so sessions
    survive worker restart. For now, in-process is fine — sessions die
    with the worker, and the operator just opens a new one.
    """

    def __init__(self, image: str = "serenade/sandbox:dev") -> None:
        self.image = image
        self._handles: dict[UUID, SessionHandle] = {}
        self._locks: dict[UUID, asyncio.Lock] = {}

    async def open_ssh(
        self,
        *,
        engagement_id: UUID,
        target_host: str,
        target_port: int = 22,
        username: str,
        password: str | None = None,
        identity_file_blob: bytes | None = None,
        network: str | None = None,
    ) -> SessionHandle:
        """Open an SSH session inside a fresh sandbox container.

        Uses `sshpass` for password auth (already in the sandbox image).
        For key-based auth, the caller passes the key bytes; we write
        them into the container via stdin to a temp file.
        """
        session_id = uuid.uuid4()
        container_name = f"serenade-ses-{str(engagement_id)[:8]}-{session_id.hex[:8]}"
        env_args = ["-e", f"ENGAGEMENT_ID={engagement_id}"]
        # NEVER let a session container fall onto Docker's default bridge (it
        # NATs to the host + LAN). When the caller doesn't pass an explicit
        # network, derive the per-engagement network from engagement_id —
        # mirroring SandboxRunner. There is always a network attached.
        from sandbox.network import engagement_network_name

        net = network or engagement_network_name(engagement_id)
        # Use a long-running shell so we can `docker exec` commands.
        docker_cmd = _build_session_run_argv(
            container_name=container_name,
            env_args=env_args,
            network=net,
            image=self.image,
        )
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to spawn session container: {err.decode('utf-8', errors='replace')}"
            )

        # sandbox-10 residual: register the session container in the SAME
        # in-flight registry the SandboxRunner uses, so an
        # ``engagement.abort_requested`` event can ``docker kill`` an in-flight
        # SSH session container too — not just per-call sandbox containers.
        # Without this the abort consumer had no handle on session containers.
        from sandbox.runner import get_inflight_registry

        await get_inflight_registry().register(str(engagement_id), container_name)

        handle = SessionHandle(
            id=session_id,
            container_name=container_name,
            type="ssh",
            target_host=target_host,
            engagement_id=engagement_id,
            opened_at=datetime.now(UTC),
        )
        self._handles[session_id] = handle
        self._locks[session_id] = asyncio.Lock()
        log.info(
            "session.opened",
            session_id=str(session_id),
            container=container_name,
            target=target_host,
            type="ssh",
        )

        # Validate auth by running a single `whoami` over SSH from inside the container.
        # This both confirms the credentials work and seeds the transcript.
        _validate = await self._exec_locked(
            handle,
            self._ssh_command(target_host, target_port, username, password, "whoami"),
            timeout_seconds=30,
            redacted_command=self._ssh_command(
                target_host, target_port, username, "***REDACTED***" if password else None, "whoami"
            ),
        )
        if _validate.exit_code != 0:
            await self.close(session_id)
            raise RuntimeError(
                f"SSH auth failed (exit {_validate.exit_code}): {_validate.stderr.strip()[:200]}"
            )

        # Store the auth command formula on the handle for subsequent exec()s.
        handle.transcript.append(f"# session opened to {username}@{target_host}:{target_port}")
        handle._ssh_user = username  # type: ignore[attr-defined]
        handle._ssh_password = password  # type: ignore[attr-defined]
        handle._ssh_port = target_port  # type: ignore[attr-defined]
        return handle

    async def exec(
        self,
        session_id: UUID,
        command: str,
        *,
        timeout_seconds: int = 60,
    ) -> ExecResult:
        """Execute a command in an open session."""
        handle = self._handles.get(session_id)
        if handle is None:
            raise KeyError(f"session not found: {session_id}")
        if handle.type != "ssh":
            raise NotImplementedError(
                f"exec for {handle.type} sessions not yet implemented"
            )
        user = getattr(handle, "_ssh_user", None)
        password = getattr(handle, "_ssh_password", None)
        port = getattr(handle, "_ssh_port", 22)
        argv = self._ssh_command(handle.target_host, port, user, password, command)
        redacted = self._ssh_command(
            handle.target_host,
            port,
            user,
            "***REDACTED***" if password else None,
            command,
        )
        return await self._exec_locked(
            handle, argv, timeout_seconds=timeout_seconds, redacted_command=redacted
        )

    async def close(self, session_id: UUID) -> None:
        handle = self._handles.pop(session_id, None)
        if handle is None:
            return
        self._locks.pop(session_id, None)
        # sandbox-10 residual: drop the session container from the in-flight
        # abort registry as we tear it down so the registry stays bounded and a
        # later abort doesn't try to kill an already-removed container.
        from sandbox.runner import get_inflight_registry

        await get_inflight_registry().unregister(
            str(handle.engagement_id), handle.container_name
        )
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            handle.container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        log.info("session.closed", session_id=str(session_id), container=handle.container_name)

    def list_open(self, engagement_id: UUID | None = None) -> list[SessionHandle]:
        if engagement_id is None:
            return list(self._handles.values())
        return [h for h in self._handles.values() if h.engagement_id == engagement_id]

    # ---- internals ----

    def _ssh_command(
        self,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
        command: str,
    ) -> list[str]:
        """Build the `docker exec` argv that runs an SSH command inside the session container.

        Uses sshpass when a password is set; -o BatchMode=yes when key-only.
        StrictHostKeyChecking=no because pentest targets won't be in known_hosts.
        """
        ssh_args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes" if not password else "BatchMode=no",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(port),
            f"{user}@{host}",
            command,
        ]
        if password:
            ssh_args = ["sshpass", "-p", password, *ssh_args]
        return ssh_args

    async def _exec_locked(
        self,
        handle: SessionHandle,
        argv: list[str],
        *,
        timeout_seconds: int,
        redacted_command: list[str] | None = None,
    ) -> ExecResult:
        lock = self._locks[handle.id]
        async with lock:
            docker_exec = ["docker", "exec", handle.container_name, *argv]
            log.info(
                "session.exec",
                session_id=str(handle.id),
                command=(redacted_command or argv),
            )
            import time as _time

            t0 = _time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *docker_exec,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"[serenade] session exec timeout"
            duration = _time.monotonic() - t0
            so = stdout.decode("utf-8", errors="replace")
            se = stderr.decode("utf-8", errors="replace")
            handle.transcript.append(f"$ {shlex.join(redacted_command or argv)}")
            handle.transcript.append(so)
            return ExecResult(
                stdout=so,
                stderr=se,
                exit_code=proc.returncode if proc.returncode is not None else -1,
                duration_seconds=duration,
                command=shlex.join(redacted_command or argv),
            )


# Process-wide singleton; the worker can swap this for testing.
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
