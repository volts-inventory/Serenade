"""Sandbox runner — runs a tool argv inside an isolated container.

Every call spawns a fresh container (`docker run --rm`) instead of
`docker exec` into a shared long-lived one. This eliminates cross-engagement
leakage via shared /tmp, /proc, nuclei cache, etc.

In dev the network defaults to a per-engagement bridge name; in prod the
runner can be configured to spawn Firecracker microVMs (not yet
implemented — drop-in replacement at the `run()` method).
"""

from __future__ import annotations

import asyncio
import shlex
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel

log = structlog.get_logger(__name__)


# Container-hardening flags applied to every sandbox `docker run`/`create`.
# Defence-in-depth that is orthogonal to the network ACLs:
#   --cap-drop=ALL           drop Docker's default capability set...
#   --cap-add=NET_RAW        ...then re-add only NET_RAW (nmap SYN/ICMP needs
#                            raw sockets); everything else (SETUID, MKNOD,
#                            DAC_OVERRIDE, CHOWN, ...) stays dropped.
#   --security-opt=no-new-privileges  block setuid-based privilege escalation
#                            (a fetched ExploitDB binary can't gain caps).
#   --pids-limit             cap process count so a fork-bomb payload can't
#                            starve the host.
#   --memory / --memory-swap cap memory so a memory-bomb can't OOM the host;
#                            equal values disable swap.
# The container still runs as its image's default user; we do NOT force a
# non-root --user because several pentest tools (nmap SYN, impacket raw) need
# root-in-container for their granted caps. The capability/no-new-privs/limit
# trio is the cheap, high-value subset.
_HARDENING_FLAGS: list[str] = [
    "--cap-drop=ALL",
    "--cap-add=NET_RAW",
    "--security-opt=no-new-privileges",
    "--pids-limit=512",
    "--memory=2g",
    "--memory-swap=2g",
]


class _SettingsUnavailable(RuntimeError):
    """Raised when ``Settings`` cannot be loaded to resolve L3 enforcement."""


def _l3_required() -> bool:
    """Resolve ``Settings.sandbox_l3_required`` lazily.

    Imported here (vs at module top) so a sandbox-only deploy doesn't pay the
    pydantic-settings import cost on every ``run()`` — and so test fixtures
    that monkeypatch ``get_settings`` are honoured.

    The documented DEFAULT (flag unset) is ``False`` — L3 is best-effort and
    we don't flip the default for operators who never opted in. BUT a
    settings/parse *error* is NOT a clean "False": we previously swallowed it
    and returned ``False``, which silently disabled the egress ACL on an
    INTERNAL error (a config bug masquerading as "L3 not required"). That is a
    fail-OPEN bug. We now raise ``_SettingsUnavailable`` so the L3 path treats
    "can't determine the requirement" as fail-CLOSED whenever a ruleset was
    actually requested.
    """
    from orchestrator.settings import get_settings

    try:
        settings = get_settings()
    except Exception as e:
        raise _SettingsUnavailable(
            "could not load Settings to resolve sandbox_l3_required"
        ) from e
    return bool(settings.sandbox_l3_required)


def _l3_required_failclosed() -> bool:
    """``_l3_required`` for the L3-requested path: a settings-load error is
    treated as 'required' (fail-closed) rather than silently disabling the
    egress ACL. Used only when a ruleset was actually compiled/requested."""
    try:
        return _l3_required()
    except _SettingsUnavailable:
        log.error(
            "sandbox.l3_settings_unavailable",
            advice=(
                "Settings could not be loaded to resolve sandbox_l3_required; "
                "an L3 ruleset WAS requested, so failing CLOSED (treating L3 as "
                "required) rather than silently downgrading to L2-only."
            ),
        )
        return True


class InFlightContainerRegistry:
    """Per-engagement in-flight container tracker.

    The abort consumer needs to be able to look up every container
    currently running for a given engagement and ``terminate()`` them in
    parallel. Without this registry the published
    ``engagement.abort_requested`` event has nowhere to land — the
    sandbox runners hold the container names ephemerally inside their
    ``run()`` coroutine and the consumer can't see them.

    Concurrency: every public mutator acquires the registry's lock so
    two ``run()`` coroutines starting on the same event-loop tick don't
    corrupt the per-engagement set. ``terminate_for_engagement`` SNAPSHOTS
    and pops in one critical section so a follow-up ``run()`` registering
    a brand-new container during the terminate loop doesn't get killed
    twice (and so a re-published abort event is a no-op once the slate's
    been cleared).
    """

    def __init__(self) -> None:
        self._by_engagement: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def register(self, engagement_id: str | None, container_name: str) -> None:
        if not engagement_id or not container_name:
            return
        async with self._lock:
            self._by_engagement.setdefault(engagement_id, set()).add(container_name)

    async def unregister(self, engagement_id: str | None, container_name: str) -> None:
        if not engagement_id or not container_name:
            return
        async with self._lock:
            bucket = self._by_engagement.get(engagement_id)
            if bucket is None:
                return
            bucket.discard(container_name)
            if not bucket:
                # Drop the empty bucket so the dict doesn't grow without
                # bound across long-running workers.
                self._by_engagement.pop(engagement_id, None)

    async def snapshot_and_clear(self, engagement_id: str) -> set[str]:
        """Atomically grab the current set of containers for this
        engagement AND drop them from the registry. Used by the abort
        consumer so a second abort publish for the same engagement
        doesn't re-terminate the same containers (the first publish
        already drained them; the second sees an empty snapshot)."""
        async with self._lock:
            return self._by_engagement.pop(engagement_id, set())

    async def known_engagements(self) -> set[str]:
        async with self._lock:
            return set(self._by_engagement.keys())


_REGISTRY_SINGLETON: InFlightContainerRegistry | None = None


def get_inflight_registry() -> InFlightContainerRegistry:
    """Process-wide singleton — both the worker's SandboxRunner and the
    abort consumer must read/write the SAME registry instance."""
    global _REGISTRY_SINGLETON
    if _REGISTRY_SINGLETON is None:
        _REGISTRY_SINGLETON = InFlightContainerRegistry()
    return _REGISTRY_SINGLETON


class SandboxResult(BaseModel):
    exit_code: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    started_at: datetime
    ended_at: datetime
    truncated: bool = False
    container_name: str | None = None


class SandboxRunner:
    """Per-call container spawner.

    Each `run()` invocation:
    1. Picks a unique container name `serenade-sb-{engagement_id}-{run_id}`.
    2. `docker run --rm` into that container, hard-removes on exit.
    3. Mounts no host paths. No shared `/tmp`, no shared `/proc`.
    4. Optional network attachment for the engagement's bridge.

    Tests inject a mock `_runner` that doesn't actually call docker.
    """

    def __init__(
        self,
        image: str = "serenade/sandbox:dev",
        max_output_bytes: int = 20 * 1024 * 1024,
        default_network: str | None = None,
        engagement_id: str | None = None,
        # If set, the nftables ruleset is applied to each spawned container's
        # netns before the tool runs. Pass via `worker.runtime.run_engagement`
        # from `Scope` -> compile_ruleset.
        nftables_ruleset: str | None = None,
    ) -> None:
        self.image = image
        self.max_output_bytes = max_output_bytes
        self.engagement_id = engagement_id
        self.nftables_ruleset = nftables_ruleset
        # If a caller passes only engagement_id (no explicit default_network),
        # auto-derive the per-engagement network name. Closes the
        # OperatorExec/Windows-via-WMI gap where containers spawned without a
        # network landed on Docker's default bridge (host + LAN access).
        if default_network is None and engagement_id:
            try:
                from uuid import UUID as _UUID

                from sandbox.network import engagement_network_name

                default_network = engagement_network_name(_UUID(engagement_id))
            except Exception:
                default_network = None
        self.default_network = default_network

    def _container_name(self) -> str:
        eng = self.engagement_id or "noeng"
        return f"serenade-sb-{eng[:8]}-{uuid.uuid4().hex[:8]}"

    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int = 300,
        cwd: str = "/work",
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        network: str | None = None,
        redacted_argv: Sequence[str] | None = None,
    ) -> SandboxResult:
        if not argv:
            raise ValueError("argv must not be empty")
        container_name = self._container_name()
        # The abort consumer needs to be able to find this container by
        # engagement_id and SIGTERM it mid-run. Wrap the work in try/finally
        # so even an exceptional exit (e.g. asyncio.CancelledError from the
        # consumer's terminate()) does not leak a stale registry entry.
        # ``register`` is a no-op when ``engagement_id`` is None (mock tests
        # / unscoped invocations).
        registry = get_inflight_registry()
        await registry.register(self.engagement_id, container_name)
        completed = False
        try:
            result = await self._run_inner(
                argv=argv,
                container_name=container_name,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
                stdin=stdin,
                network=network,
                redacted_argv=redacted_argv,
            )
            completed = True
            return result
        finally:
            # On a normal return the container is already gone (`--rm`, or the
            # explicit `docker rm -f` on the L3 path). But on an exceptional
            # exit — most importantly ``asyncio.CancelledError`` from an abort
            # or worker shutdown while we're awaiting ``communicate()`` — the
            # finally clauses inside ``_run_inner`` may be skipped (the L3
            # ``docker rm -f`` lives AFTER the await), and a non-``--rm`` L3
            # container, or a ``docker run`` the daemon still owns, can leak.
            # Force-remove unconditionally on the abnormal path so no
            # container survives a cancel/exception.
            if not completed:
                await self._force_remove(container_name)
            await registry.unregister(self.engagement_id, container_name)

    @staticmethod
    async def _force_remove(container_name: str) -> None:
        """Best-effort ``docker rm -f`` — used on the cancel/exception path so
        a container can't outlive its ``run()`` coroutine. Swallows all errors
        (the container may already be gone, or docker may be unreachable)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except BaseException:
            pass

    async def _run_inner(
        self,
        *,
        argv: Sequence[str],
        container_name: str,
        timeout_seconds: int,
        cwd: str,
        env: dict[str, str] | None,
        stdin: bytes | None,
        network: str | None,
        redacted_argv: Sequence[str] | None,
    ) -> SandboxResult:
        # Always set ENGAGEMENT_ID env in the sandbox so any tool that
        # respects it (or libraries we control) can stamp the
        # X-Serenade-Engagement header onto outbound HTTP requests.
        effective_env = {**(env or {})}
        if self.engagement_id and "ENGAGEMENT_ID" not in effective_env:
            effective_env["ENGAGEMENT_ID"] = self.engagement_id
        env_args: list[str] = []
        for k, v in effective_env.items():
            env_args += ["-e", f"{k}={v}"]
        net = network or self.default_network
        net_args = ["--network", net] if net else []
        docker_cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "-w",
            cwd,
            *_HARDENING_FLAGS,
            *env_args,
            *net_args,
            self.image,
            *argv,
        ]
        log.info(
            "sandbox.run",
            container=container_name,
            argv=list(redacted_argv or argv),
            network=net,
            nftables=bool(self.nftables_ruleset),
        )
        started = datetime.now(UTC)
        t0 = time.monotonic()
        # If an nftables ruleset is configured, apply it to the container's
        # netns BEFORE the work starts. We use docker create + sidecar apply
        # + docker start so the ruleset is in place when the tool's first
        # packet is sent.
        if self.nftables_ruleset:
            from sandbox.nftables import apply_ruleset_to_sandbox

            create_cmd = [c if c != "run" else "create" for c in docker_cmd]
            create_cmd = [c for c in create_cmd if c != "--rm"]
            create_proc = await asyncio.create_subprocess_exec(
                *create_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, create_err = await create_proc.communicate()
            if create_proc.returncode != 0:
                # The two failure modes (create failure, ruleset apply
                # failure) BOTH skip the L3 ruleset. Previously the runner
                # silently fell through to a fresh ``docker run`` below,
                # giving the operator a sandbox without any per-engagement
                # nftables ACLs — a silent isolation downgrade. Now we
                # check ``Settings.sandbox_l3_required``: when True, raise
                # so the worker surfaces the failure to the operator;
                # when False, log and keep the legacy fallback (with the
                # explicit warning so it's visible in logs).
                err_text = create_err.decode("utf-8", errors="replace")
                if _l3_required_failclosed():
                    raise RuntimeError(
                        f"sandbox L3 setup failed (docker create returned "
                        f"{create_proc.returncode}); Settings.sandbox_l3_"
                        f"required is True so this is fatal. err={err_text}"
                    )
                log.warning(
                    "sandbox.create_failed",
                    err=err_text,
                    advice=(
                        "nftables ruleset NOT applied; container will run "
                        "with L2-only isolation. Set SANDBOX_L3_REQUIRED=true "
                        "to make this fatal instead."
                    ),
                )
            else:
                applied = await apply_ruleset_to_sandbox(
                    sandbox_container_name=container_name,
                    ruleset=self.nftables_ruleset,
                )
                if not applied:
                    if _l3_required_failclosed():
                        # Clean up the orphan container before raising — the
                        # ``docker create`` succeeded but we never ``docker
                        # start``ed it, so it'd sit as a zombie otherwise.
                        await asyncio.create_subprocess_exec(
                            "docker", "rm", "-f", container_name,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        raise RuntimeError(
                            "sandbox L3 ruleset failed to apply and Settings."
                            "sandbox_l3_required is True — refusing to run "
                            "with L2-only isolation"
                        )
                    log.warning(
                        "sandbox.nftables_not_applied",
                        container=container_name,
                        advice="L3 ruleset failed to apply; container will run with L2 isolation only",
                    )
                # Start the prepared container
                start_proc = await asyncio.create_subprocess_exec(
                    "docker", "start", "-a", container_name,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        start_proc.communicate(input=stdin), timeout=timeout_seconds
                    )
                except TimeoutError:
                    kill_proc = await asyncio.create_subprocess_exec(
                        "docker", "kill", container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await kill_proc.wait()
                    stdout, stderr = b"", b"sandbox timeout"
                # docker start auto-leaves the container; clean up
                await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                return SandboxResult(
                    exit_code=start_proc.returncode if start_proc.returncode is not None else -1,
                    stdout=stdout[: self.max_output_bytes],
                    stderr=stderr[: self.max_output_bytes],
                    duration_seconds=time.monotonic() - t0,
                    started_at=started,
                    ended_at=datetime.now(UTC),
                    truncated=(len(stdout) > self.max_output_bytes or len(stderr) > self.max_output_bytes),
                    container_name=container_name,
                )
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            # Best-effort: kill the container externally.
            kill_proc = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await kill_proc.wait()
            proc.kill()
            await proc.wait()
            ended = datetime.now(UTC)
            return SandboxResult(
                exit_code=124,
                stdout=b"",
                stderr=f"sandbox timeout after {timeout_seconds}s".encode(),
                duration_seconds=time.monotonic() - t0,
                started_at=started,
                ended_at=ended,
                container_name=container_name,
            )
        truncated = False
        if len(stdout) > self.max_output_bytes:
            stdout = stdout[: self.max_output_bytes]
            truncated = True
        if len(stderr) > self.max_output_bytes:
            stderr = stderr[: self.max_output_bytes]
            truncated = True
        return SandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - t0,
            started_at=started,
            ended_at=datetime.now(UTC),
            truncated=truncated,
            container_name=container_name,
        )

    async def run_shell(self, command: str, **kwargs: Any) -> SandboxResult:
        return await self.run(["/bin/sh", "-c", command], **kwargs)

    async def terminate(
        self,
        container_name: str,
        *,
        timeout_s: int = 30,
    ) -> bool:
        """SIGTERM → grace → SIGKILL a running sandbox container.

        Called by the abort path on every in-flight tool run so a customer
        kill switch actually stops work in progress instead of waiting for
        the natural per-tool timeout (which can be ~5 minutes by default).

        Behaviour:
          1. ``docker stop --time=timeout_s`` (sends SIGTERM, waits up to
             ``timeout_s`` for the process to exit cleanly).
          2. If the container is still alive after grace, ``docker kill``
             (SIGKILL).
          3. Always cleans up with ``docker rm -f`` so the container slot
             isn't held open.

        Best-effort: returns True on success, False on any docker failure
        (logged). Callers should treat False as "the kill switch fired but
        we can't confirm the container died" — the engagement state has
        already been marked aborted so the worker won't dispatch new work
        regardless.
        """
        try:
            stop_proc = await asyncio.create_subprocess_exec(
                "docker", "stop", "--time", str(timeout_s), container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stop_err = await stop_proc.communicate()
            if stop_proc.returncode != 0:
                log.warning(
                    "sandbox.terminate.stop_failed",
                    container=container_name,
                    err=stop_err.decode("utf-8", errors="replace"),
                )
            # Belt-and-braces: SIGKILL anything that survived the grace.
            kill_proc = await asyncio.create_subprocess_exec(
                "docker", "kill", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await kill_proc.wait()
            rm_proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await rm_proc.wait()
            return True
        except Exception as e:
            log.warning("sandbox.terminate.failed", container=container_name, err=str(e))
            return False


def quote(s: str) -> str:
    return shlex.quote(s)
