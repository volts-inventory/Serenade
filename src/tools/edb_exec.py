"""EDB exploit execution — fetch and run ExploitDB scripts in the sandbox.

This is the "discovery-tier execution" path. Unlike curated PoCs (which are
hand-reviewed and CI-tested), EDB scripts are fetched on-demand from the local
ExploitDB mirror and run inside the sandbox with full isolation.

Safety model (NodeZero-style):
- The sandbox provides runtime isolation (ephemeral container, per-engagement
  network, nftables scope enforcement). The EDB script CANNOT escape the
  sandbox — this is the real safety boundary.
- The native policy (policy/scope.py) enforces scope (target must be in
  in-scope CIDRs/domains).
- The LLM Critic reviews high-risk calls (RCE, destructive categories).
- The operator monitors between phases and approves stage transitions
  (guided mode); autonomous mode auto-approves.
- Non-destructive exploit types run — RCE, local privesc, webapps, bypass,
  etc. Destructive modules (DoS, or a body that statically scans as
  destructive via tools/destructive_scan.py — file removal, disk wipe, etc.)
  are reported as a finding and not executed unless the engagement set
  destructive_opt_in. The body-read scan fails closed when the script can't
  be read.

This tool is exposed to ExploitAgent alongside curated_poc_exec. The agent
prefers curated PoCs when available (higher confidence) but falls back to EDB
execution for any candidate that has a fetchable script.
"""

from __future__ import annotations

import ipaddress
import json
import re
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from tools.destructive_scan import scan_destructive
from worldstate.schema import RiskClass, Stage

# target_host flows into the sandbox argv (as a positional RHOST). It is
# never shell-interpolated anymore (see _build_edb_argv), but we still
# validate it as an IP or hostname so a leading-dash / metachar value can't
# turn into an argument-injection token to the interpreter or be mistaken for
# a flag. Hostnames: RFC-952/1123 label syntax. IPs: ipaddress parse.
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
)
# Env-var keys/values injected into the container. Keys must be a safe env
# identifier; values must not contain NUL/newline (which would break the
# `-e k=v` docker token or smuggle additional env). Everything else is fine
# because values are passed via docker `-e`, never via a shell string.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_edb_target_host(v: str) -> str:
    try:
        ipaddress.ip_address(v)
        return v
    except ValueError:
        pass
    if _HOSTNAME_RE.match(v):
        return v
    raise ValueError(
        f"target_host must be a valid IP or hostname (no metacharacters), got {v!r}"
    )


class EdbExecInput(BaseModel):
    edb_id: int = Field(ge=1, description="ExploitDB ID (e.g., 49112)")
    target_host: str = Field(min_length=1, description="In-scope target IP or hostname")
    target_port: int | None = Field(default=None, ge=0, le=65535)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Runtime parameters for the exploit. Common keys: "
            "RHOST, RPORT, LHOST, LPORT, URI, CMD. The wrapper "
            "auto-injects RHOST and RPORT from target_host/target_port. "
            "Params are passed to the script as ENVIRONMENT VARIABLES via the "
            "container env mechanism — never interpolated into a shell string."
        ),
    )
    candidate_id: str | None = Field(
        default=None,
        description="ExploitCandidate UUID this attempt originates from",
    )

    @field_validator("target_host")
    @classmethod
    def _check_target_host(cls, v: str) -> str:
        return _validate_edb_target_host(v)

    @field_validator("params")
    @classmethod
    def _check_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Keys must be valid env identifiers; values are stringified and must
        # not carry NUL/newline (which would corrupt the docker `-e k=v`
        # token). No shell interpolation happens, so the value alphabet is
        # otherwise unrestricted.
        for k, val in v.items():
            if not isinstance(k, str) or not _ENV_KEY_RE.match(k):
                raise ValueError(
                    f"param key {k!r} is not a valid environment-variable name"
                )
            sval = str(val)
            if "\x00" in sval or "\n" in sval or "\r" in sval:
                raise ValueError(
                    f"param {k!r} value must not contain NUL or newline characters"
                )
        return v


class EdbExecOutput(BaseModel):
    outcome: str  # "success" | "failure" | "error" | "skipped"
    summary: str
    error_message: str | None = None
    edb_id: int
    edb_title: str = ""
    edb_type: str = ""
    exit_code: int = -1
    stdout_excerpt: str = ""
    candidate_id: str | None = None


class EdbExec(Tool[EdbExecInput, EdbExecOutput]):
    name = "edb_exploit_exec"
    description = (
        "Fetch and execute an ExploitDB script against an in-scope target. "
        "The script runs ENTIRELY inside the sandbox container — isolated "
        "network, no host access, ephemeral filesystem. Supports all exploit "
        "types (remote, local, webapps). Subject to native scope + Critic gates. "
        "Prefer curated_poc_exec when a curated PoC matches; use this for "
        "candidates that only have an EDB reference."
    )
    stage = Stage.EXPLOIT
    risk_class = RiskClass.EXPLOIT
    # NOT marked destructive at the tool level. EDB execution is a supported
    # capability: scripts run inside the sandbox, gated by the edb_auto_exploit
    # toggle + PreExploitCritic + scope policy. Destructiveness is enforced
    # PER-MODULE inside run() — a DoS module (type="dos") is never executed and
    # is surfaced as a finding instead. Marking the whole tool destructive=True
    # would make the policy gate DENY every EDB call before run() is reached,
    # killing the feature.
    Input: ClassVar[type[BaseModel]] = EdbExecInput
    Output: ClassVar[type[BaseModel]] = EdbExecOutput

    def extract_targets(self, input: EdbExecInput) -> list[str]:
        return [input.target_host]

    def summarize_for_model(
        self, output: EdbExecOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        err_state = "yes" if output.error_message else "no"
        cand = f" candidate={output.candidate_id}" if output.candidate_id else ""
        summary = (
            f"{self.name}: edb_id={output.edb_id} type={output.edb_type} "
            f"outcome={output.outcome} exit_code={output.exit_code} "
            f"error={err_state}{cand} persisted={persisted}. "
            f"tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: EdbExecInput, ctx: ToolRunContext
    ) -> ToolResult[EdbExecOutput]:
        from orchestrator.evidence import EvidenceStore
        from orchestrator.settings import get_settings
        from sandbox.runner import SandboxRunner

        settings = get_settings()
        try:
            evidence = EvidenceStore(settings)
        except Exception:
            evidence = None  # type: ignore[assignment]
        executor = ToolExecutor(SandboxRunner(), evidence)

        started = datetime.now(UTC)

        # Check engagement context for toggle and destructive opt-in
        engagement: dict[str, Any] = {}
        eng_json = ctx.extra_env.get("__engagement_json__")
        if eng_json:
            try:
                engagement = json.loads(eng_json)
            except Exception:
                pass

        # EDB auto-exploit toggle: when False (default), surface the exploit
        # to the operator for manual approval instead of auto-executing.
        if not engagement.get("edb_auto_exploit", False):
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=EdbExecOutput(
                    outcome="skipped",
                    summary=(
                        f"EDB-{input.edb_id} available but edb_auto_exploit=false. "
                        f"Surfaced to operator for manual approval."
                    ),
                    error_message="edb_auto_exploit_disabled",
                    edb_id=input.edb_id,
                    candidate_id=input.candidate_id,
                ),
                exit_code=0,
                started_at=iso(started),
                ended_at=iso(datetime.now(UTC)),
            )

        # Step 1: Fetch exploit metadata from the local searchsploit mirror
        # to get the file path and type. This is a read-only local query.
        meta_result, _, _ = await executor.run_and_capture(
            ["searchsploit", "--json", "--edb", str(input.edb_id)],
            ctx=ctx,
            tool_name=f"{self.name}/meta",
            timeout_seconds=15,
        )
        meta = _parse_edb_meta(meta_result.stdout.decode("utf-8", errors="replace"), input.edb_id)
        if meta is None:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=EdbExecOutput(
                    outcome="error",
                    summary=f"EDB-{input.edb_id} not found in local mirror",
                    error_message="edb_not_found",
                    edb_id=input.edb_id,
                    candidate_id=input.candidate_id,
                ),
                exit_code=1,
                started_at=iso(started),
                ended_at=iso(datetime.now(UTC)),
            )

        # DoS modules are destructive: surfaced as a finding (not executed)
        # unless the engagement explicitly opts in to destructive actions,
        # mirroring the curated-PoC destructive gate. Non-DoS EDB modules
        # (remote/local/webapps) run normally.
        if meta["type"].lower() == "dos" and not engagement.get("destructive_opt_in", False):
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=EdbExecOutput(
                    outcome="skipped",
                    summary="DoS exploit not executed (destructive; set destructive_opt_in to run)",
                    error_message="destructive_without_optin",
                    edb_id=input.edb_id,
                    edb_title=meta["title"],
                    edb_type=meta["type"],
                    candidate_id=input.candidate_id,
                ),
                exit_code=0,
                started_at=iso(started),
                ended_at=iso(datetime.now(UTC)),
            )

        # Step 2: Build the sandbox execution command.
        # The EDB mirror stores scripts at known paths. We use the
        # serenade/sandbox image which has searchsploit + the mirror
        # pre-installed, so the script is available at the path from metadata.
        edb_path = meta.get("path", "")
        if not edb_path:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=EdbExecOutput(
                    outcome="error",
                    summary=f"EDB-{input.edb_id} has no script path",
                    error_message="no_script_path",
                    edb_id=input.edb_id,
                    edb_title=meta["title"],
                    edb_type=meta["type"],
                    candidate_id=input.candidate_id,
                ),
                exit_code=1,
                started_at=iso(started),
                ended_at=iso(datetime.now(UTC)),
            )

        # Destructive-operation scan: a script that deletes/overwrites target
        # files or wipes disks is destructive even when it isn't typed "dos".
        # Read the script body and gate it behind destructive_opt_in (same as
        # DoS). Skipped when opted in.
        if not engagement.get("destructive_opt_in", False):
            try:
                cat_result, _, _ = await executor.run_and_capture(
                    ["cat", edb_path],
                    ctx=ctx,
                    tool_name=f"{self.name}/scan",
                    timeout_seconds=10,
                )
            except Exception:
                cat_result = None
            if cat_result is None or cat_result.exit_code != 0 or not cat_result.stdout:
                return ToolResult(
                    tool_run_id=ctx.tool_run_id,
                    name=self.name,
                    output=EdbExecOutput(
                        outcome="skipped",
                        summary=(
                            f"EDB-{input.edb_id} script body could not be read for the "
                            f"destructive scan; not executed (set destructive_opt_in to run)"
                        ),
                        error_message="destructive_scan_unreadable",
                        edb_id=input.edb_id,
                        edb_title=meta["title"],
                        edb_type=meta["type"],
                        candidate_id=input.candidate_id,
                    ),
                    exit_code=0,
                    started_at=iso(started),
                    ended_at=iso(datetime.now(UTC)),
                )
            script_text = cat_result.stdout.decode("utf-8", errors="replace")
            hits = scan_destructive(script_text)
            if hits:
                return ToolResult(
                    tool_run_id=ctx.tool_run_id,
                    name=self.name,
                    output=EdbExecOutput(
                        outcome="skipped",
                        summary=(
                            f"EDB-{input.edb_id} performs destructive operations "
                            f"({', '.join(hits[:3])}); not executed "
                            f"(set destructive_opt_in to run)"
                        ),
                        error_message="destructive_without_optin",
                        edb_id=input.edb_id,
                        edb_title=meta["title"],
                        edb_type=meta["type"],
                        candidate_id=input.candidate_id,
                    ),
                    exit_code=0,
                    started_at=iso(started),
                    ended_at=iso(datetime.now(UTC)),
                )

        # Build execution argv based on script type
        params = dict(input.params)
        params.setdefault("RHOST", input.target_host)
        if input.target_port is not None:
            params.setdefault("RPORT", str(input.target_port))

        argv, edb_env = _build_edb_argv(edb_path, params, meta)

        # Step 3: Execute in sandbox with full isolation. Params are passed as
        # container environment variables via the runner's `-e` mechanism
        # (NEVER interpolated into a shell string), so a param value containing
        # `;`, `$(...)`, backticks, quotes, etc. cannot break out of argv.
        result, stdout_id, _ = await executor.run_and_capture(
            argv,
            ctx=ctx,
            tool_name=self.name,
            timeout_seconds=min(ctx.timeout_seconds, 180),
            env=edb_env,
        )
        ended = datetime.now(UTC)

        stdout_text = result.stdout.decode("utf-8", errors="replace")
        outcome = _classify_outcome(result.exit_code, stdout_text, result.stderr.decode("utf-8", errors="replace"))

        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=EdbExecOutput(
                outcome=outcome,
                summary=_build_summary(meta, outcome, stdout_text),
                edb_id=input.edb_id,
                edb_title=meta["title"],
                edb_type=meta["type"],
                exit_code=result.exit_code,
                stdout_excerpt=stdout_text[:500],
                candidate_id=input.candidate_id,
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(started),
            ended_at=iso(ended),
        )


def _parse_edb_meta(stdout: str, edb_id: int) -> dict[str, str] | None:
    """Extract metadata for a specific EDB-ID from searchsploit JSON output."""
    if not stdout.strip():
        return None
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    for entry in doc.get("RESULTS_EXPLOIT") or []:
        raw_id = entry.get("EDB-ID") or entry.get("EDB_ID") or "0"
        try:
            if int(raw_id) == edb_id:
                return {
                    "edb_id": str(edb_id),
                    "title": str(entry.get("Title") or ""),
                    "type": str(entry.get("Type") or "remote"),
                    "platform": str(entry.get("Platform") or ""),
                    "path": str(entry.get("Path") or ""),
                    "verified": str(entry.get("Verified", "")),
                }
        except (TypeError, ValueError):
            continue
    return None


def _build_edb_argv(
    edb_path: str, params: dict[str, Any], meta: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """Construct the sandbox argv + env to execute an EDB script.

    Returns ``(argv, env)``. The argv is a proper argument vector — NEVER a
    ``/bin/sh -c`` string interpolated with untrusted ``params``/``target_host``.
    All params are returned as the ``env`` dict and injected via the runner's
    container ``-e`` mechanism; the script reads them as environment
    variables. RHOST/RPORT are additionally appended as positional argv tokens
    (a common EDB pattern) — as discrete list elements, so a value containing
    shell metacharacters is inert.

    Strategy by file extension:
    - .py  → python3 <path> [rhost] [rport]
    - .rb  → ruby <path> [rhost] [rport]
    - .sh  → bash <path>
    - .c/.cpp → compile with gcc/g++ then execute the binary [rhost] [rport]
    - .pl  → perl <path> [rhost] [rport]
    - default → chmod +x then execute directly (shebang-based) [rhost] [rport]
    """
    import shlex

    ext = edb_path.rsplit(".", 1)[-1].lower() if "." in edb_path else ""

    # Every param becomes an environment variable. Stringify values; the
    # pydantic validator already guaranteed safe keys and NUL/newline-free
    # values. These are passed to docker via `-e k=v`, NOT a shell prefix.
    env: dict[str, str] = {k: str(v) for k, v in params.items()}

    rhost = str(params.get("RHOST", "") or "")
    rport = str(params.get("RPORT", "") or "")
    positional: list[str] = []
    if rhost:
        positional.append(rhost)
    if rport:
        positional.append(rport)

    if ext == "py":
        argv = ["python3", edb_path, *positional]
    elif ext == "rb":
        argv = ["ruby", edb_path, *positional]
    elif ext in ("sh", "bash"):
        argv = ["bash", edb_path, *positional]
    elif ext in ("c", "cpp"):
        # Compile-then-run needs a shell for `&&`, but we build the command
        # ONLY from shlex.quoted tokens — no raw untrusted interpolation. The
        # compiler/path are quoted; rhost/rport (already validated) are quoted
        # too. Params still travel via `env`, not the shell string.
        binary = "/tmp/edb_exploit"
        compiler = "g++" if ext == "cpp" else "gcc"
        run_part = " ".join(shlex.quote(t) for t in [binary, *positional])
        cmd = (
            f"{compiler} -o {shlex.quote(binary)} {shlex.quote(edb_path)} "
            f"&& {run_part}"
        )
        argv = ["/bin/sh", "-c", cmd]
    elif ext == "pl":
        argv = ["perl", edb_path, *positional]
    else:
        # Shebang-based — chmod +x then exec. Needs a shell for `&&`; build it
        # from shlex.quoted tokens only.
        run_part = " ".join(shlex.quote(t) for t in [edb_path, *positional])
        cmd = f"chmod +x {shlex.quote(edb_path)} && {run_part}"
        argv = ["/bin/sh", "-c", cmd]

    return argv, env


def _classify_outcome(exit_code: int, stdout: str, stderr: str) -> str:
    """Heuristic outcome classification from exit code and output."""
    combined = (stdout + stderr).lower()
    if exit_code == 124:
        return "failure"  # timeout
    # Success indicators
    success_signals = [
        "exploit successful",
        "shell opened",
        "session opened",
        "got shell",
        "command executed",
        "rce confirmed",
        "authenticated successfully",
        "access granted",
        "uid=",
        "root@",
        "nt authority\\system",
        "administrator",
        "payload delivered",
        "reverse shell",
    ]
    if exit_code == 0 and any(sig in combined for sig in success_signals):
        return "success"
    # Error indicators
    error_signals = [
        "connection refused",
        "connection timed out",
        "no route to host",
        "name resolution failed",
        "syntax error",
        "compilation error",
        "modulenotfounderror",
        "importerror",
        "traceback",
    ]
    if any(sig in combined for sig in error_signals):
        return "error"
    # Exit code 0 without clear success signals — inconclusive, call it failure
    if exit_code == 0:
        return "failure"
    return "failure"


def _build_summary(meta: dict[str, str], outcome: str, stdout: str) -> str:
    """Human-readable summary for the exploit attempt."""
    title = meta.get("title", "unknown")
    edb_type = meta.get("type", "unknown")
    lines = stdout.strip().split("\n")
    last_lines = "\n".join(lines[-3:]) if lines else ""
    return (
        f"EDB-{meta.get('edb_id', '?')} ({edb_type}): {title}\n"
        f"Outcome: {outcome}\n"
        f"Output tail:\n{last_lines[:200]}"
    )
