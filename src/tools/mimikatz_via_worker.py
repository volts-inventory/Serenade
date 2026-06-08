"""Mimikatz tool that dispatches via the Windows worker (Path B).

Unlike `windows_via_wmi.py` (Path A), this tool does not require
wmiexec or pre-staged binaries on the target — it pushes the job onto
a Redis queue consumed by a Windows worker VM. See
`windows_worker/README.md` for the operator-side setup.

Gating:
- Stage.LATERAL, RiskClass.POST_EXPLOIT (same as Path A)
- PreLateralCritic + OPA gate at submission time
- Worker-side rejects on auth_token mismatch
- Worker rejects on heartbeat-stale / isolation=none (pre-flight)
- Falls back with a clear error if the feature is disabled

Output now exposes a `rows` field with extracted credential records
so LateralAgent._parse_and_persist can wire mimikatz output back into
the world-state graph (Credential nodes + HAS_SESSION_ON edges to
historic LOCAL_ADMIN_ON hosts).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import ClassVar
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

from orchestrator.windows_worker import (
    WindowsJobAuthMissing,
    WindowsJobQueue,
    WindowsJobQueueDisabled,
    WindowsWorkerNotReady,
    WindowsWorkerUnsafeIsolation,
    is_enabled,
)
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class MimikatzViaWorkerInput(BaseModel):
    target: str = Field(
        min_length=1,
        description="Target host the credentials originate from (for scoping). "
        "The actual mimikatz runs on the Serenade Windows worker, NOT on "
        "this host — the operator supplies the LSASS dump separately or "
        "uses sekurlsa::pth via worker chaining.",
    )
    args: list[str] = Field(
        min_length=1,
        description="Argv to pass to mimikatz.exe on the worker. The operator "
        "staged the binary at C:\\Serenade\\tools\\mimikatz.exe.",
    )
    timeout_seconds: int = Field(default=120, ge=10, le=900)


class MimikatzCredentialRow(BaseModel):
    """One credential record extracted from mimikatz output.

    Shape compatible with LateralAgent's _parse_and_persist credential path:
    `username`, `domain`, optional `password`, optional `ntlm`. `kind` is
    always `"credential"` so the agent can distinguish from netexec-style
    success rows.
    """

    kind: str = "credential"
    username: str
    domain: str | None = None
    password: str | None = None
    ntlm: str | None = None
    source: str = "mimikatz"


class MimikatzViaWorkerOutput(BaseModel):
    exit_code: int
    stdout_excerpt: str
    stderr_excerpt: str
    summary: str
    worker_disabled: bool = False
    # terminal_error signals to the LateralAgent that retrying this
    # tool with the same input will fail the same way (config / heartbeat /
    # isolation gate). The agent's tool-result interpreter suppresses the
    # tool from the next turn's candidate list when this is True so the LLM
    # pivots to Path A or a different lateral approach.
    terminal_error: bool = False
    rows: list[MimikatzCredentialRow] = Field(default_factory=list)


# Regex patterns for mimikatz `sekurlsa::logonpasswords` and similar output.
# These are intentionally permissive — we want to extract every cred we can
# without requiring exact whitespace.
_USERNAME_RE = re.compile(r"^\s*\*?\s*Username\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^\s*\*?\s*Domain\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_PASSWORD_RE = re.compile(r"^\s*\*?\s*Password\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_NTLM_RE = re.compile(r"^\s*\*?\s*NTLM\s*:\s*([a-fA-F0-9]{32})", re.MULTILINE)


_MAX_STDOUT_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_LINES = 50_000
_MAX_CREDENTIALS = 200


def parse_mimikatz_output(stdout: str) -> list[MimikatzCredentialRow]:
    """Best-effort parse of mimikatz `sekurlsa::logonpasswords` output.

    Bounded against adversary-controlled stdout. If the target host
    floods the worker with garbage, we cap the input at 5 MB / 50k lines /
    200 credentials so the engagement doesn't burn CPU on noise.

    The format is loosely structured blocks per session. We walk the stdout
    line by line, accumulate fields until we see the next Username:, then
    emit a row. Empty/null/(null)/`(null)` fields become None.
    """
    truncated_reason: str | None = None
    original_len = len(stdout)
    if original_len > _MAX_STDOUT_BYTES:
        stdout = stdout[:_MAX_STDOUT_BYTES]
        truncated_reason = "stdout_byte_cap"
    rows: list[MimikatzCredentialRow] = []
    current: dict[str, str | None] = {}

    def _flush():
        if current.get("username") and current["username"].lower() != "(null)":
            rows.append(
                MimikatzCredentialRow(
                    username=current["username"],
                    domain=_clean(current.get("domain")),
                    password=_clean(current.get("password")),
                    ntlm=_clean(current.get("ntlm")),
                )
            )

    line_count = 0
    hit_line_cap = False
    hit_cred_cap = False
    for line in stdout.splitlines():
        line_count += 1
        if line_count > _MAX_LINES:
            hit_line_cap = True
            break
        if len(rows) >= _MAX_CREDENTIALS:
            hit_cred_cap = True
            break
        stripped = line.strip().lstrip("*").strip()
        if not stripped or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "username":
            if current:
                _flush()
            current = {"username": value}
        elif key in ("domain", "password", "ntlm") and current:
            current[key] = value
    if current and len(rows) < _MAX_CREDENTIALS:
        _flush()
    # Log a structured warning when caps fire so the operator can
    # see in the audit log that mimikatz output was truncated. An adversary
    # who controls the target host could otherwise hide a high-value
    # credential at the tail of the stdout.
    if truncated_reason or hit_line_cap or hit_cred_cap:
        log.warning(
            "mimikatz.parser_truncated",
            stdout_bytes=original_len,
            truncated_to_bytes=len(stdout),
            lines_seen=line_count,
            credentials_kept=len(rows),
            hit_byte_cap=truncated_reason == "stdout_byte_cap",
            hit_line_cap=hit_line_cap,
            hit_credential_cap=hit_cred_cap,
        )
    # Dedupe identical rows (same username+domain+ntlm).
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[MimikatzCredentialRow] = []
    for r in rows:
        key_tuple = (r.username, r.domain, r.ntlm)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        out.append(r)
    return out


def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if not v or v.lower() in ("(null)", "null", "n.a.", "n/a"):
        return None
    return v


class MimikatzViaWorker(Tool[MimikatzViaWorkerInput, MimikatzViaWorkerOutput]):
    name = "mimikatz_via_worker"
    description = (
        "Run mimikatz on the Serenade Windows worker. Requires "
        "SERENADE_WINDOWS_WORKER_ENABLED=1 plus a running worker with a fresh "
        "heartbeat and a non-'none' isolation mode. PreLateralCritic + OPA "
        "gate; worker rejects unauth'd jobs and unsigned assemblies."
    )
    stage = Stage.LATERAL
    risk_class = RiskClass.POST_EXPLOIT
    Input: ClassVar[type[BaseModel]] = MimikatzViaWorkerInput
    Output: ClassVar[type[BaseModel]] = MimikatzViaWorkerOutput

    def extract_targets(self, input: MimikatzViaWorkerInput) -> list[str]:
        return [input.target]

    def summarize_for_model(
        self, output: MimikatzViaWorkerOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # Highlight the headline: count of extracted creds, plus the first
        # interesting user. Never include passwords or NTLM verbatim — the
        # planner doesn't need them and the agent already persisted them.
        #
        # also never include `output.summary` (built in run()
        # from `input.args!r` — an operator passing sekurlsa::pth /ntlm:HASH
        # or /password:X would leak that into planner context), and never
        # include stdout/stderr excerpts (mimikatz's raw stdout is the
        # canonical leak surface).
        with_password = sum(1 for r in output.rows if r.password)
        with_ntlm = sum(1 for r in output.rows if r.ntlm)
        sample_names = ", ".join(
            (f"{r.domain}\\{r.username}" if r.domain else r.username)
            for r in output.rows[:3]
        ) or "none"
        terminal = " TERMINAL" if output.terminal_error else ""
        disabled = " DISABLED" if output.worker_disabled else ""
        summary = (
            f"{self.name}: exit={output.exit_code} creds={len(output.rows)} "
            f"(pwd={with_password}, ntlm={with_ntlm}). users: {sample_names}."
            f"{terminal}{disabled} "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: MimikatzViaWorkerInput, ctx: ToolRunContext
    ) -> ToolResult[MimikatzViaWorkerOutput]:
        started = datetime.now(UTC)
        if not is_enabled():
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MimikatzViaWorkerOutput(
                    exit_code=-100,
                    stdout_excerpt="",
                    stderr_excerpt="windows worker not enabled",
                    summary="windows worker disabled (SERENADE_WINDOWS_WORKER_ENABLED!=1)",
                    worker_disabled=True,
                    terminal_error=True,
                ),
                exit_code=-100,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        try:
            queue = WindowsJobQueue()
        except (WindowsJobQueueDisabled, WindowsJobAuthMissing) as e:
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MimikatzViaWorkerOutput(
                    exit_code=-101,
                    stdout_excerpt="",
                    stderr_excerpt=str(e),
                    summary="windows worker config error",
                    worker_disabled=True,
                    terminal_error=True,
                ),
                exit_code=-101,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        try:
            result = await queue.submit_mimikatz(
                engagement_id=ctx.engagement_id,
                args=input.args,
                timeout_seconds=input.timeout_seconds,
            )
        except WindowsWorkerNotReady as e:
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MimikatzViaWorkerOutput(
                    exit_code=-102,
                    stdout_excerpt="",
                    stderr_excerpt=str(e),
                    summary="worker heartbeat stale; dispatch refused",
                    worker_disabled=True,
                    terminal_error=True,
                ),
                exit_code=-102,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        except WindowsWorkerUnsafeIsolation as e:
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MimikatzViaWorkerOutput(
                    exit_code=-103,
                    stdout_excerpt="",
                    stderr_excerpt=str(e),
                    summary="worker reports isolation=none; refused (set SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION=1 to override)",
                    worker_disabled=True,
                    terminal_error=True,
                ),
                exit_code=-103,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        ended = datetime.now(UTC)
        creds = parse_mimikatz_output(result.stdout) if result.exit_code == 0 else []
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=MimikatzViaWorkerOutput(
                exit_code=result.exit_code,
                stdout_excerpt=result.stdout[:2000],
                stderr_excerpt=result.stderr[:1000],
                summary=(
                    f"mimikatz via worker ({input.target}): args={input.args!r}; "
                    f"exit {result.exit_code}; timed_out={result.timed_out}; "
                    f"creds_extracted={len(creds)}"
                ),
                rows=creds,
            ),
            exit_code=result.exit_code,
            started_at=result.started_at,
            ended_at=result.ended_at,
        )
