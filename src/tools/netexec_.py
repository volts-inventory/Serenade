"""NetExec (formerly CrackMapExec) adapter.

Single adapter for the four core protocols pentesters use it for: SMB, WinRM,
LDAP, MSSQL. The protocol is a typed input field; credentials are typed too.

When credentials are present in argv we pass `redacted_argv` to
`SandboxRunner.run` so the password never lands in logs verbatim.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from orchestrator.evidence import EvidenceStore
from sandbox.network import engagement_network_name
from sandbox.runner import SandboxRunner
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage

_NETEXEC_BINARY = "netexec"  # `netexec` is the canonical entry point on recent installs


class NetExecInput(BaseModel):
    protocol: Literal["smb", "winrm", "ldap", "mssql", "ssh"] = "smb"
    targets: list[str] = Field(min_length=1, description="IPs or hostnames")
    username: str | None = None
    password: str | None = None
    nt_hash: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{32}$",
        description="NT hash (32 hex chars) for pass-the-hash",
    )
    domain: str | None = None
    local_auth: bool = False
    # Action selector — what to do once authenticated (or unauthenticated)
    action: Literal[
        "auth_check",  # bare auth attempt; report success/failure per host
        "list_shares",  # SMB share enumeration
        "list_users",  # RID-cycling / LDAP users
        "loggedon_users",  # who's logged in (requires admin on SMB)
        "sessions",  # active sessions (requires admin)
    ] = "auth_check"
    timeout_seconds: int = Field(default=60, ge=10, le=600)

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v: list[str]) -> list[str]:
        # Targets are positional after the protocol; a leading-dash value would
        # be parsed as a netexec flag (sandbox-09).
        for t in v:
            if t.startswith("-"):
                raise ValueError(
                    f"target {t!r} must not start with '-' (would be parsed as "
                    "a netexec flag)"
                )
        return v

    @field_validator("username", "password", "domain")
    @classmethod
    def _check_credlike(cls, v: str | None) -> str | None:
        # username/password/domain flow into `-u/-p/-d <value>`. A value
        # beginning with `-` can be misparsed as the next flag (argparse), and
        # NUL/newline corrupt the redacted-argv log line. Reject both.
        if v is None:
            return v
        if v.startswith("-"):
            raise ValueError(
                "value must not start with '-' (would be parsed as a netexec flag)"
            )
        if "\x00" in v or "\n" in v or "\r" in v:
            raise ValueError("value must not contain NUL or newline characters")
        return v


class NetExecAuthRow(BaseModel):
    target: str
    protocol: str
    status: Literal["success", "failure", "error"]
    admin: bool = False
    domain: str | None = None
    username: str | None = None
    raw: str  # original line, kept for evidence


class NetExecOutput(BaseModel):
    rows: list[NetExecAuthRow] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    raw_stderr_blob_id: str | None = None
    command: str
    action: str


def _build_executor():
    from orchestrator.settings import get_settings
    from tools._helpers import ToolExecutor

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    return ToolExecutor(SandboxRunner(), evidence)


class NetExec(Tool[NetExecInput, NetExecOutput]):
    name = "netexec"
    description = (
        "NetExec (formerly CrackMapExec) for SMB/WinRM/LDAP/MSSQL/SSH "
        "enumeration and authenticated actions. Provide credentials (password "
        "OR nt_hash) for authenticated runs; omit them for null-session / "
        "anonymous enumeration. `action` selects what to do once auth resolves."
    )
    stage = Stage.ENUMERATION
    # Default classification — actions that require valid creds promote to
    # post-exploit, but at tool-definition time we treat this as enum.
    risk_class = RiskClass.ENUM
    destructive = False
    Input: ClassVar[type[BaseModel]] = NetExecInput
    Output: ClassVar[type[BaseModel]] = NetExecOutput

    def extract_targets(self, input: NetExecInput) -> list[str]:
        return list(input.targets)

    def summarize_for_model(
        self, output: NetExecOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        succ = [r for r in output.rows if r.status == "success"]
        admin = [r for r in succ if r.admin]
        # Show up to 3 successful hosts (admin first).
        ranked = sorted(succ, key=lambda r: not r.admin)
        top_parts: list[str] = []
        for r in ranked[:3]:
            who = (
                f"{r.domain}\\{r.username}"
                if r.domain and r.username
                else (r.username or "anon")
            )
            badge = " (Pwn3d!)" if r.admin else ""
            top_parts.append(f"{r.protocol}//{r.target} {who}{badge}")
        top = "; ".join(top_parts) or "none"
        summary = (
            f"{self.name} action={output.action}: {len(output.rows)} rows, "
            f"{len(succ)} success ({len(admin)} admin). top: {top}. "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: NetExecInput, ctx: ToolRunContext) -> ToolResult[NetExecOutput]:
        argv = [_NETEXEC_BINARY, input.protocol, *input.targets]
        redacted = list(argv)
        if input.username:
            argv += ["-u", input.username]
            redacted += ["-u", input.username]
        if input.password:
            argv += ["-p", input.password]
            redacted += ["-p", "***REDACTED***"]
        if input.nt_hash:
            argv += ["-H", input.nt_hash]
            redacted += ["-H", "***REDACTED-NT-HASH***"]
        if input.domain:
            argv += ["-d", input.domain]
            redacted += ["-d", input.domain]
        if input.local_auth:
            argv += ["--local-auth"]
            redacted += ["--local-auth"]
        # Action-specific flag
        if input.action == "list_shares":
            argv += ["--shares"]
            redacted += ["--shares"]
        elif input.action == "list_users":
            argv += ["--users"]
            redacted += ["--users"]
        elif input.action == "loggedon_users":
            argv += ["--loggedon-users"]
            redacted += ["--loggedon-users"]
        elif input.action == "sessions":
            argv += ["--sessions"]
            redacted += ["--sessions"]

        executor = _build_executor()
        # Mirror ToolExecutor.run_and_capture (canonical pattern in
        # _helpers.py): bind BOTH engagement_id AND default_network so the
        # spawned container lands on the per-engagement bridge instead of
        # Docker's default network.
        executor.sandbox.engagement_id = str(ctx.engagement_id)
        executor.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        # Run via the executor's sandbox-bind + evidence-capture path.
        result = await executor.sandbox.run(
            argv,
            timeout_seconds=input.timeout_seconds,
            redacted_argv=redacted,
        )
        stdout_id: str | None = None
        if executor.evidence and result.stdout:
            blob_id, _ = executor.evidence.put(
                result.stdout, engagement_id=ctx.engagement_id, kind=f"{self.name}/stdout"
            )
            stdout_id = str(blob_id)
        rows = _parse(result.stdout.decode("utf-8", errors="replace"), input.protocol)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=NetExecOutput(
                rows=rows,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(redacted),
                action=input.action,
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


# NetExec doesn't have stable JSON output across versions; we parse the
# coloured terminal output (with --no-color or ANSI stripping). Each line
# typically looks like:
#   SMB    10.10.10.5    445   HOSTNAME    [*] Windows ...
#   SMB    10.10.10.5    445   HOSTNAME    [+] DOMAIN\user:Password (Pwn3d!)
#   SMB    10.10.10.5    445   HOSTNAME    [-] DOMAIN\user:WrongPass
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_AUTH_LINE = re.compile(
    r"^\s*(?P<proto>SMB|WINRM|LDAP|MSSQL|SSH)\s+"
    r"(?P<host>\S+)\s+\S+\s+\S+\s+"
    r"\[(?P<sigil>[+\-*!])\]\s+"
    r"(?:(?P<domain>[^\\]+)\\)?(?P<user>[^:\s]+)?(?::(?P<rest>.*))?$",
    re.IGNORECASE,
)


def _parse(text: str, protocol: str) -> list[NetExecAuthRow]:
    out: list[NetExecAuthRow] = []
    for raw_line in text.splitlines():
        line = _ANSI.sub("", raw_line).rstrip()
        if not line.strip():
            continue
        m = _AUTH_LINE.match(line)
        if not m:
            continue
        sigil = m.group("sigil")
        if sigil == "+":
            status = "success"
        elif sigil == "-":
            status = "failure"
        else:
            continue  # informational [*] / [!] lines aren't auth results
        admin = "Pwn3d!" in line or "(Admin)" in line
        out.append(
            NetExecAuthRow(
                target=m.group("host"),
                protocol=protocol,
                status=status,  # type: ignore[arg-type]
                admin=admin,
                domain=m.group("domain"),
                username=m.group("user"),
                raw=line,
            )
        )
    return out
