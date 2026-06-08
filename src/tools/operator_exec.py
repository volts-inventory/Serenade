"""OperatorExec — operator-driven one-off command execution.

The curated-only exploit tier is intentionally a wall, not a fence.
OperatorExec is the audited escape hatch: the operator can run arbitrary
argv in the sandbox, with:

- Full sandbox isolation (per-engagement Docker network, no host access)
- OPA scope check still fires (target must be in scope)
- PreExploitCritic still fires (LLM reviews the proposed action)
- Mandatory `justification` field (audit-visible WHY)
- Credentials redaction in logs
- `tool_run_id` provenance chain to evidence + report

There are TWO bypass tiers operator can opt into, each adds an audit trail:
- `bypass_critic=true` requires a SECOND `bypass_justification` field
- `bypass_scope=true` is REFUSED — scope can never be bypassed

The tool is registered but the planner LLM is NEVER allowed to propose it.
Only the orchestrator's `POST /engagements/{id}/operator_exec` endpoint
can invoke it. See orchestrator/api/operator_exec.py.
"""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class OperatorExecInput(BaseModel):
    argv: list[str] = Field(min_length=1, description="Command argv to run in the sandbox")
    target: str = Field(min_length=1, description="In-scope target host or URL")
    justification: str = Field(
        min_length=20,
        max_length=2000,
        description=(
            "WHY the operator is running this. Audit-visible. Renders in the "
            "report's audit appendix. Must be at least 20 chars."
        ),
    )
    timeout_seconds: int = Field(default=120, ge=1, le=900)
    bypass_critic: bool = False
    bypass_justification: str | None = Field(
        default=None,
        description=(
            "Required when bypass_critic=true. Operator's reasoning for "
            "skipping the LLM review (e.g., 'I've already verified this on a "
            "lab clone of this host'). Audit-logged."
        ),
    )

    @model_validator(mode="after")
    def _bypass_needs_justification(self) -> OperatorExecInput:
        if self.bypass_critic and (
            not self.bypass_justification or len(self.bypass_justification) < 20
        ):
            raise ValueError(
                "bypass_justification required (min 20 chars) when bypass_critic=true"
            )
        return self


class OperatorExecOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    truncated: bool
    target: str
    justification: str
    bypass_critic: bool
    bypass_justification: str | None = None


class OperatorExec(Tool[OperatorExecInput, OperatorExecOutput]):
    name = "operator_exec"
    description = (
        "OPERATOR-ONLY escape hatch for one-off commands. NEVER invoke this from "
        "an LLM planner; the orchestrator's /operator_exec endpoint is the only "
        "valid invoker. Subject to OPA scope check + PreExploitCritic (unless "
        "bypassed with a second justification). Full audit chain."
    )
    stage = Stage.EXPLOIT
    risk_class = RiskClass.EXPLOIT
    destructive = False  # Per-call; the agent can't reach this tool anyway
    Input: ClassVar[type[BaseModel]] = OperatorExecInput
    Output: ClassVar[type[BaseModel]] = OperatorExecOutput

    def extract_targets(self, input: OperatorExecInput) -> list[str]:
        # The declared `target` is the operator's stated intent, but the argv
        # is what actually produces egress. Previously only `target` was
        # scope-checked, so an in-scope `target` with an out-of-scope host in
        # `argv` (e.g. argv=["curl","http://out-of-scope/"]) slipped past the
        # gate entirely (sandbox-02). Enumerate every host-bearing token in
        # the argv too so the scope policy sees ALL reachable destinations.
        targets = [input.target, *_argv_egress_targets(input.argv)]
        # De-dup, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for t in targets:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def summarize_for_model(
        self, output: OperatorExecOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # OperatorExec is operator-driven; the LLM never sees it in the
        # planner. Still emit a structured summary in case the result
        # flows into a downstream agent's context (e.g. report).
        #
        # NEVER inline stdout/stderr bytes. Operator-supplied
        # commands routinely emit plaintext credentials, password hashes,
        # Kerberos tickets, API tokens, etc. The raw stdout is persisted to
        # the evidence store (with credential scrubbing) — the planner sees
        # only structured counts. Operator opens the blob explicitly.
        bypass = " bypass_critic=Y" if output.bypass_critic else ""
        stdout_lines = output.stdout.count("\n") + (1 if output.stdout else 0)
        stderr_lines = output.stderr.count("\n") + (1 if output.stderr else 0)
        summary = (
            f"{self.name}: target={output.target} exit={output.exit_code} "
            f"dur={output.duration_seconds:.1f}s truncated={output.truncated}"
            f"{bypass}. stdout_lines={stdout_lines} stderr_lines={stderr_lines} "
            f"(content redacted from planner context). "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: OperatorExecInput, ctx: ToolRunContext) -> ToolResult[OperatorExecOutput]:
        from datetime import UTC, datetime

        from sandbox.runner import SandboxRunner

        started = datetime.now(UTC)
        # Build a per-engagement-bound SandboxRunner so the container attaches
        # to the per-engagement network (and inherits L3 nftables when present).
        runner = SandboxRunner(engagement_id=str(ctx.engagement_id))
        # Redact obvious credential patterns from argv for logging.
        redacted = _redact_argv(input.argv)
        result = await runner.run(
            input.argv,
            timeout_seconds=input.timeout_seconds,
            redacted_argv=redacted,
        )
        ended = datetime.now(UTC)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=OperatorExecOutput(
                exit_code=result.exit_code,
                stdout=result.stdout.decode("utf-8", errors="replace")[:50_000],
                stderr=result.stderr.decode("utf-8", errors="replace")[:50_000],
                duration_seconds=result.duration_seconds,
                truncated=result.truncated,
                target=input.target,
                justification=input.justification,
                bypass_critic=input.bypass_critic,
                bypass_justification=input.bypass_justification,
            ),
            exit_code=result.exit_code,
            started_at=iso(started),
            ended_at=iso(ended),
        )


import ipaddress as _ipaddress
import re as _re
from urllib.parse import urlparse as _urlparse

# Host-shaped argv token: a URL, an IPv4 (optionally :port), or a dotted
# hostname (optionally :port). Used to enumerate egress destinations carried
# in an operator's argv so the scope gate covers them.
_HOST_TOKEN_RE = _re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9+.\-]*://)?"  # optional scheme
    r"(?:[^@/]*@)?"  # optional userinfo
    r"(?P<host>\[[0-9a-fA-F:]+\]|[A-Za-z0-9._\-]+)"  # host (v6 in brackets, or v4/name)
    r"(?::\d{1,5})?"  # optional port
    r"(?:/.*)?$"  # optional path
)


# Common network/egress binaries whose host operand we can locate with a
# per-binary, flag-aware grammar. For each binary we encode which short/long
# option flags TAKE A VALUE (so the following token is a flag value, never the
# host). Everything else after the flags is a positional; the host is the first
# host-shaped positional (``nc internalhost 4444`` → ``internalhost``;
# ``ssh -p 2222 internalbox`` → ``internalbox``).
#
# sandboxN-02: without this, a bare single-label out-of-scope host reachable on
# the engagement's internal network slips past the scope gate because the
# generic host heuristic only fires on IPs / dotted names / schemes. Locating
# the real host operand closes that gap (the scope policy then accepts
# in-scope, rejects out-of-scope) WITHOUT surfacing flag operands / subcommands
# / ports as false targets (sandbox3-01).

# Short value-taking flags per binary. A short flag may be bundled with its
# value (``-p2222``); if it appears bare (``-p 2222``) the next token is its
# value and is skipped. Long ``--opt value`` / ``--opt=value`` flags are in the
# corresponding long set.
_VALUE_FLAGS_SHORT: dict[str, frozenset[str]] = {
    "ssh": frozenset({"p", "i", "l", "o", "F", "c", "m", "b", "e", "L", "R", "D", "W", "Q", "J", "w", "S"}),
    "scp": frozenset({"P", "i", "l", "o", "F", "c", "S", "J"}),
    "sftp": frozenset({"P", "i", "o", "F", "c", "S", "B", "D", "l", "R", "b", "J"}),
    "nc": frozenset({"w", "W", "i", "p", "s", "X", "x", "b", "q", "O", "T", "g", "G", "m", "M"}),
    "ncat": frozenset({"w", "i", "p", "s", "g", "G", "m", "c", "e", "o", "x", "O", "T"}),
    "netcat": frozenset({"w", "W", "i", "p", "s", "X", "x", "b", "q"}),
    "socat": frozenset({"b", "s", "T", "l", "x"}),
    "rsync": frozenset({"e", "f", "T", "B", "M", "z"}),
    "curl": frozenset({"X", "H", "d", "F", "u", "A", "e", "b", "c", "o", "T", "E", "K", "m", "y", "Y", "C", "w", "U", "x", "D", "G", "h", "z", "Z"}),
    "wget": frozenset({"O", "o", "a", "P", "U", "t", "T", "Q", "w", "A", "R", "D", "I", "X", "e", "i", "B", "header"}),
    "nmap": frozenset({"p", "e", "g", "S", "D", "T", "b"}),
    "telnet": frozenset({"b", "e", "l", "n"}),
    "ftp": frozenset({}),
    "ping": frozenset({"c", "i", "s", "t", "W", "w", "I", "p", "Q", "S", "l", "M"}),
    "ping6": frozenset({"c", "i", "s", "t", "W", "w", "I", "p", "Q", "S", "l", "M"}),
    "telnet6": frozenset({"b", "e", "l", "n"}),
}

# Long value-taking flags per binary (matched without the leading ``--``).
_VALUE_FLAGS_LONG: dict[str, frozenset[str]] = {
    "curl": frozenset({
        "request", "header", "data", "data-raw", "data-binary", "data-urlencode",
        "form", "user", "user-agent", "referer", "cookie", "cookie-jar", "output",
        "upload-file", "cert", "key", "config", "max-time", "connect-timeout",
        "proxy", "url", "range", "write-out", "resolve", "interface",
    }),
    "wget": frozenset({
        "output-document", "output-file", "append-output", "user-agent", "tries",
        "timeout", "wait", "header", "post-data", "post-file", "directory-prefix",
        "referer", "user", "password", "bind-address", "input-file",
    }),
    "nmap": frozenset({
        "script", "script-args", "script-args-file", "ttl", "data", "data-string",
        "data-length", "ip-options", "spoof-mac", "source-port", "proxies",
        "min-rate", "max-rate", "max-retries", "host-timeout", "scan-delay",
        "max-scan-delay", "min-rtt-timeout", "max-rtt-timeout", "initial-rtt-timeout",
        "min-hostgroup", "max-hostgroup", "min-parallelism", "max-parallelism",
        "exclude", "excludefile", "iflist", "datadir",
    }),
    "rsync": frozenset({
        "rsh", "rsync-path", "exclude", "exclude-from", "include", "include-from",
        "files-from", "filter", "address", "port", "sockopts", "bwlimit",
        "log-file", "log-file-format", "out-format", "password-file", "timeout",
        "contimeout", "block-size", "compress-level", "skip-compress", "backup-dir",
        "suffix", "temp-dir", "compare-dest", "copy-dest", "link-dest", "chmod",
        "chown", "usermap", "groupmap",
    }),
    "ssh": frozenset({}),
    "nc": frozenset({}),
    "ncat": frozenset({"sh-exec", "exec", "wait", "delay", "output", "ssl-cert", "ssl-key", "proxy", "proxy-type", "proxy-auth", "source", "source-port", "max-conns", "idle-timeout"}),
}

# Per-binary value-flag families that are spelled as a prefix + free-form
# suffix and whose following token is a VALUE (often a FILE), never a host.
# nmap's input/output options (``-iL targets.txt``, ``-oA scan``, ``-oN out``)
# don't fit the short-cluster (last-char) model, so match them by prefix
# (sandbox4-04: ``-iL targets.txt`` must NOT surface ``targets.txt``).
_VALUE_FLAG_PREFIXES: dict[str, tuple[str, ...]] = {
    "nmap": ("-iL", "-iR", "-iX", "-oN", "-oX", "-oS", "-oG", "-oA"),
}

# Binaries that use scp/sftp/rsync-style ``[user@]host:path`` remote specs. For
# these we must split a positional on the FIRST ``:`` (that is not part of an
# IPv6 literal) and surface the host part (sandbox4-01).
_REMOTE_SPEC_BINARIES = frozenset({"scp", "sftp", "rsync"})

# Binaries that use socat/openssl-style address specifications
# (``TCP:host:port``, ``OPENSSL:host:port``, bare ``host:port``) rather than a
# bare host positional (sandbox4-02).
_ADDR_SPEC_BINARIES = frozenset({"socat", "openssl"})

# socat/openssl address-spec scheme prefixes that precede ``host:port``.
_ADDR_SPEC_SCHEMES = frozenset({
    "TCP", "TCP4", "TCP6", "UDP", "UDP4", "UDP6", "OPENSSL", "OPENSSL-CONNECT",
    "SSL", "SOCKS4", "SOCKS4A", "SOCKS5", "PROXY", "OPENSSL-LISTEN",
})

# HTTP verbs (curl ``-X`` operands etc.) must never be surfaced as hosts even if
# our flag grammar is imperfect.
_HTTP_VERBS = frozenset({
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT",
})

_NETWORK_BINARIES = frozenset(set(_VALUE_FLAGS_SHORT) | {"telnet6"})


def _binary_basename(tok: str) -> str:
    """Strip any leading path so ``/usr/bin/nc`` matches ``nc``."""
    return tok.rsplit("/", 1)[-1]


def _strip_userinfo_and_port(tok: str) -> str:
    """Return the bare host from a ``user@host`` / ``host:port`` token."""
    if "@" in tok:
        tok = tok.rsplit("@", 1)[-1]
    # IPv6 literal in brackets, optionally with :port after the closing bracket.
    if tok.startswith("["):
        return tok.split("]", 1)[0].lstrip("[")
    # Trailing :port (but not for bare IPv6 without brackets — those have many
    # colons; only strip a single trailing :<digits>).
    m = _re.match(r"^(.*):(\d{1,5})$", tok)
    if m and tok.count(":") == 1:
        return m.group(1)
    return tok


def _is_host_shaped(tok: str) -> bool:
    """True for a token whose *bare* form is a valid hostname/FQDN/IPv4/IPv6.

    Explicitly rejects HTTP verbs, pure numbers, port ranges, and tokens with
    obvious non-host punctuation. Single-label names (``internalbox``) are
    accepted so internal short-name hosts still reach the scope gate, but
    subcommand-looking words are filtered by the verb/number guards plus the
    requirement that the token be host-legal.
    """
    if not tok or tok.startswith("-"):
        return False
    if "://" in tok or "/" in tok or "=" in tok:
        return False
    host = _strip_userinfo_and_port(tok)
    if not host:
        return False
    # IPv6 literal?
    try:
        _ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.upper() in _HTTP_VERBS:
        return False
    # Pure number or port-range (``4444``, ``1-65535``) is not a host.
    if _re.match(r"^\d+(?:-\d+)?$", host):
        return False
    # Host-legal label alphabet: letters/digits/dot/dash/underscore, each label
    # non-empty and not starting/ending with a dash. Allows single-label and
    # FQDN. Underscore is accepted for internal Docker/SMB/NetBIOS service names
    # (``db_server``) which are DNS-illegal but real on internal nets
    # (sandbox4-05).
    if not _re.match(r"^(?=.{1,253}$)[A-Za-z0-9_]([A-Za-z0-9._\-]*[A-Za-z0-9_])?$", host):
        return False
    return True


def _flag_takes_value(binary: str, flag: str) -> bool:
    """True if *flag* (a raw argv token starting with ``-``) consumes the next
    token as its value for *binary*."""
    # Prefix-spelled value-flag families (nmap ``-iL``/``-oA``/...): the flag is
    # the whole prefix token and the next token is its (file) value.
    for prefix in _VALUE_FLAG_PREFIXES.get(binary, ()):
        if flag == prefix:
            return True
    if flag.startswith("--"):
        name = flag[2:].split("=", 1)[0]
        return name in _VALUE_FLAGS_LONG.get(binary, frozenset())
    # Short flag cluster like ``-p`` or ``-vp``. If any char in the cluster is a
    # value-flag and it is the LAST char, the next token is its value. If the
    # value-flag is not last (``-p2222``), the value is bundled (no skip).
    short = _VALUE_FLAGS_SHORT.get(binary, frozenset())
    body = flag[1:]
    if not body:
        return False
    last = body[-1]
    return last in short


def _split_first_colon_outside_ipv6(tok: str) -> tuple[str, str] | None:
    """Split *tok* on the FIRST ``:`` that is not inside an IPv6 ``[...]``
    literal. Returns ``(before, after)`` or ``None`` if there is no such colon.

    Used for scp/sftp/rsync ``[user@]host:path`` remote specs, where the colon
    separates host from path (``user@evil.org:/p`` → ``("user@evil.org", "/p")``).
    """
    if tok.startswith("["):
        end = tok.find("]")
        if end != -1:
            rest = tok[end + 1 :]
            if rest.startswith(":"):
                return tok[: end + 1], rest[1:]
            return None
    idx = tok.find(":")
    if idx == -1:
        return None
    return tok[:idx], tok[idx + 1 :]


def _host_from_remote_spec(tok: str) -> str | None:
    """Extract the host from an scp/sftp/rsync ``[user@]host:path`` remote spec.

    ``user@evil.org:/path`` → ``evil.org``; ``host:dir`` → ``host``;
    ``[2001:db8::1]:/p`` → ``2001:db8::1``. A local path with no colon (or a
    Windows-ish ``C:\\...`` that is not host-shaped) yields ``None`` so local
    files are NOT surfaced as hosts (sandbox4-01).
    """
    if "://" in tok:
        return None
    split = _split_first_colon_outside_ipv6(tok)
    if split is None:
        return None
    before, _after = split
    # Strip optional ``user@`` userinfo.
    hostpart = before.rsplit("@", 1)[-1] if "@" in before else before
    bare = hostpart.strip("[]")
    if not bare:
        return None
    # IPv6 literal in brackets is host-shaped by definition.
    try:
        _ipaddress.ip_address(bare)
        return bare
    except ValueError:
        pass
    if _is_host_shaped(hostpart):
        return _strip_userinfo_and_port(hostpart)
    return None


def _host_from_addr_spec(tok: str) -> str | None:
    """Extract the host from a socat/openssl address spec.

    ``TCP:evil.org:80`` / ``OPENSSL:host:443`` / ``TCP4:host:port`` → the host
    component; bare ``host:port`` → the host (sandbox4-02). Address specs that
    have no host (``-``, ``STDIO``, ``EXEC:/bin/sh``, ``FILE:...``) yield
    ``None``.
    """
    if "://" in tok or "/" in tok:
        return None
    parts = tok.split(":")
    if len(parts) >= 3 and parts[0].upper() in _ADDR_SPEC_SCHEMES:
        # SCHEME:host:port[:...]
        candidate = parts[1]
    elif len(parts) == 2 and parts[1].isdigit():
        # bare host:port
        candidate = parts[0]
    else:
        return None
    bare = candidate.strip("[]")
    try:
        _ipaddress.ip_address(bare)
        return bare
    except ValueError:
        pass
    if _is_host_shaped(candidate):
        return _strip_userinfo_and_port(candidate)
    return None


def _hosts_from_positional(binary: str, tok: str) -> list[str]:
    """Apply the appropriate extractor(s) to a single positional token and
    return every host found. Tries, in order: scheme/URL (handled by the
    generic pass, so skipped here), remote-spec (scp/sftp/rsync), address-spec
    (socat/openssl), and plain host / ``user@host``.
    """
    if "://" in tok:
        # URLs are surfaced by the generic pass (which extracts the URL host).
        return []
    if binary in _REMOTE_SPEC_BINARIES:
        # scp/sftp/rsync: only a ``[user@]host:path`` remote spec (a positional
        # containing ``:``) carries a host. A bare positional with NO colon is a
        # LOCAL file (the SRC/DST on the operator's box), never an egress host —
        # it must NOT be surfaced or legit ``scp file user@inscope:/p`` would
        # scope-check the local filename ``file`` and over-block (sandbox4-01).
        if ":" in tok:
            host = _host_from_remote_spec(tok)
            return [host] if host else []
        return []
    if binary in _ADDR_SPEC_BINARIES:
        host = _host_from_addr_spec(tok)
        return [host] if host else []
    if _is_host_shaped(tok):
        return [_strip_userinfo_and_port(tok)]
    return []


def _flag_value_indices(argv: list[str]) -> frozenset[int]:
    """Return the set of argv indices that are VALUES consumed by a preceding
    value-flag for a known network binary (``nmap -iL targets.txt`` → index of
    ``targets.txt``). The generic host pass uses this to avoid surfacing a
    flag's file/value operand as a host (sandbox4-04: ``-iL targets.txt`` must
    not surface ``targets.txt``). Empty for unknown binaries (we don't model
    their flags, so we don't suppress anything there)."""
    if not argv:
        return frozenset()
    binary = _binary_basename(argv[0])
    if binary not in _NETWORK_BINARIES and binary not in _VALUE_FLAG_PREFIXES:
        return frozenset()
    consumed: set[int] = set()
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("-") and tok != "-":
            if "=" not in tok and _flag_takes_value(binary, tok) and i + 1 < n:
                consumed.add(i + 1)
                i += 2
            else:
                i += 1
            continue
        i += 1
    return frozenset(consumed)


def _host_from_known_binary(argv: list[str]) -> list[str]:
    """Flag-aware host extraction for a known network binary. Walks argv,
    skipping flags and their values, and surfaces host-shaped destinations from
    EVERY remaining positional — not just the first.

    Philosophy (sandbox4): missing a real destination is a scope BYPASS, which
    is worse than over-blocking. So for a known network binary we surface every
    host-shaped positional (``nmap box1 box2`` → both), the host inside an
    scp/sftp/rsync ``[user@]host:path`` remote spec (``scp f user@evil.org:/p``
    → ``evil.org``, never the local file ``f``), and the host inside a
    socat/openssl ``TCP:host:port`` address spec — while the sandbox3-01
    verb/number/range/flag-value guards keep false targets out.
    """
    binary = _binary_basename(argv[0])
    short_flags = _VALUE_FLAGS_SHORT.get(binary)
    long_flags = _VALUE_FLAGS_LONG.get(binary)
    prefixes = _VALUE_FLAG_PREFIXES.get(binary)
    if short_flags is None and long_flags is None and prefixes is None:
        return []

    out: list[str] = []
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("-") and tok != "-":
            if "=" not in tok and _flag_takes_value(binary, tok):
                i += 2  # skip the flag and its value
            else:
                i += 1
            continue
        out.extend(_hosts_from_positional(binary, tok))
        i += 1
    return out


def _argv_egress_targets(argv: list[str]) -> list[str]:
    """Best-effort: pull host/URL tokens out of an arbitrary argv so the scope
    gate can evaluate every destination the command might reach, not just the
    operator's declared `target` (sandbox-02).

    Two passes:

    1. **Flag-aware host pass** (known network binaries). Using a per-binary
       grammar of which option flags consume a value, locate the real host
       operand — even when it is a bare single-label name behind value-flags
       (``ssh -p 2222 internalbox`` → ``internalbox``;
       ``nc -w 3 internalhost 4444`` → ``internalhost``; ``ssh user@host`` →
       ``host``). Flag operands, subcommands (curl ``-X POST``), and
       ports/ranges (``nmap -p 1-65535`` operand) are NOT surfaced
       (sandbox3-01).

    2. **Generic host/URL pass**. Any token that is an IP, a dotted hostname, or
       carries a scheme is surfaced (for curl/wget this captures the URL's
       host, which the scope engine extracts itself).

    For an UNKNOWN binary we never positionally guess: only clearly host-shaped
    standalone tokens (IP / dotted name / ``user@host``) from the generic pass
    are surfaced, never raw subcommand words.
    """
    out: list[str] = []

    if argv:
        out.extend(_host_from_known_binary(argv))

    # Indices that are a known binary's value-flag operand (a file/value, not a
    # host) — the generic pass must skip them so ``nmap -iL targets.txt`` does
    # not surface the dotted list-file ``targets.txt`` (sandbox4-04).
    skip_idx = _flag_value_indices(argv)

    for idx, tok in enumerate(argv):
        if idx in skip_idx:
            continue
        if not tok or tok.startswith("-"):
            continue
        m = _HOST_TOKEN_RE.match(tok)
        if not m:
            continue
        host = m.group("host")
        bracketed = host.strip("[]")
        is_ip = False
        try:
            _ipaddress.ip_address(bracketed)
            is_ip = True
        except ValueError:
            is_ip = False
        has_scheme = "://" in tok
        # Only IP / scheme / dotted-name tokens go through the generic pass;
        # bare single-label words are NOT surfaced here (they'd be subcommands
        # for unknown binaries). The flag-aware pass handles legitimate bare
        # hosts for known binaries.
        if is_ip or has_scheme or "." in host:
            if has_scheme:
                parsed = _urlparse(tok)
                if parsed.hostname:
                    out.append(tok)
            else:
                out.append(tok)
    # De-dup preserving order (the binary pass may overlap the generic pass).
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


_REDACT_FLAGS = {"-p", "--password", "--pass", "-H", "-hash", "--hashes", "--secret"}


def _redact_argv(argv: list[str]) -> list[str]:
    """Best-effort: redact the value after any flag in `_REDACT_FLAGS`,
    plus any `key=value` token where key looks credential-shaped."""
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            out.append("***REDACTED***")
            skip_next = False
            continue
        if tok in _REDACT_FLAGS:
            out.append(tok)
            skip_next = True
            continue
        # Authorization: Bearer / Basic ... headers are common in -H values
        lower = tok.lower()
        if "password=" in lower or "token=" in lower or "secret=" in lower or "authorization:" in lower:
            key = tok.split("=", 1)[0] if "=" in tok else tok.split(":", 1)[0]
            out.append(f"{key}=***REDACTED***" if "=" in tok else f"{key}: ***REDACTED***")
            continue
        out.append(tok)
    return out
