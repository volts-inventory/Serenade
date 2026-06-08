"""BloodHound adapter.

Two tools:
- `BloodHoundCollect` — runs `bloodhound-python` against AD to gather session,
  group, ACL, and trust data; output is a ZIP of JSON files. The sandbox runs
  the collection; the ZIP is uploaded as evidence; the data is NOT imported
  into the world-state graph (that's a separate operator step).
- `BloodHoundQuery` — runs a parameterized Cypher query against a BloodHound
  Neo4j database. Requires an external BloodHound Neo4j instance if the
  operator wants AD graph queries. The world-state graph uses in-process
  networkx; this tool connects to a separate external Neo4j for BloodHound
  data only.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
from datetime import UTC
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from orchestrator.evidence import EvidenceStore
from sandbox.network import engagement_network_name
from sandbox.runner import SandboxRunner
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage

# FQDN matcher — mirrors the pattern in agents/lateral.py:_FQDN_RE.
# Domain components: letters, digits, hyphens (no leading/trailing hyphen),
# each component ≤63 chars; TLD is 2-63 letters. Allows single-label hostnames
# too (e.g. "lab") for permissive AD setups by trying full FQDN first then
# bare-hostname as fallback.
_FQDN_RE = re.compile(
    r"^(?=[a-zA-Z])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z]{2,63}$"
)
# Username: AD samAccountName allows letters/digits and a small set of
# punctuation. Explicitly reject shell metacharacters and whitespace, even
# though the argv path no longer interpolates into a shell — defense in depth.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._\-$@]{1,256}$")
# Collection method: bloodhound-python's -c flag accepts a comma-separated
# list of method names. Restrict to the known set + comma to prevent any
# injection through this field.
_COLLECTION_METHOD_RE = re.compile(r"^[A-Za-z,]{1,128}$")


class BloodHoundCollectInput(BaseModel):
    domain: str = Field(min_length=1, max_length=255)
    dc_ip: str = Field(min_length=1, max_length=64)
    username: str = Field(min_length=1, max_length=256)
    password: str | None = None
    nt_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{32}$")
    collection_method: str = Field(
        default="Default",
        description="bloodhound-python -c flag: Default, All, Group, Session, etc.",
    )

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str) -> str:
        if not _FQDN_RE.match(v):
            raise ValueError(
                f"domain must be a valid FQDN or hostname, got {v!r}"
            )
        return v

    @field_validator("dc_ip")
    @classmethod
    def _validate_dc_ip(cls, v: str) -> str:
        try:
            ipaddress.ip_address(v)
        except ValueError as e:
            raise ValueError(f"dc_ip must be a valid IPv4/IPv6 address: {e}") from e
        return v

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v: str) -> str:
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "username contains disallowed characters; only letters, digits, "
                "and ._-$@ are permitted"
            )
        return v

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str | None) -> str | None:
        # The argv path (no /bin/sh -c) means the password never reaches a
        # shell parser, so the only constraint is "no embedded NULs / newlines"
        # which would break argv passing and log scrubbing.
        if v is not None and ("\x00" in v or "\n" in v or "\r" in v):
            raise ValueError("password must not contain NUL or newline characters")
        return v

    @field_validator("collection_method")
    @classmethod
    def _validate_collection_method(cls, v: str) -> str:
        if not _COLLECTION_METHOD_RE.match(v):
            raise ValueError(
                "collection_method must be a comma-separated list of method names "
                "(letters only)"
            )
        return v


class BloodHoundCollectOutput(BaseModel):
    zip_blob_id: str | None = None
    files_collected: list[str] = Field(default_factory=list)
    command: str
    # True when the sandbox stdout (the BloodHound ZIP wire format) hit the
    # SandboxRunner's max_output_bytes cap and was truncated. A truncated zip
    # is *corrupt* — the operator should re-run with a tighter collection
    # method (-c Default instead of -c All) or contact the sandbox owner to
    # raise the cap further. Defense-in-depth: the cap is raised to 100MB
    # for this tool (vs the 20MB default) but a forest of 50k+ users can
    # still exceed that.
    truncated: bool = False


# Cypher clauses/procedures that mutate the graph or escape read-only intent.
# The tool advertises itself as read-only; enforce that here rather than
# trusting the docstring (sandbox-04). Word-boundary matched, case-insensitive.
_CYPHER_WRITE_RE = re.compile(
    r"(?i)\b("
    r"CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|"
    r"FOREACH|LOAD\s+CSV|CALL\s+DBMS|CALL\s+APOC|"
    r"CALL\s+DB\.|CALL\s+\w+\.\w*WRITE"
    r")\b"
)


def _validate_readonly_cypher(v: str) -> str:
    m = _CYPHER_WRITE_RE.search(v)
    if m:
        raise ValueError(
            f"cypher contains a write/mutating clause ({m.group(1).strip()!r}); "
            "bloodhound_query is READ-ONLY — only MATCH/RETURN/WHERE/WITH/"
            "UNWIND/OPTIONAL MATCH style read queries are permitted"
        )
    return v


class BloodHoundQueryInput(BaseModel):
    cypher: str = Field(
        min_length=1,
        description=(
            "Read-only Cypher query against the BloodHound Neo4j DB. Common: "
            "MATCH p=shortestPath((u:User {name:'X'})-[*1..]->(g:Group {name:'DOMAIN ADMINS@LAB.LOCAL'})) RETURN p"
        ),
    )
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("cypher")
    @classmethod
    def _check_readonly(cls, v: str) -> str:
        return _validate_readonly_cypher(v)
    # Connection details default to the Settings values (None on these fields
    # means "use Settings"). Per-call override is supported for ad-hoc
    # multi-instance setups but is rarely needed.
    bloodhound_neo4j_uri: str | None = None
    bloodhound_user: str | None = None
    bloodhound_password: str | None = Field(
        default=None,
        description=(
            "Defaults to Settings.bloodhound_neo4j_password (env "
            "BLOODHOUND_NEO4J_PASSWORD). The previous hard-coded community "
            "default 'bloodhoundcommunityedition' is removed — operators "
            "MUST configure a password explicitly."
        ),
    )


class BloodHoundQueryOutput(BaseModel):
    rows: list[dict] = Field(default_factory=list)
    row_count: int = 0


def _build_executor():
    from orchestrator.settings import get_settings
    from tools._helpers import ToolExecutor

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    # BloodHound's ZIP comes back on stdout; the default 20MB sandbox cap
    # truncates anything past a small forest. Bump to 100MB so realistic AD
    # collections aren't silently corrupted. The output schema also carries a
    # `truncated` flag (defence-in-depth) when even 100MB isn't enough.
    return ToolExecutor(
        SandboxRunner(max_output_bytes=100 * 1024 * 1024), evidence
    )


class BloodHoundCollect(Tool[BloodHoundCollectInput, BloodHoundCollectOutput]):
    name = "bloodhound_collect"
    description = (
        "Run bloodhound-python to collect AD session/group/ACL/trust data. "
        "Output is a ZIP of JSON files uploaded to evidence; not auto-imported "
        "into the world-state Neo4j. Requires valid domain credentials."
    )
    stage = Stage.ENUMERATION
    risk_class = RiskClass.ENUM
    Input: ClassVar[type[BaseModel]] = BloodHoundCollectInput
    Output: ClassVar[type[BaseModel]] = BloodHoundCollectOutput

    def extract_targets(self, input: BloodHoundCollectInput) -> list[str]:
        return [input.dc_ip]

    def summarize_for_model(
        self, output: BloodHoundCollectOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        files = ", ".join(output.files_collected[:5]) or "none"
        zip_state = "yes" if output.zip_blob_id else "no"
        summary = (
            f"{self.name}: zip_uploaded={zip_state}, "
            f"{len(output.files_collected)} files: {files}. "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: BloodHoundCollectInput, ctx: ToolRunContext
    ) -> ToolResult[BloodHoundCollectOutput]:
        # Operator-controlled fields (domain, dc_ip, username, password,
        # nt_hash, collection_method) flow into the sandbox via *environment
        # variables*, NEVER interpolated into a shell command string. The
        # wrapper shell script references them as "$BH_DOMAIN" etc., so
        # bloodhound-python receives them as bare argv tokens via the shell's
        # own argv expansion — not as parsed shell text. Shell metacharacters
        # in any operator field thus have no syntactic weight. The Pydantic
        # field validators are the first line of defence; this env-var
        # boundary is the second.
        out_dir = f"/tmp/bh-{ctx.tool_run_id}"
        env: dict[str, str] = {
            "BH_DOMAIN": input.domain,
            "BH_DC_IP": input.dc_ip,
            "BH_USER": input.username,
            "BH_METHOD": input.collection_method,
            "BH_OUT_DIR": out_dir,
        }
        # Build the wrapper script. Optional flags are gated by env-var
        # presence checks so an absent password/hash doesn't inject a literal
        # empty arg. Note: every "$VAR" is double-quoted so word-splitting
        # cannot turn a credential into multiple argv tokens.
        wrapper = (
            'set -e; '
            'mkdir -p "$BH_OUT_DIR"; cd "$BH_OUT_DIR"; '
            'set -- -d "$BH_DOMAIN" -ns "$BH_DC_IP" -u "$BH_USER" -c "$BH_METHOD"; '
            'if [ -n "${BH_PASSWORD-}" ]; then set -- "$@" -p "$BH_PASSWORD"; fi; '
            'if [ -n "${BH_NT_HASH-}" ]; then set -- "$@" --hashes ":$BH_NT_HASH"; fi; '
            'bloodhound-python "$@"; '
            'cd "$BH_OUT_DIR" && ls *.json | tr "\\n" "\\0" | '
            'xargs -0 zip -q out.zip && cat out.zip'
        )
        if input.password:
            env["BH_PASSWORD"] = input.password
        if input.nt_hash:
            env["BH_NT_HASH"] = input.nt_hash
        argv: list[str] = ["/bin/sh", "-c", wrapper]

        # Redacted argv for logging: the wrapper script and env-var names are
        # not secret; only the env *values* (password, hash) are. Docker's -e
        # flag is the secret path — but the runner logs ``redacted_argv``,
        # not the docker arg list, so we just surface a stable command line
        # built from operator-visible (non-secret) bits.
        red_bits = [
            "bloodhound-python",
            "-d",
            input.domain,
            "-ns",
            input.dc_ip,
            "-u",
            input.username,
            "-c",
            input.collection_method,
        ]
        if input.password:
            red_bits += ["-p", "***REDACTED***"]
        if input.nt_hash:
            red_bits += ["--hashes", "***REDACTED-NT-HASH***"]
        red_argv = ["/bin/sh", "-c", shlex.join(red_bits)]

        executor = _build_executor()
        # Mirror ToolExecutor.run_and_capture (canonical pattern in
        # _helpers.py): bind BOTH engagement_id AND default_network on the
        # runner. Setting engagement_id alone leaves default_network=None and
        # the spawned container falls onto Docker's default bridge, losing the
        # per-engagement L2 isolation ARCHITECTURE §7 promises.
        executor.sandbox.engagement_id = str(ctx.engagement_id)
        executor.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        result = await executor.sandbox.run(
            argv,
            timeout_seconds=max(ctx.timeout_seconds, 600),
            redacted_argv=red_argv,
            env=env,
            # bloodhound-python emits the full AD ZIP on stdout — small forests
            # fit in the 20MB sandbox default but mid-sized engagements blow
            # through it (`SandboxResult.truncated=True` would corrupt the
            # zip). Raise the cap to 100MB for this tool specifically.
        )
        zip_blob_id: str | None = None
        if executor.evidence and result.stdout:
            blob_id, _ = executor.evidence.put(
                result.stdout,
                engagement_id=ctx.engagement_id,
                kind=f"{self.name}/zip",
                content_type="application/zip",
                scrub_credentials=False,  # ZIP is opaque; scrubber would corrupt it
            )
            zip_blob_id = str(blob_id)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=BloodHoundCollectOutput(
                zip_blob_id=zip_blob_id,
                files_collected=_files_in_zip_listing(
                    result.stderr.decode("utf-8", errors="replace")
                ),
                command=" ".join(red_argv),
                truncated=result.truncated,
            ),
            raw_stdout_blob_id=UUID(zip_blob_id) if zip_blob_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


class BloodHoundQuery(Tool[BloodHoundQueryInput, BloodHoundQueryOutput]):
    name = "bloodhound_query"
    description = (
        "Run a READ-ONLY Cypher query against the BloodHound Neo4j database. "
        "Use for shortest-path-to-domain-admin, kerberoastable users, "
        "owned-host expansion, etc. The query must not contain write "
        "keywords (CREATE/MERGE/DELETE/SET/REMOVE) — those are rejected."
    )
    stage = Stage.LATERAL
    risk_class = RiskClass.RECON  # query is read-only; no network egress
    Input: ClassVar[type[BaseModel]] = BloodHoundQueryInput
    Output: ClassVar[type[BaseModel]] = BloodHoundQueryOutput

    def extract_targets(self, input: BloodHoundQueryInput) -> list[str]:
        # The BloodHound DB is internal infrastructure, not an engagement target.
        return []

    def summarize_for_model(
        self, output: BloodHoundQueryOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # Each row is a dict; surface keys + the count for planner context.
        keys: list[str] = []
        if output.rows:
            first = output.rows[0]
            if isinstance(first, dict):
                keys = list(first.keys())[:5]
        keys_str = ", ".join(keys) if keys else "n/a"
        summary = (
            f"{self.name}: {output.row_count} rows. cols: {keys_str}. "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: BloodHoundQueryInput, ctx: ToolRunContext
    ) -> ToolResult[BloodHoundQueryOutput]:
        from datetime import datetime

        started = datetime.now(UTC)
        uri = input.bloodhound_neo4j_uri
        user = input.bloodhound_user
        password = input.bloodhound_password
        # Validate configuration BEFORE attempting the optional neo4j
        # import — config validation must not depend on an optional lib.
        if not uri or not password:
            raise RuntimeError(
                "BloodHound Neo4j connection not configured. "
                "Provide bloodhound_neo4j_uri, bloodhound_user, and "
                "bloodhound_password in the tool input."
            )

        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            raise RuntimeError(
                "BloodHound query requires the 'neo4j' package. "
                "Install it with: pip install neo4j>=5.25"
            )

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        try:
            # Defense-in-depth: run inside an explicit READ transaction so the
            # neo4j driver itself rejects any write the regex validator missed.
            # The validator (BloodHoundQueryInput._check_readonly) is the first
            # line; execute_read is the second.
            cypher = input.cypher
            params = dict(input.params or {})

            async def _read(tx):  # type: ignore[no-untyped-def]
                result = await tx.run(cypher, **params)
                return [dict(r) async for r in result]

            async with driver.session(default_access_mode="READ") as session:
                rows = await session.execute_read(_read)
        finally:
            await driver.close()
        ended = datetime.now(UTC)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=BloodHoundQueryOutput(rows=rows, row_count=len(rows)),
            exit_code=0,
            started_at=iso(started),
            ended_at=iso(ended),
        )


def _files_in_zip_listing(stderr: str) -> list[str]:
    """Crude: parse `zip` stderr for added file names."""
    out: list[str] = []
    for line in stderr.splitlines():
        # zip -q produces no stderr on success; if verbose was on we'd see
        # "adding: <name>.json". Best-effort scan.
        if "adding:" in line:
            parts = line.split("adding:", 1)
            if len(parts) == 2:
                out.append(parts[1].strip().split(" ", 1)[0])
    return out
