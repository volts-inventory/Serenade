"""World-state graph node and edge schemas.

Every node carries provenance — a list of `tool_run_id`s that produced it.
The orchestrator enforces this on write; agents cannot insert findings
without an evidence chain.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Deterministic node-id namespace.
#
# Node ids for duplicate-risk kinds (Endpoint / Finding / Credential / ...)
# are ``uuid5(namespace, "<kind>:<engagement>:<natural_key>")`` so the same
# logical row produces the same id across parse passes and resume/retry,
# and Neo4j's MERGE-on-id collapses re-writes to a no-op. Legacy uuid4
# ids keep working; new writes use the deterministic form.
#
# Namespace is the zero UUID by convention (also used in tests) — there is no
# semantic reason to pick a different value, and using the well-known zero
# namespace keeps the SHA inputs predictable so a future ops script can
# reproduce expected ids without consulting Serenade source.
_SERENADE_NS: UUID = UUID("00000000-0000-0000-0000-000000000000")


def deterministic_node_id(kind: str, engagement_id: UUID, natural_key: str) -> UUID:
    """Derive a stable UUID for a (kind, engagement, natural_key) triple.

    Same triple → same UUID across processes and restarts; different
    ``engagement_id`` → different UUID so node identity does not leak
    across engagements. The natural-key string is the caller's
    responsibility — choose a representation that uniquely identifies the
    entity *within* its kind (see the per-kind ``natural_key_for_*``
    helpers below).
    """
    if not kind:
        raise ValueError("kind must be a non-empty string")
    return uuid5(_SERENADE_NS, f"{kind}:{engagement_id}:{natural_key}")


def natural_key_for_endpoint(*, webapp_id: UUID | None, method: str, path: str) -> str:
    """Endpoint identity: a (webapp, method, path) triple. ``webapp_id``
    is allowed to be None during pre-resolution; we render it as the
    string ``"-"`` so two endpoints with unknown webapps still collide on
    path+method (the common case during a re-parse of the same ffuf run)."""
    return f"{webapp_id or '-'}|{method.upper()}|{path}"


def natural_key_for_finding(
    *, category: str, affected_node_id: UUID | None, title: str
) -> str:
    """Finding identity: a (category, affected_node, title-hash) triple.

    The full title is hashed (sha256 first-16 chars) so a title with
    URL-bound noise — e.g. ``"nuclei XSS at https://...?id=42"`` — does
    not blow up the natural-key string and so two findings with the
    same conceptual subject collide cleanly. ``affected_node_id`` is the
    one Host / Service / WebApp the finding hangs off; ``None`` is
    allowed for findings that don't yet have an asset edge."""
    title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
    return f"{category}|{affected_node_id or '-'}|{title_hash}"


def natural_key_for_credential(
    *, username: str | None, domain: str | None, source_host_id: UUID | None
) -> str:
    """Credential identity: a (username, domain, source_host) triple.

    ``source_host_id`` is the host the credential was extracted from
    (``None`` for credentials produced by a non-host source — searchsploit,
    a static wordlist). A username/domain pair captured twice from the
    same host should collide on this key (the second parse is a no-op);
    captured from a *different* host stays distinct because lateral
    pivots can validate the same logical credential against multiple
    targets with different evidence chains."""
    return f"{username or '-'}|{domain or '-'}|{source_host_id or '-'}"


def natural_key_for_exploit_candidate(
    *,
    finding_id: UUID | None,
    source: str,
    source_ref: str,
    target_node_id: UUID | None = None,
) -> str:
    """ExploitCandidate identity: a (finding, source, source_ref, target) tuple.

    Two ``searchsploit`` runs against the same finding will produce the
    same candidate set; without a deterministic id they would write two
    copies of every candidate with different ids.

    ws-05: when ``finding_id`` is ``None`` (the producer has no real Finding
    wiring yet) the key would degrade to ``("-", source, source_ref)`` — so two
    *distinct* findings/services that merely share an EDB-ID collapse onto one
    node and last-writer-wins clobbers the first. To keep distinct targets
    distinct, pass ``target_node_id`` (the affected Service/Host id that
    triggered the lookup); it is folded into the key so per-target candidates
    stay separate even before real Finding ids exist. When a real
    ``finding_id`` is available it already distinguishes the candidates and
    ``target_node_id`` is redundant (but harmless)."""
    distinguisher = finding_id or target_node_id or "-"
    return f"{distinguisher}|{source}|{source_ref}"


def natural_key_for_attack_path(*, objective: str, step_node_ids: list[UUID]) -> str:
    """AttackPath identity: (objective, ordered step list).

    A path A → B → C captured twice for the same objective should
    collide. Order matters — A → C → B is a distinct path; we render
    the list with a separator so a pair of paths sharing a prefix do not
    collide accidentally on the hashed key. The hash is sha256-first-16
    because step lists can be long enough to bloat the natural-key
    string."""
    chain = ",".join(str(s) for s in step_node_ids)
    chain_hash = hashlib.sha256(chain.encode("utf-8")).hexdigest()[:16]
    return f"{objective}|{chain_hash}"


def natural_key_for_potential_finding(
    *, candidate_id: UUID | None, source: str, title: str
) -> str:
    """PotentialFinding identity: (candidate, source, title-hash) triple.

    Two passes that promote the same unmatched candidate into a potential
    finding must collide on identity so the graph doesn't accumulate
    duplicates across resume/retry."""
    title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
    return f"{candidate_id or '-'}|{source}|{title_hash}"


# Host / Service / WebApp deterministic identity.
#
# Each kind has a stable natural key (host: engagement_id + ip; service:
# host_id + port + protocol; webapp: host_id + url) so
# ``deterministic_node_id`` collapses re-parses (interrupt-resume,
# nmap-XML re-import, retry) onto the same node id. Without this, RECON's
# resume path would write duplicate nodes with different UUIDs but the
# same identity, breaking the report's "affected assets" pivot and the
# deduplicated topology view.


def natural_key_for_host(engagement_id: UUID, ip: str) -> str:
    """Host identity: a (engagement, ip) pair.

    The same engagement scanning the same IP twice (resume, retry,
    nmap-XML re-import) MUST collide on this key so the second parse is
    a MERGE-on-id no-op. The engagement_id is part of the key so two
    engagements that happen to share an IP (lab fixture, RFC1918
    collision across customers) keep distinct Host identity — the same
    invariant the deterministic-id helpers preserve for other node kinds.
    """
    return f"{engagement_id}:host:{ip}"


def natural_key_for_service(host_id: UUID, port: int, protocol: str) -> str:
    """Service identity: a (host, port, protocol) triple.

    A re-parse of the same nmap service-scan against the same host
    must collide on this key. Protocol is lowercased so the key is
    insensitive to ``"TCP"`` vs ``"tcp"`` differences across parsers
    (the schema's ``proto`` field is already ``Literal["tcp","udp"]``
    but defensive normalization keeps the key robust to upstream
    capitalization drift).
    """
    return f"{host_id}:service:{port}:{protocol.lower()}"


def natural_key_for_webapp(service_id: UUID, base_url: str) -> str:
    """WebApp identity: a (service, base_url) pair.

    A re-probe of the same httpx target produces the same WebApp.
    The base_url is lowercased + rstripped of trailing slash so
    ``https://h/`` and ``https://h`` and ``HTTPS://H`` collide on the
    same WebApp identity. ``service_id`` anchors the WebApp to the
    Service that hosts it; when the Service id cannot be resolved the
    recon agent stamps a deterministic fallback id derived from the
    same probe (see recon._parse_and_persist) so the WebApp identity
    is still stable across re-parses.
    """
    return f"{service_id}:webapp:{base_url.lower().rstrip('/')}"


# ExploitAttempt deterministic identity.
#
# Mirrors the per-kind pattern above. ExploitAttempts are identified by
# (engagement, poc, target_host, parameters) — a re-parse of the same
# curated_poc_exec output against the same target with the same params
# must collide so the attempt is not double-written. We hash the
# parameters blob (sha256 first 16) to keep the natural key compact
# regardless of how large the params dict is.


def natural_key_for_exploit_attempt(
    *,
    engagement_id: UUID,
    poc_id: str,
    target_host_id: UUID,
    parameters: dict[str, Any],
) -> str:
    """ExploitAttempt identity: (engagement, poc, target_host, params-hash).

    The same PoC run against the same target with the same parameters
    is the same attempt; a re-parse of the tool output must produce
    the same id (MERGE-on-id no-op) instead of writing a fresh
    attempt with a new uuid4 each time. Two attempts that share
    poc+target but differ in params (e.g. a different payload string)
    stay distinct.

    We import ``json`` lazily; ``sort_keys=True`` keeps the hash
    deterministic under dict-key reordering.
    """
    import json

    params_blob = json.dumps(parameters, sort_keys=True, default=str)
    params_hash = hashlib.sha256(params_blob.encode("utf-8")).hexdigest()[:16]
    return f"{engagement_id}|{poc_id}|{target_host_id}|{params_hash}"


# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #


class Stage(StrEnum):
    SCOPE_INTAKE = "scope_intake"
    RECON = "recon"
    ENUMERATION = "enumeration"
    VULN_TRIAGE = "vuln_triage"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post_exploit"
    PRIVESC = "privesc"
    LATERAL = "lateral"
    LOOT = "loot"
    REPORT = "report"


class RiskClass(StrEnum):
    RECON = "recon"
    ENUM = "enum"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post-exploit"
    DESTRUCTIVE = "destructive"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EngagementMode(StrEnum):
    RECON_ONLY = "recon_only"  # recon + enum only, no triage/exploit, auto transitions
    ASSESS = "assess"  # full pipeline, no exploitation, chain planning + potential findings, auto transitions
    GUIDED = "guided"  # full pipeline, exploitation allowed with operator approval, phase gates
    AUTONOMOUS = "autonomous"  # full pipeline, auto execution (curated + EDB), no gates


# Backwards-compat alias for code that still references the old name.
ExploitationMode = EngagementMode
_LEGACY_MODE_MAP: dict[str, str] = {
    "passive": "recon_only",
    "normal": "guided",
    "aggressive": "autonomous",
    "lab": "autonomous",
}


def normalize_mode(mode: str) -> str:
    """Map legacy mode names to the new EngagementMode values."""
    return _LEGACY_MODE_MAP.get(mode, mode)


class SessionType(StrEnum):
    SHELL = "shell"
    MSF = "msf"
    WINRM = "winrm"
    SSH = "ssh"
    HTTP_PROXY = "http_proxy"


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #


class WorldStateNode(BaseModel):
    """Common fields for every node in the world-state graph."""

    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)

    id: UUID = Field(default_factory=uuid4)
    engagement_id: UUID
    provenance: list[UUID] = Field(
        default_factory=list,
        description="tool_run_ids that produced or last updated this node",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    first_seen: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    tags: list[str] = Field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.__class__.__name__


# --------------------------------------------------------------------------- #
# Engagement-level                                                             #
# --------------------------------------------------------------------------- #


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cidrs: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    excluded_cidrs: list[str] = Field(default_factory=list)
    excluded_domains: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    allowed_hours_utc: tuple[int, int] | None = None  # (start_hour, end_hour) inclusive


class Engagement(WorldStateNode):
    """Top-level engagement. Not stored in graph; lives in Postgres. Mirrored here for typing."""

    engagement_id: UUID  # self-id alias for graph join
    name: str
    scope: Scope
    objectives: list[str] = Field(default_factory=list)
    mode: EngagementMode = EngagementMode.GUIDED
    time_window: TimeWindow
    token_budget: int = 2_000_000
    wallclock_hours: int = 8
    authorization_artifact_sha256: str = ""


# --------------------------------------------------------------------------- #
# Network                                                                      #
# --------------------------------------------------------------------------- #


class Target(WorldStateNode):
    """Operator-supplied target before discovery resolves it to a Host."""

    value: str  # CIDR | IP | hostname | URL
    target_kind: Literal["cidr", "ip", "hostname", "url"]
    in_scope: bool = True


class Host(WorldStateNode):
    ip: str
    hostnames: list[str] = Field(default_factory=list)
    os_family: Literal["linux", "windows", "macos", "bsd", "network", "unknown"] = "unknown"
    os_detail: str | None = None
    mac: str | None = None
    uptime_seconds: int | None = None


class Service(WorldStateNode):
    host_id: UUID
    port: int = Field(ge=0, le=65535)
    proto: Literal["tcp", "udp"]
    name: str | None = None  # e.g. "http", "ssh"
    product: str | None = None  # e.g. "Apache httpd"
    version: str | None = None
    banner: str | None = None
    tls: bool = False
    cpes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Web                                                                          #
# --------------------------------------------------------------------------- #


class WebApp(WorldStateNode):
    service_id: UUID
    base_url: str
    vhost: str | None = None
    technologies: list[str] = Field(default_factory=list)  # ["nginx", "PHP/7.4", "WordPress"]
    title: str | None = None


class Endpoint(WorldStateNode):
    webapp_id: UUID
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    path: str
    params: list[str] = Field(default_factory=list)
    status_code: int | None = None
    content_type: str | None = None
    requires_auth: bool = False


# --------------------------------------------------------------------------- #
# Identity & access                                                            #
# --------------------------------------------------------------------------- #


class Credential(WorldStateNode):
    username: str | None = None
    # Secret fields are marked ``secret=True`` so they are encrypted at
    # serialization and redacted from LLM-facing slices. Add the marker to
    # any new secret field so it is covered by default (see
    # ``CREDENTIAL_SECRET_FIELDS`` below).
    password: str | None = Field(default=None, json_schema_extra={"secret": True})
    nt_hash: str | None = Field(default=None, json_schema_extra={"secret": True})
    aes256_key: str | None = Field(default=None, json_schema_extra={"secret": True})
    ssh_private_key_ref: str | None = Field(default=None, json_schema_extra={"secret": True})
    domain: str | None = None
    source: str  # "smb_null", "kerberoast", "secretsdump", "default_creds", etc.
    valid: bool | None = None


def _credential_secret_fields() -> tuple[str, ...]:
    """Field names on ``Credential`` marked ``secret=True`` in their schema.

    Single source of truth for which credential fields get encrypted at rest
    and redacted from LLM slices -- derived from the model so adding a marked
    field automatically extends both controls.
    """
    out: list[str] = []
    for name, field in Credential.model_fields.items():
        extra = field.json_schema_extra
        if isinstance(extra, dict) and extra.get("secret") is True:
            out.append(name)
    return tuple(out)


CREDENTIAL_SECRET_FIELDS: tuple[str, ...] = _credential_secret_fields()


class Identity(WorldStateNode):
    sam_name: str
    domain: str
    sid: str | None = None
    identity_kind: Literal["user", "computer", "group", "service"] = "user"
    enabled: bool = True


class Domain(WorldStateNode):
    name: str
    netbios: str | None = None
    domain_controllers: list[UUID] = Field(default_factory=list)  # host ids
    functional_level: str | None = None


# --------------------------------------------------------------------------- #
# Findings & exploitation                                                      #
# --------------------------------------------------------------------------- #


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_run_id: UUID
    blob_id: UUID | None = None
    excerpt: str | None = None  # short quote, MUST come verbatim from blob
    byte_offset: int | None = None


class Finding(WorldStateNode):
    title: str
    description: str
    severity: Severity
    cvss_v3_1: float | None = Field(default=None, ge=0.0, le=10.0)
    cvss_v4: float | None = Field(default=None, ge=0.0, le=10.0)
    cwe: str | None = None
    cve: list[str] = Field(default_factory=list)
    affected_node_ids: list[UUID] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    remediation: str | None = None
    references: list[str] = Field(default_factory=list)


class ExploitCandidate(WorldStateNode):
    """Output of VulnTriageAgent: a candidate for ExploitAgent to consider."""

    finding_id: UUID
    source: Literal["exploitdb", "metasploit", "nuclei", "curated", "manual"]
    source_ref: str  # EDB-ID, module path, template id, manifest id
    match_strength: float = Field(ge=0.0, le=1.0)
    auto_runnable: bool = False  # true iff a curated entry exists with auto_run.allowed
    risk_class: RiskClass
    notes: str | None = None


class ExploitAttempt(WorldStateNode):
    candidate_id: UUID
    target_host_id: UUID
    poc_manifest_id: str  # e.g. "CVE-2021-44228-log4shell"
    poc_manifest_sha256: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outcome: Literal["success", "failure", "error", "skipped"]
    session_id: UUID | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    error_message: str | None = None


class Session(WorldStateNode):
    host_id: UUID
    type: SessionType
    user: str | None = None
    privileged: bool = False
    pid: int | None = None
    transcript_blob_id: UUID | None = None
    active: bool = True
    parent_session_id: UUID | None = None  # for pivot chains


class AttackPath(WorldStateNode):
    """Materialized sequence of edges from initial access to an objective."""

    objective: str  # "domain_admin" | "data_exfil_sim" | etc.
    step_node_ids: list[UUID]  # ordered: Finding | Credential | Session | Identity
    cost: float = 1.0
    feasible: bool = True


class PotentialFinding(WorldStateNode):
    """Unconfirmed finding — a plausible vulnerability or attack chain that was
    not validated via curated PoC execution.

    Sources:
    - Unmatched ExploitCandidates (discovered via searchsploit but no curated
      PoC available to confirm)
    - LLM-reasoned attack chains (the exploit agent identified a plausible
      multi-step path but did not execute it)
    - Destructive attack paths that are never executed (finding-only)

    These appear in the report under a dedicated "Potential Findings" section,
    clearly marked as unconfirmed.
    """

    title: str
    description: str
    severity: Severity
    cvss_v3_1: float | None = Field(default=None, ge=0.0, le=10.0)
    cwe: str | None = None
    cve: list[str] = Field(default_factory=list)
    affected_node_ids: list[UUID] = Field(default_factory=list)
    source: Literal["unmatched_candidate", "attack_chain_plan"] = "unmatched_candidate"
    candidate_id: UUID | None = None
    chain_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason_unconfirmed: str = "no curated PoC available"
    remediation: str | None = None
    references: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Edges                                                                        #
# --------------------------------------------------------------------------- #


class EdgeKind(StrEnum):
    RUNS_ON = "RUNS_ON"  # Service -> Host
    OF_WEBAPP = "OF_WEBAPP"  # Endpoint -> WebApp
    BELONGS_TO = "BELONGS_TO"  # Identity -> Domain
    MEMBER_OF = "MEMBER_OF"  # Identity -> Identity (group)
    HAS_SESSION_ON = "HAS_SESSION_ON"  # Credential -> Host
    LOCAL_ADMIN_ON = "LOCAL_ADMIN_ON"  # Identity -> Host
    EVIDENCED_BY = "EVIDENCED_BY"  # Finding -> EvidenceRef (modeled as prop in practice)
    LEADS_TO = "LEADS_TO"  # Finding -> Finding
    PIVOTED_VIA = "PIVOTED_VIA"  # Session -> Session
    AFFECTS = "AFFECTS"  # Finding -> Host | Service | WebApp
    EXPLOITS = "EXPLOITS"  # ExploitAttempt -> Finding
    YIELDED = "YIELDED"  # ExploitAttempt -> Session | Credential
    DISCOVERED_BY = "DISCOVERED_BY"  # any node -> ToolRun (logical, by provenance)


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EdgeKind
    src_id: UUID
    dst_id: UUID
    engagement_id: UUID
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: list[UUID] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #


NODE_TYPES: dict[str, type[WorldStateNode]] = {
    "Target": Target,
    "Host": Host,
    "Service": Service,
    "WebApp": WebApp,
    "Endpoint": Endpoint,
    "Credential": Credential,
    "Identity": Identity,
    "Domain": Domain,
    "Finding": Finding,
    "ExploitCandidate": ExploitCandidate,
    "ExploitAttempt": ExploitAttempt,
    "Session": Session,
    "AttackPath": AttackPath,
    "PotentialFinding": PotentialFinding,
}
