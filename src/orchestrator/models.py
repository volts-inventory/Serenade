"""SQLAlchemy ORM models — engagement, tool runs, evidence index, llm calls,
audit log, approvals."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from orchestrator.db import Base


class GUID(TypeDecorator):
    """String(36) id column that transparently accepts ``UUID`` *or* ``str``.

    The SQLite/aiosqlite DBAPI cannot bind a Python ``UUID`` parameter
    (``sqlite3.ProgrammingError: type 'UUID' is not supported``). Every
    ledger/audit writer is routinely called with a ``UUID`` engagement /
    tool-run id, and the best-effort ``except Exception`` around those
    writers silently swallowed the bind failure — so audit_log / llm_calls /
    tool_runs / evidence / regressions persisted ZERO rows in production
    (finding db-01).

    This decorator coerces the bound value to its canonical ``str`` form so
    passing a ``UUID`` or a ``str`` both persist identically. The stored
    column type is unchanged (``VARCHAR(36)``) so no schema migration is
    needed — only the Python-side bind behaviour changes. Read-back returns
    the stored ``str`` (callers already compare/concat ids as strings).
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, UUID):
            return str(value)
        return str(value)

    def process_result_value(self, value, dialect):
        return value


class TZDateTime(TypeDecorator):
    """``DateTime(timezone=True)`` that round-trips tz-AWARE UTC.

    SQLite has no native tz-aware datetime, so a value written as
    ``datetime.now(UTC)`` (aware) reads back tz-NAIVE, leaking an
    aware/naive inconsistency to every consumer (finding db-05). This
    decorator normalises any aware value to UTC on write and re-stamps
    ``tzinfo=UTC`` on read so stored datetimes are always aware-UTC on
    read-back. Naive inputs are assumed to already be UTC.

    The audit-chain canonicaliser independently normalises tz away before
    hashing, so this does not disturb the hash chain.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(UTC)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


# Regression-mechanism defaults. Read by the stage controller, the
# BudgetCritic, and the audit row so every layer sees the same numbers.
# The cap is now a per-engagement column on ``EngagementRow``; the module
# constant is a fallback used only when an engagement row predates the
# column (legacy NULL) or when no engagement context is wired (tests).
REGRESSION_BUDGET_MAX_DEFAULT: int = 50
"""Runaway-loop circuit breaker, not a design constraint.

Loop safety lives in :class:`NoveltyCritic` per-fingerprint replay
rejection plus :class:`ScopeCritic`. Real engagements should set this
to ``0`` (unlimited) via the advanced intake form. The 50 default
exists so a bug that causes a producer agent to fire ``REGRESS_TO`` in
a tight loop trips a stop within a few minutes instead of burning the
whole token budget.

A value of ``0`` means "no cap" (matching the existing
``token_budget == 0`` semantics in :class:`BudgetCritic`)."""

REGRESSION_TOKEN_RESERVE_PCT: float = 0.15
"""Fraction of the engagement's ``token_budget`` reserved for regression
work. Regression LLM calls (council deliberation + delta-scoped agent
re-runs) bill against this reserve. Default reserve is 15% of the budget;
the BudgetCritic refuses a new regression when the projected spend would
breach the reserve."""


GLOBAL_AUDIT_CHAIN_KEY = "__global__"
"""Sentinel scope key for audit rows whose ``engagement_id IS NULL``.

All such rows form a single shared hash chain. The literal is used when
computing chain scope / ``seq`` so the global chain behaves exactly like
a per-engagement one (a None engagement_id is never compared against a
real UUID for chain purposes)."""


def _now() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid4())


class EngagementRow(Base):
    __tablename__ = "engagements"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    scope: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    objectives: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # Single config axis. recon_only | assess | guided (default) |
    # autonomous. allowed_risk_classes and approval_policy are derived from
    # this in policy.engagement_dict. Legacy values (passive/normal/
    # aggressive/lab) are normalized at read time via normalize_mode().
    mode: Mapped[str] = mapped_column(String(32), default="guided", nullable=False)
    time_window_start: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    time_window_end: Mapped[datetime] = mapped_column(TZDateTime, nullable=False)
    token_budget: Mapped[int] = mapped_column(Integer, default=2_000_000, nullable=False)
    wallclock_hours: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    authorization_artifact_sha256: Mapped[str] = mapped_column(
        String(64), default="", nullable=False
    )
    # Per-tool token ceilings. Map of tool.name -> max-tokens-per-run.
    per_tool_token_budget: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    current_stage: Mapped[str] = mapped_column(String(32), default="scope_intake", nullable=False)
    # Authoring operator's username at create-time. NULL for
    # rows that predate this column. Drives whose LLM config the
    # snapshot was taken from + future per-engagement ACL.
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Snapshot of the creator's LLM config taken at engagement
    # create-time. Worker reads this to build the router. Keeps engagements
    # reproducible — a user editing their config (or being disabled) doesn't
    # disturb in-flight engagements. Today only the backend selector is
    # snapshotted (shape: {"backend": ...}); no API key is stored here.
    # Empty dict {} means "fall back to env defaults" (used when the
    # creator has no operator_llm_config row).
    llm_config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # Customer-facing kill-switch state. ``abort_requested_at``
    # is set when ``POST /engagements/{id}/abort`` fires; the LangGraph
    # worker checks the engagement row's ``status == "aborted"`` at each
    # step entry point and short-circuits with ``ESCALATE_HUMAN`` if so.
    # ``abort_reason`` is the operator's typed explanation (>=10 chars; the
    # API enforces it). ``aborted_by`` is the authenticated principal that
    # called the endpoint. All three are NULL on engagements that were
    # never aborted; the operator UX renders them on the engagement detail
    # page when ``status=aborted``.
    abort_requested_at: Mapped[datetime | None] = mapped_column(
        TZDateTime, nullable=True
    )
    abort_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    aborted_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Runaway-loop circuit breaker for stage regressions. ``0`` (the
    # default) means unlimited — loop safety lives in NoveltyCritic
    # per-fingerprint replay rejection and ScopeCritic literal
    # authorisation, not in this number. Operators with a paranoid
    # operational posture can set a positive cap via the advanced intake
    # form; ``BudgetCritic`` then VETOes with rationale
    # ``runaway_loop_circuit_breaker`` once ``regressions_used`` meets
    # the cap.
    regression_budget_max: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    # Operator opt-in for destructive actions (tracking issue #112, P2 #3).
    # A separate axis from mode, fail-closed default False. When False,
    # destructive tools are DENIED by the policy gate
    # (``destructive_not_authorized``), destructive curated PoCs are gated
    # (``destructive_without_optin``), and DoS EDB modules are reported-not-run.
    # When True, all three are allowed but always routed through approval
    # (policy returns ``require_approval``) — never executed silently.
    destructive_opt_in: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, onupdate=_now, nullable=False
    )


class WorldStateSnapshotRow(Base):
    """Durable snapshot of the per-engagement world-state graph.

    The world-state graph lives in memory on ``AgentBundle.repo`` during a
    run; it is ALSO snapshotted here after every agent step that wrote nodes
    (``InMemoryGraphRepo.to_json()``) and rehydrated on resume
    (``load_json``). One row per engagement (UPSERT) — the latest snapshot
    wins. ``graph_json`` is the node-link serialization with Credential
    secret fields Fernet-encrypted at rest (see
    ``worldstate.neo4j_repo._encrypt_credential_secrets``).

    Atomicity note: this snapshot write and the LangGraph checkpoint are
    SEPARATE writes (not one transaction). Because world-state node ids are
    deterministic ``uuid5``, a slightly-stale snapshot plus an agent re-run
    re-converges onto the same nodes — eventual consistency, not corruption.
    We deliberately do NOT try to make the two writes atomic.
    """

    __tablename__ = "world_state_snapshots"

    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        primary_key=True,
    )
    graph_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, onupdate=_now, nullable=False
    )


class ToolRunRow(Base):
    __tablename__ = "tool_runs"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool: Mapped[str] = mapped_column(String(100), nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    sandbox_id: Mapped[str] = mapped_column(String(100), default="dev", nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stdout_blob_id: Mapped[str | None] = mapped_column(GUID, nullable=True)
    stderr_blob_id: Mapped[str | None] = mapped_column(GUID, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(100), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    blob_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(100), default="application/octet-stream", nullable=False
    )
    redacted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )


class LLMCallRow(Base):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent: Mapped[str] = mapped_column(String(100), nullable=False)
    site: Mapped[str] = mapped_column(String(32), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )


class AuditLogRow(Base):
    """Append-only audit row, tamper-evident via a per-engagement HMAC hash
    chain (tracking issue #112, P0 #2).

    Each chain is scoped by ``chain_scope``; rows with
    ``engagement_id IS NULL`` share one global chain under the sentinel
    scope ``"__global__"`` (see :data:`GLOBAL_AUDIT_CHAIN_KEY`). Within a
    chain ``seq`` is a 1-based monotonic counter, ``prev_hash`` is the
    predecessor row's ``row_hash`` (NULL only for the genesis row), and
    ``row_hash`` is the master-key HMAC over the canonical serialization
    of the row (including ``prev_hash``, which is what links the chain).

    ``chain_scope`` is a NON-NULL mirror of the chain key: the
    engagement-id string for per-engagement chains, or
    ``GLOBAL_AUDIT_CHAIN_KEY`` for the global (``engagement_id IS NULL``)
    chain. The race backstop ``UNIQUE(chain_scope, seq)`` is placed on
    this column rather than ``engagement_id`` because SQLite (and ANSI
    SQL) treat NULLs as DISTINCT in a unique index — so a
    ``UNIQUE(engagement_id, seq)`` would NOT block two global rows from
    claiming the same ``seq`` slot, silently forking the global chain
    (finding db-02). ``engagement_id`` stays nullable so the FK to
    ``engagements`` and the ``engagement_id IS NULL`` read semantics are
    preserved; the hashed canonical bytes still fold in ``engagement_id``
    (``None`` for global), so existing rows verify unchanged.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        UniqueConstraint("chain_scope", "seq", name="uq_audit_log_chain_seq"),
    )

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str | None] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # Non-null chain key. Mirrors ``engagement_id`` for per-engagement
    # chains; carries ``GLOBAL_AUDIT_CHAIN_KEY`` for global rows so the
    # UNIQUE(chain_scope, seq) backstop actually covers the global chain
    # (NULLs are distinct in a UNIQUE index). Written by
    # ``ledger.append_audit_row``, never by callers.
    chain_scope: Mapped[str] = mapped_column(
        String(36),
        default=GLOBAL_AUDIT_CHAIN_KEY,
        server_default=GLOBAL_AUDIT_CHAIN_KEY,
        nullable=False,
        index=True,
    )
    actor: Mapped[str] = mapped_column(String(100), default="system", nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    ts: Mapped[datetime] = mapped_column(TZDateTime, default=_now, nullable=False)
    # Hash-chain columns. ``seq`` is per-chain monotonic starting at 1;
    # ``prev_hash`` is NULL only for the genesis row; ``row_hash`` is the
    # master-key HMAC over the canonical serialization. These are written
    # by ``ledger.append_audit_row`` (and ``ledger.audit``), never by
    # callers directly.
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class SuppressedToolRow(Base):
    """First-class durable row for tool-suppression state.

    Previously the suppression list lived in two stores:

    1. ``EngagementState.suppressed_tools`` (LangGraph checkpoint) — the
       agent loop's working copy. Read at ``_candidate_tools`` time to
       drop muted tools out of the planner's tool list.
    2. The audit_log (``action="tool.suppressed"``) — reconstructed by
       ``_suppressed_tools_for`` whenever the API needed to surface the
       muted-tools card on the Overview tab.

    Two stores can disagree (a stage-boundary reset clears the checkpoint
    but the audit row stays; a checkpoint rollback restores entries the
    audit log never saw). Promote suppression to its own table so there
    is ONE source of truth — both the agent (write) and the API (read)
    go through the same rows.

    ``cleared_at`` is the soft-delete cursor: stage-boundary reset (H3)
    stamps ``cleared_at = now()`` on every active row for that engagement
    so the same tool can be re-suppressed in a later stage (new row with
    ``cleared_at = NULL``). API reads filter on ``cleared_at IS NULL``.
    The audit log keeps its ``tool.suppressed`` action for historical
    visibility — but the API does not read it anymore.
    """

    __tablename__ = "suppressed_tools"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(String(200), default="unknown", nullable=False)
    stage: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    suppressed_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )
    # NULL while the row is "active"; stamped when the stage-boundary
    # reset clears the row (so a future suppression in a new stage
    # inserts a fresh row instead of mutating history).
    cleared_at: Mapped[datetime | None] = mapped_column(
        TZDateTime, nullable=True
    )




class ApprovalRow(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    requested_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    decision: Mapped[str | None] = mapped_column(String(50), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )


    updated_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, onupdate=_now, nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(100), default="bootstrap", nullable=False)


class ReflectionMemoryRow(Base):
    """Episodic verbal memory for strategy-level lessons.

    Written after agent steps that scored low; the prompt renderer reads
    recent rows to inject a "Lessons learned" section.
    """

    __tablename__ = "reflection_memory"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )


class RegressionRow(Base):
    """First-class audit row for a stage-regression event.

    One row per regression request, written by ``record_regression`` at
    the moment the agent emits the ``StepDecision.REGRESS_TO`` signal.
    The StageRegressionCouncil's verdict is patched into the same row
    when it deliberates, and ``applied_at`` is stamped when the
    controller actually writes the backwards transition.

    Field shape rationale:
    - ``delta_json`` / ``evidence_json`` are JSON so the council can
      read them back during deliberation and the report renderer can
      surface them in the methodology section without a Python-side
      schema.
    - ``council_decision`` / ``council_rationale`` / ``applied_at`` are
      nullable because the row is created BEFORE the council runs. They
      get filled in atomically by the council + controller.
    - No FK to a parent regression row: a regression doesn't have an
      ordering relationship with its peers. The "per-target-stage cap"
      is implemented as a count query at request time, not a
      parent-child relation.
    """

    __tablename__ = "regressions"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=_new_uuid)
    engagement_id: Mapped[str] = mapped_column(
        GUID,
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_at: Mapped[datetime] = mapped_column(
        TZDateTime, default=_now, nullable=False
    )
    target_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    delta_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    evidence_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # sha256 hex of the canonical-JSON delta payload. Indexed jointly
    # with (engagement_id, target_stage, trigger_kind) so
    # ``NoveltyCritic``'s replay query is O(log N). NULL on legacy rows
    # written before the column existed; the critic treats NULL as
    # "no fingerprint match against any current request".
    delta_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Filled in by the council during deliberation; nullable until then.
    council_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    council_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Stamped by the controller when the backwards transition actually
    # fires. Nullable for "requested but not yet applied" rows.
    applied_at: Mapped[datetime | None] = mapped_column(TZDateTime, nullable=True)
