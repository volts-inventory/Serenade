"""Persistence helpers for the LLM call ledger and tool run ledger.

The router and agents call these directly with an active SQLAlchemy session;
no middleware magic — keeps the data path explicit and auditable.
"""

from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime
from typing import Any, NamedTuple
from uuid import UUID

import structlog
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator import crypto
from orchestrator.db import session_scope
from orchestrator.models import (
    GLOBAL_AUDIT_CHAIN_KEY,
    AuditLogRow,
    EvidenceRow,
    LLMCallRow,
    RegressionRow,
    SuppressedToolRow,
    ToolRunRow,
)

log = structlog.get_logger(__name__)

# Number of optimistic read-compute-insert attempts before giving up on an
# audit-row append. The UNIQUE(engagement_id, seq) constraint is the race
# backstop: a concurrent appender that wins the seq slot raises
# IntegrityError, we re-read the tail and try the next slot.
_AUDIT_APPEND_ATTEMPTS = 3


async def record_llm_call(
    *,
    engagement_id: UUID,
    agent: str,
    site: str,
    backend: str,
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
) -> None:
    """Insert an LLMCallRow. Best-effort: if Postgres is unreachable, log and continue —
    we never block the agent loop on the ledger."""
    try:
        async with session_scope() as s:
            s.add(
                LLMCallRow(
                    engagement_id=engagement_id,
                    agent=agent,
                    site=site,
                    backend=backend,
                    model=model,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )
            )
    except Exception as e:
        log.warning("ledger.llm_call_persist_failed", err=str(e))


async def record_tool_run(
    *,
    tool_run_id: UUID,
    engagement_id: UUID,
    tool: str,
    input_json: dict[str, Any],
    sandbox_id: str,
    started_at: Any,
    ended_at: Any,
    exit_code: int,
    stdout_blob_id: UUID | None,
    stderr_blob_id: UUID | None,
    error: str | None = None,
) -> None:
    try:
        async with session_scope() as s:
            s.add(
                ToolRunRow(
                    id=tool_run_id,
                    engagement_id=engagement_id,
                    tool=tool,
                    input_json=input_json,
                    sandbox_id=sandbox_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    exit_code=exit_code,
                    stdout_blob_id=stdout_blob_id,
                    stderr_blob_id=stderr_blob_id,
                    error=error,
                )
            )
    except Exception as e:
        log.warning("ledger.tool_run_persist_failed", err=str(e))


async def record_evidence(
    *,
    engagement_id: UUID,
    kind: str,
    sha256: str,
    blob_uri: str,
    content_type: str = "application/octet-stream",
) -> None:
    try:
        async with session_scope() as s:
            s.add(
                EvidenceRow(
                    engagement_id=engagement_id,
                    kind=kind,
                    sha256=sha256,
                    blob_uri=blob_uri,
                    content_type=content_type,
                )
            )
    except Exception as e:
        log.warning("ledger.evidence_persist_failed", err=str(e))


async def record_regression(
    *,
    engagement_id: UUID,
    target_stage: str,
    trigger_kind: str,
    delta: dict[str, Any],
    evidence: list[str],
    rationale: str,
    delta_fingerprint: str | None = None,
) -> UUID | None:
    """Persist a regression request as a ``RegressionRow``.

    Best-effort, mirroring ``record_tool_run`` / ``record_evidence``: a
    Postgres outage logs a warning and returns ``None`` instead of
    breaking the agent loop. The stage controller publishes the
    in-memory ``regression.requested`` event regardless so the UI still
    surfaces the attempt; only the durable audit row is lost.

    The row is created BEFORE the StageRegressionCouncil deliberates.
    The council's verdict and the controller's ``applied_at`` stamp are
    patched onto the same row later.

    ``delta_fingerprint`` is the sha256 hex of the canonical-JSON delta;
    the council computes it at evaluate-time and threads it through so
    every row carries a stable replay key. Legacy callers that omit it
    write NULL; ``NoveltyCritic`` treats NULL rows as
    "no fingerprint match against any current request".

    Returns the new row's UUID on success so the caller can later
    UPDATE the same row with the council verdict; ``None`` on
    persistence failure.
    """
    try:
        async with session_scope() as s:
            row = RegressionRow(
                engagement_id=engagement_id,
                target_stage=target_stage,
                trigger_kind=trigger_kind,
                delta_json=delta,
                evidence_json=evidence,
                rationale=rationale,
                delta_fingerprint=delta_fingerprint,
            )
            s.add(row)
            await s.flush()
            return row.id
    except Exception as e:
        log.warning("ledger.regression_persist_failed", err=str(e))
        return None


async def update_regression_with_verdict(
    *,
    row_id: UUID,
    council_decision: str,
    council_rationale: str,
    applied_at: Any | None,
) -> None:
    """Patch the
    council's verdict (and optionally the controller's ``applied_at``
    stamp) onto a previously-inserted ``RegressionRow``.

    Sibling helper to :func:`record_regression`. Best-effort like its
    sibling: a Postgres outage logs a warning and silently returns so
    the agent loop survives. The in-memory event-bus publish in
    ``stage_controller._handle_regress_to`` is the redundant durability
    path for the council verdict — losing the audit row is recoverable;
    crashing the agent loop is not.

    ``applied_at`` is non-``None`` only on the APPROVE branch (when the
    controller commits the backwards transition). VETO / ESCALATE pass
    ``None`` so the row preserves "the council looked at this and
    rejected / escalated, but the transition never fired".
    """
    try:
        async with session_scope() as s:
            row = await s.get(RegressionRow, row_id)
            if row is None:
                log.warning("ledger.regression_update_row_missing", row_id=str(row_id))
                return
            row.council_decision = council_decision
            row.council_rationale = council_rationale
            if applied_at is not None:
                row.applied_at = applied_at
    except Exception as e:
        log.warning("ledger.regression_update_persist_failed", err=str(e))


# --------------------------------------------------------------------------- #
# Audit-log hash chain (tracking issue #112, P0 #2)                            #
# --------------------------------------------------------------------------- #


def _chain_scope(engagement_id: Any) -> str:
    """Return the NON-NULL ``chain_scope`` key for ``engagement_id``.

    Rows with ``engagement_id IS NULL`` form one shared chain under the
    :data:`GLOBAL_AUDIT_CHAIN_KEY` sentinel, so the verifier and writer
    agree on which rows belong together. Real engagement ids are coerced
    to ``str`` (the writers may pass a ``UUID``). This value is stored in
    the non-null ``chain_scope`` column so ``UNIQUE(chain_scope, seq)``
    actually covers the global chain (SQLite treats NULLs as distinct in
    a unique index, so a NULL-scoped constraint would not — finding
    db-02)."""
    return str(engagement_id) if engagement_id else GLOBAL_AUDIT_CHAIN_KEY


def _canonical_chain_scope(engagement_id: Any) -> str:
    """The chain-scope value as it appears inside the *hashed* bytes.

    None engagement ids hash under the literal sentinel so the global
    chain is self-describing in its own canonical form."""
    return str(engagement_id) if engagement_id else GLOBAL_AUDIT_CHAIN_KEY


def _canonical_default(value: Any) -> Any:
    """Injective fallback encoder for non-JSON-native values (crypto-03).

    ``json.dumps(default=str)`` is lossy: it coerces e.g. a ``UUID`` or a
    ``datetime`` to its ``str()`` form *before* hashing, so a value ``X`` and
    the plain string ``str(X)`` produce identical canonical bytes and thus an
    identical ``row_hash`` — two semantically different rows can share one
    HMAC, and a tampered payload that round-trips through ``str()`` re-verifies
    clean.

    Instead we type-TAG every coerced value as ``{"__pytype__": <qualname>,
    "value": <str>}``. A real JSON string can never produce that two-key dict,
    so the encoding stays injective: ``str(X)`` (a bare string) and ``X`` (a
    tagged object) now hash differently. Callers SHOULD pass JSON-native
    payloads; this only governs the rare non-native value.
    """
    return {
        "__pytype__": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def audit_canonical_bytes(
    *,
    seq: int,
    engagement_id: Any,
    actor: str,
    action: str,
    payload: Any,
    ts: datetime,
    prev_hash: str | None,
) -> bytes:
    """Canonical serialization of an audit row, used by BOTH the writer and
    the verifier (and the export renderer) so the HMAC is reproducible.

    ``prev_hash`` is folded into the hashed bytes — that linkage is what
    makes the rows a chain rather than a bag of independently-signed rows.
    Keys are sorted and separators are tight so the encoding is stable
    across Python versions / refactors.

    ``ts`` is normalized to a tz-naive UTC isoformat before hashing. The
    audit_log column is ``DateTime(timezone=True)`` but SQLite drops the
    offset on read-back (a tz-aware ``datetime.now(UTC)`` returns naive on
    the next SELECT), so we MUST canonicalise the timezone away or the
    writer and verifier would disagree on otherwise-identical rows.
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return json.dumps(
        {
            "seq": seq,
            "engagement_id": str(engagement_id) if engagement_id else None,
            "actor": actor,
            "action": action,
            "payload": payload,
            "ts": ts.isoformat(),
            "prev_hash": prev_hash or "",
        },
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_default,
    ).encode()


async def _read_chain_tail(
    s: AsyncSession, engagement_id: Any, *, for_update: bool
) -> AuditLogRow | None:
    """Read the current tail (highest seq) row for the chain.

    ``.with_for_update()`` takes a row lock on Postgres so a concurrent
    appender blocks until we commit; it is a documented NO-OP on the
    ``sqlite+aiosqlite`` backend this system runs on (``SELECT ... FOR
    UPDATE`` is silently ignored — finding async-09). Concurrency safety
    on SQLite therefore rests ENTIRELY on two things, both of which hold:

    1. ``UNIQUE(chain_scope, seq)`` — a concurrent appender that wins a
       ``seq`` slot makes the loser's INSERT raise ``IntegrityError``;
       and because the scope key is non-null (sentinel for the global
       chain), this constraint actually covers the global chain too.
    2. The bounded read-compute-insert retry in
       :func:`append_audit_row` (``_AUDIT_APPEND_ATTEMPTS``), which
       re-reads the tail and tries the next slot on collision.

    The query filters on the non-null ``chain_scope`` column so the
    global chain (engagement_id IS NULL) is selected by sentinel."""
    scope = _chain_scope(engagement_id)
    stmt = (
        select(AuditLogRow)
        .where(AuditLogRow.chain_scope == scope)
        .order_by(AuditLogRow.seq.desc())
        .limit(1)
    )
    if for_update:
        try:
            stmt = stmt.with_for_update()
        except Exception:  # pragma: no cover - dialect without row locking
            pass
    return (await s.execute(stmt)).scalars().first()


async def append_audit_row(
    s: AsyncSession,
    *,
    engagement_id: Any,
    actor: str,
    action: str,
    payload: dict[str, Any],
    ts: datetime | None = None,
    row_id: UUID | str | None = None,
) -> AuditLogRow:
    """Append one tamper-evident row to the engagement's audit hash chain.

    Reads the chain tail, computes ``seq = tail.seq + 1`` (or 1 for
    genesis), links ``prev_hash = tail.row_hash`` (or None), builds the
    canonical bytes and stamps ``row_hash = HMAC(master_key, canonical)``,
    then inserts. The whole read-compute-insert runs inside the caller's
    transaction; a concurrent appender that wins the seq slot trips the
    ``UNIQUE(engagement_id, seq)`` constraint and we retry on a fresh tail
    read.

    ``ts`` is computed ONCE and used for both the stored row and the hash.
    """
    if ts is None:
        ts = datetime.now(UTC)

    last_err: Exception | None = None
    for _ in range(_AUDIT_APPEND_ATTEMPTS):
        tail = await _read_chain_tail(s, engagement_id, for_update=True)
        seq = (tail.seq + 1) if tail is not None else 1
        prev_hash = tail.row_hash if tail is not None else None
        canonical = audit_canonical_bytes(
            seq=seq,
            engagement_id=engagement_id,
            actor=actor,
            action=action,
            payload=payload,
            ts=ts,
            prev_hash=prev_hash,
        )
        row = AuditLogRow(
            engagement_id=engagement_id,
            # Non-null mirror of the chain key. MUST be set here (not left to
            # the column default) so per-engagement rows carry their own
            # engagement id and only the genuine global chain carries the
            # sentinel — otherwise the UNIQUE(chain_scope, seq) backstop and
            # the tail read would lump every engagement under "__global__".
            chain_scope=_chain_scope(engagement_id),
            actor=actor,
            action=action,
            payload=payload,
            ts=ts,
            seq=seq,
            prev_hash=prev_hash,
            row_hash=crypto.audit_hmac(canonical),
        )
        # Honour an explicit pre-minted id (the report-veto path mints one
        # so the finding's ``audit_row_id`` resolves back to this row);
        # otherwise let the ORM column default mint one.
        if row_id is not None:
            row.id = row_id
        try:
            # SAVEPOINT around just this INSERT. ``append_audit_row`` is often
            # called with a SHARED session that already holds the caller's
            # business writes (e.g. services/engagements._audit); a full
            # ``s.rollback()`` on a seq collision would silently discard THOSE
            # too. A nested transaction rolls back only the failed INSERT and
            # leaves the caller's pending state intact.
            async with s.begin_nested():
                s.add(row)
                await s.flush()
            return row
        except IntegrityError as e:
            # Someone else grabbed our seq slot; the savepoint already undid
            # this INSERT. Re-read the tail and try the next slot.
            last_err = e
            continue
    assert last_err is not None
    raise last_err


async def audit(
    *,
    engagement_id: UUID | None,
    actor: str = "system",
    action: str,
    payload: dict[str, Any],
) -> None:
    """Append-only, tamper-evident audit log (issue #112, P0 #2).

    Writes a row to the per-engagement HMAC hash chain via
    :func:`append_audit_row`. Best-effort outer envelope like the other
    ledger writers — a hard failure (Postgres down, schema mismatch) logs
    and continues so the agent loop is never blocked; the structlog event
    below is the redundant write path. But we never insert a row with a
    wrong/NULL ``prev_hash`` when a predecessor exists: the chain compute
    is inside the same transaction and the UNIQUE constraint is the race
    backstop.
    """
    log.info(
        "audit",
        action=action,
        engagement_id=str(engagement_id) if engagement_id else None,
        **payload,
    )
    try:
        async with session_scope() as s:
            await append_audit_row(
                s,
                engagement_id=engagement_id,
                actor=actor,
                action=action,
                payload=payload,
            )
    except Exception as e:
        log.warning("ledger.audit_persist_failed", err=str(e))


class AuditChainResult(NamedTuple):
    """Outcome of :func:`verify_audit_chain`.

    ``ok`` is True only for a fully-intact chain. On failure
    ``first_bad_seq`` is the seq position where verification first broke
    and ``reason`` is one of: ``hash-mismatch`` (a row's content was
    edited), ``broken-link`` (prev_hash doesn't match the predecessor's
    row_hash — a row was inserted or reordered), ``seq-gap`` (a row was
    deleted), ``seq-dup`` (two rows claim the same seq), or
    ``genesis-prev-hash`` (the first row carries a non-NULL prev_hash).
    An empty chain is ``ok=True``.
    """

    ok: bool
    first_bad_seq: int | None
    reason: str


async def verify_audit_chain(engagement_id: UUID | str | None) -> AuditChainResult:
    """Recompute and verify the HMAC hash chain for one chain scope.

    Loads every row for the chain ordered by ``seq`` ascending, then for
    each row: re-derives the canonical bytes, recomputes the master-key
    HMAC and compares it to the stored ``row_hash`` (hash-mismatch =
    edited); checks ``prev_hash`` links to the predecessor's stored
    ``row_hash`` (broken-link = inserted/reordered); and checks ``seq`` is
    contiguous from 1 with no gaps (seq-gap = deleted) or duplicates
    (seq-dup). Works for the ``engagement_id=None`` global chain too.
    """
    scope = _chain_scope(engagement_id)
    async with session_scope() as s:
        stmt = (
            select(AuditLogRow)
            .where(AuditLogRow.chain_scope == scope)
            .order_by(AuditLogRow.seq.asc())
        )
        rows = (await s.execute(stmt)).scalars().all()

    expected_seq = 1
    prev_row_hash: str | None = None
    for row in rows:
        # Contiguity: seq must increase by exactly 1 from 1.
        if row.seq == expected_seq - 1:
            return AuditChainResult(False, row.seq, "seq-dup")
        if row.seq != expected_seq:
            # A higher-than-expected seq means a row was deleted from the
            # middle (gap); the missing slot is ``expected_seq``.
            return AuditChainResult(False, expected_seq, "seq-gap")

        # Genesis must not carry a prev_hash; later rows must.
        if row.seq == 1:
            if row.prev_hash is not None:
                return AuditChainResult(False, row.seq, "genesis-prev-hash")
        elif row.prev_hash != prev_row_hash:
            return AuditChainResult(False, row.seq, "broken-link")

        canonical = audit_canonical_bytes(
            seq=row.seq,
            engagement_id=row.engagement_id,
            actor=row.actor,
            action=row.action,
            payload=row.payload,
            ts=row.ts,
            prev_hash=row.prev_hash,
        )
        # Constant-time MAC comparison (crypto-04): ``!=`` short-circuits on
        # the first differing byte, leaking a timing oracle. ``compare_digest``
        # runs in time independent of where the mismatch is.
        if not hmac.compare_digest(crypto.audit_hmac(canonical), row.row_hash):
            return AuditChainResult(False, row.seq, "hash-mismatch")

        prev_row_hash = row.row_hash
        expected_seq += 1

    return AuditChainResult(True, None, "ok")


# --------------------------------------------------------------------------- #
# suppressed_tools persistence helpers                                        #
# --------------------------------------------------------------------------- #


async def record_tool_suppression(
    *,
    engagement_id: UUID,
    tool_name: str,
    reason: str,
    stage: str,
) -> None:
    """Insert a ``SuppressedToolRow`` for a tool the agent is muting.

    Idempotent against the *active* set: a tool that's already
    suppressed (``cleared_at IS NULL``) for the engagement is a no-op
    so the agent loop can call this on every parse pass without
    inflating the row count. Best-effort — Postgres unreachable logs and
    returns; the agent loop survives without persistence.

    Pairs with the existing ``tool.suppressed`` audit-log write so the
    historical action stays visible; the new table is the live source
    of truth for "is this tool currently muted?" reads.
    """
    try:
        async with session_scope() as s:
            stmt = (
                select(SuppressedToolRow)
                .where(SuppressedToolRow.engagement_id == engagement_id)
                .where(SuppressedToolRow.tool_name == tool_name)
                .where(SuppressedToolRow.cleared_at.is_(None))
                .limit(1)
            )
            existing = (await s.execute(stmt)).scalars().first()
            if existing is not None:
                return
            s.add(
                SuppressedToolRow(
                    engagement_id=engagement_id,
                    tool_name=tool_name,
                    reason=reason or "unknown",
                    stage=stage or "unknown",
                )
            )
    except Exception as e:
        log.warning("ledger.tool_suppression_persist_failed", err=str(e))


async def clear_tool_suppressions(
    *,
    engagement_id: UUID,
    stage: str | None = None,
) -> int:
    """Soft-delete every active suppression row for the engagement.

    ``stage`` is an optional scope filter: when provided, only the
    rows whose ``stage`` matches are cleared (matches the
    stage-boundary reset semantics in the stage controller where a new
    stage clears the previous stage's mutes). When ``stage`` is None,
    every active row is cleared — used by tests / admin tooling.

    Returns the number of rows that transitioned ``cleared_at`` to
    non-null (0 on persistence failure or no active rows).
    """
    cleared_at = datetime.now(UTC)
    try:
        async with session_scope() as s:
            stmt = (
                update(SuppressedToolRow)
                .where(SuppressedToolRow.engagement_id == engagement_id)
                .where(SuppressedToolRow.cleared_at.is_(None))
                .values(cleared_at=cleared_at)
            )
            if stage is not None:
                stmt = stmt.where(SuppressedToolRow.stage == stage)
            res = await s.execute(stmt)
            # ``rowcount`` is the standard SQLAlchemy attribute for
            # UPDATE statements; the typeshed stub on ``Result`` doesn't
            # advertise it because it's specific to the dialect, so we
            # access it dynamically.
            return int(getattr(res, "rowcount", 0) or 0)
    except Exception as e:
        log.warning("ledger.tool_suppression_clear_failed", err=str(e))
        return 0


async def list_active_suppressions(
    *, engagement_id: UUID
) -> list[dict[str, Any]]:
    """Return the active suppression rows for an engagement.

    Read-side helper used by the engagement-detail API. Returns plain
    dicts (not ORM objects) so the caller doesn't need a session
    context. Empty list on persistence failure — the API surface
    treats it as "no active mutes" which is the healthy default.
    """
    try:
        async with session_scope() as s:
            stmt = (
                select(SuppressedToolRow)
                .where(SuppressedToolRow.engagement_id == engagement_id)
                .where(SuppressedToolRow.cleared_at.is_(None))
                .order_by(SuppressedToolRow.suppressed_at.asc())
            )
            rows = (await s.execute(stmt)).scalars().all()
    except Exception as e:
        log.warning("ledger.tool_suppression_list_failed", err=str(e))
        return []
    return [
        {
            "tool_name": r.tool_name,
            "reason": r.reason,
            "stage": r.stage,
            "suppressed_at": r.suppressed_at,
        }
        for r in rows
    ]


async def list_approved_regressions(
    *, engagement_id: UUID, limit: int = 50
) -> list[dict[str, Any]]:
    """Return approved regression rows for an engagement as plain dicts.

    Used by the stage controller to build a compact regression-history
    digest for the LLM prompt. Best-effort: returns an empty list on
    persistence failure so the agent loop is never blocked.
    """
    try:
        async with session_scope() as s:
            stmt = (
                select(RegressionRow)
                .where(RegressionRow.engagement_id == engagement_id)
                .where(RegressionRow.council_decision == "approve")
                .order_by(RegressionRow.requested_at)
                .limit(limit)
            )
            rows = (await s.execute(stmt)).scalars().all()
    except Exception as e:
        log.warning("ledger.approved_regressions_list_failed", err=str(e))
        return []
    return [
        {
            "target_stage": r.target_stage,
            "trigger_kind": r.trigger_kind,
            "delta_json": r.delta_json or {},
        }
        for r in rows
    ]
