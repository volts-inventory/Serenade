"""Engagement CRUD + lifecycle — plain async functions.

Extracted from ``orchestrator.api.engagements``.  No FastAPI, no Depends,
no HTTPException.  Callers get domain exceptions from ``services.errors``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db import session_scope
from orchestrator.events.bus import get_bus
from orchestrator.models import (
    ApprovalRow,
    AuditLogRow,
    EngagementRow,
    LLMCallRow,
    RegressionRow,
    SuppressedToolRow,
    ToolRunRow,
)
from orchestrator.services.errors import (
    NotFoundError,
    ValidationError,
)
from worldstate.schema import EngagementMode, Scope, TimeWindow, normalize_mode

_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


# --------------------------------------------------------------------------- #
# DTOs — kept as-is (Pydantic models, not FastAPI-specific)                   #
# --------------------------------------------------------------------------- #


class EngagementCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    scope: Scope
    objectives: list[str] = Field(default_factory=list)
    mode: EngagementMode = EngagementMode.GUIDED
    time_window: TimeWindow | None = None
    token_budget: int = Field(default=2_000_000, ge=0)
    wallclock_hours: int = 8
    authorization_artifact_sha256: str = ""
    per_tool_token_budget: dict[str, int] = Field(default_factory=dict)
    auto_start: bool = False
    regression_budget_max: int = Field(default=0, ge=0, le=10_000)

    @field_validator("authorization_artifact_sha256")
    @classmethod
    def _validate_hex_sha256(cls, v: str) -> str:
        if v == "":
            return v
        if not _HEX_SHA256.match(v):
            raise ValueError("authorization_artifact_sha256 must be a 64-char hex sha256 or empty")
        return v.lower()


class EngagementAbortRequest(BaseModel):
    reason: str = Field(min_length=10, max_length=2000)


class EngagementUpdate(BaseModel):
    status: Literal["draft", "running", "paused", "completed", "aborted"] | None = None
    mode: EngagementMode | None = None


class EngagementSuppressedToolsRead(BaseModel):
    tool_name: str
    reason: str
    stage: str


class EngagementRead(BaseModel):
    id: UUID
    name: str
    scope: Scope
    objectives: list[str]
    mode: EngagementMode
    time_window: TimeWindow
    token_budget: int
    wallclock_hours: int
    authorization_artifact_sha256: str
    status: str
    current_stage: str
    created_by: str | None = None
    llm_backend_used: str | None = None
    llm_model_used: str | None = None
    created_at: datetime
    updated_at: datetime
    regressions_used: int = 0
    regression_budget_max: int = 0
    suppressed_tools: list[EngagementSuppressedToolsRead] = Field(default_factory=list)


class RegressionRead(BaseModel):
    id: UUID
    requested_at: datetime
    target_stage: str
    trigger_kind: str
    delta: dict
    evidence: list | dict
    rationale: str
    council_decision: str | None
    council_rationale: str | None
    applied_at: datetime | None


class ApprovalRead(BaseModel):
    id: UUID
    engagement_id: UUID
    kind: str
    requested_payload: dict
    decision: str | None
    decided_by: str | None
    decided_at: datetime | None
    created_at: datetime


class EngagementSummary(BaseModel):
    tool_calls_total: int = 0
    tool_calls_auto_approved: int = 0
    tool_calls_critic_vetoed: int = 0
    tool_calls_human_approved_gated: int = 0
    tool_calls_ungated: int = 0
    suppressed_tools_count: int = 0
    llm_cost_usd: float = 0.0
    tool_cost_usd: float = 0.0
    wallclock_minutes_used: float = 0.0
    regressions_used: int = 0
    regressions_blocked: int = 0
    regressions_blocked_by_mode: dict[str, int] = Field(default_factory=dict)
    findings_count: int = 0
    warnings_fired: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    model_config = {"extra": "ignore"}


class BulkApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    kinds: list[str] = Field(min_length=1)
    model_config = {"extra": "ignore"}


class BulkApprovalResult(BaseModel):
    decided: int
    kinds_seen: list[str]


class EngagementCloneRequest(BaseModel):
    name_suffix: str = Field(default=" (retest)", max_length=40)
    new_authorization_artifact_sha256: str | None = None


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


# Per-call-site model fields snapshotted per backend. These key names are the
# canonical contract: the write side (``_build_llm_config_snapshot``), the read
# side (``_row_to_read``), and the router builder (``llm.router.snapshot_from_settings``
# / ``build_router_async``) all agree on these exact keys. See findings llm-04,
# llm-05/db-06.
_SNAPSHOT_ANTHROPIC_MODEL_KEYS = (
    "anthropic_model_planner",
    "anthropic_model_synthesizer",
    "anthropic_model_parser",
    "anthropic_model_triage",
    "anthropic_model_critic",
)
_SNAPSHOT_LOCAL_MODEL_KEYS = (
    "local_model_planner",
    "local_model_synthesizer",
    "local_model_parser",
    "local_model_triage",
    "local_model_critic",
)


def _build_llm_config_snapshot(settings) -> dict:
    """Capture the LLM backend + per-call-site models for an engagement.

    Written under the SAME keys ``_row_to_read`` and the router builder read,
    so the snapshot actually pins the backend/models for the engagement's
    lifetime (findings llm-04, llm-05/db-06).
    """
    snap: dict = {"llm_backend": settings.llm_backend}
    keys = (
        _SNAPSHOT_ANTHROPIC_MODEL_KEYS
        if settings.llm_backend == "anthropic"
        else _SNAPSHOT_LOCAL_MODEL_KEYS
    )
    for key in keys:
        val = getattr(settings, key, None)
        if val is not None:
            snap[key] = val
    return snap


def _row_to_read(
    row: EngagementRow,
    *,
    regressions_used: int = 0,
    suppressed_tools: list[EngagementSuppressedToolsRead] | None = None,
) -> EngagementRead:
    snapshot = getattr(row, "llm_config_snapshot", {}) or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    # Back-compat: very old snapshots stored the backend under "backend".
    llm_backend_used = snapshot.get("llm_backend") or snapshot.get("backend")
    if llm_backend_used in ("mlx", "ollama"):
        llm_model_used = snapshot.get("local_model_planner") or snapshot.get("mlx_model")
    else:
        llm_model_used = snapshot.get("anthropic_model_planner")
    return EngagementRead(
        id=row.id,
        name=row.name,
        scope=Scope.model_validate(row.scope),
        objectives=row.objectives,
        mode=EngagementMode(normalize_mode(row.mode)),
        time_window=TimeWindow(start=row.time_window_start, end=row.time_window_end),
        token_budget=row.token_budget,
        wallclock_hours=row.wallclock_hours,
        authorization_artifact_sha256=row.authorization_artifact_sha256,
        status=row.status,
        current_stage=row.current_stage,
        created_by=getattr(row, "created_by", None),
        llm_backend_used=llm_backend_used,
        llm_model_used=llm_model_used or None,
        created_at=row.created_at,
        updated_at=row.updated_at,
        regressions_used=regressions_used,
        regression_budget_max=int(getattr(row, "regression_budget_max", 0) or 0),
        suppressed_tools=suppressed_tools or [],
    )


async def _suppressed_tools_for(
    session: AsyncSession, engagement_id: UUID
) -> list[EngagementSuppressedToolsRead]:
    try:
        stmt = (
            select(SuppressedToolRow)
            .where(SuppressedToolRow.engagement_id == engagement_id)
            .where(SuppressedToolRow.cleared_at.is_(None))
            .order_by(SuppressedToolRow.suppressed_at.asc())
        )
        rows = (await session.execute(stmt)).scalars().all()
    except Exception:
        return []
    out: list[EngagementSuppressedToolsRead] = []
    seen: set[str] = set()
    for r in rows:
        if r.tool_name in seen:
            continue
        seen.add(r.tool_name)
        out.append(
            EngagementSuppressedToolsRead(
                tool_name=r.tool_name,
                reason=r.reason or "unknown",
                stage=r.stage or "unknown",
            )
        )
    return out


async def _count_regressions_used(session: AsyncSession, engagement_id: UUID) -> int:
    try:
        stmt = (
            select(func.count())
            .select_from(RegressionRow)
            .where(RegressionRow.engagement_id == engagement_id)
            .where(RegressionRow.applied_at.is_not(None))
        )
        result = await session.execute(stmt)
        return int(result.scalar() or 0)
    except Exception:
        return 0


async def _audit(
    session: AsyncSession, *, engagement_id: UUID | None, action: str, payload: dict
) -> None:
    # Route through the ledger so the row joins the tamper-evident hash
    # chain (seq / prev_hash / row_hash) instead of bypassing it with a
    # bare insert (issue #112, P0 #2).
    from orchestrator import ledger

    await ledger.append_audit_row(
        session, engagement_id=engagement_id, actor="system", action=action, payload=payload
    )


# --------------------------------------------------------------------------- #
# Service functions                                                           #
# --------------------------------------------------------------------------- #


async def create_engagement(
    payload: EngagementCreate, created_by: str = "local"
) -> EngagementRead:
    from orchestrator.settings import get_settings

    async with session_scope() as session:
        if payload.token_budget == 0:
            is_local = get_settings().llm_backend in ("mlx", "ollama")
            if not is_local:
                raise ValidationError(
                    "token_budget=0 (unlimited) requires a local LLM backend."
                )

        snapshot = _build_llm_config_snapshot(get_settings())
        now = datetime.now(UTC)
        tw = payload.time_window or TimeWindow(
            start=now, end=now + timedelta(hours=payload.wallclock_hours)
        )
        row = EngagementRow(
            name=payload.name,
            scope=payload.scope.model_dump(),
            objectives=payload.objectives,
            mode=payload.mode.value,
            time_window_start=tw.start,
            time_window_end=tw.end,
            token_budget=payload.token_budget,
            wallclock_hours=payload.wallclock_hours,
            authorization_artifact_sha256=payload.authorization_artifact_sha256,
            regression_budget_max=payload.regression_budget_max,
            created_by=created_by,
            llm_config_snapshot=snapshot,
        )
        session.add(row)
        await session.flush()
        await _audit(
            session,
            engagement_id=row.id,
            action="engagement.created",
            payload={"name": row.name, "mode": row.mode},
        )
        if payload.auto_start:
            if not row.authorization_artifact_sha256:
                raise ValidationError("auto_start requires authorization_artifact_sha256")
            if not _HEX_SHA256.match(row.authorization_artifact_sha256):
                raise ValidationError("auto_start requires a valid 64-char hex sha256")
            row.status = "running"
            row.current_stage = "recon"
            await _audit(
                session,
                engagement_id=row.id,
                action="engagement.started",
                payload={"via": "auto_start"},
            )
        return _row_to_read(row)


async def list_engagements() -> list[EngagementRead]:
    async with session_scope() as session:
        rows = (
            (await session.execute(select(EngagementRow).order_by(EngagementRow.created_at.desc())))
            .scalars()
            .all()
        )
        counts: dict[UUID, int] = {}
        if rows:
            try:
                ids = [r.id for r in rows]
                stmt = (
                    select(RegressionRow.engagement_id, func.count())
                    .where(RegressionRow.engagement_id.in_(ids))
                    .where(RegressionRow.applied_at.is_not(None))
                    .group_by(RegressionRow.engagement_id)
                )
                for eid, c in (await session.execute(stmt)).all():
                    counts[eid] = int(c)
            except Exception:
                counts = {}
        return [_row_to_read(r, regressions_used=counts.get(r.id, 0)) for r in rows]


async def get_engagement(engagement_id: UUID) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")
        used = await _count_regressions_used(session, engagement_id)
        suppressed = await _suppressed_tools_for(session, engagement_id)
        return _row_to_read(row, regressions_used=used, suppressed_tools=suppressed)


async def update_engagement(
    engagement_id: UUID, payload: EngagementUpdate
) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")

        if payload.status is not None:
            row.status = payload.status
        if payload.mode is not None:
            row.mode = payload.mode.value

        audit_payload = payload.model_dump(exclude_none=True)
        await _audit(
            session,
            engagement_id=row.id,
            action="engagement.updated",
            payload=audit_payload,
        )
        return _row_to_read(row)


async def start_engagement(engagement_id: UUID) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")
        if not row.authorization_artifact_sha256:
            raise ValidationError("authorization artifact required to start")
        if not _HEX_SHA256.match(row.authorization_artifact_sha256):
            raise ValidationError("authorization artifact must be valid hex sha256")
        row.status = "running"
        row.current_stage = "recon"
        await _audit(session, engagement_id=row.id, action="engagement.started", payload={})
        return _row_to_read(row)


async def pause_engagement(engagement_id: UUID) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")
        row.status = "paused"
        await _audit(session, engagement_id=row.id, action="engagement.paused", payload={})
        return _row_to_read(row)


async def resume_engagement(engagement_id: UUID) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")
        if row.status not in ("paused", "running"):
            raise ValidationError(f"cannot resume engagement in status {row.status}")
        row.status = "running"
        await _audit(session, engagement_id=row.id, action="engagement.resumed", payload={})
        return _row_to_read(row)


async def record_terminal_status(
    engagement_id: UUID,
    status: Literal["completed", "failed", "paused", "aborted"],
) -> None:
    """Persist a terminal/parked status for an engagement run (asyncN-01).

    The worker dispatch loop selects ``status='running'`` rows and dispatches
    each exactly once. When the run finishes, *something* must move the row
    out of ``running`` or the next poll re-dispatches it from a fresh state
    forever. ``run_engagement`` calls this in its finally with the run outcome:
    ``completed`` on success, ``failed`` on exception, ``paused`` when the
    graph interrupted, ``aborted`` when an abort was observed.

    Idempotent / non-clobbering: a row already in a terminal state
    (``completed``/``failed``/``aborted``) is left untouched. In particular an
    operator ``abort`` that landed mid-run must not be overwritten by a late
    ``completed``/``paused``. Missing rows are a no-op (best-effort).
    """
    _TERMINAL = {"completed", "failed", "aborted"}
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if row is None:
            return
        if row.status in _TERMINAL:
            return
        row.status = status
        await _audit(
            session,
            engagement_id=engagement_id,
            action="engagement.terminal",
            payload={"status": status},
        )


async def abort_engagement(
    engagement_id: UUID, payload: EngagementAbortRequest, aborted_by: str = "local"
) -> EngagementRead:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")

        now = datetime.now(UTC)
        row.status = "aborted"
        row.abort_requested_at = now
        row.abort_reason = payload.reason
        row.aborted_by = aborted_by

        await _audit(
            session,
            engagement_id=row.id,
            action="engagement.aborted",
            payload={
                "aborted_by": aborted_by,
                "reason": payload.reason,
                "abort_requested_at": now.isoformat(),
            },
        )

        try:
            await get_bus().publish(
                engagement_id=engagement_id,
                kind="engagement.abort_requested",
                payload={
                    "aborted_by": aborted_by,
                    "reason": payload.reason,
                },
            )
        except Exception:
            pass

        return _row_to_read(row)


# ---- approvals ----


async def list_approvals(engagement_id: UUID) -> list[ApprovalRead]:
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(ApprovalRow)
                    .where(ApprovalRow.engagement_id == engagement_id)
                    .order_by(ApprovalRow.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
        return [
            ApprovalRead(
                id=r.id,
                engagement_id=r.engagement_id,
                kind=r.kind,
                requested_payload=r.requested_payload,
                decision=r.decision,
                decided_by=r.decided_by,
                decided_at=r.decided_at,
                created_at=r.created_at,
            )
            for r in rows
        ]


async def decide_approval(
    engagement_id: UUID,
    approval_id: UUID,
    payload: ApprovalDecision,
    decided_by: str = "local",
) -> ApprovalRead:
    async with session_scope() as session:
        row = await session.get(ApprovalRow, approval_id)
        if not row or row.engagement_id != engagement_id:
            raise NotFoundError("approval not found")
        row.decision = payload.decision
        row.decided_by = decided_by
        row.decided_at = datetime.now(UTC)
        await _audit(
            session,
            engagement_id=engagement_id,
            action="approval.decided",
            payload={
                "approval_id": str(approval_id),
                "decision": payload.decision,
                "decided_by": decided_by,
            },
        )
        return ApprovalRead(
            id=row.id,
            engagement_id=row.engagement_id,
            kind=row.kind,
            requested_payload=row.requested_payload,
            decision=row.decision,
            decided_by=row.decided_by,
            decided_at=row.decided_at,
            created_at=row.created_at,
        )


async def bulk_decide_approvals(
    engagement_id: UUID,
    payload: BulkApprovalDecision,
    decided_by: str = "local",
) -> BulkApprovalResult:
    async with session_scope() as session:
        stmt = (
            select(ApprovalRow)
            .where(ApprovalRow.engagement_id == engagement_id)
            .where(ApprovalRow.decision.is_(None))
            .where(ApprovalRow.kind.in_(payload.kinds))
        )
        rows = (await session.execute(stmt)).scalars().all()
        now = datetime.now(UTC)
        seen: set[str] = set()
        for row in rows:
            row.decision = payload.decision
            row.decided_by = decided_by
            row.decided_at = now
            seen.add(row.kind)
        if rows:
            await _audit(
                session,
                engagement_id=engagement_id,
                action="approval.bulk_decided",
                payload={
                    "count": len(rows),
                    "decision": payload.decision,
                    "kinds": sorted(seen),
                },
            )
        return BulkApprovalResult(decided=len(rows), kinds_seen=sorted(seen))


# ---- regressions ----


async def list_regressions(engagement_id: UUID) -> list[RegressionRead]:
    async with session_scope() as session:
        stmt = (
            select(RegressionRow)
            .where(RegressionRow.engagement_id == engagement_id)
            .order_by(RegressionRow.requested_at.desc())
        )
        rows = (await session.execute(stmt)).scalars().all()
        return [
            RegressionRead(
                id=r.id,
                requested_at=r.requested_at,
                target_stage=r.target_stage,
                trigger_kind=r.trigger_kind,
                delta=r.delta_json or {},
                evidence=r.evidence_json or [],
                rationale=r.rationale,
                council_decision=r.council_decision,
                council_rationale=r.council_rationale,
                applied_at=r.applied_at,
            )
            for r in rows
        ]


# ---- summary ----


async def get_engagement_summary(engagement_id: UUID) -> EngagementSummary:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")

        summary = EngagementSummary()

        # 1. Tool-call totals
        try:
            n_tools = (
                await session.execute(
                    select(func.count(ToolRunRow.id)).where(
                        ToolRunRow.engagement_id == engagement_id
                    )
                )
            ).scalar() or 0
            summary.tool_calls_total = int(n_tools)
        except Exception:
            summary.tool_calls_total = 0

        # 2. Approval breakdown
        try:
            stmt = (
                select(ApprovalRow.decided_by, ApprovalRow.decision, func.count())
                .where(ApprovalRow.engagement_id == engagement_id)
                .where(ApprovalRow.decision.is_not(None))
                .group_by(ApprovalRow.decided_by, ApprovalRow.decision)
            )
            for decided_by, decision, count in (await session.execute(stmt)).all():
                c = int(count or 0)
                if decided_by == "auto-policy" and decision == "approve":
                    summary.tool_calls_auto_approved += c
                elif decided_by == "critic" and decision == "reject":
                    summary.tool_calls_critic_vetoed += c
                elif decision == "approve":
                    summary.tool_calls_human_approved_gated += c
        except Exception:
            pass

        summary.tool_calls_ungated = max(
            0,
            summary.tool_calls_total
            - summary.tool_calls_auto_approved
            - summary.tool_calls_critic_vetoed
            - summary.tool_calls_human_approved_gated,
        )

        # 3. Suppressed-tools count
        try:
            suppressed = await _suppressed_tools_for(session, engagement_id)
            summary.suppressed_tools_count = len(suppressed)
        except Exception:
            summary.suppressed_tools_count = 0

        # 4. LLM cost
        try:
            cost = (
                await session.execute(
                    select(func.coalesce(func.sum(LLMCallRow.cost_usd), 0.0)).where(
                        LLMCallRow.engagement_id == engagement_id
                    )
                )
            ).scalar()
            summary.llm_cost_usd = round(float(cost or 0.0), 4)
        except Exception:
            summary.llm_cost_usd = 0.0

        # 5. Tool cost
        summary.tool_cost_usd = 0.0

        # 6. Wallclock
        try:
            start = getattr(row, "time_window_start", None)
            if start is not None:
                now = datetime.now(start.tzinfo) if getattr(start, "tzinfo", None) else datetime.utcnow()
                delta = (now - start).total_seconds() / 60.0
                summary.wallclock_minutes_used = round(max(delta, 0.0), 2)
        except Exception:
            summary.wallclock_minutes_used = 0.0

        # 7. Regressions
        try:
            summary.regressions_used = await _count_regressions_used(session, engagement_id)
        except Exception:
            summary.regressions_used = 0

        # 7b. Blocked regressions
        try:
            from agents.report import classify_regression_failure

            stmt_blocked = (
                select(
                    RegressionRow.council_decision,
                    RegressionRow.council_rationale,
                )
                .where(
                    RegressionRow.engagement_id == engagement_id,
                    RegressionRow.council_decision.in_(["veto", "escalate"]),
                )
            )
            blocked_rows = (await session.execute(stmt_blocked)).all()
            blocked_total = 0
            mode_counts: dict[str, int] = {}
            for decision, rationale in blocked_rows:
                blocked_total += 1
                mode = classify_regression_failure(decision, rationale or "") or "other"
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            summary.regressions_blocked = blocked_total
            summary.regressions_blocked_by_mode = mode_counts
        except Exception:
            pass

        # 7c. Threshold-warning debounce keys
        try:
            stmt_warnings = (
                select(AuditLogRow.action, AuditLogRow.payload)
                .where(AuditLogRow.engagement_id == engagement_id)
                .where(
                    AuditLogRow.action.in_(
                        [
                            "engagement.budget_warning",
                            "engagement.veto_rate_warning",
                            "engagement.suppression_spread",
                        ]
                    )
                )
            )
            fired: set[str] = set()
            for action, payload_data in (await session.execute(stmt_warnings)).all():
                payload_data = payload_data or {}
                if action == "engagement.budget_warning":
                    kind = str(payload_data.get("kind") or "")
                    if kind:
                        fired.add(f"budget:{kind}")
                elif action == "engagement.veto_rate_warning":
                    fired.add("veto_rate")
                elif action == "engagement.suppression_spread":
                    stage = str(payload_data.get("stage") or "")
                    if stage:
                        fired.add(f"suppression_spread:{stage}")
            summary.warnings_fired = sorted(fired)
        except Exception:
            summary.warnings_fired = []

        # 8. Findings count
        try:
            from orchestrator.services.findings import get_repo

            repo = get_repo()
            from worldstate.queries import coverage as _coverage_helper

            cov = await _coverage_helper(repo, engagement_id)
            summary.findings_count = int(cov.get("findings", 0) or 0)
        except Exception:
            summary.findings_count = 0

        return summary


# ---- clone ----


async def clone_engagement(
    engagement_id: UUID,
    payload: EngagementCloneRequest,
    cloned_by: str = "local",
) -> EngagementRead:
    async with session_scope() as session:
        src = await session.get(EngagementRow, engagement_id)
        if not src:
            raise NotFoundError("engagement not found")

        auth_sha = payload.new_authorization_artifact_sha256 or ""
        if auth_sha == "":
            raise ValidationError(
                "new_authorization_artifact_sha256 is required: the prior auth "
                "doesn't carry over to a retest."
            )
        if auth_sha and not _HEX_SHA256.match(auth_sha):
            raise ValidationError("new_authorization_artifact_sha256 must be 64-char hex")

        now = datetime.now(UTC)
        new_row = EngagementRow(
            name=(src.name + payload.name_suffix)[:200],
            scope=src.scope,
            objectives=src.objectives,
            mode=src.mode,
            time_window_start=now,
            time_window_end=now + timedelta(hours=src.wallclock_hours or 8),
            token_budget=src.token_budget,
            wallclock_hours=src.wallclock_hours,
            authorization_artifact_sha256=auth_sha,
            llm_config_snapshot=src.llm_config_snapshot or {},
            created_by=cloned_by or src.created_by,
        )
        session.add(new_row)
        await session.flush()
        await _audit(
            session,
            engagement_id=new_row.id,
            action="engagement.cloned",
            payload={
                "source_engagement_id": str(engagement_id),
                "source_name": src.name,
                "name": new_row.name,
            },
        )
        return _row_to_read(new_row)


# ---- LLM info (was in api/llm_info.py and api/me_llm.py) ----


async def get_llm_info() -> dict[str, Any]:
    from orchestrator.settings import get_settings

    s = get_settings()
    is_local = s.llm_backend in ("mlx", "ollama")
    return {
        "backend": s.llm_backend,
        "is_local": is_local,
        "model": s.local_model_planner if is_local else s.anthropic_model_planner,
        "configured": True,
    }


async def get_llm_config() -> dict[str, Any]:
    from orchestrator.settings import get_settings

    s = get_settings()
    return {
        "backend": s.llm_backend,
        "configured": True,
        "is_local": s.llm_backend in ("ollama", "mlx"),
    }
