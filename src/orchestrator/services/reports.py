"""Report rendering — plain async functions.

Extracted from ``orchestrator.api.reports``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from agents.report import DraftFinding, DraftReport, RegressionAppendix
from orchestrator.db import session_scope
from orchestrator.models import AuditLogRow, EngagementRow, RegressionRow
from orchestrator.services.errors import NotFoundError, PermanentFailure, ValidationError
from orchestrator.services.findings import get_repo
from worldstate.queries import attack_paths, coverage, findings_by_severity


async def _build_draft(engagement_id: UUID) -> DraftReport:
    repo = get_repo()
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if not row:
            raise NotFoundError("engagement not found")
        name = row.name
        engagement_meta = {
            "id": str(row.id),
            "name": row.name,
            "mode": row.mode,
            "status": row.status,
        }

        try:
            stmt = (
                select(RegressionRow)
                .where(RegressionRow.engagement_id == engagement_id)
                .where(RegressionRow.applied_at.is_not(None))
                .order_by(RegressionRow.requested_at.asc())
            )
            regression_rows = (await session.execute(stmt)).scalars().all()
        except Exception:
            regression_rows = []

        member_critiques_by_row: dict[str, list[dict[str, Any]]] = {}
        try:
            audit_stmt = (
                select(AuditLogRow)
                .where(AuditLogRow.engagement_id == engagement_id)
                .where(AuditLogRow.action == "regression.council_decided")
            )
            for row_audit in (await session.execute(audit_stmt)).scalars().all():
                p = row_audit.payload or {}
                rid = p.get("regression_row_id")
                members = p.get("member_critiques")
                if rid and isinstance(members, list) and members:
                    member_critiques_by_row[str(rid)] = members
        except Exception:
            member_critiques_by_row = {}

    cov = await coverage(repo, engagement_id)
    severities = await findings_by_severity(repo, engagement_id)
    paths = await attack_paths(repo, engagement_id)

    draft = DraftReport(engagement_id=engagement_id, title=name, coverage=cov)
    for severity, items in severities.items():
        for f in items:
            draft.findings.append(
                DraftFinding(
                    title=str(f.get("title", "Untitled")),
                    description=str(f.get("description", "")),
                    severity=severity,
                    cvss_v3_1=f.get("cvss_v3_1"),
                    cwe=f.get("cwe"),
                    cve=list(f.get("cve") or []),
                    evidence_refs=list(f.get("evidence") or []),
                    remediation=f.get("remediation"),
                    references=list(f.get("references") or []),
                )
            )
    draft.attack_paths = [{"objective": p.get("objective"), "cost": p.get("cost")} for p in paths]
    draft.regressions = [
        RegressionAppendix(
            target_stage=r.target_stage,
            trigger_kind=r.trigger_kind,
            rationale=r.rationale,
            council_decision=r.council_decision or "approve",
            council_rationale=r.council_rationale or "",
            requested_at=r.requested_at.isoformat() if r.requested_at else "",
            member_critiques=member_critiques_by_row.get(str(r.id), []),
        )
        for r in regression_rows
    ]
    _ = engagement_meta
    return draft


async def render_report(engagement_id: UUID, fmt: str) -> tuple[bytes, str]:
    """Render a report in the given format.

    Returns ``(content_bytes, media_type)``.
    """
    from reports.renderer import render_html, render_json, render_markdown, render_sarif

    draft = await _build_draft(engagement_id)

    if fmt == "html":
        return render_html(draft).encode("utf-8") if isinstance(render_html(draft), str) else render_html(draft), "text/html"
    elif fmt == "markdown":
        content = render_markdown(draft)
        return (content.encode("utf-8") if isinstance(content, str) else content), "text/markdown"
    elif fmt == "json":
        content = render_json(draft)
        return (content.encode("utf-8") if isinstance(content, str) else content), "application/json"
    elif fmt == "sarif":
        content = render_sarif(draft)
        return (content.encode("utf-8") if isinstance(content, str) else content), "application/sarif+json"
    elif fmt == "pdf":
        try:
            from reports.renderer import render_pdf

            pdf = render_pdf(draft)
            return pdf, "application/pdf"
        except RuntimeError as e:
            raise PermanentFailure(str(e)) from e
    else:
        raise ValidationError(f"unsupported format: {fmt}")


async def override_finding_veto(
    engagement_id: UUID,
    finding_id: UUID,
    justification: str,
    principal: str = "local",
) -> dict[str, Any]:
    """Operator reinstates a vetoed finding with a justification."""
    from datetime import UTC, datetime

    from orchestrator.ledger import audit

    if len(justification.strip()) < 20:
        raise ValidationError("justification must be at least 20 characters")

    repo = get_repo()
    now = datetime.now(UTC).isoformat()
    ok = await repo.set_finding_override(
        engagement_id=engagement_id,
        finding_id=finding_id,
        justification=justification,
        by=principal,
        at_iso=now,
    )
    if not ok:
        raise NotFoundError("finding not found in engagement")

    await audit(
        engagement_id=engagement_id,
        actor=principal,
        action="report.finding_veto_overridden",
        payload={"finding_id": str(finding_id), "justification": justification},
    )
    return {"finding_id": str(finding_id), "overridden_at": now, "by": principal}
