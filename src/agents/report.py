"""ReportAgent — final stage. Drafts the report, then runs ReportCritic
on each finding, then hands off to the rendering pipeline.

The rendering pipeline (Jinja2 templates → HTML/PDF/MD/JSON) lives in
`reports/`. This agent is responsible for the *content* (which findings,
narrative, exec summary, attack-path graph) and for review.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from agents.base import BaseAgent, StageContext, StepDecision
from agents.critic import ReportCritic, Verdict
from agents.regression import RegressionRequest, RegressionTriggerKind
from llm.prompts.base import PERSONA_REPORT
from llm.router import CallSite, LLMRequest
from tools.base import Tool
from worldstate.queries import attack_paths, coverage, findings_by_severity
from worldstate.schema import Stage, WorldStateNode

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Evidence excerpt verification — HONEST-ASSURANCE NOTE (ws4-01)
# ---------------------------------------------------------------------------
# The report path verifies each ``EvidenceRef.excerpt`` against the raw blob it
# was supposedly quoted from (see ``_validate_evidence_refs`` ->
# ``orchestrator.provenance.verify_evidence_excerpts``). This catches an agent
# that cites a valid ``tool_run_id`` but attaches a FABRICATED quote.
#
# IMPORTANT — effective verification REQUIRES producers to populate ``blob_id``.
# An excerpt can only be confirmed when its backing blob is fetchable, and a
# blob is only fetchable when the producing ``EvidenceRef`` carries a
# ``blob_id``. TODAY no production producer sets ``blob_id`` (web_enum, the
# nuclei importer, exploit, privesc, loot all emit ``tool_run_id`` + ``excerpt``
# only), so the check currently only ever ENGAGES for refs that cite a fetchable
# blob — in practice tests and any future blob-carrying producer. For every real
# ref the blob is absent, the excerpt cannot be checked, and the ref is KEPT but
# stamped ``excerpt_unverified: True`` (NOT dropped — that was the ws3-01
# regression).
#
# To make the contract "excerpt MUST come verbatim from blob" actively enforced
# rather than dormant, producers must attach a real ``blob_id`` + verbatim
# excerpt. That is a tracked follow-up. Until then the marker keeps the report
# HONEST: a surviving excerpt that could not be blob-confirmed is labelled
# ``excerpt_unverified`` and is NEVER labelled ``excerpt_verified``, so no
# renderer or SARIF/JSON consumer can imply such a quote was verified.
class _CachingBlobFetcher:
    """Wrap a ``(ref) -> bytes | None`` blob resolver with a per-pass cache so a
    given ref's blob is fetched at most ONCE during a single validation pass
    (ws4-02).

    ``verify_evidence_excerpts`` fetches every excerpt-bearing ref's blob to
    check the quote; the report's kept-loop then needs the same blob again to
    decide the verified/unverified stamp. Without caching, the
    EvidenceStore-backed fetcher would re-instantiate the store and re-glob the
    engagement directory a second time for every surviving ref. The cache keys
    on ``blob_id`` when present (multiple refs can share a blob) and otherwise
    on ``id(ref)`` (no-blob refs always resolve to ``None`` but we still avoid a
    redundant call). The instance is short-lived — one per
    ``_validate_evidence_refs`` invocation — so it does not retain blob bytes
    beyond report assembly.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: dict[Any, bytes | None] = {}

    @staticmethod
    def _key(ref: Any) -> Any:
        blob_id = ref.get("blob_id") if isinstance(ref, dict) else getattr(ref, "blob_id", None)
        # Distinct namespaces for blob_id-keyed vs identity-keyed entries so a
        # blob_id that happens to equal an id() can't collide.
        return ("blob", str(blob_id)) if blob_id else ("ref", id(ref))

    async def __call__(self, ref: Any) -> bytes | None:
        key = self._key(ref)
        if key in self._cache:
            return self._cache[key]
        blob = await self._inner(ref)
        self._cache[key] = blob
        return blob


def classify_regression_failure(council_decision: str, council_rationale: str) -> str | None:
    """Classify a regression row's failure mode from its council verdict.

    Returns None for approved regressions. Returns a short tag for vetoed/
    escalated ones so the report and summary can show why a regression
    was blocked.
    """
    if council_decision == "approve":
        return None
    r = (council_rationale or "").lower()
    if "circuit breaker" in r or "budget" in r or "runaway" in r:
        return "budget_exhausted"
    if "replay" in r or "already applied" in r:
        return "novelty_replay"
    if "out of scope" in r or "outside" in r or "excluded" in r or "out_of_scope" in r:
        return "scope_violation"
    if "empty delta" in r or "empty_delta" in r:
        return "empty_delta"
    if "storm" in r:
        return "regression_storm"
    if "token reserve" in r or "token_reserve" in r:
        return "token_reserve_exhausted"
    if council_decision == "escalate":
        return "escalated"
    return "other"


class DraftFinding(BaseModel):
    """In-memory finding draft sent to ReportCritic and the renderer."""

    title: str
    description: str
    severity: str
    cvss_v3_1: float | None = None
    cwe: str | None = None
    cve: list[str] = []
    affected_assets: list[str] = []
    evidence_refs: list[dict[str, Any]] = []
    remediation: str | None = None
    references: list[str] = []
    critic_verdict: str = "pending"
    critic_concern: str = ""
    # An auditor needs to trace from a vetoed-finding entry
    # in the report back to the audit log row that recorded the rejection.
    # These three fields are populated in ``_review_findings`` when a
    # critic vetoes/escalates a finding; they remain empty for accepted
    # findings (which don't need an audit trail at this level).
    audit_row_id: str | None = None
    critic_name: str | None = None
    critic_confidence: float | None = None
    # The tool_run_ids that backed the (vetoed) claim. The
    # auditor uses this to inspect the raw tool output and decide whether
    # the critic was right to veto. Derived from ``evidence_refs`` so it
    # stays in sync.
    tool_run_ids: list[str] = []
    # Operator can reinstate a vetoed finding with a justification.
    # When set, the renderer moves the finding back to the main body with
    # a footnote citing this override.
    override_justification: str | None = None
    override_by: str | None = None
    override_at: str | None = None


class RegressionAppendix(BaseModel):
    """A single applied regression, surfaced in the report's
    "Pipeline regressions" section.

    Only regressions the council APPROVED + the controller committed
    (i.e. ``applied_at`` was non-null) flow into this list. Vetoed and
    escalated requests stay operator-internal; the customer-facing
    report is about "what we did", not "what we tried and the council
    blocked".
    """

    target_stage: str
    trigger_kind: str
    rationale: str
    council_decision: str = "approve"
    council_rationale: str = ""
    requested_at: str = ""  # ISO-8601 timestamp (or "" if missing)
    failure_mode: str | None = None
    # Per-critic breakdown so a customer auditor sees which
    # of the 3 council critics dissented (e.g. 2 approve + 1 veto). Mirrors
    # the shape of ``CouncilVerdict.member_critiques`` after a small
    # projection — one dict per critic with name/verdict/confidence/
    # primary_concern. Empty list when the regression was decided in
    # single-critic mode (legacy / fallback).
    member_critiques: list[dict[str, Any]] = []


class DraftReport(BaseModel):
    engagement_id: UUID
    title: str
    executive_summary: str = ""
    findings: list[DraftFinding] = []
    # Findings vetoed by ReportCritic live here, not in `findings`. The
    # HTML/PDF/MD renderers show them in a clearly-separate "rejected during
    # review" appendix; SARIF/JSON include them as a distinct array. The
    # executive summary counts them so the reader knows what was filtered out.
    vetoed_findings: list[DraftFinding] = []
    attack_paths: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}
    # Applied regressions (target_stage, trigger, rationale,
    # council verdict). The template only renders the section when this
    # list is non-empty, so most engagements stay clean.
    regressions: list[RegressionAppendix] = []
    regressions_blocked: list[RegressionAppendix] = Field(default_factory=list)


class ReportAgent(BaseAgent):
    stage = Stage.REPORT
    system_persona = PERSONA_REPORT
    # Report stage runs a small set of drafting steps then hands
    # off to the rendering pipeline. Few tools, few turns; 6 is plenty
    # and keeps token spend low at the engagement's tail end.
    max_loop_turns = 6

    def __init__(self, *, report_critic: ReportCritic | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.report_critic = report_critic

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        findings = await findings_by_severity(self.repo, ctx.engagement_id)
        paths = await attack_paths(self.repo, ctx.engagement_id)
        # Count findings that lack evidence_refs — these are the ones
        # the critic will catch; surfacing the count in the slice lets
        # the playbook tell the model "evidence is missing, REGRESS_TO
        # the relevant stage to re-back the claim".
        findings_without_evidence = 0
        for sev_bucket in findings.values():
            for f in sev_bucket:
                if not f.get("evidence"):
                    findings_without_evidence += 1
        return {
            "coverage": cov,
            "findings_by_severity": {k: len(v) for k, v in findings.items()},
            "findings_without_evidence": findings_without_evidence,
            "attack_paths": len(paths),
            "turn": ctx.turn,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior report writer's per-turn workflow.

        PERSONA_REPORT: every finding must cite a tool_run_id; CVSS must
        follow described impact; group by asset, not by tool. The
        branching:
          1. findings without evidence → REGRESS_TO the stage that
             should have produced the backing (we don't ship a
             evidence-less claim; the persona forbids it).
          2. evidence-backed findings exist → draft exec summary, run
             ReportCritic, advance.
        """
        cov = slice_.get("coverage", {})
        fbs = slice_.get("findings_by_severity", {})
        n_findings = sum(int(v) for v in fbs.values())
        missing_evidence = int(slice_.get("findings_without_evidence", 0))
        n_paths = int(slice_.get("attack_paths", 0))
        if n_findings == 0:
            return (
                f"PLAYBOOK [report · empty]. State: findings=0, "
                f"attack_paths={n_paths}, coverage={cov}. There are no "
                "findings to report. Draft a one-paragraph executive "
                "summary that LEADS with 'no significant findings' — do not "
                "exaggerate to manufacture severity. REFUSE: marketing "
                "tone, hedge phrases ('might potentially allow'); say "
                "exactly what was observed. Emit a final-state message "
                "after the summary is drafted."
            )
        if missing_evidence > 0:
            return (
                f"PLAYBOOK [report · evidence gap]. State: findings={n_findings}, "
                f"missing_evidence={missing_evidence}, attack_paths={n_paths}. "
                "Some findings cite a tool_run_id that resolves to nothing "
                "concrete OR have no evidence_refs at all. The persona "
                "forbids shipping evidence-less claims. If you find a "
                "finding whose tool_run_ids point at nothing concrete, "
                "emit it via the controller's regression mechanism — the "
                "platform will re-enter the originating stage with "
                "`{finding_id, regenerate_evidence: True}` and you'll get "
                "a chance to re-render the section after. Findings "
                "without evidence belong in the rejected appendix, not "
                "the report body."
            )
        return (
            f"PLAYBOOK [report · drafting]. State: findings={n_findings}, "
            f"by_severity={fbs}, attack_paths={n_paths}. STEP: draft the "
            "executive summary based on coverage and findings counts. Lead "
            "with the most severe finding. Group findings by affected "
            "asset, not by tool. CVSS scores must follow from described "
            "impact — beat the ReportCritic to severity-drift judgement. "
            "Tone: factual, no marketing, no hedge phrases. REFUSE: "
            "including findings the ReportCritic vetoed unless the "
            "operator explicitly overrode. EXIT: once the draft is ready "
            "for the rendering pipeline. ESCALATE: any finding whose "
            "evidence_ref tool_run_id resolves to nothing — the controller "
            "may REGRESS_TO the stage that should have produced the "
            "evidence so the claim can be backed."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        # Report stage is a regression *producer* — when a
        # finding's tool_run_ids reference tool_runs that no longer exist
        # OR the finding has no evidence at all, the playbook asks the
        # controller to re-enter the originating stage to back the claim.
        # The check runs FIRST so a missing-evidence finding doesn't get
        # swallowed by the ADVANCE path. Producer-side budget cap is
        # enforced inside the helper.
        # Reuse the already-built slice from ``step()``.
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)
        regress_request = await self._maybe_build_regression_request(ctx, slice_)
        if regress_request is not None:
            ctx.pending_regression_request = regress_request
            return StepDecision.REGRESS_TO

        # Report stage advances when every asset has at least one
        # evidence-backed finding (or there are no findings at all — the
        # "no significant findings" path). Terminal semantics: the report
        # stage always advances once it runs.
        missing = int(slice_.get("findings_without_evidence", 0))
        if missing > 0 and ctx.turn < 2:
            # Give the planner a chance to surface the gap before
            # advancing — most engagements never hit this branch.
            return StepDecision.CONTINUE
        return StepDecision.ADVANCE

    async def _maybe_build_regression_request(
        self, ctx: StageContext, slice_: dict[str, Any]
    ) -> RegressionRequest | None:
        """Scan findings for evidence gaps and assemble a
        typed :class:`RegressionRequest` so the controller can re-enter
        the originating stage to re-back the claim.

        A finding counts as an evidence gap when EITHER:

        1. it has ``tool_run_ids`` but every id fails the Postgres
           ``validate_evidence_refs`` check (the finding cites tool runs
           that no longer exist), OR
        2. its ``evidence`` list is empty (no backing at all).

        Producer-side budget check (``regressions_used <
        regression_budget_max``) gates the request — the BudgetCritic in
        the council re-checks this, but the producer-side cut keeps a
        known-doomed request off the bus / audit log / Postgres.

        Returns the *first* such finding's regression request; multiple
        gaps regress one at a time (re-enter origin stage, re-render,
        come back, surface the next gap on a subsequent step). Returns
        ``None`` when no gap exists or the budget is exhausted.
        """
        # Producer-side budget cut. ``regression_budget_max == 0`` =
        # unlimited (the default); skip the check. Include in-step
        # pending requests so a future multi-emit step sees the live
        # count.
        cap = ctx.regression_budget_max
        if cap > 0 and (
            ctx.regressions_used + ctx.regressions_pending >= cap
        ):
            return None

        findings_buckets = await findings_by_severity(self.repo, ctx.engagement_id)
        # Flatten severity buckets while preserving the cvss-desc order
        # within each bucket (Neo4j query already orders). Walk severity
        # in priority order so a critical evidence-less finding regresses
        # before a low-severity one.
        ordered: list[dict[str, Any]] = []
        for sev in ("critical", "high", "medium", "low", "info"):
            ordered.extend(findings_buckets.get(sev, []))
        # Append any unknown severity bucket so nothing silently drops.
        seen_sevs = {"critical", "high", "medium", "low", "info"}
        for sev, items in findings_buckets.items():
            if sev not in seen_sevs:
                ordered.extend(items)

        try:
            from orchestrator.provenance import validate_evidence_refs
        except ImportError:
            validate_evidence_refs = None  # type: ignore[assignment]

        for finding in ordered:
            tool_run_ids = self._extract_tool_run_ids(finding)
            evidence = finding.get("evidence") or []
            # Gap kind 1: tool_run_ids cite tool_runs that don't exist.
            gap = False
            if tool_run_ids and validate_evidence_refs is not None:
                try:
                    missing = await validate_evidence_refs(
                        ctx.engagement_id, tool_run_ids
                    )
                except Exception:
                    missing = []
                if missing and len(missing) == len(tool_run_ids):
                    # Every cited tool_run is gone -> evidence cannot be
                    # reconstructed from the report stage; regress.
                    gap = True
            # Gap kind 2: no evidence_refs at all (regardless of tool_run
            # validation outcome). The persona forbids shipping
            # evidence-less claims, so this drives a regression too.
            if not evidence:
                gap = True
            if not gap:
                continue

            finding_id = finding.get("id")
            if finding_id is None:
                continue
            target_stage = self._origin_stage_for_finding(finding)
            # Bump in-step pending counter so a chained
            # producer in the same step sees the live cap.
            ctx.regressions_pending += 1
            return RegressionRequest(
                target_stage=target_stage,
                trigger_kind=RegressionTriggerKind.MISCONFIGURATION_OPENS_PATH,
                delta={
                    "finding_id": str(finding_id),
                    "regenerate_evidence": True,
                },
                evidence=[str(finding_id)],
                rationale=(
                    f"report: finding {finding.get('title', '<untitled>')!r} "
                    "has no resolvable evidence — re-entering the "
                    f"originating stage ({target_stage.value}) with "
                    "regenerate_evidence so the claim can be backed."
                )[:1000],
            )
        return None

    @staticmethod
    def _extract_tool_run_ids(finding: dict[str, Any]) -> list[UUID]:
        """Extract tool_run_id UUIDs from a finding row's ``evidence`` or
        ``tool_run_ids`` field. Tolerates missing fields, non-list values,
        and malformed UUIDs (they're filtered out, not raised on)."""
        ids: list[UUID] = []
        raw_ids: list[Any] = []
        # Two surfaces produce tool_run_ids on a finding row: the
        # ``evidence`` list (EvidenceRef-shaped dicts) and a direct
        # ``tool_run_ids`` list (some agents persist this denormalized).
        evidence = finding.get("evidence")
        if isinstance(evidence, list):
            for ref in evidence:
                if isinstance(ref, dict):
                    tr = ref.get("tool_run_id")
                    if tr is not None:
                        raw_ids.append(tr)
        direct = finding.get("tool_run_ids")
        if isinstance(direct, list):
            raw_ids.extend(direct)
        for r in raw_ids:
            try:
                ids.append(UUID(str(r)))
            except (ValueError, TypeError):
                continue
        return ids

    @staticmethod
    def _origin_stage_for_finding(finding: dict[str, Any]) -> Stage:
        """Pick the stage to regress to for a missing-evidence finding.

        Priority:
          1. The finding's own ``origin_stage`` field (canonical when set).
          2. The finding's ``category`` — exploit-class categories map to
             EXPLOIT; web categories map to ENUMERATION (the web_enum
             agent's stage); others default to ENUMERATION which is the
             closest re-discovery surface.
        """
        origin = finding.get("origin_stage")
        if isinstance(origin, str) and origin:
            try:
                return Stage(origin)
            except ValueError:
                pass
        category = str(finding.get("category") or "").lower()
        if "exploit" in category or "rce" in category or "cve" in category:
            return Stage.EXPLOIT
        if "web" in category or "http" in category or "xss" in category or "sqli" in category:
            return Stage.ENUMERATION
        return Stage.ENUMERATION

    async def _parse_and_persist(
        self,
        ctx: StageContext,
        tool: Tool[Any, Any],
        tool_input: dict[str, Any],
        tool_output: BaseModel,
        tool_run_id: UUID,
    ) -> list[WorldStateNode]:
        return []

    # ---- Report-specific public API ----

    @staticmethod
    async def _fetch_regressions(engagement_id: UUID) -> tuple[list[dict], list[dict]]:
        """Return (approved, blocked) regression rows for the engagement."""
        try:
            from sqlalchemy import select

            from orchestrator.db import session_scope
            from orchestrator.models import RegressionRow

            async with session_scope() as s:
                stmt = select(RegressionRow).where(
                    RegressionRow.engagement_id == engagement_id
                ).order_by(RegressionRow.requested_at)
                rows = (await s.execute(stmt)).scalars().all()

                approved: list[dict] = []
                blocked: list[dict] = []
                for r in rows:
                    entry: dict = {
                        "target_stage": r.target_stage,
                        "trigger_kind": r.trigger_kind,
                        "rationale": r.rationale or "",
                        "council_decision": r.council_decision or "",
                        "council_rationale": r.council_rationale or "",
                        "requested_at": str(r.requested_at) if r.requested_at else "",
                    }
                    if r.applied_at is not None:
                        approved.append(entry)
                    elif r.council_decision in ("veto", "escalate"):
                        entry["failure_mode"] = classify_regression_failure(
                            r.council_decision, r.council_rationale or ""
                        )
                        blocked.append(entry)
                return approved, blocked
        except Exception:
            return [], []

    async def build_draft(self, ctx: StageContext) -> DraftReport:
        findings = await findings_by_severity(self.repo, ctx.engagement_id)
        paths = await attack_paths(self.repo, ctx.engagement_id)
        cov = await coverage(self.repo, ctx.engagement_id)

        draft = DraftReport(
            engagement_id=ctx.engagement_id,
            title=str(ctx.engagement.get("name", "Pentest Report")),
            coverage=cov,
        )

        for severity, items in findings.items():
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

        draft.attack_paths = [
            {"objective": p.get("objective"), "cost": p.get("cost")} for p in paths
        ]

        # Populate approved and blocked regression lists from Postgres.
        approved_regs, blocked_regs = await self._fetch_regressions(ctx.engagement_id)
        draft.regressions = [RegressionAppendix(**r) for r in approved_regs]
        draft.regressions_blocked = [RegressionAppendix(**r) for r in blocked_regs]

        # Drop any EvidenceRefs whose tool_run_id doesn't exist in Postgres
        # for this engagement. Keeps the report from citing 404 evidence links.
        await self._validate_evidence_refs(ctx, draft)

        # Narrative
        draft.executive_summary = await self._draft_exec_summary(ctx, draft)

        # Critic review of each finding
        if self.report_critic:
            await self._review_findings(ctx, draft)

        return draft

    async def _validate_evidence_refs(self, ctx: StageContext, draft: DraftReport) -> None:
        from uuid import UUID

        try:
            from orchestrator.provenance import (
                validate_evidence_refs,
                verify_evidence_excerpts,
            )
        except ImportError:
            return
        base_fetcher = self._make_blob_fetcher(ctx)
        # ws4-02: the blob backing each excerpt-bearing ref was being fetched
        # TWICE per surviving ref — once inside ``verify_evidence_excerpts`` and
        # again in the kept-loop below to decide the ``excerpt_unverified``
        # stamp. For the EvidenceStore-backed fetcher each fetch re-instantiates
        # the store + re-globs the engagement dir, so that doubled the
        # filesystem work on every report. We wrap the fetcher in a per-pass
        # cache so each distinct ref/blob is resolved at most once across both
        # the verification call and the marker loop.
        blob_fetcher = _CachingBlobFetcher(base_fetcher)
        for finding in draft.findings:
            ids: list[UUID] = []
            for ref in finding.evidence_refs:
                tr = ref.get("tool_run_id") if isinstance(ref, dict) else None
                if tr:
                    try:
                        ids.append(UUID(str(tr)))
                    except ValueError:
                        continue
            if ids:
                missing = await validate_evidence_refs(ctx.engagement_id, ids)
                if missing:
                    log.warning(
                        "report.evidence_ref_unknown",
                        finding=finding.title,
                        missing=[str(m) for m in missing],
                    )
                    # Drop refs whose tool_run_id doesn't validate.
                    missing_set = {str(m) for m in missing}
                    finding.evidence_refs = [
                        r
                        for r in finding.evidence_refs
                        if not (
                            isinstance(r, dict)
                            and str(r.get("tool_run_id", "")) in missing_set
                        )
                    ]

            # ws-08 / wsN-03 / ws3-01: a ref can cite a perfectly valid
            # tool_run_id and still carry a *fabricated* ``excerpt`` (the field
            # is documented as "MUST come verbatim from blob" but nothing
            # enforced it). We verify each excerpt against its backing blob, but
            # only DROP a ref when there is POSITIVE evidence of tampering — a
            # blob IS fetchable AND the excerpt demonstrably does not match it.
            #
            # We must NOT fail closed: every production EvidenceRef producer
            # (web_enum, nuclei importer, exploit, privesc, loot) leaves
            # ``blob_id`` unset and writes a synthesized/truncated excerpt, so
            # dropping unverifiable excerpts would strip 100% of legitimate
            # evidence from every report. Refs whose blob can't be fetched are
            # KEPT and marked ``excerpt_unverified`` for transparency.
            if not finding.evidence_refs:
                continue
            bad_idx = await verify_evidence_excerpts(
                finding.evidence_refs, blob_fetcher
            )
            bad_set = set(bad_idx)
            if bad_idx:
                dropped = [finding.evidence_refs[i] for i in bad_idx]
                log.warning(
                    "report.evidence_excerpt_fabricated",
                    finding=finding.title,
                    dropped=len(bad_idx),
                    tool_run_ids=[
                        str(r.get("tool_run_id"))
                        for r in dropped
                        if isinstance(r, dict)
                    ],
                )
            # Rebuild the ref list: drop only the provably-fabricated indices;
            # for surviving refs that carry an excerpt, stamp an EXPLICIT
            # verification status so no consumer can imply an unchecked excerpt
            # was blob-confirmed (ws4-01). A surviving excerpt-bearing ref is:
            #   * ``excerpt_unverified: True``  when its blob couldn't be
            #     fetched (today: ALL real refs — no producer sets ``blob_id``,
            #     see the module docstring); the quote was NOT blob-confirmed.
            #   * ``excerpt_verified: True``     only when a blob WAS fetched and
            #     the excerpt matched it verbatim.
            # The blob fetch here reuses the per-pass cache populated by
            # ``verify_evidence_excerpts`` above, so no ref is fetched twice
            # (ws4-02).
            kept: list[Any] = []
            for i, r in enumerate(finding.evidence_refs):
                if i in bad_set:
                    continue
                if isinstance(r, dict) and r.get("excerpt"):
                    blob = None
                    try:
                        blob = await blob_fetcher(r)
                    except Exception:
                        blob = None
                    if blob is None:
                        # Keep the ref but mark the quote as unverified so the
                        # renderer/auditor knows it wasn't blob-confirmed. Strip
                        # any stale ``excerpt_verified`` an upstream producer may
                        # have set so the marker can't lie.
                        r = {k: v for k, v in r.items() if k != "excerpt_verified"}
                        r = {**r, "excerpt_unverified": True}
                    else:
                        # A blob was fetched AND the excerpt survived the
                        # ``verify_evidence_excerpts`` pass above (it's not in
                        # ``bad_set``), so it matched verbatim: honestly verified.
                        r = {k: v for k, v in r.items() if k != "excerpt_unverified"}
                        r = {**r, "excerpt_verified": True}
                kept.append(r)
            finding.evidence_refs = kept

    def _make_blob_fetcher(self, ctx: StageContext):
        """Build the ``(ref) -> bytes | None`` resolver that
        :func:`verify_evidence_excerpts` uses to fetch the blob backing an
        evidence ref.

        A test (or a caller wanting custom resolution) may set
        ``self._blob_fetcher`` to an async callable to override this; otherwise
        we resolve through the on-disk :class:`EvidenceStore`. The stored
        layout is ``{engagement}/{kind}/{blob_id}`` and an ``EvidenceRef``
        carries no ``kind``, so we glob across the engagement's kind subdirs
        for the blob id. Any failure (missing blob, store error) yields
        ``None`` — :func:`verify_excerpt_against_blob` then treats the excerpt
        as unverifiable and KEEPS it (ws3-01: we only drop on a positive
        mismatch against a fetchable blob, never on absence).
        """
        override = getattr(self, "_blob_fetcher", None)
        if override is not None:
            return override

        engagement_id = ctx.engagement_id

        async def _fetch(ref: Any) -> bytes | None:
            blob_id = ref.get("blob_id") if isinstance(ref, dict) else getattr(ref, "blob_id", None)
            if not blob_id:
                return None
            try:
                from orchestrator.evidence import EvidenceStore
                from orchestrator.settings import get_settings

                store = EvidenceStore(get_settings())
                eng_dir = store.evidence_dir / str(engagement_id)
                for match in eng_dir.glob(f"*/{blob_id}"):
                    return match.read_bytes()
            except Exception as e:  # pragma: no cover - defensive
                log.warning("report.blob_fetch_error", err=str(e), blob_id=str(blob_id))
            return None

        return _fetch

    async def _draft_exec_summary(self, ctx: StageContext, draft: DraftReport) -> str:
        sev_counts = {s: 0 for s in ("critical", "high", "medium", "low", "info")}
        for f in draft.findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        system = (
            "You write executive summaries for pentest reports. One paragraph, "
            "no marketing tone, no exaggeration. Lead with the most severe finding "
            "or with 'no significant findings' if appropriate. Cite numbers."
        )
        user = (
            f"Coverage: {draft.coverage}\n"
            f"Findings: {sev_counts}\n"
            f"Attack paths identified: {len(draft.attack_paths)}\n\n"
            f"Write the executive summary."
        )
        resp = await self.router.complete(
            LLMRequest(
                system=system,
                user=user,
                site=CallSite.SYNTHESIZER,
                max_tokens=600,
                temperature=0.2,
                engagement_id=str(ctx.engagement_id),
            )
        )
        return resp.text.strip()

    async def _review_findings(self, ctx: StageContext, draft: DraftReport) -> None:
        assert self.report_critic is not None
        survivors: list[DraftFinding] = []
        for finding in draft.findings:
            subject = finding.model_dump()
            critique = await self.report_critic.review(subject, engagement_id=ctx.engagement_id)
            finding.critic_verdict = critique.verdict.value
            finding.critic_concern = critique.primary_concern
            # Surface the critic identity on the finding so
            # an auditor can trace "who vetoed this and at what confidence".
            finding.critic_name = critique.critic_name
            finding.critic_confidence = critique.confidence
            # Derive tool_run_ids from evidence_refs so the
            # auditor can pivot back to the raw tool output.
            ids: list[str] = []
            for ref in finding.evidence_refs:
                if isinstance(ref, dict):
                    tr = ref.get("tool_run_id")
                    if tr:
                        ids.append(str(tr))
            finding.tool_run_ids = ids
            if critique.verdict == Verdict.VETO:
                log.warning(
                    "report.finding_vetoed",
                    finding=finding.title,
                    concern=critique.primary_concern,
                )
                # Write an audit row that records the veto
                # and stash its ACTUAL row id on the finding. Best-effort
                # — a failed audit-row insert must not block the report
                # build; on failure the field stays None and the report
                # still renders.
                # Mint a single uuid here and insert the row with that
                # id explicitly so the finding's ``audit_row_id`` resolves
                # to a real ``audit_log`` row. Letting the SQLA default on
                # ``AuditLogRow.id`` mint a separate uuid would leave the
                # finding carrying a uuid that points to nothing.
                try:
                    from uuid import uuid4

                    from orchestrator import ledger
                    from orchestrator.db import session_scope

                    audit_row_id = uuid4()
                    payload = {
                        "audit_row_id": str(audit_row_id),
                        "finding_title": finding.title,
                        "critic_name": critique.critic_name,
                        "critic_confidence": critique.confidence,
                        "primary_concern": critique.primary_concern,
                        "tool_run_ids": ids,
                    }
                    # Emit the structlog event under the ``ledger_audit``
                    # event name that operators grep on.
                    log.info(
                        "audit",
                        action="report.finding_vetoed",
                        engagement_id=str(ctx.engagement_id),
                        **payload,
                    )
                    # Route through the ledger so the row joins the
                    # tamper-evident hash chain. We pass the pre-minted
                    # uuid as ``row_id`` so the finding's ``audit_row_id``
                    # still resolves to the inserted row (issue #112).
                    async with session_scope() as s:
                        await ledger.append_audit_row(
                            s,
                            engagement_id=ctx.engagement_id,
                            actor=f"critic:{critique.critic_name}",
                            action="report.finding_vetoed",
                            payload=payload,
                            row_id=audit_row_id,
                        )
                    # Only stamp the id on the finding AFTER the insert
                    # succeeded. If the session_scope context manager
                    # raised (Postgres down, schema mismatch, etc.) we
                    # leave ``audit_row_id`` as None — better a null
                    # field than a dangling pointer.
                    finding.audit_row_id = str(audit_row_id)
                except Exception as e:
                    log.warning("report.audit_row_emit_failed", err=str(e))
                # Vetoed findings move to a dedicated bucket; they don't
                # render alongside accepted findings.
                draft.vetoed_findings.append(finding)
            else:
                survivors.append(finding)
        draft.findings = survivors
