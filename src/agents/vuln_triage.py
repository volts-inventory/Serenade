"""VulnTriageAgent — match service+version against ExploitDB; rank candidates.

Tools: searchsploit_lookup.
Writes: ExploitCandidate nodes; reviewed by triage_council before persistence.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from agents.council import Council
from agents.critic import Verdict
from llm.prompts.base import PERSONA_VULN_TRIAGE
from tools.base import Tool
from tools.searchsploit_ import SearchsploitOutput
from worldstate.queries import coverage, hosts_with_open_services
from worldstate.schema import (
    ExploitCandidate,
    RiskClass,
    Stage,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_exploit_candidate,
)

log = structlog.get_logger(__name__)


class VulnTriageAgent(BaseAgent):
    stage = Stage.VULN_TRIAGE
    system_persona = PERSONA_VULN_TRIAGE
    # Triage is bounded by service count — one searchsploit lookup
    # per distinct product+version. 12 turns covers a typical scope.
    max_loop_turns = 12

    def __init__(
        self,
        *,
        triage_council: Council | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.triage_council = triage_council

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        hs = await hosts_with_open_services(self.repo, ctx.engagement_id)
        # Compact: per-host list of services with product+version present.
        # Build the full per-host list, then cap so the
        # planner sees a slice prefix + a count of dropped hosts; the
        # ``total_versioned_services`` count remains exact (the planner
        # uses it for completion criteria).
        compact: list[dict[str, Any]] = []
        total_versioned = 0
        for entry in hs:
            host = entry["host"]
            services = [
                {
                    "port": s.get("port"),
                    "name": s.get("name"),
                    "product": s.get("product"),
                    "version": s.get("version"),
                }
                for s in entry["services"]
                if s.get("product") or s.get("version")
            ]
            if services:
                compact.append({"ip": host.get("ip"), "services": services})
                total_versioned += len(services)
        return {
            "coverage": cov,
            "hosts_with_versions": _truncate_slice(compact, kind_label="Host"),
            "total_versioned_services": total_versioned,
            "turn": ctx.turn,
            "max_turns": self.budget.max_turns,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior vuln-triage analyst's per-turn workflow.

        The discipline from PERSONA_VULN_TRIAGE: rank, score, never
        emit a finding (that's report-stage). Branch:
          - no versioned services → nothing to triage; if there ARE
            services without versions, that's a REGRESS_TO recon/web_enum
            signal because the upstream stage didn't fingerprint deeply
            enough.
          - versioned services exist → one searchsploit per distinct
            (product, version), prefer candidates the curated PoC library
            can actually exercise.
        """
        compact = slice_.get("hosts_with_versions", [])
        cov = slice_.get("coverage", {})
        services = int(cov.get("services", 0))
        candidates = int(cov.get("candidates", 0))
        if not compact:
            return (
                f"PLAYBOOK [vuln_triage · idle]. State: services={services}, "
                "but zero have product+version data. There is nothing to score. "
                "If services>0 the upstream recon/web_enum did not produce "
                "version fingerprints — emit a final-state message; the "
                "controller may REGRESS_TO an earlier stage to deepen "
                "fingerprinting. REFUSE: emitting a finding from triage. "
                "Findings come from confirmed exploitation or the report "
                "stage — triage only ranks ExploitCandidate nodes."
            )
        n_hosts = len(compact)
        return (
            f"PLAYBOOK [vuln_triage · ranking]. State: hosts={n_hosts}, "
            f"versioned_services={slice_.get('total_versioned_services', 0)}, "
            f"candidates_so_far={candidates}. STEP: for each distinct "
            "(product, version) pair across all hosts, call "
            'searchsploit_lookup with query="<product> <version>". Skip '
            "services where version is missing. One tool call per pair — "
            "not per host (dedupe across the fleet). Score each match: "
            "vendor agreement + version-range fit + confirmable preconditions; "
            "two-of-three = candidate with confidence < 0.7. Prefer matches "
            "the curated PoC library can service. REFUSE: emitting a finding, "
            "guessing CVE IDs from banner alone, ranking inferred versions "
            "equal to directly-observed versions. EXIT: when every distinct "
            "versioned service has a triage score."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        cov = await coverage(self.repo, ctx.engagement_id)
        # Reuse the already-built slice from ``step()``.
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)
        compact = slice_.get("hosts_with_versions", [])
        # Advance when every versioned service has been triaged
        # (candidates produced) OR when there's nothing to triage at all
        # OR when the cap is hit. Approximate "every service triaged" by
        # candidates >= total_versioned_services (one searchsploit may
        # return zero or many ExploitCandidates per service; >= is the
        # right ceiling-style check).
        total_versioned = int(slice_.get("total_versioned_services", 0))
        if total_versioned == 0:
            return StepDecision.ADVANCE
        if cov["candidates"] >= max(1, total_versioned):
            return StepDecision.ADVANCE
        if not compact:
            return StepDecision.ADVANCE
        # Legacy safety net: at least some candidates and we've spent
        # enough turns — advance to exploit so the next stage can act.
        if cov["candidates"] >= 1 and ctx.turn >= 3:
            return StepDecision.ADVANCE
        if ctx.turn >= self.budget.max_turns:
            return StepDecision.ADVANCE
        return StepDecision.CONTINUE

    async def _parse_and_persist(
        self,
        ctx: StageContext,
        tool: Tool[Any, Any],
        tool_input: dict[str, Any],
        tool_output: BaseModel,
        tool_run_id: UUID,
    ) -> list[WorldStateNode]:
        nodes: list[WorldStateNode] = []
        if isinstance(tool_output, SearchsploitOutput):
            # ws-05: with no real Finding wiring yet, the candidate natural key
            # would degrade to ("-", source, edb_id) — so two distinct versioned
            # services that merely share an EDB-ID collapse onto one node and
            # last-writer-wins clobbers the first. The searchsploit query
            # ("<product> <version>") is the per-target signal available at this
            # call site; fold a deterministic id derived from it into the key so
            # candidates for distinct services stay distinct, while a re-run of
            # the *same* query still dedups idempotently.
            query = str(tool_input.get("query", "")).strip()
            target_node_id = (
                deterministic_node_id("ServiceQuery", ctx.engagement_id, query)
                if query
                else None
            )
            for cand in tool_output.candidates:
                ec = ExploitCandidate(
                    id=deterministic_node_id(
                        "ExploitCandidate",
                        ctx.engagement_id,
                        natural_key_for_exploit_candidate(
                            finding_id=None,
                            source="exploitdb",
                            source_ref=str(cand.edb_id),
                            target_node_id=target_node_id,
                        ),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    # Mint a stable placeholder finding_id so re-parses
                    # don't change this field. When real Finding wiring is
                    # in place the placeholder is replaced by the real id.
                    #
                    # wsN-02: fold the same per-target distinguisher
                    # (``target_node_id``, the searchsploit query id) into the
                    # placeholder finding_id that the node-id natural key uses.
                    # Otherwise two distinct candidate nodes (different queries,
                    # same EDB-ID) would carry an *identical* placeholder
                    # finding_id and downstream join-by-finding_id (e.g.
                    # lateral ``covered_finding_ids``) would over-merge them.
                    finding_id=deterministic_node_id(
                        "Finding",
                        ctx.engagement_id,
                        f"placeholder|exploitdb|{target_node_id or '-'}|{cand.edb_id}",
                    ),
                    source="exploitdb",
                    source_ref=str(cand.edb_id),
                    match_strength=0.6 if cand.verified else 0.4,
                    auto_runnable=False,  # only curated tier auto-runs
                    risk_class=RiskClass.EXPLOIT,
                    notes=cand.title,
                )
                nodes.append(ec)

        # Council review of the proposed candidate list (skip if no council injected).
        if nodes and self.triage_council:
            subject = {
                "stage": "vuln_triage",
                "candidates": [
                    {
                        "source": n.source,
                        "source_ref": n.source_ref,
                        "title": n.notes,
                        "match_strength": n.match_strength,
                    }
                    for n in nodes
                    if isinstance(n, ExploitCandidate)
                ],
            }
            # Fast mode for large batches (>10 candidates).
            triage_mode = ctx.engagement.get("triage_mode", "auto")
            if triage_mode == "always_full":
                mode = "full"
            elif triage_mode == "always_fast":
                mode = "fast"
            else:
                mode = "fast" if len(nodes) > 10 else "full"
            verdict = await self.triage_council.deliberate(
                subject, engagement_id=ctx.engagement_id, mode=mode
            )
            log.info(
                "vuln_triage.council",
                verdict=verdict.decision.value,
                confidence=verdict.confidence,
                rationale=verdict.rationale,
            )
            if verdict.decision == Verdict.VETO:
                # Drop the candidates for this turn; they can be re-proposed next turn.
                log.warning("vuln_triage.candidates_vetoed", reason=verdict.rationale)
                return []
        return nodes
