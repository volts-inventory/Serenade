"""WebEnumAgent — directory brute, parameter fuzzing, nuclei templating.

Tools: ffuf_dir_brute, nuclei_scan.
Writes: Endpoint, Finding nodes; OF_WEBAPP, AFFECTS edges.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from llm.prompts.base import PERSONA_WEB_ENUM
from tools.base import Tool
from tools.ffuf_ import FfufOutput
from tools.nuclei_ import NucleiOutput
from worldstate.queries import coverage, webapps_summary
from worldstate.schema import (
    Endpoint,
    EvidenceRef,
    Finding,
    Severity,
    Stage,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_endpoint,
    natural_key_for_finding,
)

log = structlog.get_logger(__name__)


_SEVERITY_MAP = {
    "info": Severity.INFO,
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


class WebEnumAgent(BaseAgent):
    stage = Stage.ENUMERATION
    system_persona = PERSONA_WEB_ENUM
    # Web enumeration fans out per webapp (fingerprint, then per-stack
    # wordlist + paths, then vuln template scan). 18 turns covers a typical
    # mid-size scope without burning the cap when one wordlist takes longer.
    max_loop_turns = 18

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        webapps = await webapps_summary(self.repo, ctx.engagement_id)
        # Cap the webapps list so the slice JSON stays
        # within prompt-cache-friendly bounds even on a large engagement.
        compact_webapps = [
            {
                "id": w["webapp"].get("id"),
                "base_url": w["webapp"].get("base_url"),
                "title": w["webapp"].get("title"),
                "technologies": w["webapp"].get("technologies", []),
                "endpoints": w["endpoints"],
            }
            for w in webapps
        ]
        return {
            "coverage": cov,
            "webapps": _truncate_slice(compact_webapps, kind_label="WebApp"),
            "turn": ctx.turn,
            "max_turns": self.budget.max_turns,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior web-recon specialist's per-turn workflow.

        Branch order: fingerprint stack → run a stack-appropriate
        directory wordlist → run template scans. The persona's "stack
        identification first" discipline maps to: when a webapp has
        endpoints==0 the next action is ffuf (paths give us the stack
        signal); when paths exist but no findings the next action is
        nuclei. SSO / mTLS surfaces escalate to operator.
        """
        webapps = slice_.get("webapps", [])
        n = len(webapps)
        if not webapps:
            return (
                f"PLAYBOOK [web_enum · idle]. State: webapps={n}. No WebApp nodes "
                "to fingerprint — recon hasn't populated them yet (or the scope "
                "has no HTTP services). Emit a final-state message with no tool "
                "call; this terminates the loop and lets the controller advance."
            )

        # Branch 1: webapps without endpoints — fingerprint via ffuf path brute.
        unfingerprinted = [w for w in webapps if w.get("endpoints", 0) == 0]
        # Branch 2: webapps with endpoints but no tech tags — same: ffuf seeds the stack.
        no_tech = [
            w for w in webapps
            if w.get("endpoints", 0) > 0 and not w.get("technologies")
        ]
        if unfingerprinted or no_tech:
            target = unfingerprinted or no_tech
            urls = [w.get("base_url") for w in target[:5]]
            return (
                f"PLAYBOOK [web_enum · stage 1 of 2: fingerprint + paths]. State: "
                f"webapps={n}, unscanned={len(unfingerprinted)}, "
                f"no_tech={len(no_tech)}. STEP: run ffuf_dir_brute against the "
                f"unscanned URLs (one at a time): {urls}. "
                "Use wordlist='common', match codes [200,204,301,302,401,403]. "
                "Pick a stack-specific wordlist ONLY after the fingerprint "
                "confirms (a WordPress wordlist on a Spring Boot app wastes "
                "budget). Throttle — a WAF block hurts every later stage. "
                "REFUSE: aggressive directory recursion, parameter fuzzing on "
                "endpoints we haven't catalogued yet. "
                "ESCALATE: any endpoint that returns an SSO / mTLS / custom "
                "auth-header response — that's a credential question."
            )

        # Branch 3: paths + tech known — run targeted vuln template scan.
        return (
            f"PLAYBOOK [web_enum · stage 2 of 2: template scan]. State: "
            f"webapps={n}, all have endpoints + technologies. STEP: run "
            "nuclei_scan with severity=['medium','high','critical'] against "
            "each webapp using technology-matched templates (e.g. "
            "tags=['wordpress'] for a WP stack). One scan per webapp; do "
            "not re-run a finished scan with louder flags. "
            "REFUSE: brute-forcing login forms (that's lateral-stage); "
            "fuzzing parameters that require auth we don't have. "
            "EXIT: once every webapp has a Finding-or-no-finding result the "
            "stage is done — emit a final-state message to terminate."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        cov = await coverage(self.repo, ctx.engagement_id)
        # Nothing to enumerate.
        if cov["webapps"] == 0:
            return StepDecision.ADVANCE
        # Advance when every WebApp has at least one Endpoint
        # recorded — i.e. fingerprinting succeeded for every webapp.
        # We use endpoints >= webapps as a coarse "each webapp has at
        # least one path" check (matches the per-stage exit rule:
        # "every WebService node has tech_tags + at least one path").
        if cov["endpoints"] >= cov["webapps"]:
            return StepDecision.ADVANCE
        if ctx.turn >= self.budget.max_turns:
            return StepDecision.ESCALATE_HUMAN
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
        if isinstance(tool_output, FfufOutput):
            # Build a URL-prefix -> WebApp.id index so
            # Endpoint.webapp_id is real rather than a placeholder UUID.
            webapp_index = await _index_webapps_by_base_url(self.repo, ctx.engagement_id)
            base_url_from_input = str(tool_input.get("url", "")).rstrip("/").replace("/FUZZ", "")
            for hit in tool_output.hits:
                webapp_id = _match_webapp_for_url(hit.url, base_url_from_input, webapp_index)
                ep_path = (
                    hit.url.split("//", 1)[-1].split("/", 1)[-1]
                    if "//" in hit.url
                    else hit.url
                )
                # Derive a deterministic id from the
                # (webapp, method, path) triple so a re-parse of the same
                # ffuf output writes the same id (MERGE-on-id dedupes) instead
                # of minting a fresh uuid4 each time. The fallback uuid4 for
                # the webapp_id is preserved — but it's stamped INTO the
                # natural key so two ffuf runs against the same unmatched URL
                # still collide.
                resolved_webapp_id = webapp_id or uuid4()
                ep = Endpoint(
                    id=deterministic_node_id(
                        "Endpoint",
                        ctx.engagement_id,
                        natural_key_for_endpoint(
                            webapp_id=resolved_webapp_id,
                            method="GET",
                            path=ep_path,
                        ),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    webapp_id=resolved_webapp_id,
                    method="GET",
                    path=ep_path,
                    status_code=hit.status,
                    content_type=hit.content_type,
                    requires_auth=hit.status in (401, 403),
                )
                nodes.append(ep)
        elif isinstance(tool_output, NucleiOutput):
            for f in tool_output.findings:
                sev = _SEVERITY_MAP.get(f.severity.lower(), Severity.INFO)
                evidence = [
                    EvidenceRef(
                        tool_run_id=tool_run_id,
                        excerpt=f"{f.template_id} matched at {f.matched_url}",
                    )
                ]
                cves = f.classification.get("cve-id", []) if f.classification else []
                cwes = f.classification.get("cwe-id", []) if f.classification else []
                title = f.name or f.template_id
                # Deterministic id keyed on the nuclei template
                # category, matched URL, and finding title so two scans of the
                # same target write the same Finding id.
                finding = Finding(
                    id=deterministic_node_id(
                        "Finding",
                        ctx.engagement_id,
                        natural_key_for_finding(
                            category=f.template_id or "nuclei",
                            affected_node_id=None,
                            title=f"{title}|{f.matched_url}",
                        ),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    title=title,
                    description=f.description or f"nuclei template {f.template_id} matched",
                    severity=sev,
                    cve=cves,
                    cwe=cwes[0] if cwes else None,
                    evidence=evidence,
                    references=[f.matched_url],
                )
                nodes.append(finding)
        return nodes


async def _index_webapps_by_base_url(repo, engagement_id) -> dict[str, UUID]:
    """Build a `base_url.rstrip('/')` -> WebApp.id index."""
    webapps = await repo.get_nodes_by_kind(engagement_id, "WebApp")
    out: dict[str, UUID] = {}
    for w in webapps:
        b = w.get("base_url")
        wid = w.get("id")
        if b and wid:
            try:
                out[str(b).rstrip("/")] = UUID(str(wid))
            except (ValueError, TypeError):
                continue
    return out


def _match_webapp_for_url(
    hit_url: str, base_url_from_input: str, index: dict[str, UUID]
) -> UUID | None:
    """Find the WebApp whose base_url is the longest prefix match of `hit_url`.

    Tries the ffuf -u input first (most likely match), then falls back to
    the longest-prefix scan over the index.
    """
    if base_url_from_input and base_url_from_input in index:
        return index[base_url_from_input]
    candidates = sorted(
        (k for k in index if hit_url.startswith(k)),
        key=len,
        reverse=True,
    )
    if candidates:
        return index[candidates[0]]
    return None
