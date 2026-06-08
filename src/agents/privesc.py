"""PrivEscAgent — Linux privesc enumeration via in-session linpeas.

Stage: PRIVESC. Tools: `linpeas` (runs through SessionManager). Critic:
PreExploitCritic already wired for POST_EXPLOIT risk class.

Per unscanned unprivileged session:
  1. Propose `linpeas` with sections=[suid,sudo,kernel,cron]
  2. Parse `LinPEASOutput.findings` into `Finding` nodes
  3. Cross-link Finding -> Session via AFFECTS edge

Targeted escalation (SUID-via-GTFOBins, sudo NOPASSWD chains) is not
attempted by this agent; it produces high-severity findings and the
operator routes them to OperatorExec for approval.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from agents.regression import RegressionRequest, RegressionTriggerKind
from llm.prompts.base import PERSONA_PRIVESC
from tools.base import Tool
from worldstate.queries import coverage
from worldstate.schema import (
    Edge,
    EdgeKind,
    EvidenceRef,
    Finding,
    Severity,
    Stage,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_finding,
)

log = structlog.get_logger(__name__)


class PrivEscAgent(BaseAgent):
    stage = Stage.PRIVESC
    system_persona = PERSONA_PRIVESC
    # PrivEsc is enumerate (linpeas) → exploit cycles; per-session
    # not per-engagement, so 12 turns covers a typical small set of
    # unprivileged sessions without overspending on one host.
    max_loop_turns = 12

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        sessions = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Session")
        linpeas_done = await self._sessions_linpeas_done(ctx.engagement_id)
        unprivileged_unscanned = [
            {
                "session_id": s.get("id"),
                "host_id": s.get("host_id"),
                "user": s.get("user"),
                "type": s.get("type"),
            }
            for s in sessions
            if not s.get("privileged", False) and str(s.get("id")) not in linpeas_done
        ]
        # Count credentials produced by privesc (post-escalation artifacts).
        # An escalation should produce CredentialMaterial; we approximate
        # "escalation succeeded" by the presence of credentials.
        creds = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Credential")
        return {
            "coverage": cov,
            "sessions_total": len(sessions),
            # Cap with sentinel so a fleet with 100+
            # unscanned sessions doesn't blow the prompt.
            "unprivileged_unscanned": _truncate_slice(
                unprivileged_unscanned, kind_label="Session"
            ),
            "credentials_count": len(creds),
            "turn": ctx.turn,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior privesc operator's per-turn workflow.

        PERSONA_PRIVESC's discipline: enumerate first, exploit second;
        prefer documented misconfigs (SUID, sudo, writable units) over
        kernel CVEs (crash blast radius). Every escalation must produce
        a CredentialMaterial artifact. NO persistence — that's lateral
        territory and needs destructive-opt-in.
        """
        unscanned = slice_.get("unprivileged_unscanned", [])
        n_sessions = int(slice_.get("sessions_total", 0))
        n_creds = int(slice_.get("credentials_count", 0))
        if n_sessions == 0:
            return (
                "PLAYBOOK [privesc · idle]. State: sessions=0. No foothold "
                "to escalate from; this stage runs only on hosts we have a "
                "shell on. Emit a final-state message with no tool call."
            )
        if not unscanned:
            return (
                f"PLAYBOOK [privesc · enumeration complete]. State: "
                f"sessions={n_sessions}, credentials={n_creds}, all "
                "unprivileged sessions have been linpeas'd. The enumerated "
                "findings (SUID/sudo/cron) have been surfaced; targeted "
                "escalation via SUID-GTFOBins / sudo-NOPASSWD chains is "
                "OperatorExec territory for the operator to approve. Emit "
                "a final-state message; lateral stage takes over from here. "
                "REFUSE: persistence (cron, systemd, registry run keys) — "
                "that requires the engagement's destructive_opt_in."
            )
        target_ids = [s["session_id"] for s in unscanned[:3]]
        return (
            f"PLAYBOOK [privesc · stage 1: enumerate]. State: sessions={n_sessions}, "
            f"unprivileged_unscanned={len(unscanned)}, credentials={n_creds}. "
            f"STEP: run `linpeas` on each unscanned session id: {target_ids}. "
            "Use sections=['suid','sudo','kernel','cron']. Findings will be "
            "persisted automatically. Stage 2 (escalation) only after the "
            "enumeration output is parsed: prefer documented misconfigs "
            "(SUID with known GTFOBins shell-out, world-writable systemd "
            "units, NOPASSWD sudo chains) over kernel exploits. Kernel CVEs "
            "are last resort — crash blast radius is high. ESCALATE: a "
            "host where the only available paths are kernel-only — operator "
            "should sign off on the crash risk. REFUSE: any persistence "
            "operation (cron, systemd, registry run keys) — that's "
            "lateral-stage territory and requires destructive_opt_in. Every "
            "successful escalation must produce a CredentialMaterial node "
            "with provenance; no \"I escalated\" without an artifact."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        # Reuse the already-built slice from ``step()``.
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)

        # Check for regression before deciding to advance.
        regress_request = await self._maybe_build_regression_request(ctx)
        if regress_request is not None:
            ctx.pending_regression_request = regress_request
            return StepDecision.REGRESS_TO

        # Nothing to do.
        if int(slice_.get("sessions_total", 0)) == 0:
            return StepDecision.ADVANCE
        if not slice_["unprivileged_unscanned"]:
            # All sessions enumerated.
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
        findings = getattr(tool_output, "findings", None)
        if not findings:
            return nodes

        session_id_str = tool_input.get("session_id", "")
        try:
            session_id = UUID(session_id_str) if session_id_str else None
        except (ValueError, TypeError):
            session_id = None

        for lp in findings:
            section = getattr(lp, "section", "unknown")
            detail = getattr(lp, "detail", "")
            excerpt = getattr(lp, "raw_excerpt", "")
            sev = _section_severity(section)
            title = f"linpeas {section}: {detail[:100]}"
            # Deterministic id so re-running linpeas on the
            # same session doesn't double-insert findings under different
            # ids. ``session_id`` is the affected node when we have it.
            finding = Finding(
                id=deterministic_node_id(
                    "Finding",
                    ctx.engagement_id,
                    natural_key_for_finding(
                        category=f"linpeas:{section}",
                        affected_node_id=session_id,
                        title=title,
                    ),
                ),
                engagement_id=ctx.engagement_id,
                provenance=[tool_run_id],
                title=title,
                description=f"linpeas flagged a {section} privesc vector. Detail: {detail}",
                severity=sev,
                evidence=[
                    EvidenceRef(
                        tool_run_id=tool_run_id,
                        excerpt=excerpt[:500],
                    )
                ],
                remediation=_remediation_for_section(section),
            )
            nodes.append(finding)
            if session_id is not None:
                await self.repo.upsert_edge(
                    Edge(
                        kind=EdgeKind.AFFECTS,
                        src_id=finding.id,
                        dst_id=session_id,
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                    )
                )
        return nodes

    async def _sessions_linpeas_done(self, engagement_id: UUID) -> set[str]:
        """Return session ids that already have linpeas-sourced Findings."""
        findings = await self.repo.get_nodes_by_kind(engagement_id, "Finding")
        done: set[str] = set()
        for f in findings:
            title = f.get("title", "")
            if not isinstance(title, str) or not title.startswith("linpeas "):
                continue
            fid = f.get("id")
            if not fid:
                continue
            # Follow AFFECTS edges from this Finding to Session nodes.
            sessions = await self.repo.get_successors_by_edge_type(
                UUID(str(fid)), "AFFECTS"
            )
            for s in sessions:
                if s.get("kind") == "Session":
                    sid = s.get("id")
                    if sid:
                        done.add(str(sid))
        return done

    async def _maybe_build_regression_request(
        self, ctx: StageContext
    ) -> RegressionRequest | None:
        """Emit a regression to VULN_TRIAGE when linpeas produces HIGH/CRITICAL
        findings that don't already have corresponding ExploitCandidates.

        This ensures that SUID binaries, kernel vulns, sudo NOPASSWD, and
        writable cron findings get triaged into ExploitCandidates for the
        exploit stage to act on.
        """
        cap = ctx.regression_budget_max
        if cap > 0 and (
            ctx.regressions_used + ctx.regressions_pending >= cap
        ):
            return None

        untriaged = await self._high_findings_without_candidates(ctx.engagement_id)
        if not untriaged:
            return None

        # Collect finding IDs and the host IDs they affect.
        finding_ids: list[str] = []
        host_ids: set[str] = set()
        for finding_id, host_id in untriaged:
            finding_ids.append(finding_id)
            if host_id:
                host_ids.add(host_id)

        delta: dict[str, Any] = {
            "finding_ids": finding_ids,
        }
        if host_ids:
            delta["host_ids"] = sorted(host_ids)

        rationale = (
            f"privesc: linpeas produced {len(finding_ids)} HIGH/CRITICAL "
            f"finding(s) on host(s) {sorted(host_ids)} that have no "
            "corresponding ExploitCandidates — regress to VULN_TRIAGE "
            "so these privesc vectors get triaged into actionable candidates."
        )

        ctx.regressions_pending += 1
        return RegressionRequest(
            target_stage=Stage.VULN_TRIAGE,
            trigger_kind=RegressionTriggerKind.PRIVESC_FINDING_DISCOVERED,
            delta=delta,
            evidence=finding_ids,
            rationale=rationale[:1000],
        )

    async def _high_findings_without_candidates(
        self, engagement_id: UUID
    ) -> list[tuple[str, str | None]]:
        """Return ``(finding_id, host_id)`` pairs for HIGH/CRITICAL linpeas
        findings that do NOT yet have a corresponding ExploitCandidate.

        An ExploitCandidate corresponds to a Finding when
        ``ExploitCandidate.finding_id == Finding.id``. We only consider
        linpeas-sourced findings (title starts with ``'linpeas '``).
        """
        findings = await self.repo.get_nodes_by_kind(engagement_id, "Finding")
        candidates = await self.repo.get_nodes_by_kind(engagement_id, "ExploitCandidate")
        # ExploitCandidate corresponds to a Finding via the finding_id field.
        covered_finding_ids = {
            str(c.get("finding_id")) for c in candidates if c.get("finding_id")
        }

        results: list[tuple[str, str | None]] = []
        seen_fids: set[str] = set()
        for f in findings:
            title = f.get("title", "")
            if not isinstance(title, str) or not title.startswith("linpeas "):
                continue
            if str(f.get("severity", "")).lower() not in ("high", "critical"):
                continue
            fid = f.get("id")
            if not fid:
                continue
            fid = str(fid)
            if fid in covered_finding_ids or fid in seen_fids:
                continue
            seen_fids.add(fid)

            # Best-effort host resolution: Finding -AFFECTS-> Host, or
            # Finding -AFFECTS-> Session -HAS_SESSION_ON-> Host. None if absent.
            hid: str | None = None
            try:
                affected = await self.repo.get_successors_by_edge_type(UUID(fid), "AFFECTS")
                for n in affected:
                    if n.get("kind") == "Host" and n.get("id"):
                        hid = str(n["id"])
                        break
                if hid is None:
                    for n in affected:
                        if n.get("kind") == "Session" and n.get("id"):
                            for h in await self.repo.get_successors_by_edge_type(
                                UUID(str(n["id"])), "HAS_SESSION_ON"
                            ):
                                if h.get("kind") == "Host" and h.get("id"):
                                    hid = str(h["id"])
                                    break
                        if hid is not None:
                            break
            except (ValueError, TypeError):
                hid = None
            results.append((fid, hid))
        return results


_SEVERITY_BY_SECTION = {
    "suid": Severity.HIGH,
    "sudo": Severity.HIGH,
    "kernel": Severity.HIGH,
    "cron": Severity.MEDIUM,
    "writable": Severity.MEDIUM,
    "caps": Severity.MEDIUM,
}


def _section_severity(section_label: str) -> Severity:
    lower = section_label.lower()
    for keyword, sev in _SEVERITY_BY_SECTION.items():
        if keyword in lower:
            return sev
    return Severity.LOW


def _remediation_for_section(section_label: str) -> str:
    s = section_label.lower()
    if "suid" in s:
        return (
            "Audit SUID binaries; remove the bit from non-essential ones. "
            "For required SUID binaries, ensure they are patched and not listed in GTFOBins."
        )
    if "sudo" in s:
        return (
            "Review /etc/sudoers; remove NOPASSWD entries and broad ALL=(ALL) grants "
            "for non-admin users."
        )
    if "kernel" in s:
        return "Patch the kernel to the latest LTS revision; track CVE feeds for the kernel build."
    if "cron" in s:
        return (
            "Audit /etc/cron* and /etc/anacrontab for writable scripts/dirs; "
            "constrain to root-owned, root-only-write."
        )
    return "Review the linpeas output; this finding requires manual triage."
