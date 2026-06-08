"""LootAgent — in-session catalog of cleartext creds / keys / PII.

Stage: LOOT. Catalog ONLY — never exfiltrate full files. Output flows
through `EvidenceStore.put` which scrubs credentials by default, so even
the evidence blobs don't contain raw passwords.

Per open session, runs a series of read-only grep commands via
`SessionManager.exec` looking for the standard cred-leak paths:
  - /etc/passwd, /etc/shadow (readability check)
  - $HOME/.aws/credentials, .ssh/id_*, .git-credentials
  - /var/www/**/wp-config.php, .env, config.{php,js,yaml}
  - /etc/cron* readable scripts

Hits become high-severity Findings tied to the session via AFFECTS.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from agents.regression import RegressionRequest, RegressionTriggerKind
from llm.prompts.base import PERSONA_LOOT
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


class LootAgent(BaseAgent):
    stage = Stage.LOOT
    system_persona = PERSONA_LOOT
    # Loot is focused on already-compromised hosts; a per-session
    # read-only sweep is bounded. 8 turns is enough for a small fleet.
    max_loop_turns = 8

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        sessions = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Session")
        looted = await self._sessions_looted(ctx.engagement_id)
        not_yet = [
            {"session_id": s.get("id"), "host_id": s.get("host_id"), "user": s.get("user")}
            for s in sessions
            if str(s.get("id")) not in looted
        ]
        return {
            "coverage": cov,
            "sessions": len(sessions),
            "objectives": ctx.engagement.get("objectives", []),
            # Cap with sentinel so a fleet with many open
            # sessions doesn't blow the prompt.
            "unlooted_sessions": _truncate_slice(not_yet, kind_label="Session"),
            "turn": ctx.turn,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior post-compromise investigator's per-turn workflow.

        PERSONA_LOOT: catalog in-place, never exfiltrate. Tag every loot
        finding with host + privileged account. Never stage or write data
        outside the sandbox. The branch:
          1. unlooted sessions → run read-only enumeration on each.
          2. all sessions cataloged → stop, advance to report.
        """
        unlooted = slice_.get("unlooted_sessions", [])
        n_sessions = int(slice_.get("sessions", 0))
        if n_sessions == 0:
            return (
                "PLAYBOOK [loot · idle]. State: sessions=0. No foothold to "
                "loot from. Emit a final-state message with no tool call to "
                "terminate the loop."
            )
        if not unlooted:
            return (
                f"PLAYBOOK [loot · complete]. State: sessions={n_sessions}, "
                "all cataloged. Every open session has been enumerated for "
                "credential stores / configs / PII paths; findings exist for "
                "the report stage to format. Emit a final-state message."
            )
        target_ids = [s.get("session_id") for s in unlooted[:3]]
        return (
            f"PLAYBOOK [loot · in-place catalog]. State: sessions={n_sessions}, "
            f"unlooted={len(unlooted)}. STEP: "
            f"for each unlooted session id ({target_ids}), run READ-ONLY "
            "checks via SessionManager.exec: cat /etc/passwd, ls -la "
            "~/.ssh ~/.aws, find /var/www -name '.env' -o -name "
            "'wp-config.php', cat /etc/crontab. Each hit creates a "
            "Finding. Prefer in-place enumeration commands that leave no "
            "artifact (PowerShell Get-Content is quieter than copying). "
            "Tag every loot finding with the host it came from AND a "
            "privileged account that could access it — findings without a "
            "source host are unactionable. REFUSE: exfiltrate full files "
            "(read only); copy data outside the sandbox; create working "
            "directories to stage a dump — destructive actions are never "
            "permitted. ESCALATE: any write outside the sandbox. "
            "EvidenceStore.put already "
            "scrubs credential values before MinIO; we capture paths + "
            "snippets, never raw secrets."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        # Reuse the already-built slice from ``step()``.
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)

        # Structural REGRESS_TO producer. After persisting credentials
        # during loot, check whether any credential implies access to a
        # known host via a service class that host doesn't have
        # enumerated. If so, regress to ENUMERATION. This check runs
        # FIRST so the discovery isn't swallowed by the ADVANCE path.
        regress_request = await self._maybe_build_regression_request(ctx)
        if regress_request is not None:
            ctx.pending_regression_request = regress_request
            return StepDecision.REGRESS_TO

        if int(slice_.get("sessions", 0)) == 0:
            return StepDecision.ADVANCE
        if not slice_["unlooted_sessions"]:
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
        # Two output shapes we expect: LinPEASOutput (re-use for `loot` sections)
        # or a raw shell ExecResult flowing through a hypothetical SessionExec tool.
        # For now: pattern-match against any tool output that has a `.stdout` field.
        stdout_text = getattr(tool_output, "stdout", None)
        if not isinstance(stdout_text, str):
            return nodes

        session_id_str = tool_input.get("session_id", "")
        try:
            session_id = UUID(session_id_str) if session_id_str else None
        except (ValueError, TypeError):
            session_id = None

        hits = _scan_for_loot_hits(stdout_text)
        for hit in hits:
            title = f"loot {hit['kind']}: {hit['path'][:100]}"
            # Deterministic id so re-running the same loot
            # sweep against the same session does not duplicate findings.
            finding = Finding(
                id=deterministic_node_id(
                    "Finding",
                    ctx.engagement_id,
                    natural_key_for_finding(
                        category=f"loot:{hit['kind']}",
                        affected_node_id=session_id,
                        title=f"{hit['kind']}|{hit['path']}",
                    ),
                ),
                engagement_id=ctx.engagement_id,
                provenance=[tool_run_id],
                title=title,
                description=(
                    f"Cleartext {hit['kind']} discovered at {hit['path']}. "
                    "Raw value scrubbed in evidence per orchestrator/evidence_scrub.py."
                ),
                severity=Severity.HIGH,
                evidence=[
                    EvidenceRef(
                        tool_run_id=tool_run_id,
                        excerpt=hit["snippet"][:300],
                    )
                ],
                remediation=_remediation_for_loot(hit["kind"]),
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

    async def _maybe_build_regression_request(
        self, ctx: StageContext
    ) -> RegressionRequest | None:
        """Check whether a newly-persisted credential during loot
        implies access to a known host via a service class that host
        doesn't have enumerated. When found, emit a
        :class:`RegressionRequest` targeting ENUMERATION so the
        re-entered stage can catalog the implied service.

        Producer-side budget check gates the request — the BudgetCritic
        in the council re-checks, but the early exit avoids pushing a
        known-doomed request onto the bus / audit log.

        Returns ``None`` when no credential-implied pivot is found, the
        budget is exhausted, or the world-state slice doesn't carry the
        fields needed for this check (``discovered_credentials`` with
        host / service-class data is not yet populated by all loot
        output shapes).
        """
        # Producer-side budget check.
        cap = ctx.regression_budget_max
        if cap > 0 and (
            ctx.regressions_used + ctx.regressions_pending >= cap
        ):
            return None

        slice_ = ctx.last_world_state_slice
        if slice_ is None:
            return None

        # The loot slice carries ``discovered_credentials`` when
        # credential-shaped findings reference known hosts with a
        # service-class hint. Not all loot output shapes populate this
        # field yet — return None conservatively when absent. The
        # structural wiring is in place; the check fires as soon as the
        # slice carries the needed data.
        discovered_credentials = slice_.get("discovered_credentials", [])
        if not discovered_credentials:
            return None

        # Look for credentials that reference known hosts with
        # unenumerated service classes.
        for cred in discovered_credentials:
            cred_id = cred.get("cred_id")
            host_id = cred.get("host_id")
            if not host_id:
                continue
            service_class_hint = cred.get("service_class_hint", "")
            if not service_class_hint:
                continue
            host_services = cred.get("host_services", [])
            known_classes = {
                str(svc.get("service_class", "")).lower()
                for svc in host_services
                if svc.get("service_class")
            }
            if service_class_hint.lower() in known_classes:
                continue

            evidence = [str(host_id)]
            if cred_id:
                evidence.insert(0, str(cred_id))

            ctx.regressions_pending += 1
            return RegressionRequest(
                target_stage=Stage.ENUMERATION,
                trigger_kind=RegressionTriggerKind.CREDENTIALS_IMPLY_PIVOT,
                delta={
                    "host_id": str(host_id),
                    "service_class_hint": service_class_hint,
                },
                evidence=evidence,
                rationale=(
                    f"loot: credential implies access to host {host_id} "
                    f"via service class {service_class_hint!r} which is "
                    "not in the host's enumerated Service nodes — "
                    "regress to ENUMERATION to catalog the implied "
                    "service."
                )[:1000],
            )

        return None

    async def _sessions_looted(self, engagement_id: UUID) -> set[str]:
        """Return session ids that already have loot-sourced Findings."""
        findings = await self.repo.get_nodes_by_kind(engagement_id, "Finding")
        looted: set[str] = set()
        for f in findings:
            title = f.get("title", "")
            if not isinstance(title, str) or not title.startswith("loot "):
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
                        looted.add(str(sid))
        return looted


def _scan_for_loot_hits(text: str) -> list[dict[str, str]]:
    """Best-effort scan of session stdout for loot evidence.

    Returns a list of {kind, path, snippet} dicts. The evidence-scrub layer
    redacts the actual cred values; we only capture path + an N-char snippet
    of context to prove the file exists / contains the kind of data.
    """
    import re

    hits: list[dict[str, str]] = []
    patterns = [
        ("aws_credentials", re.compile(r"(\.aws/credentials\b[^\n]*)", re.IGNORECASE)),
        ("ssh_private_key", re.compile(r"(\.ssh/id_(?:rsa|ed25519|ecdsa)\b[^\n]*)", re.IGNORECASE)),
        # `.env` file mention OR a `KEY=secret` line that looks env-shaped
        ("env_file", re.compile(r"(\.env\b[^\n]*)", re.IGNORECASE)),
        ("wp_config", re.compile(r"(wp-config\.php[^\n]*)", re.IGNORECASE)),
        ("git_credentials", re.compile(r"(\.git-credentials\b[^\n]*)", re.IGNORECASE)),
        ("kubeconfig", re.compile(r"(\.kube/config\b[^\n]*)", re.IGNORECASE)),
    ]
    for kind, regex in patterns:
        for m in regex.finditer(text):
            snippet = text[max(0, m.start() - 80) : m.end() + 80]
            hits.append({"kind": kind, "path": m.group(1)[:200], "snippet": snippet})
    return hits


_REM = {
    "aws_credentials": "Rotate the AWS access key; enforce IMDSv2; use IAM roles instead of static keys.",
    "ssh_private_key": "Remove the key from disk; rotate the keypair; add a passphrase to remaining keys.",
    "env_file": "Move secrets out of .env into a secret manager (Vault / AWS Secrets Manager); .gitignore the file.",
    "wp_config": "Restrict wp-config.php to 0400 owned by the webserver user; rotate DB password.",
    "git_credentials": "Delete ~/.git-credentials; configure git to use SSH or a credential helper backed by the OS keychain.",
    "kubeconfig": "Restrict ~/.kube/config to 0600; rotate the cluster's certificate authority; consider short-lived tokens.",
}


def _remediation_for_loot(kind: str) -> str:
    return _REM.get(
        kind,
        "Audit the discovered file; remove or restrict access; rotate any embedded credentials.",
    )
