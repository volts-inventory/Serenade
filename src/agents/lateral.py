"""LateralAgent — credential reuse + BloodHound shortest-path pivoting.

`PreLateralCritic` is wired by default to veto credential reuse without
validation, account-lockout risk, and detection-sensitive auth.

Behavior:
  1. For each Credential not yet validated against a known host, propose
     `netexec` with `action=auth_check` against that host
  2. On success → persist the validation as a HAS_SESSION_ON edge AND
     update `Credential.valid=True`
  3. If admin status was confirmed (`Pwn3d!` in the NetExec row), add
     LOCAL_ADMIN_ON edge
  4. Run `bloodhound_query` for shortest-path from owned principals to
     `Group: DOMAIN ADMINS@*` when ≥1 valid domain cred exists
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from agents.critic import BaseCritic
from agents.regression import RegressionRequest, RegressionTriggerKind
from llm.prompts.base import PERSONA_LATERAL
from tools.base import Tool
from worldstate.queries import coverage, credentials_with_known_hosts
from worldstate.schema import (
    Credential,
    Edge,
    EdgeKind,
    RiskClass,
    Stage,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_credential,
)

log = structlog.get_logger(__name__)

# Regex matchers used by ``_scan_credentials_for_pivot_targets``.
# These run against ``description`` / ``source`` / ``notes`` payload fields
# of recently-captured Credential nodes; an IP or FQDN match that falls
# outside the engagement's recon coverage but inside the engagement's
# scope motivates a ``REGRESS_TO recon`` to discover hosts in the implied
# subnet before pivoting.
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FQDN_RE = re.compile(
    r"\b(?=[a-zA-Z])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,63}\b"
)


def _ip_in_any_cidr(ip: str, cidrs: list[str]) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if addr in net:
            return True
    return False


def _ip_in_known_hosts(ip: str, host_ips: list[str]) -> bool:
    return ip in set(host_ips)


def _cidr_for_ip(ip: str) -> str | None:
    """Return a canonical ``/24`` (IPv4) or ``/64`` (IPv6) network string
    that covers ``ip``. The re-entered recon stage will sweep this CIDR
    to discover sibling hosts the credential implied are reachable."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv4Address):
        net = ipaddress.ip_network(f"{ip}/24", strict=False)
    else:
        net = ipaddress.ip_network(f"{ip}/64", strict=False)
    return str(net)


def _fqdn_in_scope(fqdn: str, scope_domains: list[str]) -> bool:
    """``fqdn`` falls under ``scope_domains`` if it equals one of them
    OR is a subdomain (matching from the right). e.g. ``db.corp.example.com``
    is in scope when ``corp.example.com`` is in scope."""
    f = fqdn.lower().rstrip(".")
    for d in scope_domains:
        d_lower = d.lower().rstrip(".")
        if f == d_lower or f.endswith("." + d_lower):
            return True
    return False


class LateralAgent(BaseAgent):
    stage = Stage.LATERAL
    system_persona = PERSONA_LATERAL
    critic_required_for_risk_classes = (RiskClass.POST_EXPLOIT, RiskClass.EXPLOIT)
    # Lateral is validate-then-move cycles; one cred + one host per
    # turn pair. 10 turns covers a small set of pivots without enabling
    # spray. PreLateralCritic + lockout-risk veto keep the loop tight.
    max_loop_turns = 10
    # Explicit slice cap for lateral's targetable-host list.
    #
    # PreLateralCritic vetoes credential spray with a 20-host lockout cap.
    # Pin the slice cap explicitly here so the lockout protection doesn't
    # depend on the stage-agnostic ``_SLICE_CAP_DEFAULT``. If that default
    # were tuned below 20 the spray protection would silently degrade.
    LOCKOUT_SAFE_SLICE_CAP: int = 20

    def __init__(
        self,
        *,
        pre_lateral_critic: BaseCritic | None = None,
        enable_debate_gates: bool = False,
        **kwargs: Any,
    ) -> None:
        if pre_lateral_critic is not None:
            kwargs["pre_action_critic"] = pre_lateral_critic
        super().__init__(**kwargs)

        if enable_debate_gates and self.pre_action_council is None:
            from agents.council import debate_council  # noqa: PLC0415

            self.pre_action_council = debate_council(self.router)
            self._debate_rounds = 2
        else:
            self._debate_rounds = 1

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        creds_with_hosts = await credentials_with_known_hosts(self.repo, ctx.engagement_id)
        all_creds = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Credential")
        validated_cred_ids = {
            str(c["credential"].get("id")) for c in creds_with_hosts
        }
        untested = [
            {
                "id": c.get("id"),
                "username": c.get("username"),
                "domain": c.get("domain"),
                "source": c.get("source"),
                "has_password": bool(c.get("password")),
                "has_hash": bool(c.get("nt_hash")),
            }
            for c in all_creds
            if str(c.get("id")) not in validated_cred_ids and c.get("valid") is not True
        ]
        hosts = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Host")
        targetable_hosts = [
            {"id": h.get("id"), "ip": h.get("ip"), "os_family": h.get("os_family")}
            for h in hosts
        ]
        return {
            "coverage": cov,
            "untested_credentials": untested[:10],
            "targetable_hosts": _truncate_slice(
                targetable_hosts,
                cap=self.LOCKOUT_SAFE_SLICE_CAP,
                kind_label="Host",
            ),
            "validated_credentials_count": len(creds_with_hosts),
            "turn": ctx.turn,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Per-turn playbook branching on credential validation state."""
        untested = slice_.get("untested_credentials", [])
        hosts = slice_.get("targetable_hosts", [])
        n_validated = int(slice_.get("validated_credentials_count", 0))
        if not untested and n_validated == 0:
            return (
                "PLAYBOOK [lateral · idle]. State: untested=0, validated=0. No "
                "credentials to pivot with — nothing for lateral to do. Emit a "
                "final-state message with no tool call."
            )
        if not untested:
            return (
                f"PLAYBOOK [lateral · stage 2 of 2: pivot]. State: untested=0, "
                f"validated={n_validated}. STEP: every cred is validated; "
                "now propose `bloodhound_query` for shortest-path from owned "
                "principals to DOMAIN ADMINS (re-verify the edge in real time "
                "before relying on it — an old admin can be removed, a "
                "delegation revoked). After a path is confirmed, move to the "
                "next hop via the native protocol (Kerberos with a real ticket "
                "is quieter than NTLM spray). REFUSE: pass-the-hash when "
                "Kerberos is available; out-of-scope targets. ESCALATE: any "
                "auth method that triggers high-confidence detection "
                "(S4U2Self, RBCD, golden ticket without a strong reason). "
                "EXIT: when no edges remain to traverse."
            )
        if not hosts:
            return (
                f"PLAYBOOK [lateral · blocked]. State: untested={len(untested)}, "
                "but no targetable hosts in scope. The next-hop targets do not "
                "exist in graph; REGRESS_TO net_enum may be needed to "
                "discover internal hosts. Emit a final-state message."
            )
        return (
            f"PLAYBOOK [lateral · stage 1 of 2: validate]. State: "
            f"untested={len(untested)}, targetable={len(hosts)}, "
            f"validated={n_validated}. STEP: for each untested credential, "
            "run `netexec` with protocol=smb action=auth_check against ONE "
            "host at a time. PreLateralCritic vetoes high-lockout-risk "
            "attempts. Validate the cred BEFORE using it on the target — a "
            "failed auth is louder than a successful one. After a success, "
            "stage 2 (BloodHound + move) begins. REFUSE: spraying multiple "
            "hosts with the same cred (lockout risk); brute-force; using a "
            "cred whose validity has not been confirmed. ESCALATE: when "
            "the account-lockout policy is unknown (the persona's veto "
            "rule). EXIT: when no untested credentials remain."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)

        # Check RECON regression (new subnets from credential payloads).
        regress_request = await self._maybe_build_regression_request(ctx, slice_)
        if regress_request is not None:
            ctx.pending_regression_request = regress_request
            return StepDecision.REGRESS_TO

        # Check VULN_TRIAGE regression (authenticated access on known hosts).
        vuln_triage_request = await self._maybe_build_vuln_triage_regression(ctx)
        if vuln_triage_request is not None:
            ctx.pending_regression_request = vuln_triage_request
            return StepDecision.REGRESS_TO

        # State-driven advance: all creds validated, no hosts, or hard cap.
        if not slice_["untested_credentials"]:
            return StepDecision.ADVANCE
        if not slice_["targetable_hosts"]:
            return StepDecision.ADVANCE
        if ctx.turn >= self.budget.stage_step_cap:
            return StepDecision.ADVANCE
        return StepDecision.CONTINUE

    async def _maybe_build_regression_request(
        self, ctx: StageContext, slice_: dict[str, Any]
    ) -> RegressionRequest | None:
        """Emit a regression request if credential payloads reference
        in-scope subnets/domains not yet covered by recon."""
        cap = ctx.regression_budget_max
        if cap > 0 and (
            ctx.regressions_used + ctx.regressions_pending >= cap
        ):
            return None

        targets = await self._scan_credentials_for_pivot_targets(ctx)
        if not targets:
            return None

        n_validated = int(slice_.get("validated_credentials_count", 0))
        has_pivot_signal = bool(targets)
        if n_validated >= 1:
            progress_branch = "validated"
        elif has_pivot_signal and ctx.turn >= 2:
            progress_branch = "fresh_phished"
        else:
            return None

        scope = ctx.engagement.get("scope") or {}
        scope_cidrs = list(scope.get("cidrs") or [])
        scope_domains = list(scope.get("domains") or [])
        excluded_cidrs = list(scope.get("excluded_cidrs") or [])
        excluded_domains = list(scope.get("excluded_domains") or [])

        # Existing recon coverage.
        host_ips = await self._known_host_ips(ctx.engagement_id)
        known_domains = await self._known_domain_names(ctx.engagement_id)

        new_cidrs: list[str] = []
        new_domains: list[str] = []
        evidence_ids: list[str] = []
        seen_cidrs: set[str] = set()
        seen_domains: set[str] = set()

        for cred_id, ips, fqdns in targets:
            cred_contributed = False
            for ip in ips:
                if not _ip_in_any_cidr(ip, scope_cidrs):
                    continue
                if _ip_in_any_cidr(ip, excluded_cidrs):
                    continue
                if _ip_in_known_hosts(ip, host_ips):
                    continue
                cidr = _cidr_for_ip(ip)
                if cidr is None:
                    continue
                cred_contributed = True
                if cidr in seen_cidrs:
                    continue
                seen_cidrs.add(cidr)
                new_cidrs.append(cidr)
            for fqdn in fqdns:
                f = fqdn.lower()
                if not _fqdn_in_scope(f, scope_domains):
                    continue
                if _fqdn_in_scope(f, excluded_domains):
                    continue
                if f in known_domains:
                    continue
                cred_contributed = True
                if f in seen_domains:
                    continue
                seen_domains.add(f)
                new_domains.append(f)
            if cred_contributed and cred_id and cred_id not in evidence_ids:
                evidence_ids.append(cred_id)

        if not new_cidrs and not new_domains:
            return None

        delta: dict[str, Any] = {}
        if new_cidrs:
            delta["cidrs"] = new_cidrs
        if new_domains:
            delta["domains"] = new_domains

        rationale_parts = []
        if new_cidrs:
            rationale_parts.append(
                f"credential payloads reference uncovered in-scope CIDRs {new_cidrs}"
            )
        if new_domains:
            rationale_parts.append(
                f"credential payloads reference uncovered in-scope domains {new_domains}"
            )
        if progress_branch == "validated":
            branch_note = (
                f"progress: {n_validated} cred(s) already validated"
            )
        else:
            branch_note = (
                "progress: 0 validated creds yet, but a fresh cred with a "
                f"clear pivot signal exists (turn={ctx.turn})"
            )
        rationale = (
            "lateral: " + "; ".join(rationale_parts) +
            f" [{branch_note}]" +
            " — recon never enumerated this surface; "
            "regress to discover hosts before pivoting."
        )

        ctx.regressions_pending += 1
        return RegressionRequest(
            target_stage=Stage.RECON,
            trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
            delta=delta,
            evidence=evidence_ids,
            rationale=rationale[:1000],
        )

    async def _maybe_build_vuln_triage_regression(
        self, ctx: StageContext
    ) -> RegressionRequest | None:
        """Emit a regression to VULN_TRIAGE when credentials are validated
        (``valid=True``) on hosts already in the graph.

        Authenticated access changes the attack surface for a host — services
        that were previously unexploitable (e.g. authenticated-only SMB shares,
        WMI, RDP with NLA) now become candidates. The VULN_TRIAGE stage should
        re-evaluate the host's services in light of the new credential.

        This is separate from the existing RECON regression that fires for
        new subnets.
        """
        cap = ctx.regression_budget_max
        if cap > 0 and (
            ctx.regressions_used + ctx.regressions_pending >= cap
        ):
            return None

        validated = await self._recently_validated_on_known_hosts(ctx.engagement_id)
        if not validated:
            return None

        # Build evidence and delta from validated (host_id, credential_id) pairs.
        host_ids: list[str] = []
        credential_ids: list[str] = []
        evidence_ids: list[str] = []
        seen_hosts: set[str] = set()
        seen_creds: set[str] = set()
        for host_id, cred_id in validated:
            if host_id and host_id not in seen_hosts:
                seen_hosts.add(host_id)
                host_ids.append(host_id)
            if cred_id and cred_id not in seen_creds:
                seen_creds.add(cred_id)
                credential_ids.append(cred_id)
            # Evidence: both host and credential contribute.
            if host_id and host_id not in evidence_ids:
                evidence_ids.append(host_id)
            if cred_id and cred_id not in evidence_ids:
                evidence_ids.append(cred_id)

        if not host_ids:
            return None

        delta: dict[str, Any] = {
            "host_ids": host_ids,
            "credential_ids": credential_ids,
        }

        rationale = (
            f"lateral: credentials validated on {len(host_ids)} known host(s) "
            f"{host_ids} — authenticated access changes the attack surface; "
            "regress to VULN_TRIAGE to re-evaluate services with the new "
            "credential context (authenticated SMB, WMI, RDP+NLA, etc.)."
        )

        ctx.regressions_pending += 1
        return RegressionRequest(
            target_stage=Stage.VULN_TRIAGE,
            trigger_kind=RegressionTriggerKind.AUTHENTICATED_ACCESS_GAINED,
            delta=delta,
            evidence=evidence_ids,
            rationale=rationale[:1000],
        )

    async def _recently_validated_on_known_hosts(
        self, engagement_id: UUID
    ) -> list[tuple[str, str]]:
        """Return ``(host_id, credential_id)`` pairs where a credential has
        been validated (``valid=True``) and has a ``HAS_SESSION_ON`` edge to a
        host, but there is no prior ``AUTHENTICATED_ACCESS_GAINED`` regression
        already covering that (host, credential) pair.

        To avoid duplicate regression loops, we check that no ExploitCandidate
        nodes exist that were produced by a prior VULN_TRIAGE regression for
        the same host. We approximate this by checking whether any
        ExploitCandidate already references a Finding that AFFECTS a Service
        on the host — if so, triage has already evaluated the host.
        """
        # Find validated credentials with HAS_SESSION_ON edges to hosts.
        creds = await self.repo.get_nodes_by_kind(engagement_id, "Credential")
        pairs: list[tuple[str, str]] = []  # (host_id, cred_id)
        seen: set[tuple[str, str]] = set()
        for c in creds:
            if c.get("valid") is not True:
                continue
            cid = c.get("id")
            if not cid:
                continue
            hosts = await self.repo.get_successors_by_edge_type(
                UUID(str(cid)), "HAS_SESSION_ON"
            )
            for h in hosts:
                if h.get("kind") != "Host" or not h.get("id"):
                    continue
                key = (str(h["id"]), str(cid))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        if not pairs:
            return []

        # Filter out hosts that already have ExploitCandidates linked through
        # findings → services on that host (meaning triage already evaluated
        # the host with credential context).
        hosts_already_triaged = await self._hosts_with_authed_candidates(engagement_id)

        return [(hid, cid) for hid, cid in pairs if hid not in hosts_already_triaged]

    async def _hosts_with_authed_candidates(
        self, engagement_id: UUID
    ) -> set[str]:
        """Return host IDs that already have ExploitCandidates linked through
        a credential-session chain, indicating that VULN_TRIAGE has already
        evaluated the host with authenticated context."""
        # Hosts reachable from a validated credential via HAS_SESSION_ON.
        creds = await self.repo.get_nodes_by_kind(engagement_id, "Credential")
        session_hosts: set[str] = set()
        for c in creds:
            if c.get("valid") is not True or not c.get("id"):
                continue
            for h in await self.repo.get_successors_by_edge_type(
                UUID(str(c["id"])), "HAS_SESSION_ON"
            ):
                if h.get("kind") == "Host" and h.get("id"):
                    session_hosts.add(str(h["id"]))
        if not session_hosts:
            return set()

        # Hosts AFFECTS-ed by a Finding that already has an ExploitCandidate
        # (linked by ExploitCandidate.finding_id == Finding.id).
        candidates = await self.repo.get_nodes_by_kind(engagement_id, "ExploitCandidate")
        covered_finding_ids = {
            str(c.get("finding_id")) for c in candidates if c.get("finding_id")
        }
        if not covered_finding_ids:
            return set()
        findings = await self.repo.get_nodes_by_kind(engagement_id, "Finding")
        hosts_with_authed_candidate: set[str] = set()
        for f in findings:
            fid = f.get("id")
            if not fid or str(fid) not in covered_finding_ids:
                continue
            for n in await self.repo.get_successors_by_edge_type(UUID(str(fid)), "AFFECTS"):
                if n.get("kind") == "Host" and n.get("id"):
                    hosts_with_authed_candidate.add(str(n["id"]))

        return session_hosts & hosts_with_authed_candidate

    async def _scan_credentials_for_pivot_targets(
        self, ctx: StageContext
    ) -> list[tuple[str, list[str], list[str]]]:
        """Extract ``(credential_id, [ips], [fqdns])`` from credential
        ``source`` and ``username`` (LHS of UPN only) fields."""
        creds = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Credential")
        out: list[tuple[str, list[str], list[str]]] = []
        for c in creds:
            source_blob = str(c.get("source") or "")
            username_raw = str(c.get("username") or "")
            username_lhs = username_raw.split("@", 1)[0] if username_raw else ""

            ips_text = " ".join(b for b in (source_blob, username_lhs) if b)
            fqdns_text = source_blob

            if not ips_text.strip() and not fqdns_text.strip():
                continue
            ips = [m.group(0) for m in _IPV4_RE.finditer(ips_text)]
            fqdns = [m.group(0) for m in _FQDN_RE.finditer(fqdns_text)]
            valid_ips: list[str] = []
            for ip in ips:
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    continue
                valid_ips.append(ip)
            if not valid_ips and not fqdns:
                continue
            cred_id = str(c.get("id") or "")
            out.append((cred_id, valid_ips, fqdns))
        return out

    async def _known_host_ips(self, engagement_id: UUID) -> list[str]:
        hosts = await self.repo.get_nodes_by_kind(engagement_id, "Host")
        return [str(h.get("ip") or "") for h in hosts if h.get("ip")]

    async def _known_domain_names(self, engagement_id: UUID) -> set[str]:
        domains = await self.repo.get_nodes_by_kind(engagement_id, "Domain")
        out: set[str] = set()
        for d in domains:
            n = str(d.get("name") or "").lower()
            if n:
                out.add(n)
        return out

    async def _parse_and_persist(
        self,
        ctx: StageContext,
        tool: Tool[Any, Any],
        tool_input: dict[str, Any],
        tool_output: BaseModel,
        tool_run_id: UUID,
    ) -> list[WorldStateNode]:
        nodes: list[WorldStateNode] = []
        rows = getattr(tool_output, "rows", None)
        if rows is None:
            return nodes

        for row in rows:
            row_kind = getattr(row, "kind", None)
            if row_kind == "credential":
                node = await self._persist_credential(
                    ctx, row, tool_run_id, tool_input.get("target")
                )
                if node is not None:
                    nodes.append(node)

        cred_id = await self._resolve_credential_id(
            ctx.engagement_id,
            username=tool_input.get("username"),
            domain=tool_input.get("domain"),
        )
        for row in rows:
            if getattr(row, "kind", None) == "credential":
                continue
            if getattr(row, "status", None) != "success":
                continue
            target_host_id = await self._host_id_for_ip(
                ctx.engagement_id, getattr(row, "target", "")
            )
            if cred_id and target_host_id:
                await self.repo.upsert_edge(
                    Edge(
                        kind=EdgeKind.HAS_SESSION_ON,
                        src_id=cred_id,
                        dst_id=target_host_id,
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                    )
                )
                if getattr(row, "admin", False):
                    await self.repo.upsert_edge(
                        Edge(
                            kind=EdgeKind.LOCAL_ADMIN_ON,
                            src_id=cred_id,
                            dst_id=target_host_id,
                            engagement_id=ctx.engagement_id,
                            provenance=[tool_run_id],
                        )
                    )
        return nodes

    async def _persist_credential(
        self,
        ctx: StageContext,
        row: Any,
        tool_run_id: UUID,
        from_host: str | None,
    ) -> WorldStateNode | None:
        """Persist a Credential and wire a HAS_SESSION_ON edge to the source host."""
        username = getattr(row, "username", None)
        if not username:
            return None
        domain = getattr(row, "domain", None)
        host_id: UUID | None = None
        if from_host:
            host_id = await self._host_id_for_ip(ctx.engagement_id, from_host)
        cred = Credential(
            id=deterministic_node_id(
                "Credential",
                ctx.engagement_id,
                natural_key_for_credential(
                    username=username,
                    domain=domain,
                    source_host_id=host_id,
                ),
            ),
            engagement_id=ctx.engagement_id,
            provenance=[tool_run_id],
            username=username,
            password=getattr(row, "password", None),
            nt_hash=getattr(row, "ntlm", None),
            domain=domain,
            source=getattr(row, "source", "mimikatz"),
            valid=None,  # un-tested until LateralAgent picks it up next turn
        )
        try:
            await self.repo.upsert_node(cred)
        except Exception:
            log.warning("lateral.credential_persist_failed", username=username, exc_info=True)
            return None
        if host_id is not None:
            try:
                await self.repo.upsert_edge(
                    Edge(
                        kind=EdgeKind.HAS_SESSION_ON,
                        src_id=cred.id,
                        dst_id=host_id,
                        engagement_id=ctx.engagement_id,
                        provenance=[tool_run_id],
                    )
                )
            except Exception:
                log.warning(
                    "lateral.credential_session_edge_failed",
                    cred_id=str(cred.id),
                    exc_info=True,
                )
        return cred

    async def _resolve_credential_id(
        self, engagement_id: UUID, *, username: str | None, domain: str | None
    ) -> UUID | None:
        if not username:
            return None
        # Use secondary index for O(1) lookup by username.
        creds = await self.repo.get_nodes_by_kind_field(
            engagement_id, "Credential", "username", username
        )
        if not creds:
            return None
        for c in creds:
            if domain and c.get("domain") != domain:
                continue
            cid = c.get("id")
            if cid:
                try:
                    return UUID(str(cid))
                except (ValueError, TypeError):
                    continue
        # If domain was specified but no match, return None.
        if domain:
            return None
        # No domain filter -- return the first match.
        cid = creds[0].get("id")
        if cid:
            try:
                return UUID(str(cid))
            except (ValueError, TypeError):
                pass
        return None

    async def _host_id_for_ip(self, engagement_id: UUID, ip: str) -> UUID | None:
        if not ip:
            return None
        try:
            hosts = await self.repo.get_nodes_by_kind_field(
                engagement_id, "Host", "ip", ip
            )
        except Exception:
            return None
        if not hosts:
            return None
        hid = hosts[0].get("id")
        if hid:
            try:
                return UUID(str(hid))
            except (ValueError, TypeError):
                pass
        return None
