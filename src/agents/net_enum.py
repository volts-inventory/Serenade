"""NetEnumAgent — AD/SMB/LDAP-targeted enumeration.

Sibling to WebEnumAgent. Runs in the ENUMERATION stage when the world-state
shows Windows hosts with 445/389/5985 open. Tools: netexec (smb/ldap/winrm).

Followups it can request from the operator (via approval queue):
- After listing users via LDAP, propose AS-REP roast on the user list
- After enumerating Kerberoastable SPNs, propose impacket_kerberoast
- After confirming creds work via NetExec, propose BloodHound collection
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision, _truncate_slice
from llm.prompts.base import PERSONA_NET_ENUM
from tools.base import Tool
from worldstate.queries import coverage, hosts_with_open_services
from worldstate.schema import (
    Credential,
    Stage,
    WorldStateNode,
    deterministic_node_id,
    natural_key_for_credential,
)

log = structlog.get_logger(__name__)


_AD_PORTS = {445, 389, 636, 5985, 5986, 88, 3268}


class NetEnumAgent(BaseAgent):
    """Stage: ENUMERATION (shares the stage with WebEnumAgent; both feed candidates)."""

    stage = Stage.ENUMERATION
    system_persona = PERSONA_NET_ENUM
    # NetEnum progresses anonymous → authenticated → BloodHound. Each
    # phase can fan out across multiple DC candidates; 18 turns lets the
    # progression complete without forcing the stage to advance early.
    max_loop_turns = 18

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        cov = await coverage(self.repo, ctx.engagement_id)
        hosts = await hosts_with_open_services(self.repo, ctx.engagement_id)
        # Build the full AD-candidate list, then cap; this
        # gives the planner an ordered prefix + a count of dropped
        # candidates rather than silently looking at only the first 20.
        ad_candidates: list[dict[str, Any]] = []
        for entry in hosts:
            services = entry["services"]
            ad_ports = [
                int(s.get("port", 0)) for s in services if int(s.get("port", 0)) in _AD_PORTS
            ]
            if ad_ports:
                ad_candidates.append(
                    {
                        "ip": entry["host"].get("ip"),
                        "os_family": entry["host"].get("os_family"),
                        "ad_ports": ad_ports,
                    }
                )
        creds = await self.repo.get_nodes_by_kind(ctx.engagement_id, "Credential")
        return {
            "coverage": cov,
            "ad_candidates": _truncate_slice(ad_candidates, kind_label="Host"),
            "known_credentials": [
                {
                    "username": c.get("username"),
                    "domain": c.get("domain"),
                    "source": c.get("source"),
                }
                for c in creds[:10]
            ],
            "turn": ctx.turn,
            "max_turns": self.budget.max_turns,
        }

    def _pentester_playbook(self, slice_: dict[str, Any]) -> str:
        """Senior AD-enumeration specialist's per-turn workflow.

        Three-phase progression encoded as branches:
          1. anonymous / null session — list shares, look for SMB signing,
             enumerate the domain controller from outside.
          2. authenticated — once we have a verified credential, list
             users / groups / kerberoastable SPNs.
          3. BloodHound collection — only after the authenticated phase
             produced ground truth.

        Hard refuses brute-force per persona (that's lateral-stage).
        """
        ad = slice_.get("ad_candidates", [])
        creds = slice_.get("known_credentials", [])
        n_ad = len(ad)
        n_creds = len(creds)
        if not ad:
            return (
                f"PLAYBOOK [net_enum · idle]. State: ad_candidates={n_ad}. No host "
                "exposes the AD port set (445/389/636/5985/5986/88/3268). "
                "Nothing for NetEnum to enumerate; emit a final-state message "
                "with no tool call to terminate the loop."
            )
        if n_creds == 0:
            ips = [c.get("ip") for c in ad[:5]]
            return (
                f"PLAYBOOK [net_enum · phase 1 of 3: anonymous]. State: "
                f"ad_candidates={n_ad}, known_credentials=0. STEP: run `netexec` "
                f"with protocol=smb against the AD hosts {ips}: "
                "action=auth_check (null-session probe, no creds), then "
                "action=list_shares. Be specific about which DC and which "
                "protocol — SMB / LDAP / RPC / Kerberos surfaces are not "
                "interchangeable. "
                "REFUSE: brute-force of any account, even one — that is a "
                "lateral-movement decision, not enumeration. ESCALATE: if "
                "the account-lockout policy is unknown (we cannot bound "
                "spray risk before knowing it)."
            )
        # Authenticated phase: we have at least one cred; enumerate properly.
        ips = [c.get("ip") for c in ad[:5]]
        return (
            f"PLAYBOOK [net_enum · phase 2 of 3: authenticated]. State: "
            f"ad_candidates={n_ad}, known_credentials={n_creds}. STEP: run "
            f"`netexec` against {ips} with the known creds — "
            "protocol=smb action=loggedon_users, then protocol=ldap "
            "action=list_users / list_groups / kerberoastable_spns. Prefer "
            "ldapsearch / NetExec over wmiexec (noise comes last). "
            "BloodHound collection runs after net_enum: only "
            "propose bloodhound_query after the user+group inventory is "
            "captured. REFUSE: brute-force, password spray. ESCALATE: any "
            "request that requires a credential whose validity we have not "
            "confirmed."
        )

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return self._pentester_playbook(slice_)

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        cov = await coverage(self.repo, ctx.engagement_id)
        # Nothing to enumerate.
        if cov["hosts"] == 0:
            return StepDecision.ADVANCE
        # Reuse the already-built slice from ``step()``.
        slice_ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)
        ad = slice_.get("ad_candidates", [])
        creds = slice_.get("known_credentials", [])
        # Advance when we have either no AD surface to enumerate
        # OR the authenticated phase has produced creds — the inventory
        # exists, and any deeper work (BloodHound) can run from the
        # post_exploit stage. Otherwise CONTINUE so the anonymous /
        # authenticated phases can finish their work.
        if not ad:
            return StepDecision.ADVANCE
        if creds and ctx.turn >= 2:
            # Authenticated phase has produced creds and we've had at
            # least two turns to inventory — advance.
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
        # Inspect by attribute presence; avoid importing NetExecOutput at top level.
        rows = getattr(tool_output, "rows", None)
        if not rows:
            return nodes
        for row in rows:
            if getattr(row, "status", None) == "success":
                # Confirmed working credentials -> persist as Credential.
                # Deterministic id keyed on (username,
                # domain, source_host) so the same successful auth via
                # netexec doesn't double-write the cred on a re-parse.
                username = getattr(row, "username", None) or tool_input.get("username")
                domain = getattr(row, "domain", None) or tool_input.get("domain")
                cred = Credential(
                    id=deterministic_node_id(
                        "Credential",
                        ctx.engagement_id,
                        natural_key_for_credential(
                            username=username,
                            domain=domain,
                            # netexec validates against a target host but the
                            # row doesn't always carry that host's UUID; we
                            # leave it unset so cred identity is consistent
                            # across discovery vs. validation paths.
                            source_host_id=None,
                        ),
                    ),
                    engagement_id=ctx.engagement_id,
                    provenance=[tool_run_id],
                    username=username,
                    password=tool_input.get("password"),
                    nt_hash=tool_input.get("nt_hash"),
                    domain=domain,
                    source=f"netexec_{getattr(row, 'protocol', 'smb')}",
                    valid=True,
                )
                nodes.append(cred)
        return nodes
