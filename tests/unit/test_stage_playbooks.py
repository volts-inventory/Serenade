"""R3 #G Phase G — each stage agent embodies the senior pentester role.

Each agent now ships a ``_pentester_playbook(world_state_slice) -> str``
method that returns the stage-specific workflow description the
``_objective_for_turn`` feeds to the planner. The playbook encodes:

* the senior-operator mental model for the stage,
* a branching decision tree over current world-state,
* tool preference order,
* exit/escalation criteria,
* refusal rules (matching the persona).

These tests lock in three contracts:

1. Every stage agent ships a non-trivial ``_pentester_playbook`` whose
   output references stage-appropriate vocabulary. New agents that
   skip the playbook (or copy-paste from another stage) fail loud.
2. Per-stage ``_decide_advance`` returns ADVANCE on a representative
   "done" world-state and CONTINUE on a representative "still working"
   state. This is the structural exit criterion the playbook talks about.
3. Per-stage ``max_loop_turns`` matches the table in the Phase G plan —
   a casual override that drifts the value fails the test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import agents.net_enum as net_enum_mod
import agents.vuln_triage as vuln_triage_mod
import agents.web_enum as web_enum_mod
from agents.base import StageContext, StepDecision
from agents.exploit import ExploitAgent
from agents.lateral import LateralAgent
from agents.loot import LootAgent
from agents.net_enum import NetEnumAgent
from agents.poc_runtime import PoCLibrary
from agents.privesc import PrivEscAgent
from agents.recon import ReconAgent
from agents.report import ReportAgent
from agents.vuln_triage import VulnTriageAgent
from agents.web_enum import WebEnumAgent
from worldstate.schema import Stage

# --------------------------------------------------------------------------- #
# Per-stage max_loop_turns table                                              #
# --------------------------------------------------------------------------- #
# Forcing function: drift this value in the agent class without updating the
# Phase G plan and the test fails. The values come from the table in the
# Phase G prompt — workflow-shape-driven, not arbitrary.

EXPECTED_MAX_LOOP_TURNS: dict[str, int] = {
    "recon": 20,
    "web_enum": 18,
    "net_enum": 18,
    "vuln_triage": 12,
    "exploit": 8,
    "privesc": 12,
    "lateral": 10,
    "loot": 8,
    "report": 6,
}


# --------------------------------------------------------------------------- #
# Per-stage "playbook must mention" keywords                                  #
# --------------------------------------------------------------------------- #
# Each agent's playbook must reference its stage's vocabulary so a
# copy-paste between stages fails immediately. Keywords are chosen so
# they appear in the playbook but NOT in another agent's playbook.

EXPECTED_PLAYBOOK_KEYWORDS: dict[str, list[str]] = {
    "recon": ["discovery", "service-version", "ports"],
    "web_enum": ["fingerprint", "wordlist", "ffuf"],
    "net_enum": ["anonymous", "authenticated", "BloodHound"],
    "vuln_triage": ["rank", "candidate", "version"],
    "exploit": ["curated", "PoC", "verify"],
    "privesc": ["enumerate", "linpeas", "kernel"],
    "lateral": ["validate", "credential", "lockout"],
    "loot": ["in-place", "exfiltrate", "read"],
    "report": ["evidence", "CVSS", "tool_run_id"],
}


# --------------------------------------------------------------------------- #
# Agent construction helpers                                                  #
# --------------------------------------------------------------------------- #


def _stub_repo() -> MagicMock:
    """A repo mock that returns benign defaults for any query the
    agents' ``_build_world_state_slice`` / ``_decide_advance`` use."""
    repo = MagicMock()
    repo.count = AsyncMock(return_value={
        "Host": 0,
        "Service": 0,
        "WebApp": 0,
        "Endpoint": 0,
        "Finding": 0,
        "ExploitCandidate": 0,
        "Session": 0,
    })
    repo.get_nodes_by_kind = AsyncMock(return_value=[])
    repo.run_cypher = AsyncMock(return_value=[])
    repo.run_read_cypher = AsyncMock(return_value=[])
    return repo


def _agent_instances() -> dict[str, Any]:
    """Build one instance of each stage agent with default constructors.

    VulnTriage and Exploit / Lateral / Report take extra kwargs; the
    playbook test does not exercise them so None defaults are fine.
    """
    shared = {
        "router": MagicMock(),
        "repo": _stub_repo(),
        "registry": MagicMock(),
        "policy": MagicMock(),
        "sandbox": MagicMock(),
    }

    return {
        "recon": ReconAgent(**shared),
        "web_enum": WebEnumAgent(**shared),
        "net_enum": NetEnumAgent(**shared),
        "vuln_triage": VulnTriageAgent(triage_council=None, **shared),
        "exploit": ExploitAgent(library=PoCLibrary(), **shared),
        "privesc": PrivEscAgent(**shared),
        "lateral": LateralAgent(**shared),
        "loot": LootAgent(**shared),
        "report": ReportAgent(**shared),
    }


def _ctx(stage: Stage = Stage.RECON) -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual", "scope": {"cidrs": ["10.0.0.0/24"]}},
        stage=stage,
    )


# --------------------------------------------------------------------------- #
# 1. Playbook present + stage-vocabulary present                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(EXPECTED_PLAYBOOK_KEYWORDS))
def test_each_agent_has_pentester_playbook(name: str) -> None:
    """Forcing function: every stage agent ships a callable
    ``_pentester_playbook`` that returns a non-trivial string."""
    agent = _agent_instances()[name]
    assert hasattr(agent, "_pentester_playbook"), (
        f"{type(agent).__name__} missing _pentester_playbook"
    )
    # Call with a representative empty slice — every agent must handle
    # an empty world-state without crashing (idle branch).
    out = agent._pentester_playbook(
        {
            "coverage": {
                "hosts": 0,
                "services": 0,
                "webapps": 0,
                "endpoints": 0,
                "findings": 0,
                "candidates": 0,
                "sessions": 0,
            },
            # Per-agent fields each playbook may consult:
            "scope": {},
            "webapps": [],
            "ad_candidates": [],
            "known_credentials": [],
            "hosts_with_versions": [],
            "total_versioned_services": 0,
            "library": [],
            "candidate_count": 0,
            "attempt_count": 0,
            "success_count": 0,
            "unprivileged_unscanned": [],
            "sessions_total": 0,
            "credentials_count": 0,
            "untested_credentials": [],
            "targetable_hosts": [],
            "validated_credentials_count": 0,
            "sessions": 0,
            "unlooted_sessions": [],
            "destructive_opt_in": False,
            "findings_by_severity": {},
            "findings_without_evidence": 0,
            "attack_paths": 0,
        }
    )
    assert isinstance(out, str)
    assert len(out) >= 100, f"{name} playbook is suspiciously short ({len(out)} chars)"
    # Every playbook should self-identify as a playbook so the planner
    # knows it's reading a structured workflow, not a one-liner.
    assert "PLAYBOOK" in out, f"{name} playbook missing PLAYBOOK marker"


@pytest.mark.parametrize(
    "name,keywords",
    [(n, kws) for n, kws in EXPECTED_PLAYBOOK_KEYWORDS.items()],
)
def test_playbook_uses_stage_appropriate_vocabulary(
    name: str, keywords: list[str]
) -> None:
    """Each agent's playbook must reference the vocabulary that's
    distinctive to its stage — recon mentions 'discovery'/'ports',
    exploit mentions 'curated'/'PoC', loot mentions 'in-place', etc.

    The check runs across all branches of the playbook (idle + working)
    so the keyword presence is not branch-conditional.
    """
    agent = _agent_instances()[name]
    # Combine outputs across all the relevant slice shapes for this
    # agent: idle (no work) + each non-idle branch. This catches
    # branches that drop the vocabulary by accident.
    slices = [_idle_slice(), *_branch_slices_for(name)]
    parts = [agent._pentester_playbook(s) for s in slices]
    combined = "\n".join(parts).lower()
    missing = [kw for kw in keywords if kw.lower() not in combined]
    assert not missing, (
        f"{name} playbook does not reference expected keywords {missing}. "
        f"Got: {combined[:400]!r}"
    )


def _idle_slice() -> dict[str, Any]:
    return {
        "coverage": {
            "hosts": 0,
            "services": 0,
            "webapps": 0,
            "endpoints": 0,
            "findings": 0,
            "candidates": 0,
            "sessions": 0,
        },
        "scope": {},
        "webapps": [],
        "ad_candidates": [],
        "known_credentials": [],
        "hosts_with_versions": [],
        "total_versioned_services": 0,
        "library": [],
        "candidate_count": 0,
        "attempt_count": 0,
        "success_count": 0,
        "unprivileged_unscanned": [],
        "sessions_total": 0,
        "credentials_count": 0,
        "untested_credentials": [],
        "targetable_hosts": [],
        "validated_credentials_count": 0,
        "sessions": 0,
        "unlooted_sessions": [],
        "destructive_opt_in": False,
        "findings_by_severity": {},
        "findings_without_evidence": 0,
        "attack_paths": 0,
    }


def _branch_slices_for(name: str) -> list[dict[str, Any]]:
    """Build a list of slices that together exercise every non-idle
    branch of the agent's playbook. Used by the keyword-coverage test
    so a branch that drops a stage-distinctive keyword fails the test.
    """
    if name == "net_enum":
        # Two slices: anonymous phase (creds=[]) + authenticated phase (creds present).
        anon = _idle_slice()
        anon["ad_candidates"] = [{"ip": "10.0.0.1", "os_family": "windows",
                                  "ad_ports": [445]}]
        auth = dict(anon)
        auth["known_credentials"] = [{"username": "alice", "domain": "LAB",
                                      "source": "lsa"}]
        return [anon, auth]
    if name == "lateral":
        # Two slices: untested (validate phase) + all-validated (pivot phase).
        untested = _idle_slice()
        untested["untested_credentials"] = [
            {"id": str(uuid4()), "username": "alice", "domain": "LAB"}
        ]
        untested["targetable_hosts"] = [{"id": str(uuid4()), "ip": "10.0.0.1"}]
        validated = _idle_slice()
        validated["validated_credentials_count"] = 2
        return [untested, validated]
    if name == "report":
        # Two slices: findings without evidence (gap) + findings with evidence (drafting).
        gap = _idle_slice()
        gap["findings_by_severity"] = {"high": 1}
        gap["findings_without_evidence"] = 1
        ok = _idle_slice()
        ok["findings_by_severity"] = {"high": 1}
        return [gap, ok]
    return [_busy_slice_for(name)]


def _busy_slice_for(name: str) -> dict[str, Any]:
    """Build a representative 'mid-stage' slice for one agent. Each
    branch exercises a different non-idle code path of the playbook."""
    s = _idle_slice()
    if name == "recon":
        s["coverage"]["hosts"] = 3
        s["coverage"]["services"] = 1
        s["scope"] = {"cidrs": ["10.0.0.0/24"]}
    elif name == "web_enum":
        s["webapps"] = [
            {"id": str(uuid4()), "base_url": "http://w1", "title": "t",
             "technologies": [], "endpoints": 0}
        ]
    elif name == "net_enum":
        # Include both anonymous (no creds) and authenticated (creds)
        # branch material so the playbook covers both phases when
        # concatenated across slice shapes.
        s["ad_candidates"] = [{"ip": "10.0.0.1", "os_family": "windows",
                               "ad_ports": [445]}]
        s["known_credentials"] = [{"username": "alice", "domain": "LAB",
                                   "source": "lsa"}]
    elif name == "vuln_triage":
        s["hosts_with_versions"] = [
            {"ip": "10.0.0.1",
             "services": [{"port": 80, "name": "http", "product": "nginx",
                          "version": "1.18"}]}
        ]
        s["total_versioned_services"] = 1
    elif name == "exploit":
        s["library"] = [{"id": "p1", "service": "smb", "version_range": "1.0",
                         "auto_run_allowed": True, "destructive": False}]
        s["candidate_count"] = 2
    elif name == "privesc":
        s["sessions_total"] = 2
        s["unprivileged_unscanned"] = [
            {"session_id": str(uuid4()), "host_id": str(uuid4()),
             "user": "ubuntu", "type": "shell"}
        ]
    elif name == "lateral":
        s["untested_credentials"] = [
            {"id": str(uuid4()), "username": "alice", "domain": "LAB",
             "source": "lsa", "has_password": True, "has_hash": False}
        ]
        s["targetable_hosts"] = [{"id": str(uuid4()), "ip": "10.0.0.1",
                                  "os_family": "windows"}]
    elif name == "loot":
        s["sessions"] = 1
        s["unlooted_sessions"] = [
            {"session_id": str(uuid4()), "host_id": str(uuid4()),
             "user": "root"}
        ]
    elif name == "report":
        s["findings_by_severity"] = {"high": 1}
    return s


# --------------------------------------------------------------------------- #
# 2. _decide_advance returns ADVANCE on done, CONTINUE on working             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recon_decides_advance_on_done_state() -> None:
    repo = _stub_repo()
    # 3 hosts + 3 services → 100% coverage → ADVANCE.
    repo.count = AsyncMock(return_value={"Host": 3, "Service": 3})
    agent = ReconAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                       policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.RECON)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_recon_decides_continue_on_working_state() -> None:
    repo = _stub_repo()
    # 3 hosts, 0 services — there's work left (hosts known, services to
    # enumerate). Coverage < 0.9*hosts → not ADVANCE.
    repo.count = AsyncMock(return_value={"Host": 3, "Service": 0})
    agent = ReconAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                       policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.RECON)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_web_enum_decides_advance_on_done_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo()
    # endpoints >= webapps → fingerprint succeeded for each → ADVANCE.
    repo.count = AsyncMock(return_value={"WebApp": 2, "Endpoint": 5})
    agent = WebEnumAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())

    async def _summary(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []
    monkeypatch.setattr(web_enum_mod, "webapps_summary", _summary)
    ctx = _ctx(Stage.ENUMERATION)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_web_enum_decides_continue_on_working_state() -> None:
    repo = _stub_repo()
    # 2 webapps, 0 endpoints — fingerprinting still pending.
    repo.count = AsyncMock(return_value={"WebApp": 2, "Endpoint": 0})
    agent = WebEnumAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.ENUMERATION)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_net_enum_decides_advance_when_no_ad_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo()
    # hosts exist but the slice has no ad_candidates → ADVANCE (nothing
    # for NetEnum to do).
    repo.count = AsyncMock(return_value={"Host": 2, "Service": 4})

    async def _no_ad(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"host": {"ip": "10.0.0.5"}, "services": [{"port": 80}]}]
    monkeypatch.setattr(net_enum_mod, "hosts_with_open_services", _no_ad)
    agent = NetEnumAgent(router=MagicMock(), repo=repo,
                         registry=MagicMock(), policy=MagicMock(),
                         sandbox=MagicMock())
    ctx = _ctx(Stage.ENUMERATION)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_net_enum_decides_continue_during_authenticated_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Host": 2, "Service": 4})
    repo.get_nodes_by_kind = AsyncMock(return_value=[])

    async def _ad(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"host": {"ip": "10.0.0.5", "os_family": "windows"},
                 "services": [{"port": 445}, {"port": 389}]}]
    monkeypatch.setattr(net_enum_mod, "hosts_with_open_services", _ad)
    agent = NetEnumAgent(router=MagicMock(), repo=repo,
                         registry=MagicMock(), policy=MagicMock(),
                         sandbox=MagicMock())
    ctx = _ctx(Stage.ENUMERATION)
    ctx.turn = 1
    # AD surface present, no creds, turn 1 — should CONTINUE
    # (anonymous phase still has work).
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_vuln_triage_decides_advance_when_no_versioned_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Service": 0, "ExploitCandidate": 0})

    async def _hs(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []
    monkeypatch.setattr(vuln_triage_mod, "hosts_with_open_services", _hs)
    agent = VulnTriageAgent(router=MagicMock(), repo=repo,
                            registry=MagicMock(), policy=MagicMock(),
                            sandbox=MagicMock(), triage_council=None)
    ctx = _ctx(Stage.VULN_TRIAGE)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_vuln_triage_decides_continue_with_unranked_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Service": 3, "ExploitCandidate": 0})

    async def _hs(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"host": {"ip": "10.0.0.1"},
                 "services": [
                     {"product": "nginx", "version": "1.18", "port": 80,
                      "name": "http"},
                     {"product": "openssh", "version": "8.4", "port": 22,
                      "name": "ssh"},
                 ]}]
    monkeypatch.setattr(vuln_triage_mod, "hosts_with_open_services", _hs)
    agent = VulnTriageAgent(router=MagicMock(), repo=repo,
                            registry=MagicMock(), policy=MagicMock(),
                            sandbox=MagicMock(), triage_council=None)
    ctx = _ctx(Stage.VULN_TRIAGE)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_exploit_decides_advance_on_success() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"ExploitCandidate": 2})
    # 1 candidate, 1 attempt, 1 success → ADVANCE.
    attempts = [{"outcome": "success"}]
    candidates = [{"id": str(uuid4())}, {"id": str(uuid4())}]

    async def _by_kind(_eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "ExploitCandidate":
            return candidates
        if kind == "ExploitAttempt":
            return attempts
        return []
    repo.get_nodes_by_kind = AsyncMock(side_effect=_by_kind)
    agent = ExploitAgent(library=PoCLibrary(), router=MagicMock(), repo=repo,
                         registry=MagicMock(), policy=MagicMock(),
                         sandbox=MagicMock())
    ctx = _ctx(Stage.EXPLOIT)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_exploit_decides_continue_with_unattempted_candidates() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"ExploitCandidate": 3})
    candidates = [{"id": str(uuid4())} for _ in range(3)]
    attempts: list[dict[str, Any]] = []

    async def _by_kind(_eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "ExploitCandidate":
            return candidates
        if kind == "ExploitAttempt":
            return attempts
        return []
    repo.get_nodes_by_kind = AsyncMock(side_effect=_by_kind)
    agent = ExploitAgent(library=PoCLibrary(), router=MagicMock(), repo=repo,
                         registry=MagicMock(), policy=MagicMock(),
                         sandbox=MagicMock())
    ctx = _ctx(Stage.EXPLOIT)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_privesc_decides_advance_when_no_sessions() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Session": 0})
    agent = PrivEscAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.PRIVESC)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_privesc_decides_continue_with_unscanned_sessions() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Session": 1})
    session_id = str(uuid4())
    sessions = [{"id": session_id, "privileged": False, "host_id": str(uuid4()),
                 "user": "ubuntu", "type": "shell"}]
    repo.get_nodes_by_kind = AsyncMock(return_value=sessions)
    # No linpeas-done sessions: rows=[]
    repo.run_cypher = AsyncMock(return_value=[])
    agent = PrivEscAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.PRIVESC)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_lateral_decides_advance_when_no_untested_creds() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={})
    repo.get_nodes_by_kind = AsyncMock(return_value=[])
    repo.run_cypher = AsyncMock(return_value=[])
    agent = LateralAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.LATERAL)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_lateral_decides_continue_with_untested_creds() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={})
    creds = [{"id": str(uuid4()), "username": "alice", "valid": None}]
    hosts = [{"id": str(uuid4()), "ip": "10.0.0.5", "os_family": "windows"}]

    async def _by_kind(eid: Any, kind: str) -> list[dict[str, Any]]:
        if kind == "Credential":
            return creds
        if kind == "Host":
            return hosts
        return []
    repo.get_nodes_by_kind = AsyncMock(side_effect=_by_kind)
    # Untested cred has no sessions → validated_credentials_count == 0 → CONTINUE.
    repo.get_successors_by_edge_type = AsyncMock(return_value=[])
    agent = LateralAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                         policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.LATERAL)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_loot_decides_advance_when_no_sessions() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Session": 0})
    repo.get_nodes_by_kind = AsyncMock(return_value=[])
    agent = LootAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                      policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.LOOT)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_loot_decides_continue_with_unlooted_sessions() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Session": 1})
    sessions = [{"id": str(uuid4()), "host_id": str(uuid4()), "user": "root"}]
    repo.get_nodes_by_kind = AsyncMock(return_value=sessions)
    repo.run_cypher = AsyncMock(return_value=[])
    agent = LootAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                      policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.LOOT)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


@pytest.mark.asyncio
async def test_report_decides_advance_when_findings_have_evidence() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Finding": 1})

    async def _findings(eid: Any) -> list[dict[str, Any]]:
        return [
            {"title": "f1", "severity": "high",
             "evidence": [{"tool_run_id": str(uuid4())}]}
        ]
    repo.run_cypher = AsyncMock(side_effect=lambda c, p: [{"f": _f} for _f in [
        {"title": "f1", "severity": "high",
         "evidence": [{"tool_run_id": str(uuid4())}]}
    ]] if "Finding" in c else ([{"p": {"objective": "domain admin", "cost": 1}}]
                                if "AttackPath" in c else []))
    agent = ReportAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                        policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.REPORT)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE


@pytest.mark.asyncio
async def test_report_decides_continue_when_evidence_missing_early_turn() -> None:
    repo = _stub_repo()
    repo.count = AsyncMock(return_value={"Finding": 1})
    # Finding with no evidence + turn 1 → should give planner a chance.
    repo.get_nodes_by_kind = AsyncMock(side_effect=lambda eid, kind: (
        [{"title": "f1", "severity": "high", "evidence": []}]
        if kind == "Finding" else []
    ))
    agent = ReportAgent(router=MagicMock(), repo=repo, registry=MagicMock(),
                        policy=MagicMock(), sandbox=MagicMock())
    ctx = _ctx(Stage.REPORT)
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE


# --------------------------------------------------------------------------- #
# 3. max_loop_turns matches the per-stage table                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [(n, v) for n, v in EXPECTED_MAX_LOOP_TURNS.items()],
)
def test_max_loop_turns_matches_per_stage_table(name: str, expected: int) -> None:
    """A casual override that drifts ``max_loop_turns`` away from the
    Phase G table fails this test — forcing the author to either update
    the table or revert the override."""
    agent = _agent_instances()[name]
    actual = type(agent).max_loop_turns
    assert actual == expected, (
        f"{type(agent).__name__}.max_loop_turns = {actual}, expected {expected}"
    )


def test_max_loop_turns_propagates_to_default_budget() -> None:
    """When no explicit budget is passed, the per-stage class attribute
    overrides the StepBudget default of 15. Tests / callers that pass
    an explicit StepBudget keep their own value (verified by the
    existing test_interactive_max_turns_cap_respected)."""
    instances = _agent_instances()
    for name, agent in instances.items():
        assert agent.budget.interactive_max_turns == EXPECTED_MAX_LOOP_TURNS[name], (
            f"{type(agent).__name__} budget.interactive_max_turns = "
            f"{agent.budget.interactive_max_turns}, expected "
            f"{EXPECTED_MAX_LOOP_TURNS[name]}"
        )


# --------------------------------------------------------------------------- #
# 4. Playbooks are mutually distinct (no copy-paste accidents)                #
# --------------------------------------------------------------------------- #


def test_playbooks_are_mutually_distinct() -> None:
    """Two agents must not return identical playbook strings on the
    same world-state slice. Catches accidental copy-paste between
    stages — e.g. if a refactor extracts a 'common idle' helper that
    returns the same string for all 9 agents."""
    instances = _agent_instances()
    idle = _idle_slice()
    outputs = {n: a._pentester_playbook(idle) for n, a in instances.items()}
    seen: dict[str, str] = {}
    for name, out in outputs.items():
        if out in seen:
            pytest.fail(
                f"{name} playbook is identical to {seen[out]} playbook "
                "on the idle slice (copy-paste accident)"
            )
        seen[out] = name
    assert len(set(outputs.values())) == len(outputs)
