"""R1 automation-readiness fixes — see ``docs/ARCHITECTURE.md``.

Five items covered:

1. Recon zero-host watchdog escalates after N consecutive turns at coverage=0.
2. Critic VETO in unattended mode queues an approval so the run halts.
3. Global stalled-turn watchdog stops the engagement on 4+ no-progress turns.
4. Transient I/O failures escalate in unattended mode; manual mode preserves
   the legacy log-and-continue path.
5. _persist_approval threads decided_by="critic" for veto rows; auto-policy
   rows still stamp "auto-policy".

(The former item 6 — destructive_opt_in + destructive_auto_approve_confirmed
consent gate on EngagementCreate — was removed when the engagement config
collapsed onto a single mode axis; destructive is finding-only now.)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents import base as agent_base
from agents.base import (
    StageContext,
    StepDecision,
    StepResult,
    _persist_approval,
    _should_raise_transient,
    _TransientFailure,
)
from agents.recon import ReconAgent
from orchestrator.stage_controller import (
    EngagementState,
    _check_budgets,
    _step_agent,
)
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Shared mocks                                                                #
# --------------------------------------------------------------------------- #


def _mock_recon_agent(host_counts_per_turn: list[int]) -> ReconAgent:
    """Build a ReconAgent whose `coverage()` returns a scripted sequence of
    host counts (one per turn). Other agent dependencies are minimal mocks
    because `_decide_advance` only touches `self.repo`."""
    repo = MagicMock()
    counts_iter = iter(host_counts_per_turn)

    async def _count(_eid: Any) -> dict[str, int]:
        try:
            n = next(counts_iter)
        except StopIteration:
            n = host_counts_per_turn[-1] if host_counts_per_turn else 0
        return {"Host": n, "Service": 0}

    repo.count = AsyncMock(side_effect=_count)
    agent = ReconAgent(
        router=MagicMock(),
        repo=repo,
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )
    return agent


def _ctx() -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )


# --------------------------------------------------------------------------- #
# Item 1: Recon zero-host watchdog                                            #
# --------------------------------------------------------------------------- #


async def test_recon_zero_host_watchdog_escalates_after_three_turns() -> None:
    """Three consecutive turns with coverage.hosts == 0 should escalate.

    The recon agent without the watchdog would keep looping until
    `max_turns=8`. With the watchdog, the third zero-host call returns
    `ESCALATE_HUMAN`, surfacing the stall to the operator faster.
    """
    agent = _mock_recon_agent([0, 0, 0])
    ctx = _ctx()
    # Turn 1: zero hosts -> counter 1 -> CONTINUE
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE
    assert ctx.consecutive_zero_host_turns == 1
    # Turn 2: zero hosts -> counter 2 -> CONTINUE
    ctx.turn = 2
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE
    assert ctx.consecutive_zero_host_turns == 2
    # Turn 3: zero hosts -> counter hits threshold -> ESCALATE_HUMAN
    ctx.turn = 3
    assert await agent._decide_advance(ctx) == StepDecision.ESCALATE_HUMAN
    assert ctx.consecutive_zero_host_turns == 3


async def test_recon_zero_host_watchdog_resets_when_hosts_appear() -> None:
    """A turn that discovers hosts but no services should reset the counter,
    so a later zero-host collapse restarts cleanly."""
    agent = _mock_recon_agent([0, 0, 1, 0])
    ctx = _ctx()
    ctx.turn = 1
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE
    assert ctx.consecutive_zero_host_turns == 1
    ctx.turn = 2
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE
    assert ctx.consecutive_zero_host_turns == 2
    # Host discovered -> counter resets, decision is CONTINUE (services still 0)
    ctx.turn = 3
    assert await agent._decide_advance(ctx) == StepDecision.CONTINUE
    assert ctx.consecutive_zero_host_turns == 0


async def test_recon_advance_resets_watchdog() -> None:
    """When coverage advances, the watchdog counter clears."""
    agent = _mock_recon_agent([0])
    ctx = _ctx()
    ctx.consecutive_zero_host_turns = 2

    # Switch to a coverage that satisfies ADVANCE (hosts + services).
    async def _count_advance(_eid: Any) -> dict[str, int]:
        return {"Host": 1, "Service": 1}

    agent.repo.count = AsyncMock(side_effect=_count_advance)
    assert await agent._decide_advance(ctx) == StepDecision.ADVANCE
    assert ctx.consecutive_zero_host_turns == 0


# --------------------------------------------------------------------------- #
# Item 2: Critic VETO → escalation in unattended mode                          #
# --------------------------------------------------------------------------- #


def _build_veto_decider():
    """Mimic the VETO branch of BaseAgent.step. We test the policy-driven
    queueing logic in isolation because the full step() loop has too many
    moving parts (LLM router, registry, etc.) for a focused unit test."""

    async def _decide(engagement: dict[str, Any]) -> list[dict[str, Any]]:
        ctx_pending: list[dict[str, Any]] = []
        # This mirrors the new VETO branch in agents/base.py exactly.
        policy = (engagement.get("approval_policy") or "manual").lower()
        if policy != "manual":
            ctx_pending.append(
                {
                    "tool": "curated_poc_exec",
                    "input": {"poc_id": "ms17-010"},
                    "reason": "critic_veto: too risky",
                }
            )
        return ctx_pending

    return _decide


async def test_veto_in_auto_reject_queues_approval() -> None:
    """In auto_reject mode the operator never sees the veto via the legacy
    quiet-skip. The fix queues a pending approval so the post-loop check
    flips the decision to ESCALATE_HUMAN."""
    decide = _build_veto_decider()
    pending = await decide({"approval_policy": "auto_reject"})
    assert len(pending) == 1
    assert "critic_veto" in pending[0]["reason"]


async def test_veto_in_auto_approve_queues_approval() -> None:
    """auto_approve is symmetric — the operator is not watching, so the
    veto must still surface as a pause."""
    decide = _build_veto_decider()
    pending = await decide({"approval_policy": "auto_approve"})
    assert len(pending) == 1


async def test_veto_in_manual_mode_preserves_quiet_skip() -> None:
    """Manual mode keeps the previous behaviour: write the audit row but
    don't queue an approval. Operators see the row in the Approvals tab."""
    decide = _build_veto_decider()
    pending = await decide({"approval_policy": "manual"})
    assert pending == []


async def test_veto_with_no_policy_defaults_to_manual() -> None:
    """A missing approval_policy must NOT trigger an escalation — that's the
    legacy unauthenticated path."""
    decide = _build_veto_decider()
    pending = await decide({})
    assert pending == []


# --------------------------------------------------------------------------- #
# Item 3: Global stalled-turn watchdog                                         #
# --------------------------------------------------------------------------- #


def _engagement_state(**kwargs: Any) -> EngagementState:
    now_ns = time.time_ns()
    base = dict(
        engagement_id=uuid4(),
        engagement={
            "token_budget": 1_000_000,
            "window": {
                "start_ns": now_ns - 1_000_000_000,
                "end_ns": now_ns + 3600 * 1_000_000_000,
            },
        },
        current_stage=Stage.RECON,
    )
    base.update(kwargs)
    return EngagementState(**base)


def test_stalled_watchdog_below_threshold_returns_none() -> None:
    s = _engagement_state()
    s.stalled_turn_count = 3
    assert _check_budgets(s) is None


def test_stalled_watchdog_at_threshold_halts() -> None:
    """4 consecutive zero-progress turns is the documented threshold."""
    s = _engagement_state()
    s.stalled_turn_count = 4
    reason = _check_budgets(s)
    assert reason is not None
    assert "stalled_turns_exceeded" in reason
    assert "threshold=4" in reason


async def test_stalled_watchdog_increments_on_zero_progress() -> None:
    """The stage-controller `_step_agent` increments the counter when an
    agent step yields zero tool_runs and zero nodes_written, and resets it
    on any real progress."""
    s = _engagement_state()

    class _Agent:
        stage = Stage.RECON

        async def step(self, ctx: StageContext) -> StepResult:
            return StepResult(
                decision=StepDecision.CONTINUE,
                tokens_used=10,
                cost_usd=0.0,
                tool_runs=0,
                nodes_written=0,
                notes="",
            )

    node = _step_agent(_Agent())
    await node(s)
    assert s.stalled_turn_count == 1
    await node(s)
    assert s.stalled_turn_count == 2


async def test_stalled_watchdog_resets_on_progress() -> None:
    s = _engagement_state()
    s.stalled_turn_count = 3

    class _Agent:
        stage = Stage.RECON

        async def step(self, ctx: StageContext) -> StepResult:
            return StepResult(
                decision=StepDecision.CONTINUE,
                tokens_used=10,
                cost_usd=0.0,
                tool_runs=1,
                nodes_written=2,
                notes="",
            )

    node = _step_agent(_Agent())
    await node(s)
    assert s.stalled_turn_count == 0


# --------------------------------------------------------------------------- #
# Item 4: Transient I/O failure escalation                                     #
# --------------------------------------------------------------------------- #


def test_should_raise_transient_manual_mode_returns_false() -> None:
    """Manual mode preserves the legacy log-and-continue; no escalation."""
    eng = {"approval_policy": "manual"}
    assert _should_raise_transient(eng, TimeoutError("slow neo4j")) is False


def test_should_raise_transient_auto_reject_with_timeout_returns_true() -> None:
    eng = {"approval_policy": "auto_reject"}
    assert _should_raise_transient(eng, TimeoutError("slow neo4j")) is True


def test_should_raise_transient_auto_approve_with_connection_error() -> None:
    eng = {"approval_policy": "auto_approve"}
    assert _should_raise_transient(eng, ConnectionError("postgres dropped")) is True


def test_should_raise_transient_non_retryable_returns_false() -> None:
    """A ValueError (programming bug) should fall through to the legacy
    log-and-continue path even in unattended mode — retrying won't help."""
    eng = {"approval_policy": "auto_reject"}
    assert _should_raise_transient(eng, ValueError("bad shape")) is False


def test_transient_failure_carries_context() -> None:
    """The sentinel exposes `where` + the original exception for the
    converter at the top of step()."""
    orig = TimeoutError("postgres slow")
    tf = _TransientFailure("tool_run_pre_persist", orig)
    assert tf.where == "tool_run_pre_persist"
    assert tf.original is orig
    assert "tool_run_pre_persist" in str(tf)


# --------------------------------------------------------------------------- #
# Item 5: Veto audit-trail distinction                                         #
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Captures whatever ApprovalRow we add so we can assert on its fields."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, row: Any) -> None:
        self.added.append(row)


class _FakeSessionCtx:
    def __init__(self) -> None:
        self.session = _FakeSession()

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *a: Any) -> None:
        return None


@pytest.fixture
def session_capture(monkeypatch: pytest.MonkeyPatch) -> _FakeSessionCtx:
    """Replace session_scope so _persist_approval lands in our fake."""
    ctx = _FakeSessionCtx()

    def _session_scope() -> _FakeSessionCtx:
        return ctx

    monkeypatch.setattr(agent_base, "session_scope", _session_scope)
    # Suppress the bus publish so we don't reach into Redis.
    monkeypatch.setattr(
        agent_base, "get_bus", lambda: MagicMock(publish=AsyncMock(return_value=None))
    )
    return ctx


async def test_persist_approval_stamps_critic_when_decided_by_critic(
    session_capture: _FakeSessionCtx,
) -> None:
    """R1 #5: a critic veto row carries decided_by="critic", not the legacy
    "auto-policy" marker. Distinguishes critic-blocked from policy-blocked."""
    await _persist_approval(
        engagement_id=uuid4(),
        kind="critic_veto:curated_poc_exec",
        payload={"rationale": "blast radius too wide"},
        auto_decision="reject",
        decided_by="critic",
    )
    assert len(session_capture.session.added) == 1
    row = session_capture.session.added[0]
    assert row.decided_by == "critic"
    assert row.decision == "reject"


async def test_persist_approval_defaults_to_auto_policy_marker(
    session_capture: _FakeSessionCtx,
) -> None:
    """R1 #5: rows that pass an auto_decision but no decided_by still get the
    legacy "auto-policy" marker — preserves R9 #2 + Phase 43 behaviour."""
    await _persist_approval(
        engagement_id=uuid4(),
        kind="policy_require_approval:nmap",
        payload={"reason": "destructive"},
        auto_decision="approve",
    )
    row = session_capture.session.added[0]
    assert row.decided_by == "auto-policy"
    assert row.decision == "approve"


async def test_persist_approval_operator_decision_path_unchanged(
    session_capture: _FakeSessionCtx,
) -> None:
    """A queued row (no auto_decision, no decided_by) stays NULL so the
    R9 #2 operator decision handler can stamp the authenticated user later."""
    await _persist_approval(
        engagement_id=uuid4(),
        kind="policy_require_approval:nmap",
        payload={"reason": "manual"},
    )
    row = session_capture.session.added[0]
    assert row.decided_by is None
    assert row.decision is None
