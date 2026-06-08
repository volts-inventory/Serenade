"""R3 #K3 Bug 2: mid-step budget kill in the interactive loop.

``orchestrator.stage_controller._check_budgets`` only fires at the
top of a node — when the LangGraph state machine RE-ENTERS the agent.
A single ``step()`` with a large ``interactive_llm_turn_cap`` (recon
is 20) could therefore burn 20× per-turn cost AFTER the engagement
was already over budget without anyone noticing.

The fix threads ``total_tokens_at_step_start`` / ``token_budget`` /
``wallclock_end_ns`` from ``EngagementState`` onto ``StageContext`` at
step entry; the interactive loop then calls
``_check_budget_mid_step_inline`` after every LLM turn. A violation
breaks the loop and returns ``ESCALATE_HUMAN`` with a note that
mirrors the controller's wording so audit history stays uniform.

These tests cover the two budget shapes:

* Token budget breached mid-step → loop stops after the offending
  turn, ESCALATE_HUMAN, note contains the running total.
* Wallclock window expired mid-step → same shape, wallclock note.
* Bonus: ``_check_budget_mid_step`` is the pure helper the controller
  exports; we cover its truth table at the unit level too.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepBudget, StepDecision
from llm.router import LLMRequest, LLMResponse, ToolCall
from orchestrator.stage_controller import _check_budget_mid_step
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


# --------------------------------------------------------------------------- #
# Pure helper truth-table                                                     #
# --------------------------------------------------------------------------- #


def test_check_budget_mid_step_token_breach():
    reason = _check_budget_mid_step(
        total_tokens_at_step_start=995_000,
        accumulated_step_tokens=10_000,
        token_budget=1_000_000,
        wallclock_end_ns=None,
    )
    assert reason is not None
    assert "token_budget_exceeded_mid_step" in reason
    assert "1005000" in reason  # 995_000 + 10_000


def test_check_budget_mid_step_under_token_budget():
    assert (
        _check_budget_mid_step(
            total_tokens_at_step_start=100,
            accumulated_step_tokens=200,
            token_budget=1_000_000,
            wallclock_end_ns=None,
        )
        is None
    )


def test_check_budget_mid_step_zero_token_budget_is_disabled():
    """Token budget of 0 means "no cap" (same convention as
    ``_check_budgets``) — must never violate."""
    assert (
        _check_budget_mid_step(
            total_tokens_at_step_start=10**12,
            accumulated_step_tokens=10**12,
            token_budget=0,
            wallclock_end_ns=None,
        )
        is None
    )


def test_check_budget_mid_step_wallclock_expired():
    past = time.time_ns() - 60 * 1_000_000_000  # one minute ago
    reason = _check_budget_mid_step(
        total_tokens_at_step_start=0,
        accumulated_step_tokens=0,
        token_budget=0,
        wallclock_end_ns=past,
    )
    assert reason is not None
    assert "wallclock_window_exceeded_mid_step" in reason


def test_check_budget_mid_step_wallclock_future_ok():
    future = time.time_ns() + 3600 * 1_000_000_000  # +1h
    assert (
        _check_budget_mid_step(
            total_tokens_at_step_start=0,
            accumulated_step_tokens=0,
            token_budget=0,
            wallclock_end_ns=future,
        )
        is None
    )


# --------------------------------------------------------------------------- #
# End-to-end: agent.step bails out mid-loop                                   #
# --------------------------------------------------------------------------- #


class _IO(BaseModel):
    payload: str = ""


class _Echo(Tool[_IO, _IO]):
    name = "echo"
    description = "echo"
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    destructive = False
    Input = _IO
    Output = _IO

    def extract_targets(self, input: _IO) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _IO, *, tool_run_id: UUID, persisted: int
    ) -> str:
        return f"echo ok payload={output.payload[:50]} persisted={persisted}"

    async def run(self, input: _IO, ctx: ToolRunContext) -> ToolResult[_IO]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_IO(payload=input.payload),
            exit_code=0,
            started_at="2025-01-01T00:00:00Z",
            ended_at="2025-01-01T00:00:00Z",
        )


class _Registry:
    def __init__(self, tools: list[Tool[Any, Any]]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool[Any, Any]:
        return self._tools[name]

    def for_stage(self, stage: Stage) -> list[Tool[Any, Any]]:
        return [t for t in self._tools.values() if t.stage == stage]


class _ScriptedRouter:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, req: LLMRequest, *, agent: str = "unknown") -> LLMResponse:
        self.calls.append(req)
        if not self._responses:
            raise AssertionError("router exhausted unexpectedly")
        return self._responses.pop(0)


def _mk_response(
    tool_calls: list[ToolCall] | None = None,
    text: str = "",
    *,
    input_tokens: int = 5_000,
    output_tokens: int = 5_000,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        structured=None,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=0,
        cost_usd=0.01,
        latency_ms=10,
        raw=None,
    )


def _mk_call(name: str = "echo") -> ToolCall:
    return ToolCall(name=name, input={"payload": "x"}, call_id=f"c-{uuid4()}")


class _Stub(BaseAgent):
    stage = Stage.RECON

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        return {}

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return "test"

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        return StepDecision.CONTINUE

    async def _parse_and_persist(
        self,
        ctx: StageContext,
        tool: Tool[Any, Any],
        tool_input: dict[str, Any],
        tool_output: BaseModel,
        tool_run_id: UUID,
    ) -> Iterable[WorldStateNode]:
        return []


def _build_agent(*, router: _ScriptedRouter, cap: int = 20) -> _Stub:
    registry = _Registry([_Echo()])
    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=PolicyDecision.ALLOW, reason="ok")
    )
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)
    return _Stub(
        router=router,
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        budget=StepBudget(interactive_max_turns=cap),
    )


@pytest.mark.asyncio
async def test_mid_step_token_budget_kill_stops_loop_after_breach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State has ``total_tokens=995_000`` and ``token_budget=1_000_000``.
    Each LLM turn emits ~10_000 tokens (5k input + 5k output). The loop
    must break after turn 1 (995_000 + 10_000 > 1_000_000) and return
    ``ESCALATE_HUMAN`` — NOT chew through all 20 turns of the cap."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    # Script 20 tool-call responses; if the watchdog were broken every
    # one of them would be consumed.
    router = _ScriptedRouter(
        [_mk_response(tool_calls=[_mk_call()]) for _ in range(20)]
    )
    agent = _build_agent(router=router, cap=20)

    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
        total_tokens_at_step_start=995_000,
        token_budget=1_000_000,
    )

    result = await agent.step(ctx)

    # Loop bailed after exactly one LLM call.
    assert len(router.calls) == 1
    assert result.decision == StepDecision.ESCALATE_HUMAN
    # Note carries the canonical mid-step token wording.
    assert "token_budget_exceeded_mid_step" in result.notes
    # Step tokens accounted for the one turn we did run.
    assert result.tokens_used == 10_000


@pytest.mark.asyncio
async def test_mid_step_wallclock_kill_stops_loop_after_breach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape as the token test but with the wallclock window
    already expired before step() ran. Loop must bail after the first
    LLM turn (the per-turn check sees ``time.time_ns()`` is past
    ``wallclock_end_ns``)."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [_mk_response(tool_calls=[_mk_call()]) for _ in range(20)]
    )
    agent = _build_agent(router=router, cap=20)
    past_ns = time.time_ns() - 60 * 1_000_000_000  # 1 min ago

    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
        token_budget=0,  # token-budget disabled
        wallclock_end_ns=past_ns,
    )

    result = await agent.step(ctx)

    assert len(router.calls) == 1
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert "wallclock_window_exceeded_mid_step" in result.notes


@pytest.mark.asyncio
async def test_no_budget_thread_means_no_mid_step_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``token_budget=0`` and ``wallclock_end_ns=None`` (the
    legacy test-fixture shape, with no controller threading the
    snapshot), the mid-step check must be a no-op so existing tests
    that don't set those fields keep working unchanged."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call()]),
            _mk_response(text="all done"),
        ]
    )
    agent = _build_agent(router=router, cap=20)
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
        # token_budget default 0, wallclock_end_ns default None
    )

    result = await agent.step(ctx)

    assert len(router.calls) == 2
    assert result.decision == StepDecision.CONTINUE
    assert "mid_step" not in result.notes


@pytest.mark.asyncio
async def test_token_budget_with_headroom_does_not_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token budget threaded but plenty of headroom — loop runs to the
    natural "no tool calls" exit. Guards against a thresholds-off-by-one
    bug that triggered the kill on the first turn even when under-cap."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call()]),
            _mk_response(tool_calls=[_mk_call()]),
            _mk_response(text="done"),
        ]
    )
    agent = _build_agent(router=router, cap=20)
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
        total_tokens_at_step_start=100,
        token_budget=10_000_000,  # huge cap
    )

    result = await agent.step(ctx)

    assert len(router.calls) == 3
    assert result.decision == StepDecision.CONTINUE
