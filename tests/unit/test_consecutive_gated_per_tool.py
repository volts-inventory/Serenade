"""R3 #K3 Bug 1: per-tool gate-streak escalation in the interactive loop.

The legacy ``consecutive_all_gated`` counter incremented whenever no
tool executed in a turn — regardless of which tools the model had
proposed. A critic veto on ``tool_A`` followed by a policy deny on
``tool_B`` would trip the escalate even though the model had pivoted
to a different tool entirely, and the operator-visible note would
concatenate two unrelated gate reasons.

The fix tracks per-tool gate streaks: a tool is "persistently blocked"
only when it was gated in BOTH the previous turn AND this turn. If
the model pivots to a different tool that also gets gated, that's the
model adapting, not getting stuck — keep going. The escalation note
names exactly the persistent tool(s) and surfaces both gate reasons.

These tests cover the three shapes the ship-readiness reviewer
identified:

* Same tool gated twice → ESCALATE_HUMAN, note names the tool.
* Different tools gated on each turn → CONTINUE (no false positive).
* Overlap (tool_A gated both turns; tool_B gated only first) →
  ESCALATE_HUMAN, note names tool_A only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepBudget, StepDecision
from llm.router import LLMRequest, LLMResponse, ToolCall
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #


class _IO(BaseModel):
    payload: str = ""


def _mk_tool_cls(tool_name: str) -> type[Tool[_IO, _IO]]:
    """Build a fresh Tool subclass under the requested name. We need
    distinct registry entries because the per-tool streak tracker keys
    on ``call.name`` and the test scenarios depend on the model
    proposing different tools on different turns."""

    class _T(Tool[_IO, _IO]):
        name = tool_name
        description = f"test tool {tool_name}"
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
            return f"{tool_name} ok payload={output.payload[:50]}"

        async def run(self, input: _IO, ctx: ToolRunContext) -> ToolResult[_IO]:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=_IO(payload=input.payload),
                exit_code=0,
                started_at="2025-01-01T00:00:00Z",
                ended_at="2025-01-01T00:00:00Z",
            )

    _T.__name__ = f"_T_{tool_name}"
    return _T


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
            raise AssertionError("router exhausted")
        return self._responses.pop(0)


def _mk_response(tool_calls: list[ToolCall] | None = None, text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        structured=None,
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        cached_input_tokens=0,
        cost_usd=0.001,
        latency_ms=10,
        raw=None,
    )


def _mk_call(name: str, payload: str = "x") -> ToolCall:
    return ToolCall(name=name, input={"payload": payload}, call_id=f"c-{uuid4()}")


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


def _build_agent(
    *,
    router: _ScriptedRouter,
    tools: list[Tool[Any, Any]],
    policy_side_effect: Any,
    budget: StepBudget | None = None,
) -> _Stub:
    registry = _Registry(tools)
    policy = MagicMock()
    policy.evaluate = AsyncMock(side_effect=policy_side_effect)
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)
    return _Stub(
        router=router,
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        budget=budget or StepBudget(interactive_max_turns=10),
    )


def _build_ctx() -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )


# --------------------------------------------------------------------------- #
# Scenario 1: same tool gated twice → ESCALATE_HUMAN with that tool named     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_same_tool_gated_twice_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1: tool_A gated. Turn 2: tool_A gated again. The watchdog
    must escalate ``ESCALATE_HUMAN`` and the note must surface BOTH the
    prior-turn reason and the current-turn reason for tool_A."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    ToolA = _mk_tool_cls("tool_A")
    # Always deny — different reason each turn so we can verify the
    # note carries the prior reason alongside the current one.
    call_idx = {"n": 0}

    async def _deny_always(**_kw: Any) -> PolicyResult:
        call_idx["n"] += 1
        return PolicyResult(decision=PolicyDecision.DENY, reason=f"turn-{call_idx['n']}-reason")

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("tool_A", "p1")]),
            _mk_response(tool_calls=[_mk_call("tool_A", "p2")]),
            # If the watchdog were broken (no escalate), turn 3 would
            # also fire — assertions below verify we stop at turn 2.
            _mk_response(tool_calls=[_mk_call("tool_A", "p3")]),
        ]
    )
    agent = _build_agent(
        router=router,
        tools=[ToolA()],
        policy_side_effect=_deny_always,
    )
    ctx = _build_ctx()

    result = await agent.step(ctx)

    assert len(router.calls) == 2, "must stop at turn 2, not run a third"
    assert result.decision == StepDecision.ESCALATE_HUMAN
    # Note names the persistent tool and surfaces BOTH gate reasons.
    assert "tool_A" in result.notes
    assert "turn-1-reason" in result.notes  # prev=
    assert "turn-2-reason" in result.notes  # now=


# --------------------------------------------------------------------------- #
# Scenario 2: different tools gated on each turn → CONTINUE (model adapted)   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pivot_to_different_gated_tool_does_not_escalate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1: tool_A gated. Turn 2: tool_B gated. The model pivoted
    to a different tool — that's adaptation, not stagnation. The
    legacy counter falsely escalated here; the per-tool tracker must
    NOT. Turn 3 (text only) exits cleanly with CONTINUE."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    ToolA, ToolB = _mk_tool_cls("tool_A"), _mk_tool_cls("tool_B")

    async def _deny_always(**_kw: Any) -> PolicyResult:
        return PolicyResult(decision=PolicyDecision.DENY, reason="any-reason")

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("tool_A")]),
            _mk_response(tool_calls=[_mk_call("tool_B")]),
            _mk_response(text="ok, giving up cleanly"),
        ]
    )
    agent = _build_agent(
        router=router,
        tools=[ToolA(), ToolB()],
        policy_side_effect=_deny_always,
    )
    ctx = _build_ctx()

    result = await agent.step(ctx)

    # All three turns ran — no false-positive escalate after turn 2.
    assert len(router.calls) == 3
    # _decide_advance returns CONTINUE in the test stub.
    assert result.decision == StepDecision.CONTINUE
    # No "gated twice" wording leaked into the note.
    assert "gated twice" not in result.notes


# --------------------------------------------------------------------------- #
# Scenario 3: overlap — tool_A in both turns; tool_B in turn 1 only           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_overlap_only_persistent_tool_named_in_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1: tool_A AND tool_B both proposed; both gated. Turn 2:
    only tool_A proposed; gated. The intersection is {tool_A} — the
    escalation note must name tool_A only, NOT tool_B (which the
    model already pivoted away from)."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    ToolA, ToolB = _mk_tool_cls("tool_A"), _mk_tool_cls("tool_B")

    async def _deny_always(**_kw: Any) -> PolicyResult:
        return PolicyResult(decision=PolicyDecision.DENY, reason="forbidden")

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("tool_A"), _mk_call("tool_B")]),
            _mk_response(tool_calls=[_mk_call("tool_A")]),
            _mk_response(text="should not be reached"),
        ]
    )
    agent = _build_agent(
        router=router,
        tools=[ToolA(), ToolB()],
        policy_side_effect=_deny_always,
    )
    ctx = _build_ctx()

    result = await agent.step(ctx)

    assert len(router.calls) == 2
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert "tool_A" in result.notes
    # tool_B was gated once but the model pivoted — it MUST NOT appear
    # in the persistent-tool list. The note format puts tools inside
    # ``[...]``; check the bracketed list specifically so a stray
    # mention in a downstream "reasons:" string doesn't false-fail.
    bracket_start = result.notes.find("[")
    bracket_end = result.notes.find("]")
    assert bracket_start != -1 and bracket_end != -1
    bracketed = result.notes[bracket_start + 1 : bracket_end]
    assert "tool_B" not in bracketed
    assert "tool_A" in bracketed


# --------------------------------------------------------------------------- #
# Coverage: streak resets when a tool that was gated last turn now executes   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_recovers_from_gate_does_not_escalate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1: tool_A is gated. Turn 2: tool_A is allowed (e.g. an
    earlier turn produced the world-state evidence the OPA policy
    needed). Turn 3: tool_A is gated again. This is NOT a persistent
    block — the intersection between turn-2's gated set ({}) and
    turn-3's gated set ({tool_A}) is empty, so the watchdog must NOT
    escalate. Guards against an over-eager tracker that compares to
    the all-time gate history instead of just the immediately-prior
    turn."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    ToolA = _mk_tool_cls("tool_A")
    call_count = {"n": 0}

    async def _flapping_policy(**_kw: Any) -> PolicyResult:
        call_count["n"] += 1
        # 1st call (turn 1): deny. 2nd call (turn 2): allow.
        # 3rd call (turn 3): deny. The watchdog must not fire because
        # turn 2 reset the streak.
        if call_count["n"] == 2:
            return PolicyResult(decision=PolicyDecision.ALLOW, reason="ok")
        return PolicyResult(decision=PolicyDecision.DENY, reason="forbidden")

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("tool_A", "p1")]),
            _mk_response(tool_calls=[_mk_call("tool_A", "p2")]),
            _mk_response(tool_calls=[_mk_call("tool_A", "p3")]),
            _mk_response(text="done"),
        ]
    )
    agent = _build_agent(
        router=router,
        tools=[ToolA()],
        policy_side_effect=_flapping_policy,
    )
    ctx = _build_ctx()

    result = await agent.step(ctx)

    # All four turns ran — the gate flap did not trip a false escalate.
    assert len(router.calls) == 4
    assert result.decision == StepDecision.CONTINUE
    assert "gated twice" not in result.notes
