"""R3 #H1 (HIGH): `_handle_one_tool_call` runs cheap gates before expensive ones.

The pre-fix order was:
    validation -> OPA policy -> critic/council (LLM!) -> per-tool budget -> execute

A tool that was about to be rejected by the per-tool budget gate (an O(1)
dict lookup) still paid for the critic council's 1-3 LLM-call
deliberation first. Likewise an OPA-deny was discovered AFTER the
council deliberated — wasted spend on a request that was already dead.

The new order is:
    validation -> per-tool budget -> OPA policy -> critic/council -> execute

These tests build an agent with a per-tool budget that's already exceeded
for the test tool AND with a critic council that would VETO. The right
behaviour is: budget denial returned, council `.deliberate` never called.
Similarly for an OPA DENY when the council would also VETO.

We also lock in the post-fix denial-text contract — the planner reads
``feedback_text`` to decide what to do next, so changing those strings
would silently break planner behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext
from agents.council import CouncilVerdict
from agents.critic import Verdict
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #


class _DangerInput(BaseModel):
    x: int = 0


class _DangerOutput(BaseModel):
    x: int = 0


class _DangerTool(Tool[_DangerInput, _DangerOutput]):
    """A destructive-risk tool so the critic/council gate fires."""

    name = "danger_tool"
    description = "destructive test tool"
    stage = Stage.RECON
    risk_class = RiskClass.DESTRUCTIVE
    destructive = True
    Input = _DangerInput
    Output = _DangerOutput

    def extract_targets(self, input: _DangerInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _DangerOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # R3 #I1 Bug 7: summarize_for_model is abstract; test stubs
        # must override to instantiate.
        return f"danger_tool ok: x={output.x} persisted={persisted}"

    async def run(self, input: _DangerInput, ctx: ToolRunContext) -> ToolResult[_DangerOutput]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_DangerOutput(),
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


class _TestAgent(BaseAgent):
    stage = Stage.RECON

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        return {}

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return "test"

    async def _decide_advance(self, ctx: StageContext) -> Any:
        from agents.base import StepDecision
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
    policy_decision: PolicyDecision = PolicyDecision.ALLOW,
    council_verdict: Verdict = Verdict.VETO,
) -> tuple[_TestAgent, MagicMock]:
    """Build an agent with the danger_tool registered, scripted policy
    decision, and a council that will return the given verdict. Returns
    (agent, council_mock) so callers can assert call counts on the council.
    """
    registry = _Registry([_DangerTool()])

    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=policy_decision, reason="test-policy-reason")
    )
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)

    council = MagicMock()
    council.deliberate = AsyncMock(
        return_value=CouncilVerdict(
            decision=council_verdict,
            rationale="council says no",
            confidence=0.9,
            member_critiques=[],
            total_cost_usd=0.0,
        )
    )

    agent = _TestAgent(
        router=MagicMock(),
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        pre_action_council=council,
    )
    # The default critic_required_for_risk_classes on BaseAgent is empty,
    # which would skip the council. Configure it so DESTRUCTIVE risk
    # triggers the council gate.
    agent.critic_required_for_risk_classes = (RiskClass.DESTRUCTIVE,)
    return agent, council


def _ctx(**eng_overrides: Any) -> StageContext:
    engagement = {"approval_policy": "manual"}
    engagement.update(eng_overrides)
    return StageContext(
        engagement_id=uuid4(),
        engagement=engagement,
        stage=Stage.RECON,
    )


def _mk_call(name: str = "danger_tool", input: dict[str, Any] | None = None) -> Any:
    """Build the duck-typed ToolCall the loop expects."""
    return MagicMock(name=name, input=input or {"x": 1}, call_id="c1")


# --------------------------------------------------------------------------- #
# Budget denial short-circuits before the council deliberates.                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_denial_skips_council_deliberation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 #H1 (HIGH): per-tool budget is an O(1) dict lookup; the council
    is 1-3 LLM calls. The reordered gates run budget BEFORE the council
    so we don't pay the LLM cost on a request that's already over the
    cap."""
    agent, council = _build_agent(council_verdict=Verdict.VETO)
    # Engagement has a 100-token cap on danger_tool. We've already burned
    # 100, so the next call is over the line.
    ctx = _ctx(per_tool_token_budget={"danger_tool": 100})
    ctx.tokens_by_tool["danger_tool"] = 100

    # Manually instantiate a ToolCall-like object.
    call = MagicMock()
    call.name = "danger_tool"
    call.input = {"x": 1}
    call.call_id = "c1"

    # Patch the _persist_approval helper to a no-op so we don't need
    # a real DB session.
    from agents import base as agent_base

    async def _noop_persist(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(agent_base, "_persist_approval", _noop_persist)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=call, slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    assert "tool denied by budget" in outcome.feedback_text
    assert "per-tool token cap" in outcome.feedback_text
    # The whole point of this test: council MUST NOT have been called.
    council.deliberate.assert_not_called()
    # Policy can run either before or after budget (both are cheap);
    # we don't assert on it. But we DO want to ensure the council —
    # the only LLM-priced gate — wasn't touched.


# --------------------------------------------------------------------------- #
# Policy denial short-circuits before the council deliberates.                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_policy_deny_skips_council_deliberation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPA DENY is a local sub-millisecond evaluation; the council is 1-3
    LLM calls. The reordered gates run policy BEFORE the council so a
    denied call doesn't pay the council cost."""
    agent, council = _build_agent(
        policy_decision=PolicyDecision.DENY, council_verdict=Verdict.VETO
    )
    ctx = _ctx()

    call = MagicMock()
    call.name = "danger_tool"
    call.input = {"x": 1}
    call.call_id = "c1"

    from agents import base as agent_base

    async def _noop_persist(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(agent_base, "_persist_approval", _noop_persist)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=call, slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    assert "tool denied by policy" in outcome.feedback_text
    council.deliberate.assert_not_called()


# --------------------------------------------------------------------------- #
# Sanity: when budget + policy both ALLOW, the council DOES get called.       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_council_runs_when_budget_and_policy_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inverse check: if neither cheap gate fires, the council DOES get
    invoked. Protects against an over-eager reorder that skips the gate
    when it shouldn't."""
    agent, council = _build_agent(council_verdict=Verdict.VETO)
    # No per-tool budget cap → budget gate is a no-op.
    ctx = _ctx()

    call = MagicMock()
    call.name = "danger_tool"
    call.input = {"x": 1}
    call.call_id = "c1"

    from agents import base as agent_base

    async def _noop_persist(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(agent_base, "_persist_approval", _noop_persist)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=call, slice_={}, turn_tokens=10
    )

    # Council vetoed.
    assert outcome.executed is False
    assert "tool denied by critic (veto)" in outcome.feedback_text
    # And it was actually called.
    assert council.deliberate.call_count == 1
