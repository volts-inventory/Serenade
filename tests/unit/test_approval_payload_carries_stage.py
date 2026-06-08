"""R3 #I1 Bug 2 — approval payloads carry the agent's ``stage`` so the
``ApprovalsTab`` UI can bucket rows by stage.

Background: ``clients/ui/components/tabs/ApprovalsTab.tsx::extractStage``
reads ``stage`` / ``from_stage`` / ``current_stage`` from
``requested_payload`` and falls back to ``"unknown"`` when none is
present. H3 Bug 6 added the "Approve all in <stage>" grouping but the
per-gate ``_persist_approval`` call sites in ``src/agents/base.py``
never stamped ``stage`` on the payload — so every row fell into the
``unknown`` bucket and the grouping was useless.

These tests drive each gate path (budget, policy-require-approval,
critic-veto, critic-revise/escalate) through the agent's
``_handle_one_tool_call`` and assert the payload submitted to
``_persist_approval`` (and the in-memory ``ctx.pending_approvals``
entry, where applicable) carry the agent's stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from agents.base import BaseAgent, StageContext, StepDecision
from agents.council import CouncilVerdict
from agents.critic import Verdict
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


# --------------------------------------------------------------------------- #
# Minimal Tool + Agent stubs                                                   #
# --------------------------------------------------------------------------- #


class _DangerInput(BaseModel):
    x: int = 0


class _DangerOutput(BaseModel):
    x: int = 0


class _DangerTool(Tool[_DangerInput, _DangerOutput]):
    name = "danger_tool"
    description = "destructive test tool"
    stage = Stage.EXPLOIT
    risk_class = RiskClass.DESTRUCTIVE
    destructive = True
    Input = _DangerInput
    Output = _DangerOutput

    def extract_targets(self, input: _DangerInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _DangerOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        return f"danger_tool ok persisted={persisted}"

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


class _ExploitTestAgent(BaseAgent):
    stage = Stage.EXPLOIT
    critic_required_for_risk_classes = (RiskClass.DESTRUCTIVE,)

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
    policy_decision: PolicyDecision = PolicyDecision.ALLOW,
    council_verdict: Verdict = Verdict.APPROVE,
) -> tuple[_ExploitTestAgent, MagicMock]:
    registry = _Registry([_DangerTool()])
    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=policy_decision, reason="policy-reason")
    )
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)
    council = MagicMock()
    council.deliberate = AsyncMock(
        return_value=CouncilVerdict(
            decision=council_verdict,
            rationale="council says: " + council_verdict.value,
            confidence=0.9,
            member_critiques=[],
            total_cost_usd=0.0,
        )
    )
    agent = _ExploitTestAgent(
        router=MagicMock(),
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        pre_action_council=council,
    )
    return agent, council


def _ctx(**eng_overrides: Any) -> StageContext:
    engagement = {"approval_policy": "manual"}
    engagement.update(eng_overrides)
    return StageContext(
        engagement_id=uuid4(),
        engagement=engagement,
        stage=Stage.EXPLOIT,
    )


def _mk_call() -> Any:
    call = MagicMock()
    call.name = "danger_tool"
    call.input = {"x": 1}
    call.call_id = "c1"
    return call


def _capture_persist_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``_persist_approval`` with a recorder and return the captured
    kwargs list. Each entry is the kwargs dict ``_persist_approval`` was
    called with (kind, payload, ...)."""
    captured: list[dict[str, Any]] = []

    async def _record(*args: Any, **kwargs: Any) -> None:
        captured.append(kwargs)

    from agents import base as agent_base

    monkeypatch.setattr(agent_base, "_persist_approval", _record)
    return captured


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_per_tool_budget_payload_includes_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget gate: the persisted ApprovalRow payload AND the in-memory
    ctx.pending_approvals entry must each carry ``stage`` set to the
    agent's stage."""
    agent, _ = _build_agent()
    ctx = _ctx(per_tool_token_budget={"danger_tool": 100})
    ctx.tokens_by_tool["danger_tool"] = 100  # exhausted
    captured = _capture_persist_calls(monkeypatch)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=_mk_call(), slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    # Server-side ApprovalRow payload includes stage.
    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["stage"] == Stage.EXPLOIT.value
    # In-memory pending_approvals entry also includes stage.
    assert ctx.pending_approvals
    assert ctx.pending_approvals[-1]["stage"] == Stage.EXPLOIT.value


@pytest.mark.asyncio
async def test_policy_require_approval_payload_includes_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQUIRE_APPROVAL goes through ``_resolve_approval_gate`` which
    persists a row before returning ``"queue"`` (in manual mode). Both
    that row and the in-memory pending_approvals entry must carry stage.
    """
    agent, _ = _build_agent(policy_decision=PolicyDecision.REQUIRE_APPROVAL)
    ctx = _ctx()
    captured = _capture_persist_calls(monkeypatch)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=_mk_call(), slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    # _resolve_approval_gate persists one row.
    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["stage"] == Stage.EXPLOIT.value
    assert ctx.pending_approvals
    assert ctx.pending_approvals[-1]["stage"] == Stage.EXPLOIT.value


@pytest.mark.asyncio
async def test_critic_veto_payload_includes_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critic-veto path: the auto-rejected ApprovalRow MUST carry stage.
    The pending_approvals entry is appended only in unattended mode; we
    flip the policy to auto_approve here so we can verify both surfaces.
    """
    agent, _ = _build_agent(council_verdict=Verdict.VETO)
    # Unattended mode triggers the in-memory append for VETO.
    ctx = _ctx(approval_policy="auto_approve")
    captured = _capture_persist_calls(monkeypatch)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=_mk_call(), slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    assert "veto" in outcome.feedback_text.lower()
    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["stage"] == Stage.EXPLOIT.value
    # Unattended mode → also appended to ctx.pending_approvals.
    assert ctx.pending_approvals
    assert ctx.pending_approvals[-1]["stage"] == Stage.EXPLOIT.value


@pytest.mark.asyncio
async def test_critic_revise_payload_includes_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critic-REVISE / ESCALATE: the gate-resolved ApprovalRow AND the
    in-memory pending_approvals entry (when queued in manual mode) must
    carry stage."""
    agent, _ = _build_agent(council_verdict=Verdict.REVISE)
    ctx = _ctx()
    captured = _capture_persist_calls(monkeypatch)

    outcome = await agent._handle_one_tool_call(
        ctx=ctx, call=_mk_call(), slice_={}, turn_tokens=10
    )

    assert outcome.executed is False
    assert "revise" in outcome.feedback_text.lower()
    # One row from _resolve_approval_gate.
    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["stage"] == Stage.EXPLOIT.value
    assert ctx.pending_approvals
    assert ctx.pending_approvals[-1]["stage"] == Stage.EXPLOIT.value


@pytest.mark.asyncio
async def test_stage_value_matches_agent_subclass_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: a different stage on the agent flows through to the
    payload. Confirms we read ``self.stage.value`` (not a hard-coded
    string)."""

    class _ReconTestAgent(_ExploitTestAgent):
        stage = Stage.RECON

    registry = _Registry([_DangerTool()])
    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=PolicyDecision.ALLOW, reason="ok")
    )
    council = MagicMock()
    council.deliberate = AsyncMock(
        return_value=CouncilVerdict(
            decision=Verdict.VETO,
            rationale="no",
            confidence=0.9,
            member_critiques=[],
            total_cost_usd=0.0,
        )
    )
    # NOTE: Stage.EXPLOIT tool but Stage.RECON agent — registry filters
    # on agent.stage so we must register the tool with the same stage as
    # the agent for the candidate-tool path to find it. Easier path:
    # bypass the registry and call _handle_one_tool_call directly.
    agent = _ReconTestAgent(
        router=MagicMock(),
        repo=MagicMock(),
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        pre_action_council=council,
    )
    agent.critic_required_for_risk_classes = (RiskClass.DESTRUCTIVE,)
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )
    captured = _capture_persist_calls(monkeypatch)

    await agent._handle_one_tool_call(
        ctx=ctx, call=_mk_call(), slice_={}, turn_tokens=10
    )

    assert captured
    assert captured[0]["payload"]["stage"] == Stage.RECON.value
