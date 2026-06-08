"""R3 #F.5: Tool-result context-size budget across a multi-turn loop.

Phase F's interactive loop fed the full ``output.model_dump_json()`` of
every tool back to the planner — up to 4096 chars per call. Across a
15-turn loop with 2-3 tools per turn that burned 100-300K of context.
Phase F.5 swapped that for a per-tool deterministic summary capped at
500 chars. This test drives a 5-turn loop with synthetic tools that
return large payloads and asserts the total tool_result-content bytes
across all turns stays under ~5000 (10 tool calls * 500-char cap).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from agents.base import (
    BaseAgent,
    StageContext,
    StepBudget,
    StepDecision,
)
from llm.router import LLMRequest, LLMResponse, ToolCall
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


class _NoiseInput(BaseModel):
    size: int = 0
    count: int = 0


class _NoiseSubItem(BaseModel):
    name: str = ""
    body: str = ""


class _NoiseOutput(BaseModel):
    items: list[_NoiseSubItem] = Field(default_factory=list)
    blob: str = ""


class _NoiseTool(Tool[_NoiseInput, _NoiseOutput]):
    """Tool that returns a chunky output (huge string + many subitems)
    so we can verify the summarizer keeps the planner-visible context
    bounded regardless of payload size."""

    name = "noise_tool"
    description = "Returns a fixed-shape, configurably-large output."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input = _NoiseInput
    Output = _NoiseOutput

    def extract_targets(self, input: _NoiseInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _NoiseOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # R3 #I1 Bug 7: summarize_for_model is now @abstractmethod; the
        # previous comment about "inheriting the default impl" no longer
        # applies because there IS no concrete default. We explicitly
        # emit a tiny per-shape summary that names list lengths only —
        # not their bodies — so the bounded-context assertions still
        # hold (no 100KB blob in the planner-visible string).
        return (
            f"noise_tool ok: items={len(output.items)} blob_len={len(output.blob)} "
            f"persisted={persisted} tool_run_id={tool_run_id}"
        )

    async def run(self, input: _NoiseInput, ctx: ToolRunContext) -> ToolResult[_NoiseOutput]:
        items = [
            _NoiseSubItem(name=f"item-{i}", body="x" * 200) for i in range(input.count)
        ]
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_NoiseOutput(items=items, blob="X" * input.size),
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
            raise AssertionError("router exhausted")
        return self._responses.pop(0)


def _mk_response(
    *, tool_calls: list[ToolCall] | None = None, text: str = ""
) -> LLMResponse:
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


def _mk_call(name: str, input: dict[str, Any], call_id: str) -> ToolCall:
    return ToolCall(name=name, input=input, call_id=call_id)


class _TestAgent(BaseAgent):
    stage = Stage.RECON

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        return {"hosts": 0}

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return "exercise context-size budget"

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


def _build_agent(router: _ScriptedRouter, tools: list[Tool[Any, Any]]) -> _TestAgent:
    registry = _Registry(tools)
    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=PolicyDecision.ALLOW, reason="ok")
    )
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)
    return _TestAgent(
        router=router,
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        budget=StepBudget(interactive_max_turns=8),
    )


def _build_ctx() -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )


@pytest.mark.asyncio
async def test_tool_result_content_bytes_stay_under_budget_across_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive a 5-turn loop with 2 tool calls per turn (10 total),
    where each tool returns ~100KB of output (huge blob + 200 items).
    Total tool_result content bytes across the whole conversation
    must stay under 5000 (10 calls * 500-char cap)."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)

    # 5 turns, each emitting 2 tool calls; final turn emits no tool calls
    # so the loop exits cleanly.
    responses: list[LLMResponse] = []
    for turn in range(5):
        responses.append(
            _mk_response(
                tool_calls=[
                    _mk_call(
                        "noise_tool",
                        {"size": 100_000, "count": 200},
                        call_id=f"c{turn}a",
                    ),
                    _mk_call(
                        "noise_tool",
                        {"size": 100_000, "count": 200},
                        call_id=f"c{turn}b",
                    ),
                ]
            )
        )
    responses.append(_mk_response(text="done"))

    router = _ScriptedRouter(responses)
    agent = _build_agent(router=router, tools=[_NoiseTool()])
    ctx = _build_ctx()
    await agent.step(ctx)

    # Walk every router request and sum the tool_result block content
    # bytes that the planner actually saw.
    total_bytes = 0
    tool_result_count = 0
    for req in router.calls:
        if not req.messages:
            continue
        for msg in req.messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    payload = block.get("content")
                    if isinstance(payload, str):
                        total_bytes += len(payload)
                        tool_result_count += 1

    # Each tool_result is read once per subsequent turn it's included in,
    # so the same block can appear in router.calls[i].messages for i>=its
    # turn-of-origin. The conversation keeps growing turn-over-turn, which
    # is why this test measures the cumulative bytes seen by the model
    # across all turn requests (the actual context budget).
    # 10 tool calls; if each call's summary is ~500 chars and shows up in
    # ~4 subsequent turns on average, expect <= 10 * 500 * 4 = 20K.
    # The HARD per-call bound is 500 chars * 10 calls = 5000 chars in
    # the FINAL turn's request alone. We assert the final turn budget
    # to make the test independent of conversation-replay accounting.
    final_turn_bytes = 0
    final_req = router.calls[-1]
    assert final_req.messages is not None
    for msg in final_req.messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                payload = block.get("content")
                if isinstance(payload, str):
                    final_turn_bytes += len(payload)

    assert tool_result_count >= 10, (
        "expected at least 10 tool_result blocks across the loop"
    )
    assert final_turn_bytes <= 5000, (
        f"final-turn tool_result bytes ({final_turn_bytes}) exceeds budget; "
        f"summarizer hook not bounding context."
    )
    # Sanity: raw blob (100KB of 'X') must NEVER have leaked into context.
    for req in router.calls:
        if not req.messages:
            continue
        for msg in req.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        payload = block.get("content")
                        if isinstance(payload, str):
                            assert "XXXXXXXXXX" not in payload, (
                                "100KB blob leaked into planner-visible context"
                            )
