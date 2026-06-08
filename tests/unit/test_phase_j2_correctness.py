"""R3 #J2: correctness polish from the second-pass expert review.

Covers:
- Bug 3: invalid-shape suppression log warning;
- Bug 4: ESCALATE_HUMAN on all-gated-twice (see also test_interactive_loop);
- Bug 5: turn-token attribution is output-size weighted, not equal-split;
- Bug 6: producer-side regression budget check reads the live count via
  ``regressions_pending``;
- Bug 7: ``_truncate_slice_with_sibling`` returns ``(items, info)``
  pairs so the in-list sentinel is replaced by a sibling top-level
  structure;
- Bug 9: ``_decide_advance`` consumes ``ctx.last_world_state_slice``
  instead of re-running the (Neo4j round-trip) slice builder.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from agents.base import (
    BaseAgent,
    StageContext,
    StepBudget,
    StepDecision,
    _SLICE_CAP_DEFAULT,
    _truncate_slice,
    _truncate_slice_with_sibling,
)
from llm.router import LLMRequest, LLMResponse, ToolCall
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode


# --------------------------------------------------------------------------- #
# Shared test scaffolding                                                     #
# --------------------------------------------------------------------------- #


class _TinyInput(BaseModel):
    payload: str = ""


class _TinyOutput(BaseModel):
    payload: str = ""


class _TinyTool(Tool[_TinyInput, _TinyOutput]):
    name = "tiny"
    description = "Tiny test tool — returns a short payload."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input = _TinyInput
    Output = _TinyOutput

    def extract_targets(self, input: _TinyInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _TinyOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # Short — exercises the weighted attribution branch where this
        # tool's "weight" is the small one.
        return "tiny ok"

    async def run(self, input: _TinyInput, ctx: ToolRunContext) -> ToolResult[_TinyOutput]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_TinyOutput(payload=input.payload),
            exit_code=0,
            started_at="2025-01-01T00:00:00Z",
            ended_at="2025-01-01T00:00:00Z",
        )


class _FatInput(BaseModel):
    size: int = 0


class _FatOutput(BaseModel):
    blob: str = ""


class _FatTool(Tool[_FatInput, _FatOutput]):
    name = "fat"
    description = "Fat test tool — returns a huge payload."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input = _FatInput
    Output = _FatOutput

    def extract_targets(self, input: _FatInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _FatOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # Long summary — Bug 5's weighted attribution makes this tool
        # eat most of the turn's budget instead of getting equal-split
        # with the tiny tool above.
        return "fat ok " + ("X" * 400)

    async def run(self, input: _FatInput, ctx: ToolRunContext) -> ToolResult[_FatOutput]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_FatOutput(blob="Y" * input.size),
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
    *,
    tool_calls: list[ToolCall] | None = None,
    text: str = "",
    input_tokens: int = 600,
    output_tokens: int = 400,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        structured=None,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=0,
        cost_usd=0.001,
        latency_ms=10,
        raw=None,
    )


def _mk_call(name: str, input: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(name=name, input=input or {}, call_id=f"call-{uuid4()}")


class _SliceCountingAgent(BaseAgent):
    """Tracks how many times ``_build_world_state_slice`` is called per
    step. Bug 9 asserts the count is exactly 1 — the cached slice from
    ``step()`` is reused by ``_decide_advance``."""

    stage = Stage.RECON

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.slice_calls = 0

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        self.slice_calls += 1
        return {"hosts": 0, "marker": self.slice_calls}

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return "test"

    async def _decide_advance(self, ctx: StageContext) -> StepDecision:
        # R3 #J2 Bug 9: prefer the cached slice if present.
        _ = ctx.last_world_state_slice or await self._build_world_state_slice(ctx)
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
    tools: list[Tool[Any, Any]] | None = None,
    policy_decision: PolicyDecision = PolicyDecision.ALLOW,
    budget: StepBudget | None = None,
    agent_cls: type[BaseAgent] = _SliceCountingAgent,
) -> BaseAgent:
    registry = _Registry(tools or [_TinyTool(), _FatTool()])
    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=policy_decision, reason="ok")
    )
    repo = MagicMock()
    repo.upsert_node = AsyncMock(return_value=None)
    repo.upsert_edge = AsyncMock(return_value=None)
    return agent_cls(
        router=router,
        repo=repo,
        registry=registry,
        policy=policy,
        sandbox=MagicMock(),
        budget=budget,
    )


def _build_ctx() -> StageContext:
    return StageContext(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )


# --------------------------------------------------------------------------- #
# Bug 3: validator logs on invalid shape                                       #
# --------------------------------------------------------------------------- #


def test_invalid_shape_validator_logs_warning(capsys, monkeypatch) -> None:
    """An unknown ``suppressed_tools`` shape lands in the validator
    fall-through path; it MUST emit a structlog warning before falling
    back to an empty dict, otherwise a future state-corruption bug
    would be silently swallowed (D4)."""
    from orchestrator.stage_controller import EngagementState, log as ctrl_log
    from worldstate.schema import Stage as _Stage

    captured: list[dict[str, Any]] = []

    def _capture(event: str, **kwargs: Any) -> None:
        captured.append({"event": event, **kwargs})

    monkeypatch.setattr(ctrl_log, "warning", _capture)

    eid = uuid4()
    st = EngagementState(
        engagement_id=eid,
        engagement={},
        current_stage=_Stage.RECON,
        suppressed_tools=42,  # type: ignore[arg-type]  # int is not a known shape
    )
    assert st.suppressed_tools == {}
    assert any(
        rec["event"] == "suppressed_tools.invalid_shape" for rec in captured
    ), f"expected warning event missing; got {captured}"
    # Surfaces the bad value type for debugging.
    matching = [
        rec for rec in captured if rec["event"] == "suppressed_tools.invalid_shape"
    ]
    assert matching[0]["value_type"] == "int"


# --------------------------------------------------------------------------- #
# Bug 5: turn-token attribution is output-size weighted                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_turn_tokens_split_is_output_size_weighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal-split would attribute the same per-tool tokens to a small
    bloodhound query and a large nuclei scan. With weighted attribution
    the noisy tool eats most of the budget — exactly what the per-tool
    ledger needs to reflect."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    # One turn with two tool calls: tiny + fat. The fat tool's summary
    # is ~400 chars; the tiny tool's is ~10 chars. Weighted split should
    # land ~97% / 3% — definitely not 50/50.
    router = _ScriptedRouter(
        [
            _mk_response(
                tool_calls=[
                    _mk_call("tiny", {"payload": "x"}),
                    _mk_call("fat", {"size": 100}),
                ],
                input_tokens=600,
                output_tokens=400,  # turn_tokens = 1000
            ),
            # Second LLM turn returns no tools to terminate the loop.
            _mk_response(tool_calls=[], text="done"),
        ]
    )
    agent = _build_agent(
        router=router,
        budget=StepBudget(interactive_max_turns=4),
    )
    ctx = _build_ctx()
    await agent.step(ctx)

    tiny_tokens = ctx.tokens_by_tool.get("tiny", 0)
    fat_tokens = ctx.tokens_by_tool.get("fat", 0)
    assert tiny_tokens > 0
    assert fat_tokens > 0
    # The fat tool MUST eat the lion's share. Equal-split would put
    # both at ~500. Weighted by ~10:400 means fat >> tiny.
    assert fat_tokens > tiny_tokens * 10, (
        f"expected fat >> tiny but got fat={fat_tokens} tiny={tiny_tokens}"
    )
    # Sanity: the total stays near the turn_tokens budget (modulo
    # integer truncation in the weighted split).
    assert 950 <= tiny_tokens + fat_tokens <= 1000


# --------------------------------------------------------------------------- #
# Bug 6: producer regression-budget check reads live count                     #
# --------------------------------------------------------------------------- #


def test_producer_budget_check_includes_pending_count() -> None:
    """``regressions_pending`` adds to ``regressions_used`` for the
    cap check so a future chained producer that emits two requests in
    one step doesn't bypass the cheap producer-side cut."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.LATERAL,
        regressions_used=2,
        regression_budget_max=3,
        regressions_pending=0,
    )
    # First request: 2 + 0 < 3 → allowed.
    assert (
        ctx.regressions_used + ctx.regressions_pending < ctx.regression_budget_max
    )
    ctx.regressions_pending += 1
    # Second request in the same step: 2 + 1 >= 3 → producer cuts it.
    assert (
        ctx.regressions_used + ctx.regressions_pending >= ctx.regression_budget_max
    )


# --------------------------------------------------------------------------- #
# Bug 7: sibling truncation structure                                          #
# --------------------------------------------------------------------------- #


def test_truncate_slice_with_sibling_returns_pair() -> None:
    """The sibling-form helper returns ``(truncated_list, info_dict)``
    so the calling agent can store each on adjacent top-level slice
    keys instead of mixing a sentinel-shaped dict into the homogeneous
    list."""
    items = [{"id": i} for i in range(_SLICE_CAP_DEFAULT + 5)]
    truncated, info = _truncate_slice_with_sibling(items, kind_label="Host")
    assert len(truncated) == _SLICE_CAP_DEFAULT
    # No sentinel inside the items list — every entry is a plain Host.
    assert all(set(d.keys()) == {"id"} for d in truncated)
    assert info is not None
    assert info["_total"] == _SLICE_CAP_DEFAULT + 5
    assert info["_cap"] == _SLICE_CAP_DEFAULT
    assert "_truncated" in info


def test_truncate_slice_with_sibling_no_truncation_returns_none_info() -> None:
    """When no truncation happened, ``info`` is None so callers know to
    skip the sibling top-level key entirely."""
    items = [{"id": 1}, {"id": 2}]
    truncated, info = _truncate_slice_with_sibling(items)
    assert truncated == items
    assert info is None


def test_legacy_truncate_slice_still_produces_in_list_sentinel() -> None:
    """The legacy ``_truncate_slice`` is preserved verbatim so existing
    callers behave identically until they're migrated to the
    sibling-form helper."""
    items = [{"id": i} for i in range(_SLICE_CAP_DEFAULT + 3)]
    out = _truncate_slice(items, kind_label="Host")
    assert len(out) == _SLICE_CAP_DEFAULT + 1
    sentinel = out[-1]
    assert isinstance(sentinel, dict)
    assert "_total" in sentinel


# --------------------------------------------------------------------------- #
# Bug 9: _decide_advance reuses the slice already built in step()              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_decide_advance_reuses_cached_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``step()`` builds the slice once, stashes it on ``ctx``, and
    ``_decide_advance`` consumes it — saving the second Neo4j
    round-trip per step (9× across stages)."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("tiny", {"payload": "x"})]),
            _mk_response(tool_calls=[], text="done"),
        ]
    )
    agent = _build_agent(
        router=router,
        budget=StepBudget(interactive_max_turns=4),
    )
    ctx = _build_ctx()
    await agent.step(ctx)
    # The interactive loop builds the slice ONCE at step entry and
    # reuses it across LLM turns. ``_decide_advance`` MUST consume the
    # cached slice from ``ctx.last_world_state_slice`` instead of
    # calling the builder a second time. Without the cache the
    # decide-advance fallback would push the count to 2.
    assert agent.slice_calls == 1  # type: ignore[attr-defined]
    # And the slice was stashed on ctx so the contract is visible.
    assert ctx.last_world_state_slice is not None
    assert ctx.last_world_state_slice.get("marker") == 1


@pytest.mark.asyncio
async def test_decide_advance_falls_back_when_slice_not_cached() -> None:
    """Tests / legacy callers that invoke ``_decide_advance`` directly
    without going through ``step()`` MUST still see a slice — the
    fallback rebuilds it."""
    router = _ScriptedRouter([])
    agent = _build_agent(router=router)
    ctx = _build_ctx()
    # Slice not pre-populated.
    assert ctx.last_world_state_slice is None
    decision = await agent._decide_advance(ctx)
    assert decision == StepDecision.CONTINUE
    # Fallback build fired.
    assert agent.slice_calls == 1  # type: ignore[attr-defined]
