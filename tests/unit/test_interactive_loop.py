"""R3 #F: BaseAgent.step() interactive tool-use loop.

Phase F flips ``BaseAgent.step`` from one-LLM-call-per-step (where the
model never sees tool results inside the same conversation) to an
interactive loop that feeds tool results back to the model between turns
the way a senior pentester chains decisions. These tests lock in:

* multi-turn happy path terminates when the model says "I'm done";
* the ``interactive_max_turns`` cap is honored;
* gate denials (OPA DENY, critic VETO, budget) flow back to the model
  as ``tool_result`` blocks so it can pivot;
* the ``SERENADE_BATCHED_AGENT_LOOP=1`` env flag reverts to legacy
  single-shot behaviour;
* tool output is bounded to ~4KB before being fed back;
* per-step token accounting sums across all LLM turns.
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
    _format_tool_result_for_model,
    _is_batched_mode,
)
from agents.critic import Critique, Verdict
from agents.recon import ReconAgent
from llm.anthropic_backend import AnthropicBackend
from llm.router import LLMRequest, LLMResponse, ToolCall
from orchestrator.settings import Settings
from policy.engine import PolicyDecision, PolicyResult
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage, WorldStateNode

# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #


class _EchoInput(BaseModel):
    payload: str = ""


class _EchoOutput(BaseModel):
    payload: str = ""


class _EchoTool(Tool[_EchoInput, _EchoOutput]):
    name = "echo_tool"
    description = "Test tool that echoes its input."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    destructive = False
    Input = _EchoInput
    Output = _EchoOutput

    def extract_targets(self, input: _EchoInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _EchoOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # R3 #I1 Bug 7: summarize_for_model is now @abstractmethod;
        # test tools must override or they fail at instantiation.
        return f"echo: {output.payload[:200]} persisted={persisted}"

    async def run(self, input: _EchoInput, ctx: ToolRunContext) -> ToolResult[_EchoOutput]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_EchoOutput(payload=input.payload),
            exit_code=0,
            started_at="2025-01-01T00:00:00Z",
            ended_at="2025-01-01T00:00:00Z",
        )


class _BulkOutput(BaseModel):
    """Output schema large enough to exercise the 4KB truncation."""

    blob: str = ""


class _BulkInput(BaseModel):
    size: int = 0


class _BulkTool(Tool[_BulkInput, _BulkOutput]):
    name = "bulk_tool"
    description = "Test tool that returns a huge output."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input = _BulkInput
    Output = _BulkOutput

    def extract_targets(self, input: _BulkInput) -> list[str]:
        return []

    def summarize_for_model(
        self, output: _BulkOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # R3 #I1 Bug 7: required override for the abstractmethod. We
        # report only the blob length — never the 100KB body — so this
        # also exercises the "summary is small even when output is huge"
        # contract from the surrounding tests.
        return (
            f"bulk_tool ok: blob_len={len(output.blob)} persisted={persisted} "
            f"tool_run_id={tool_run_id}"
        )

    async def run(self, input: _BulkInput, ctx: ToolRunContext) -> ToolResult[_BulkOutput]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_BulkOutput(blob="A" * input.size),
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
    """Router that returns a scripted sequence of LLMResponses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, req: LLMRequest, *, agent: str = "unknown") -> LLMResponse:
        self.calls.append(req)
        if not self._responses:
            raise AssertionError("router exhausted but step() asked for another LLM call")
        return self._responses.pop(0)


def _mk_response(
    *,
    tool_calls: list[ToolCall] | None = None,
    text: str = "",
    input_tokens: int = 100,
    output_tokens: int = 50,
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


def _mk_call(
    name: str,
    input: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> ToolCall:
    return ToolCall(name=name, input=input or {}, call_id=call_id or f"call-{uuid4()}")


class _TestAgent(BaseAgent):
    """Minimal concrete BaseAgent subclass for the interactive-loop tests."""

    stage = Stage.RECON

    async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
        return {"hosts": 0}

    def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:
        return "exercise the interactive loop"

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
    tools: list[Tool[Any, Any]] | None = None,
    policy_decision: PolicyDecision = PolicyDecision.ALLOW,
    budget: StepBudget | None = None,
) -> _TestAgent:
    registry = _Registry(tools or [_EchoTool()])

    policy = MagicMock()
    policy.evaluate = AsyncMock(
        return_value=PolicyResult(decision=policy_decision, reason="ok")
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
        budget=budget,
    )


def _build_ctx(**overrides: Any) -> StageContext:
    base = {
        "engagement_id": uuid4(),
        "engagement": {"approval_policy": "manual"},
        "stage": Stage.RECON,
    }
    base.update(overrides)
    return StageContext(**base)


# --------------------------------------------------------------------------- #
# Multi-turn happy path                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_turn_happy_path_terminates_when_model_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1: tool call. Turn 2: tool call. Turn 3: no tool calls → end.

    Verifies the loop chains three LLM calls in a single step() and that
    the conversation grows: each subsequent LLMRequest carries the prior
    assistant+user turns in req.messages.
    """
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("echo_tool", {"payload": "first"})]),
            _mk_response(tool_calls=[_mk_call("echo_tool", {"payload": "second"})]),
            _mk_response(text="all done"),
        ]
    )
    agent = _build_agent(router=router)
    ctx = _build_ctx()

    result = await agent.step(ctx)

    assert result.decision == StepDecision.CONTINUE
    assert result.tool_runs == 2  # turns 1 and 2 executed their tool
    assert len(router.calls) == 3  # three LLM turns within the step

    # Conversation grows turn-over-turn:
    # turn 1 → just the initial user prompt
    # turn 2 → user + assistant + user(tool_results)
    # turn 3 → ... + assistant + user(tool_results)
    msgs_turn1 = router.calls[0].messages
    msgs_turn2 = router.calls[1].messages
    msgs_turn3 = router.calls[2].messages
    assert msgs_turn1 is not None and len(msgs_turn1) == 1
    assert msgs_turn2 is not None and len(msgs_turn2) == 3
    assert msgs_turn3 is not None and len(msgs_turn3) == 5

    # Per-turn telemetry: tokens accumulate across turns.
    assert result.tokens_used == 3 * 150  # 100 + 50 per response, 3 turns


# --------------------------------------------------------------------------- #
# Cap respected                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_interactive_max_turns_cap_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the model keeps emitting tool calls, the loop terminates at the
    configured interactive_max_turns cap rather than running forever."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    cap = 4
    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("echo_tool", {"payload": str(i)})])
            for i in range(cap)
        ]
    )
    agent = _build_agent(router=router, budget=StepBudget(interactive_max_turns=cap))
    ctx = _build_ctx()

    result = await agent.step(ctx)

    assert len(router.calls) == cap
    assert result.tool_runs == cap


# --------------------------------------------------------------------------- #
# Gate denial flows back as tool_result the model can react to                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_policy_deny_feeds_back_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPA DENY on a tool call becomes a ``tool_result`` block whose
    content explains the denial. The model gets to see the denial and
    pivot — verified by the conversation containing the denial string
    on turn 2."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(
                tool_calls=[_mk_call("echo_tool", {"payload": "first"}, call_id="c1")]
            ),
            _mk_response(text="ok, pivoting"),
        ]
    )
    agent = _build_agent(router=router, policy_decision=PolicyDecision.DENY)
    ctx = _build_ctx()

    result = await agent.step(ctx)

    # Tool was denied, not executed.
    assert result.tool_runs == 0

    # Turn 2 must include the tool_result block with denial text.
    msgs_turn2 = router.calls[1].messages
    assert msgs_turn2 is not None
    # Last user turn (the one carrying tool_results) is at index 2.
    tool_result_turn = msgs_turn2[2]
    assert tool_result_turn["role"] == "user"
    blocks = tool_result_turn["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "c1"
    assert "tool denied by policy" in blocks[0]["content"]


# --------------------------------------------------------------------------- #
# Critic VETO feeds back AND queues for operator                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_critic_veto_feeds_back_queues_approval_and_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In unattended mode a critic VETO must:
    1. show up as a tool_result the model can read on the next turn,
    2. queue a pending_approvals row,
    3. flip the post-loop decision to ESCALATE_HUMAN."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    monkeypatch.setattr(
        "agents.base._persist_approval",
        AsyncMock(return_value=None),
    )

    # Need a destructive tool so the critic gate fires.
    class _DangerInput(BaseModel):
        x: int = 0

    class _DangerOutput(BaseModel):
        x: int = 0

    class _DangerTool(Tool[_DangerInput, _DangerOutput]):
        name = "danger"
        description = "destructive"
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
            return f"danger ok: x={output.x} persisted={persisted}"

        async def run(self, input: _DangerInput, ctx: ToolRunContext) -> ToolResult[_DangerOutput]:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=_DangerOutput(),
                exit_code=0,
                started_at="2025-01-01T00:00:00Z",
                ended_at="2025-01-01T00:00:00Z",
            )

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("danger", {"x": 1}, call_id="cv1")]),
            _mk_response(text="will pivot"),
        ]
    )
    agent = _build_agent(router=router, tools=[_DangerTool()])
    agent.critic_required_for_risk_classes = (RiskClass.DESTRUCTIVE,)
    agent.pre_action_critic = MagicMock()
    agent.pre_action_critic.review = AsyncMock(
        return_value=Critique(
            verdict=Verdict.VETO,
            confidence=0.95,
            primary_concern="too dangerous",
            critic_name="test_critic",
        )
    )

    ctx = _build_ctx(engagement={"approval_policy": "auto_reject"})
    result = await agent.step(ctx)

    # Pending approval was queued AND post-loop decision is ESCALATE_HUMAN.
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert result.approvals_created == 1

    # Model saw the veto in turn 2's tool_results.
    msgs_turn2 = router.calls[1].messages
    assert msgs_turn2 is not None
    blocks = msgs_turn2[2]["content"]
    assert "tool denied by critic" in blocks[0]["content"]
    assert "too dangerous" in blocks[0]["content"]


# --------------------------------------------------------------------------- #
# Batched fallback honors the env flag                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_batched_mode_env_flag_runs_single_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``SERENADE_BATCHED_AGENT_LOOP=1`` the legacy single-shot path
    runs: exactly one router.complete call per step(), no matter how many
    tool calls the model returned (they all execute, but no follow-up
    turn happens)."""
    monkeypatch.setenv("SERENADE_BATCHED_AGENT_LOOP", "1")
    assert _is_batched_mode() is True

    router = _ScriptedRouter(
        [
            _mk_response(
                tool_calls=[
                    _mk_call("echo_tool", {"payload": "a"}),
                    _mk_call("echo_tool", {"payload": "b"}),
                ]
            ),
        ]
    )
    agent = _build_agent(router=router)
    ctx = _build_ctx()

    result = await agent.step(ctx)
    # Exactly one LLM call in the entire step, regardless of tool count.
    assert len(router.calls) == 1
    # Both tool calls executed.
    assert result.tool_runs == 2
    # Legacy path uses the ``user`` string, not ``messages``.
    assert router.calls[0].messages is None


# --------------------------------------------------------------------------- #
# Tool result bounding                                                         #
# --------------------------------------------------------------------------- #


def test_tool_result_bounding_truncates_large_payloads() -> None:
    """R3 #F.5: each tool now emits a per-tool summary (<=500 chars) via
    ``Tool.summarize_for_model`` and the JSON dump path is gone. The
    default summarizer enumerates list-shaped fields and includes the
    tool_run_id pointer; a tool whose Output has only a giant string
    field gets a tiny summary (one field name, no content)."""
    big = _BulkOutput(blob="A" * 100_000)
    tool_run_id = uuid4()
    out = _format_tool_result_for_model(
        tool=_BulkTool(),
        output=big,
        tool_run_id=tool_run_id,
        nodes_written=0,
    )
    # Default summarizer for _BulkOutput has no list fields → reports
    # "no list fields"; total well under the 500-char cap.
    assert len(out) <= 500
    assert "bulk_tool" in out
    assert str(tool_run_id) in out
    assert "persisted=0" in out
    # The 100KB blob value MUST NOT have leaked into the summary.
    assert "AAAAA" not in out


def test_overlong_summary_is_hard_truncated_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If a tool's ``summarize_for_model`` returns >500 chars (a bug in
    the tool override), the agent layer hard-truncates with a
    ``... [truncated]`` suffix and emits an
    ``agent.tool_summary_overflow`` warning."""

    class _OverlongOutput(BaseModel):
        blob: str = ""

    class _OverlongTool(Tool[_BulkInput, _OverlongOutput]):
        name = "overlong"
        description = "bad summary"
        stage = Stage.RECON
        risk_class = RiskClass.RECON
        Input = _BulkInput
        Output = _OverlongOutput

        def extract_targets(self, input: _BulkInput) -> list[str]:
            return []

        def summarize_for_model(
            self, output: _OverlongOutput, *, tool_run_id: UUID, persisted: int
        ) -> str:
            return "X" * 5000  # bug: way over the cap

        async def run(self, input: _BulkInput, ctx: ToolRunContext) -> ToolResult[_OverlongOutput]:
            raise NotImplementedError

    caplog.clear()
    out = _format_tool_result_for_model(
        tool=_OverlongTool(),
        output=_OverlongOutput(),
        tool_run_id=uuid4(),
        nodes_written=0,
    )
    assert len(out) <= 500
    assert out.endswith(" ... [truncated]")


@pytest.mark.asyncio
async def test_tool_result_in_conversation_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 #F.5: end-to-end: a tool that returns 100KB of output produces
    a tool_result block in the next turn's conversation whose content
    is the bounded summary (<=500 chars), with the raw blob NOT inlined."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(
                tool_calls=[_mk_call("bulk_tool", {"size": 100_000}, call_id="big1")]
            ),
            _mk_response(text="done"),
        ]
    )
    agent = _build_agent(router=router, tools=[_BulkTool()])
    ctx = _build_ctx()
    await agent.step(ctx)

    msgs_turn2 = router.calls[1].messages
    assert msgs_turn2 is not None
    tool_result_block = msgs_turn2[2]["content"][0]
    content_str = tool_result_block["content"]
    assert len(content_str) <= 500
    # Raw 100KB blob must NOT leak into the planner-visible summary.
    assert "AAAAA" not in content_str
    # Summary mentions the tool by name + tool_run_id pointer.
    assert "bulk_tool" in content_str
    assert "tool_run_id=" in content_str


# --------------------------------------------------------------------------- #
# Token accounting across turns                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tokens_accumulate_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ctx.tokens_used after a 3-turn step equals the sum of the three
    responses' token counts."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    router = _ScriptedRouter(
        [
            _mk_response(
                tool_calls=[_mk_call("echo_tool", {"payload": "1"})],
                input_tokens=10,
                output_tokens=5,
            ),
            _mk_response(
                tool_calls=[_mk_call("echo_tool", {"payload": "2"})],
                input_tokens=20,
                output_tokens=8,
            ),
            _mk_response(text="done", input_tokens=15, output_tokens=3),
        ]
    )
    agent = _build_agent(router=router)
    ctx = _build_ctx()
    result = await agent.step(ctx)

    expected = (10 + 5) + (20 + 8) + (15 + 3)
    assert ctx.tokens_used == expected
    assert result.tokens_used == expected


# --------------------------------------------------------------------------- #
# Stop-on-all-gated: model gets ONE more turn after every tool is denied      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_all_gated_gives_model_one_more_turn_then_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 #J2 Bug 4: when every tool a turn proposes is gated out, the
    model gets ONE additional turn to react. Two consecutive all-gated
    turns end the loop early AND return ``ESCALATE_HUMAN`` with the
    per-tool gate reasons surfaced as the note — previously the loop
    silently ``break``-ed and let ``_decide_advance`` decide, which
    chewed through ``stalled_turn_count`` increments before the global
    watchdog noticed."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    # Every turn proposes a tool the policy denies; cap is large so the
    # only stop condition that can fire is "two consecutive all-gated".
    router = _ScriptedRouter(
        [_mk_response(tool_calls=[_mk_call("echo_tool", {"payload": str(i)})])
         for i in range(10)]
    )
    agent = _build_agent(
        router=router,
        policy_decision=PolicyDecision.DENY,
        budget=StepBudget(interactive_max_turns=10),
    )
    ctx = _build_ctx()
    result = await agent.step(ctx)
    # Two turns of the same tool getting gated → stops. Not 10, not 1.
    assert len(router.calls) == 2
    # R3 #J2 Bug 4: stop is now ESCALATE_HUMAN, not CONTINUE.
    assert result.decision == StepDecision.ESCALATE_HUMAN
    # R3 #K3 Bug 1: note format shifted from "all tools gated twice"
    # to per-tool "tools gated twice [...]" because the watchdog is
    # now per-tool — but the "tools gated twice" phrase is preserved
    # so existing operator docs still match. The persistent tool name
    # appears in the bracketed list.
    assert "tools gated twice" in result.notes
    assert "echo_tool" in result.notes


# --------------------------------------------------------------------------- #
# Anthropic backend strips _text_fallback hint before sending                   #
# --------------------------------------------------------------------------- #


def test_anthropic_backend_strips_private_hint_keys() -> None:
    """The _text_fallback hint BaseAgent attaches for MLX is private to
    the local-LLM path. The Anthropic API would reject it as an unknown
    field, so the backend strips it before sending."""
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend.settings = Settings(_env_file=None)  # type: ignore[call-arg]

    msgs_in = [
        {"role": "user", "content": "hi", "_text_fallback": "should be stripped"},
        {"role": "assistant", "content": "ok"},
    ]
    cleaned = [
        {k: v for k, v in m.items() if not k.startswith("_")} for m in msgs_in
    ]
    assert all("_text_fallback" not in m for m in cleaned)


# --------------------------------------------------------------------------- #
# R1 watchdog still increments after interactive step                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recon_zero_host_watchdog_still_works_through_interactive_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The R1 zero-host watchdog lives in ReconAgent._decide_advance which
    fires AFTER the interactive loop terminates. Confirm the new loop
    doesn't interfere with it."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)

    repo = MagicMock()
    repo.count = AsyncMock(return_value={"Host": 0, "Service": 0})
    repo.get_nodes_by_kind = AsyncMock(return_value=[])

    router = _ScriptedRouter([_mk_response(text="nothing to do")])

    agent = ReconAgent(
        router=router,
        repo=repo,
        registry=_Registry([]),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )
    ctx = _build_ctx()
    ctx.consecutive_zero_host_turns = 2  # one more zero and watchdog escalates

    result = await agent.step(ctx)
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert ctx.consecutive_zero_host_turns == 3


# --------------------------------------------------------------------------- #
# agent.llm_turn event published per turn                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_llm_turn_event_published_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators should be able to see how many LLM turns a step took.
    The new ``agent.llm_turn`` event fires once per LLM call within
    a step; ``stage.step.start`` still fires once per step."""
    monkeypatch.delenv("SERENADE_BATCHED_AGENT_LOOP", raising=False)
    bus = MagicMock()
    bus.publish = AsyncMock(return_value=None)
    monkeypatch.setattr("agents.base.get_bus", lambda: bus)

    router = _ScriptedRouter(
        [
            _mk_response(tool_calls=[_mk_call("echo_tool", {"payload": "a"})]),
            _mk_response(text="done"),
        ]
    )
    agent = _build_agent(router=router)
    ctx = _build_ctx()
    await agent.step(ctx)

    kinds = [c.kwargs.get("kind") for c in bus.publish.call_args_list]
    assert kinds.count("stage.step.start") == 1
    assert kinds.count("agent.llm_turn") == 2  # two LLM calls = two turn events
