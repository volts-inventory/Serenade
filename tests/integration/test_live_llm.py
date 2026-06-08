"""Live LLM smoke tests — verify the Anthropic / MLX backends actually round-trip.

Gated behind `RUN_LIVE_LLM=1` so they never fire in default test runs (which
would burn tokens and require live infrastructure). Run manually before any
real engagement:

    RUN_LIVE_LLM=1 LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... \
        pytest -m llm tests/integration/test_live_llm.py

    RUN_LIVE_LLM=1 LLM_BACKEND=mlx MLX_BASE_URL=http://host.docker.internal:8080/v1 \
        pytest -m llm tests/integration/test_live_llm.py

The contract under test:
- LLMRouter routes per CallSite to the right model
- A simple tool-use round-trip works (Anthropic native; MLX prompt-encoded)
- Cost / token counts are populated in the response
"""

from __future__ import annotations

import os

import pytest

from llm.router import CallSite, LLMRequest, build_router

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("RUN_LIVE_LLM", "").strip(),
        reason="Live LLM tests require RUN_LIVE_LLM=1 and a configured backend",
    ),
]


class _FakeTool:
    name = "echo"
    description = "Echo back the given text. Use this when the user asks to repeat something."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        }


@pytest.mark.asyncio
async def test_router_text_completion():
    """Basic text completion. Asserts non-empty output + token counts populated."""
    router = build_router()
    resp = await router.complete(
        LLMRequest(
            system="You are a terse assistant. Reply in one sentence.",
            user="Say 'hello world' and nothing else.",
            site=CallSite.PARSER,
            max_tokens=50,
            temperature=0.0,
        )
    )
    assert resp.text.strip(), "empty completion"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.latency_ms > 0


@pytest.mark.asyncio
async def test_router_tool_use_roundtrip():
    """The model should respond with a tool_call when given an obvious tool match."""
    router = build_router()
    resp = await router.complete(
        LLMRequest(
            system=(
                "You are a strict tool-using agent. Whenever the user asks you to "
                "repeat or echo something, you MUST use the echo tool with the "
                "text to repeat. Do not respond with prose."
            ),
            user="Please echo back: serenade-smoke-test",
            site=CallSite.PLANNER,
            tools=[_FakeTool()],  # type: ignore[list-item]
            max_tokens=300,
            temperature=0.0,
        )
    )
    assert resp.tool_calls, f"expected a tool_call, got text: {resp.text[:200]}"
    call = resp.tool_calls[0]
    assert call.name == "echo"
    assert "serenade-smoke-test" in call.input.get("text", "")


@pytest.mark.asyncio
async def test_router_cost_accounting():
    """Anthropic backend should populate cost_usd; MLX should be 0.0 (local)."""
    router = build_router()
    resp = await router.complete(
        LLMRequest(
            system="Reply in one word.",
            user="Echo: ack",
            site=CallSite.TRIAGE,
            max_tokens=20,
            temperature=0.0,
        )
    )
    backend = os.environ.get("LLM_BACKEND", "anthropic")
    if backend == "anthropic":
        assert resp.cost_usd > 0
    else:
        assert resp.cost_usd == 0.0
