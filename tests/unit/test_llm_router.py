from __future__ import annotations

import pytest

from llm.router import CallSite, LLMRequest, LLMResponse, LLMRouter, ToolCall
from orchestrator.settings import Settings


class _FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        return LLMResponse(
            text="ok",
            tool_calls=[
                ToolCall(name="nmap_service_scan", input={"targets": ["1.1.1.1"]}, call_id="x")
            ],
            structured=None,
            model="fake-model",
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=80,
            cost_usd=0.001,
            latency_ms=42,
            raw=None,
        )


@pytest.mark.asyncio
async def test_router_passes_through_and_logs():
    backend = _FakeBackend()
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    resp = await router.complete(
        LLMRequest(
            system="hello",
            user="world",
            site=CallSite.PLANNER,
            engagement_id="eng-1",
        )
    )
    assert resp.text == "ok"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "nmap_service_scan"
    assert resp.cached_input_tokens == 80
    assert backend.calls[0].site == CallSite.PLANNER


@pytest.mark.asyncio
async def test_router_routes_per_call_site():
    backend = _FakeBackend()
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    for site in (CallSite.PARSER, CallSite.SYNTHESIZER, CallSite.TRIAGE):
        await router.complete(LLMRequest(system="s", user="u", site=site))
    assert [c.site for c in backend.calls] == [
        CallSite.PARSER,
        CallSite.SYNTHESIZER,
        CallSite.TRIAGE,
    ]
