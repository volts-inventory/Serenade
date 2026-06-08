from __future__ import annotations

import json

import pytest

from agents.council import Council, triage_council
from agents.critic import BaseCritic, Verdict
from llm.router import LLMRequest, LLMResponse, LLMRouter
from orchestrator.settings import Settings


class _SeqBackend:
    """Returns a scripted sequence of responses, one per call."""

    name = "seq"

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.idx = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        text = self.texts[self.idx]
        self.idx += 1
        return LLMResponse(
            text=text,
            tool_calls=[],
            structured=None,
            model="seq",
            input_tokens=10,
            output_tokens=10,
            cached_input_tokens=0,
            cost_usd=0.0001,
            latency_ms=5,
            raw=None,
        )


def _verdict(v: Verdict, conf: float = 0.9, name: str = "x") -> str:
    return json.dumps(
        {
            "verdict": v.value,
            "confidence": conf,
            "primary_concern": "test",
            "issues": [],
            "questions_for_human": [],
            "revised_proposal": None,
        }
    )


class _StaticCritic(BaseCritic):
    def __init__(self, router: LLMRouter, name: str) -> None:
        super().__init__(router)
        self.name = name
        self.perspective = "test"

    def _system_prompt(self) -> str:
        return "test"

    def _format_subject(self, subject: dict[str, object]) -> str:
        return "test"


@pytest.mark.asyncio
async def test_council_unanimous_approve_skips_synthesizer():
    backend = _SeqBackend([_verdict(Verdict.APPROVE, 0.9, n) for n in ("a", "b", "c")])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, n) for n in ("a", "b", "c")],
    )
    v = await council.deliberate({"subject": "x"})
    assert v.decision == Verdict.APPROVE
    assert backend.idx == 3  # only 3 critic calls, no synthesizer


@pytest.mark.asyncio
async def test_council_any_veto_vetoes():
    backend = _SeqBackend(
        [
            _verdict(Verdict.APPROVE, 0.9),
            _verdict(Verdict.VETO, 0.95),
            _verdict(Verdict.APPROVE, 0.85),
        ]
    )
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, n) for n in ("a", "b", "c")],
    )
    v = await council.deliberate({"subject": "x"})
    assert v.decision == Verdict.VETO
    assert "vetoed by" in v.rationale


@pytest.mark.asyncio
async def test_council_split_runs_synthesizer():
    backend = _SeqBackend(
        [
            _verdict(Verdict.APPROVE, 0.9),
            _verdict(Verdict.REVISE, 0.7),
            _verdict(Verdict.APPROVE, 0.6),  # not high enough for unanimous shortcut
            json.dumps({"verdict": "revise", "confidence": 0.7, "rationale": "tighten params"}),
        ]
    )
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, n) for n in ("a", "b", "c")],
    )
    v = await council.deliberate({"subject": "x"})
    assert v.decision == Verdict.REVISE
    assert v.rationale == "tighten params"
    assert backend.idx == 4  # 3 critics + synthesizer


@pytest.mark.asyncio
async def test_council_requires_min_two_members():
    router = LLMRouter(_SeqBackend([_verdict(Verdict.APPROVE)]), Settings(_env_file=None))  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Council(name="t", router=router, members=[_StaticCritic(router, "a")])


def test_triage_council_has_three_members():
    router = LLMRouter(_SeqBackend([]), Settings(_env_file=None))  # type: ignore[call-arg]
    council = triage_council(router)
    assert len(council.members) == 3
    names = {m.name for m in council.members}
    assert "exploitability_focused" in names
    assert "blast_radius_focused" in names
    assert "false_positive_skeptic" in names
