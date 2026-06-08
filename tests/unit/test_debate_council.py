"""Tests for multi-agent debate council."""

from __future__ import annotations

import json

import pytest

from agents.council import Council, debate_council
from agents.critic import BaseCritic, Verdict
from agents.debate_critics import AggressivePentester, CautiousDefender
from llm.router import LLMRequest, LLMResponse, LLMRouter
from orchestrator.settings import Settings


class _SeqBackend:
    """Returns a scripted sequence of responses, one per call."""

    name = "seq"

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.idx = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        text = self.texts[self.idx % len(self.texts)] if self.texts else ""
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


def _verdict(v: Verdict, conf: float = 0.9) -> str:
    return json.dumps(
        {
            "verdict": v.value,
            "confidence": conf,
            "primary_concern": "test concern",
            "issues": [],
            "questions_for_human": [],
            "revised_proposal": None,
            "reasoning_trace": "test reasoning",
            "suggested_fix": None,
            "flaws": [],
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


# ---- Debate rounds ----


@pytest.mark.asyncio
async def test_debate_single_round_unanimous_approve():
    """debate_rounds=1 behaves like the normal council path."""
    backend = _SeqBackend([_verdict(Verdict.APPROVE, 0.9)] * 2)
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, "a"), _StaticCritic(router, "b")],
    )
    v = await council.deliberate({"subject": "x"}, debate_rounds=1)
    assert v.decision == Verdict.APPROVE
    assert backend.idx == 2  # 2 critic calls, no synthesizer


@pytest.mark.asyncio
async def test_debate_two_rounds_unanimous_approve_skips_remaining():
    """Unanimous high-confidence approve in round 1 skips round 2."""
    backend = _SeqBackend([_verdict(Verdict.APPROVE, 0.9)] * 2)
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, "a"), _StaticCritic(router, "b")],
    )
    v = await council.deliberate({"subject": "x"}, debate_rounds=3)
    assert v.decision == Verdict.APPROVE
    assert "round 1" in v.rationale
    # Only 2 calls — round 1 short-circuited before rounds 2 and 3
    assert backend.idx == 2


@pytest.mark.asyncio
async def test_debate_veto_in_first_round_short_circuits():
    """Any VETO in any round triggers immediate VETO."""
    backend = _SeqBackend([
        _verdict(Verdict.APPROVE, 0.9),
        _verdict(Verdict.VETO, 0.95),
    ])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, "a"), _StaticCritic(router, "b")],
    )
    v = await council.deliberate({"subject": "x"}, debate_rounds=3)
    assert v.decision == Verdict.VETO
    assert "round 1" in v.rationale
    # Only 2 calls — veto in round 1 short-circuits before rounds 2 and 3
    assert backend.idx == 2


@pytest.mark.asyncio
async def test_debate_veto_in_second_round():
    """A VETO in round 2 (after APPROVE in round 1) produces VETO."""
    # Round 1: both approve (but low confidence, no short-circuit)
    # Round 2: one vetoes
    backend = _SeqBackend([
        _verdict(Verdict.APPROVE, 0.6),  # r1 critic a
        _verdict(Verdict.APPROVE, 0.6),  # r1 critic b
        _verdict(Verdict.APPROVE, 0.7),  # r2 critic a
        _verdict(Verdict.VETO, 0.95),    # r2 critic b
    ])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, "a"), _StaticCritic(router, "b")],
    )
    v = await council.deliberate({"subject": "x"}, debate_rounds=2)
    assert v.decision == Verdict.VETO
    assert "round 2" in v.rationale
    # 4 calls: 2 per round, veto in round 2
    assert backend.idx == 4


@pytest.mark.asyncio
async def test_debate_split_goes_to_synthesizer():
    """When no short-circuit fires, the synthesizer resolves."""
    # Round 1: split (low confidence, no unanimous approve)
    # Round 2: still split
    # Synthesizer call
    backend = _SeqBackend([
        _verdict(Verdict.APPROVE, 0.6),  # r1 critic a
        _verdict(Verdict.REVISE, 0.7),   # r1 critic b
        _verdict(Verdict.APPROVE, 0.65), # r2 critic a
        _verdict(Verdict.REVISE, 0.65),  # r2 critic b
        # synthesizer
        json.dumps({"verdict": "revise", "confidence": 0.7, "rationale": "needs more evidence"}),
    ])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_StaticCritic(router, "a"), _StaticCritic(router, "b")],
    )
    v = await council.deliberate({"subject": "x"}, debate_rounds=2)
    assert v.decision == Verdict.REVISE
    assert v.rationale == "needs more evidence"
    # 4 critic calls + 1 synthesizer = 5
    assert backend.idx == 5
    # All critiques from both rounds should be in the member_critiques
    assert len(v.member_critiques) == 4


@pytest.mark.asyncio
async def test_debate_members_see_prior_responses():
    """In round 2+, members receive prior critiques in the subject."""
    seen_subjects: list[dict] = []

    class _SpyCritic(BaseCritic):
        def __init__(self, router: LLMRouter, name_: str) -> None:
            super().__init__(router)
            self.name = name_
            self.perspective = "test"

        def _system_prompt(self) -> str:
            return "test"

        def _format_subject(self, subject: dict[str, object]) -> str:
            seen_subjects.append(dict(subject))
            return "test"

    backend = _SeqBackend([
        _verdict(Verdict.APPROVE, 0.6),  # r1 critic a
        _verdict(Verdict.APPROVE, 0.6),  # r1 critic b
        _verdict(Verdict.APPROVE, 0.9),  # r2 critic a
        _verdict(Verdict.APPROVE, 0.9),  # r2 critic b
    ])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = Council(
        name="t",
        router=router,
        members=[_SpyCritic(router, "a"), _SpyCritic(router, "b")],
    )
    await council.deliberate({"subject": "x"}, debate_rounds=2)

    # Round 1 subjects should NOT have prior critiques
    assert "_prior_critiques" not in seen_subjects[0]
    assert "_prior_critiques" not in seen_subjects[1]

    # Round 2 subjects SHOULD have prior critiques
    assert "_prior_critiques" in seen_subjects[2]
    assert "_debate_round" in seen_subjects[2]
    assert seen_subjects[2]["_debate_round"] == 2
    assert len(seen_subjects[2]["_prior_critiques"]) == 2  # both round-1 critiques


# ---- Debate critics ----


def test_debate_critics_exist():
    """AggressivePentester and CautiousDefender are importable and have correct attrs."""
    router = LLMRouter(_SeqBackend([]), Settings(_env_file=None))  # type: ignore[call-arg]

    ap = AggressivePentester(router)
    assert ap.name == "aggressive_pentester"
    assert "viability" in ap.perspective.lower() or "exploit" in ap.perspective.lower()

    cd = CautiousDefender(router)
    assert cd.name == "cautious_defender"
    assert "risk" in cd.perspective.lower()


def test_debate_council_factory():
    """debate_council() creates a 2-member council with the debate critics."""
    router = LLMRouter(_SeqBackend([]), Settings(_env_file=None))  # type: ignore[call-arg]
    council = debate_council(router)
    assert len(council.members) == 2
    names = {m.name for m in council.members}
    assert "aggressive_pentester" in names
    assert "cautious_defender" in names


@pytest.mark.asyncio
async def test_debate_council_with_debate_rounds():
    """debate_council with debate_rounds=2 runs the full debate flow."""
    backend = _SeqBackend([
        _verdict(Verdict.APPROVE, 0.9),  # r1 aggressive
        _verdict(Verdict.APPROVE, 0.9),  # r1 cautious
    ])
    router = LLMRouter(backend, Settings(_env_file=None))  # type: ignore[call-arg]
    council = debate_council(router)
    v = await council.deliberate({"subject": "x"}, debate_rounds=2)
    assert v.decision == Verdict.APPROVE
    # Unanimous high-confidence in round 1 short-circuits
    assert backend.idx == 2
