"""R3 #J1 — taxonomy boundary between StructuralCheck and LLMCritic.

The Phase J1 critic split (Refactor 3) decomposed the old monolithic
``BaseCritic`` into two narrower types:

- :class:`StructuralCheck` — deterministic, LLM-free; subclasses
  implement ``review(subject) -> Critique`` only. The three
  StageRegressionCouncil members (NoveltyCritic, ScopeCritic,
  BudgetCritic) are the canonical instances.
- :class:`LLMCritic` — defines ``_system_prompt`` / ``_format_subject``
  and lets the base class drive the LLM call inside ``review()``.
  PreExploitCritic, PreLateralCritic, ScopeBoundaryCritic, ReportCritic
  are the canonical instances.

These tests enforce the boundary: structural-check subclasses must
never hit the router, and LLM-critic subclasses must.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agents.critic import (
    BaseCritic,
    Critique,
    LLMCritic,
    PreExploitCritic,
    PreLateralCritic,
    ReportCritic,
    ScopeBoundaryCritic,
    StructuralCheck,
    Verdict,
)
from agents.regression import RegressionRequest, RegressionTriggerKind
from agents.regression_council import (
    BudgetCritic,
    NoveltyCritic,
    ScopeCritic,
)
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# StructuralCheck subclasses never call the LLM                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_critic_never_calls_router() -> None:
    """BudgetCritic is a pure arithmetic check — its review() must
    not invoke router.complete under any verdict shape."""
    router = AsyncMock()
    critic = BudgetCritic(router)
    critic._regressions_used = 0
    critic._token_budget = 0
    critic._tokens_used = 0
    verdict = await critic.review({}, engagement_id=uuid4())
    assert verdict.verdict == Verdict.APPROVE
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_scope_critic_never_calls_router() -> None:
    """ScopeCritic does literal CIDR / domain membership — no LLM."""
    router = AsyncMock()
    critic = ScopeCritic(router)
    critic._request = RegressionRequest(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta={"cidrs": ["10.0.0.0/24"]},
        rationale="test",
    )
    critic._scope = {"cidrs": ["10.0.0.0/24"]}
    await critic.review({}, engagement_id=uuid4())
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_novelty_critic_never_calls_router() -> None:
    """NoveltyCritic queries the world-state graph; never the LLM."""
    router = AsyncMock()
    critic = NoveltyCritic(router)
    # No context seeded -> escalate-on-missing-context path is the
    # cheapest one to exercise without a fake repo. It still must
    # not hit the LLM.
    verdict = await critic.review({}, engagement_id=uuid4())
    assert verdict.verdict == Verdict.ESCALATE
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_structural_check_subclasses_have_no_system_prompt() -> None:
    """Pre-J1, NoveltyCritic / ScopeCritic / BudgetCritic carried
    pragma-no-cover ``_system_prompt`` / ``_format_subject`` overrides
    as inheritance artefacts. Phase J1 dropped them; the structural
    check subclasses should no longer declare those methods."""
    for cls in (NoveltyCritic, ScopeCritic, BudgetCritic):
        # The methods live on LLMCritic; structural-check subclasses
        # should NOT have their own definition.
        assert "_system_prompt" not in cls.__dict__, (
            f"{cls.__name__} still carries a (dead) ``_system_prompt`` override"
        )
        assert "_format_subject" not in cls.__dict__, (
            f"{cls.__name__} still carries a (dead) ``_format_subject`` override"
        )


def test_structural_checks_are_not_llm_critics() -> None:
    """A StructuralCheck must NOT be a subclass of LLMCritic — the
    whole point of the split is that the type system reflects the
    contract difference."""
    for cls in (NoveltyCritic, ScopeCritic, BudgetCritic):
        assert issubclass(cls, StructuralCheck), (
            f"{cls.__name__} should subclass StructuralCheck"
        )
        assert not issubclass(cls, LLMCritic), (
            f"{cls.__name__} must not subclass LLMCritic"
        )


# --------------------------------------------------------------------------- #
# LLMCritic subclasses DO call the LLM                                          #
# --------------------------------------------------------------------------- #


class _FakeResp:
    """Minimal LLMResponse stand-in for the critic JSON-parse path."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls",
    [PreExploitCritic, PreLateralCritic, ScopeBoundaryCritic, ReportCritic],
)
async def test_llm_critic_subclasses_invoke_router(cls: type[LLMCritic]) -> None:
    """Every LLMCritic subclass MUST go through router.complete for its
    verdict — that's the whole point of being an LLM critic."""
    router = AsyncMock()
    router.complete.return_value = _FakeResp(
        '{"verdict": "approve", "confidence": 0.9, '
        '"primary_concern": "ok", "issues": []}'
    )
    critic = cls(router)
    verdict = await critic.review({"any": "subject"}, engagement_id=uuid4())
    assert verdict.verdict == Verdict.APPROVE
    router.complete.assert_awaited_once()


def test_llm_critic_subclasses_are_llm_critics() -> None:
    """The classic LLM-backed critics still subclass LLMCritic (and
    hence the legacy ``BaseCritic`` alias)."""
    for cls in (PreExploitCritic, PreLateralCritic, ScopeBoundaryCritic, ReportCritic):
        assert issubclass(cls, LLMCritic)
        assert issubclass(cls, BaseCritic)


# --------------------------------------------------------------------------- #
# Council accepts heterogeneous members                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_council_accepts_mixed_member_types() -> None:
    """Council.members may contain both LLMCritic and StructuralCheck
    instances — the heterogeneous-list contract from Phase J1."""
    from agents.council import Council  # noqa: PLC0415 - local import

    router = AsyncMock()
    members: list[Any] = [BudgetCritic(router), PreExploitCritic(router)]
    council = Council(name="mixed", router=router, members=members)
    # Verify both types in members
    assert any(isinstance(m, StructuralCheck) for m in council.members)
    assert any(isinstance(m, LLMCritic) for m in council.members)


# --------------------------------------------------------------------------- #
# Critique type returned is unchanged                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_structural_check_returns_critique_type() -> None:
    """The verdict type must still be :class:`Critique` — the
    structural/LLM split doesn't change the output shape."""
    router = AsyncMock()
    critic = BudgetCritic(router)
    out = await critic.review({}, engagement_id=uuid4())
    assert isinstance(out, Critique)
