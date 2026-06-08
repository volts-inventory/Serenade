"""R3 #J1 Bug 4 (D7) — the dead synthesis LLM call path is gone.

Pre-fix: ``StageRegressionCouncil._synthesise`` would fire an LLM call
when critic verdicts didn't match the structural rules (any-veto /
any-escalate / all-approve). Reachable only if a member emitted
``REVISE``, which none of the three structural members (Novelty / Scope
/ Budget) does. Cost variance for unreachable code.

Post-fix: the LLM call is removed. If verdicts ever DON'T match the
structural rules (e.g. a subclass regression that reintroduces REVISE),
the council emits a hard ESCALATE so an operator sees the impossible
case.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agents.critic import Critique, Verdict
from agents.regression import RegressionRequest, RegressionTriggerKind
from agents.regression_council import (
    BudgetCritic,
    NoveltyCritic,
    ScopeCritic,
    StageRegressionCouncil,
)
from worldstate.schema import Stage


@pytest.mark.asyncio
async def test_no_synthesise_method_on_council() -> None:
    """The synth fallback method itself was removed in Phase J1."""
    router = AsyncMock()
    council = StageRegressionCouncil(router)
    assert not hasattr(council, "_synthesise"), (
        "_synthesise was the LLM-backed fallback; J1 removed it. "
        "If something needs to reintroduce it, write a new test first."
    )


@pytest.mark.asyncio
async def test_unreachable_verdict_combo_escalates_without_llm() -> None:
    """If the (impossible) mixed-verdict path is hit by a future critic
    regression, the council emits a hard ESCALATE with the
    ``unreachable mixed-verdict combination`` rationale — no LLM."""
    router = AsyncMock()
    council = StageRegressionCouncil(router)
    # Stub each member's review() to emit a verdict shape that doesn't
    # match any of the structural rules: one APPROVE, one REVISE,
    # one APPROVE. No veto, no escalate → falls through to the
    # unreachable branch.
    novelty = next(m for m in council.members if isinstance(m, NoveltyCritic))
    scope_critic = next(m for m in council.members if isinstance(m, ScopeCritic))
    budget = next(m for m in council.members if isinstance(m, BudgetCritic))

    async def _approve(*a, **k):  # type: ignore[no-untyped-def]
        return Critique(
            verdict=Verdict.APPROVE,
            confidence=0.9,
            primary_concern="ok",
            issues=[],
            critic_name="x",
        )

    async def _revise(*a, **k):  # type: ignore[no-untyped-def]
        return Critique(
            verdict=Verdict.REVISE,
            confidence=0.5,
            primary_concern="needs revision",
            issues=[],
            critic_name="x",
        )

    novelty.review = _approve  # type: ignore[method-assign]
    scope_critic.review = _revise  # type: ignore[method-assign]
    budget.review = _approve  # type: ignore[method-assign]

    request = RegressionRequest(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta={"cidrs": ["10.0.0.0/24"]},
        rationale="test",
    )
    # Minimal stand-ins for the helper args.
    state = type("S", (), {"regressions_used": 0, "total_tokens": 0})()
    repo = AsyncMock()
    verdict = await council.evaluate(
        request, state, {}, repo, uuid4()
    )
    assert verdict.decision == Verdict.ESCALATE
    assert "unreachable" in verdict.rationale.lower()
    # Crucially: no LLM call was made for the synth path.
    router.complete.assert_not_called()


@pytest.mark.asyncio
async def test_structural_members_never_emit_revise() -> None:
    """The contract underwriting the dead-code removal: none of the
    three structural critics emit REVISE under any input shape we can
    construct here. This is a smoke test, not exhaustive coverage —
    pin the documented absence."""
    router = AsyncMock()

    # NoveltyCritic with empty context: ESCALATE-missing-context. With
    # an empty delta: VETO. With CIDR-only delta + real repo: ESCALATE
    # on graph failure. None of these emit REVISE.
    nc = NoveltyCritic(router)
    out = await nc.review({}, engagement_id=uuid4())
    assert out.verdict != Verdict.REVISE

    # ScopeCritic with no request seeded: ESCALATE-missing-context.
    sc = ScopeCritic(router)
    out = await sc.review({}, engagement_id=uuid4())
    assert out.verdict != Verdict.REVISE

    # BudgetCritic: APPROVE / VETO only. Per-engagement cap is now
    # the source of truth; 0 = unlimited (the default for new
    # engagements). Set an explicit cap here to exercise the VETO path.
    bc = BudgetCritic(router)
    bc._regressions_used = 0
    bc._regression_budget_max = 5
    bc._token_budget = 1000
    bc._tokens_used = 0
    out = await bc.review({}, engagement_id=uuid4())
    assert out.verdict != Verdict.REVISE

    bc._regressions_used = 99  # over cap
    out = await bc.review({}, engagement_id=uuid4())
    assert out.verdict == Verdict.VETO
