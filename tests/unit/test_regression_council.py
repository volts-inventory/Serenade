"""R2 #C — tests for StageRegressionCouncil and its three structural critics.

Covers each critic's veto/escalate/approve paths plus the council aggregation
rules. Uses an LLM router that fails if invoked: the council members are
deterministic and must not call the LLM in any of these tests.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from agents.critic import Verdict
from agents.regression import RegressionRequest, RegressionTriggerKind
from agents.regression_council import (
    BudgetCritic,
    NoveltyCritic,
    ScopeCritic,
    StageRegressionCouncil,
    build_regression_council,
)
from llm.router import LLMRequest, LLMResponse, LLMRouter
from orchestrator.models import (
    REGRESSION_BUDGET_MAX_DEFAULT,
    REGRESSION_TOKEN_RESERVE_PCT,
)
from orchestrator.settings import Settings
from worldstate.schema import Stage

# --------------------------------------------------------------------------- #
# Test fixtures                                                               #
# --------------------------------------------------------------------------- #


class _FailingBackend:
    """LLM backend whose .complete() raises. Used to assert critics are
    deterministic and never call the LLM."""

    name = "failing"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        raise AssertionError(
            f"LLM unexpectedly called (site={req.site}); deterministic critics must not call out"
        )


def _router() -> LLMRouter:
    return LLMRouter(_FailingBackend(), Settings(_env_file=None))  # type: ignore[call-arg]


class _FakeRepo:
    """In-memory stand-in for Neo4jRepo. Implements only the surfaces the
    council touches: get_nodes_by_kind."""

    def __init__(self, nodes_by_kind: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._nodes_by_kind = nodes_by_kind or {}

    async def get_nodes_by_kind(self, engagement_id: UUID, kind: str) -> list[dict[str, Any]]:
        return list(self._nodes_by_kind.get(kind, []))


class _FakeState:
    """Minimal stand-in for EngagementState — exposes only the attributes the
    BudgetCritic reads."""

    def __init__(self, *, regressions_used: int = 0, total_tokens: int = 0) -> None:
        self.regressions_used = regressions_used
        self.total_tokens = total_tokens


def _request(
    *,
    target_stage: Stage = Stage.RECON,
    trigger_kind: RegressionTriggerKind = RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
    delta: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    rationale: str = "test",
) -> RegressionRequest:
    return RegressionRequest(
        target_stage=target_stage,
        trigger_kind=trigger_kind,
        delta=delta if delta is not None else {"cidrs": ["10.0.5.0/24"]},
        evidence=evidence or [],
        rationale=rationale,
    )


def _engagement(
    *,
    cidrs: list[str] | None = None,
    domains: list[str] | None = None,
    urls: list[str] | None = None,
    excluded_cidrs: list[str] | None = None,
    excluded_domains: list[str] | None = None,
    token_budget: int = 1_000_000,
) -> dict[str, Any]:
    return {
        "scope": {
            "cidrs": cidrs if cidrs is not None else ["10.0.0.0/8"],
            "domains": domains or [],
            "urls": urls or [],
            "excluded_cidrs": excluded_cidrs or [],
            "excluded_domains": excluded_domains or [],
        },
        "token_budget": token_budget,
    }


# --------------------------------------------------------------------------- #
# NoveltyCritic                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_novelty_vetoes_when_delta_is_empty():
    critic = NoveltyCritic(_router())
    critic._request = _request(delta={})
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "empty_delta" in c.issues


@pytest.mark.asyncio
async def test_novelty_vetoes_when_all_entities_already_in_graph():
    eid = uuid4()
    existing_host_id = str(uuid4())
    repo = _FakeRepo(nodes_by_kind={"Host": [{"id": existing_host_id, "ip": "10.0.1.1"}]})
    critic = NoveltyCritic(_router())
    critic._request = _request(
        delta={"host_ids": [existing_host_id]},
        trigger_kind=RegressionTriggerKind.NEW_SERVICE_CLASS_DISCOVERED,
    )
    critic._repo = repo
    critic._engagement_id = eid
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "delta_already_known" in c.issues


@pytest.mark.asyncio
async def test_novelty_approves_when_at_least_one_novel_entity():
    eid = uuid4()
    repo = _FakeRepo(nodes_by_kind={"Host": [{"id": str(uuid4()), "ip": "10.0.1.1"}]})
    critic = NoveltyCritic(_router())
    critic._request = _request(
        delta={"host_ids": [str(uuid4())]},  # this id is novel
        trigger_kind=RegressionTriggerKind.NEW_SERVICE_CLASS_DISCOVERED,
    )
    critic._repo = repo
    critic._engagement_id = eid
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert c.confidence == 1.0


@pytest.mark.asyncio
async def test_novelty_vetoes_on_replay_loop():
    """Identical-fingerprint replay → veto.

    The replay-loop guard is now keyed on the full
    ``(target_stage, trigger_kind, delta_fingerprint)`` triple. A
    matching triple in ``_applied_keys`` (read from Postgres at
    evaluate-time) is the only fingerprint-equal regression and counts
    as a replay; genuinely-novel deltas with the same ``(stage,
    trigger)`` sail through (see ``test_novelty_approves_distinct_cidr_after_prior_same_trigger``).
    """
    from agents.regression_council import fingerprint_delta  # local import

    delta = {"cidrs": ["10.0.99.0/24"]}
    fp = fingerprint_delta(delta)
    critic = NoveltyCritic(_router())
    critic._request = _request(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta=delta,
    )
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    critic._current_fingerprint = fp
    critic._applied_keys = {("recon", "new_subnet_discovered", fp)}
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "replay_loop" in c.issues


@pytest.mark.asyncio
async def test_novelty_approves_distinct_cidr_after_prior_same_trigger():
    """Different CIDR + same (stage, trigger) is NOT a replay.

    The canonical "three different uncovered /24s on one engagement"
    scenario. Lateral discovered ``10.0.5.0/24`` on day 3 and a council
    approved the recon re-sweep. On day 7 lateral discovers
    ``10.0.6.0/24`` and the pair-keyed guard would have wrongly vetoed
    that as a replay. With the fingerprint key, the two deltas hash
    differently and the second request is evaluated on its own merits.
    """
    from agents.regression_council import fingerprint_delta

    prior_fp = fingerprint_delta({"cidrs": ["10.0.5.0/24"]})
    new_delta = {"cidrs": ["10.0.6.0/24"]}
    new_fp = fingerprint_delta(new_delta)
    assert prior_fp != new_fp  # invariant — different CIDRs hash differently

    critic = NoveltyCritic(_router())
    critic._request = _request(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta=new_delta,
    )
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    critic._current_fingerprint = new_fp
    critic._applied_keys = {("recon", "new_subnet_discovered", prior_fp)}
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert "replay_loop" not in c.issues


@pytest.mark.asyncio
async def test_novelty_escalates_on_regression_storm():
    """>= STORM_THRESHOLD prior approvals on the same (stage, trigger)
    pair in the last STORM_WINDOW_MINUTES → ESCALATE, not VETO.

    The storm guard fails toward operator review (ESCALATE) so a
    genuinely-novel multi-segment engagement isn't silently shut down;
    a human decides whether the producer agent is misbehaving.
    """
    from agents.regression_council import (
        REGRESSION_STORM_THRESHOLD,
        fingerprint_delta,
    )

    delta = {"cidrs": ["10.0.42.0/24"]}
    critic = NoveltyCritic(_router())
    critic._request = _request(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta=delta,
    )
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    critic._current_fingerprint = fingerprint_delta(delta)
    critic._applied_keys = set()  # never replayed; the storm threshold is what trips
    critic._recent_pair_count = REGRESSION_STORM_THRESHOLD
    c = await critic.review({})
    assert c.verdict == Verdict.ESCALATE
    assert any("regression_storm" in i for i in c.issues)


@pytest.mark.asyncio
async def test_novelty_does_not_escalate_below_storm_threshold():
    """Below-threshold boundary: count = THRESHOLD - 1 proceeds to the
    graph-existence check instead of escalating. Guards against an
    off-by-one in the storm guard."""
    from agents.regression_council import (
        REGRESSION_STORM_THRESHOLD,
        fingerprint_delta,
    )

    delta = {"cidrs": ["10.0.42.0/24"]}
    critic = NoveltyCritic(_router())
    critic._request = _request(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta=delta,
    )
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    critic._current_fingerprint = fingerprint_delta(delta)
    critic._applied_keys = set()
    critic._recent_pair_count = REGRESSION_STORM_THRESHOLD - 1
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert not any("regression_storm" in i for i in c.issues)


@pytest.mark.asyncio
async def test_novelty_approves_cidr_only_delta_at_full_confidence():
    """A CIDR-only delta is the canonical novelty signal."""
    critic = NoveltyCritic(_router())
    critic._request = _request(delta={"cidrs": ["10.0.5.0/24"]})
    critic._repo = _FakeRepo()
    critic._engagement_id = uuid4()
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert c.confidence == 1.0


@pytest.mark.asyncio
async def test_novelty_partial_match_lowers_confidence():
    eid = uuid4()
    existing_id = str(uuid4())
    novel_id = str(uuid4())
    repo = _FakeRepo(nodes_by_kind={"Host": [{"id": existing_id, "ip": "10.0.1.1"}]})
    critic = NoveltyCritic(_router())
    critic._request = _request(
        delta={"host_ids": [existing_id, novel_id]},
        trigger_kind=RegressionTriggerKind.NEW_SERVICE_CLASS_DISCOVERED,
    )
    critic._repo = repo
    critic._engagement_id = eid
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert c.confidence == 0.6
    assert "partial_novelty" in c.issues


# --------------------------------------------------------------------------- #
# ScopeCritic                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scope_vetoes_out_of_scope_cidr():
    critic = ScopeCritic(_router())
    critic._request = _request(delta={"cidrs": ["8.8.8.0/24"]})  # Google DNS — not customer
    critic._scope = {"cidrs": ["10.0.0.0/8"]}
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "out_of_scope_ip" in c.issues


@pytest.mark.asyncio
async def test_scope_vetoes_excluded_cidr_overlap():
    critic = ScopeCritic(_router())
    critic._request = _request(delta={"cidrs": ["10.0.5.0/24"]})
    critic._scope = {
        "cidrs": ["10.0.0.0/8"],
        "excluded_cidrs": ["10.0.5.0/25"],
    }
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "excluded_ip" in c.issues


@pytest.mark.asyncio
async def test_scope_escalates_on_wildcard_ambiguity():
    critic = ScopeCritic(_router())
    critic._request = _request(
        delta={"domains": ["cdn.example.com"]},
        trigger_kind=RegressionTriggerKind.MISCONFIGURATION_OPENS_PATH,
    )
    critic._scope = {"domains": ["*.example.com"]}
    c = await critic.review({})
    assert c.verdict == Verdict.ESCALATE
    assert "wildcard_ambiguous" in c.issues


@pytest.mark.asyncio
async def test_scope_approves_when_unambiguously_in_scope():
    critic = ScopeCritic(_router())
    critic._request = _request(
        delta={"cidrs": ["10.0.5.0/24"], "domains": ["api.target.com"]},
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
    )
    critic._scope = {
        "cidrs": ["10.0.0.0/8"],
        "domains": ["target.com"],
    }
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert c.confidence == 1.0


@pytest.mark.asyncio
async def test_scope_vetoes_excluded_domain():
    critic = ScopeCritic(_router())
    critic._request = _request(delta={"domains": ["admin.target.com"]})
    critic._scope = {
        "domains": ["target.com"],
        "excluded_domains": ["admin.target.com"],
    }
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "excluded_domain" in c.issues


# --------------------------------------------------------------------------- #
# BudgetCritic                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_budget_vetoes_when_regressions_used_at_cap():
    """Cap is now per-engagement; the constant is just a fallback.

    Setting an explicit ``_regression_budget_max`` mirrors the
    engagement-dict pathway used in production. The rationale string
    intentionally changed from ``regression_count_cap`` to
    ``runaway_loop_circuit_breaker`` so audit-log readers understand
    this is NOT a normal stop condition.
    """
    critic = BudgetCritic(_router())
    critic._regressions_used = 8
    critic._regression_budget_max = 8
    critic._token_budget = 1_000_000
    critic._tokens_used = 0
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "runaway_loop_circuit_breaker" in c.issues


@pytest.mark.asyncio
async def test_budget_approves_when_cap_is_zero_meaning_unlimited():
    """``regression_budget_max == 0`` means unlimited (the default).

    Matches the existing ``token_budget == 0`` semantics. No matter
    how high ``regressions_used`` climbs, the BudgetCritic approves on
    the regression-count axis; the token-reserve check still applies.
    """
    critic = BudgetCritic(_router())
    critic._regressions_used = 9_999
    critic._regression_budget_max = 0
    critic._token_budget = 0
    critic._tokens_used = 0
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE


@pytest.mark.asyncio
async def test_budget_vetoes_when_token_reserve_exhausted():
    critic = BudgetCritic(_router())
    critic._regressions_used = 0
    critic._token_budget = 1_000_000
    # Burn more than (1 - 0.15) = 85% → triggers reserve-exhausted veto.
    critic._tokens_used = int(1_000_000 * (1.0 - REGRESSION_TOKEN_RESERVE_PCT)) + 1
    c = await critic.review({})
    assert c.verdict == Verdict.VETO
    assert "token_reserve_exhausted" in c.issues


@pytest.mark.asyncio
async def test_budget_approves_when_budget_healthy():
    critic = BudgetCritic(_router())
    critic._regressions_used = 1
    critic._token_budget = 1_000_000
    critic._tokens_used = 100_000  # only 10% burned
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE
    assert c.confidence == 1.0


@pytest.mark.asyncio
async def test_budget_approves_with_zero_token_budget():
    """token_budget == 0 means "no cap" per Phase 45 — skip reserve check."""
    critic = BudgetCritic(_router())
    critic._regressions_used = 0
    critic._token_budget = 0
    critic._tokens_used = 999_999_999
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE


@pytest.mark.asyncio
async def test_budget_critic_does_not_call_llm():
    """The BudgetCritic's router has a failing backend — if review() invokes
    the LLM, this test fails loudly. Combined with the test above it asserts
    the budget path is fully arithmetic."""

    class _ExplodingBackend:
        name = "exploding"

        async def complete(self, req: LLMRequest) -> LLMResponse:
            raise AssertionError("BudgetCritic must not call the LLM")

    router = LLMRouter(_ExplodingBackend(), Settings(_env_file=None))  # type: ignore[call-arg]
    critic = BudgetCritic(router)
    critic._regressions_used = 0
    critic._token_budget = 1_000_000
    critic._tokens_used = 0
    c = await critic.review({})
    assert c.verdict == Verdict.APPROVE


# --------------------------------------------------------------------------- #
# StageRegressionCouncil aggregation                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_council_vetoes_when_any_member_vetoes():
    """Out-of-scope CIDR → ScopeCritic vetoes → council vetoes."""
    council = build_regression_council(_router())
    request = _request(delta={"cidrs": ["8.8.8.0/24"]})
    state = _FakeState()
    engagement = _engagement(cidrs=["10.0.0.0/8"])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        _FakeRepo(),
        uuid4(),  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.VETO
    assert any(
        "out_of_scope" in c.primary_concern or "outside" in c.primary_concern
        for c in verdict.member_critiques
    )


@pytest.mark.asyncio
async def test_council_escalates_when_any_escalates_no_veto():
    """Wildcard ambiguity → ScopeCritic escalates; novelty + budget approve."""
    council = build_regression_council(_router())
    request = _request(
        delta={"domains": ["cdn.example.com"]},
        trigger_kind=RegressionTriggerKind.MISCONFIGURATION_OPENS_PATH,
    )
    state = _FakeState(regressions_used=0, total_tokens=0)
    engagement = _engagement(domains=["*.example.com"], cidrs=[])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        _FakeRepo(),
        uuid4(),  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.ESCALATE


@pytest.mark.asyncio
async def test_council_approves_when_all_approve_at_min_confidence():
    """All approve, novelty is partial (0.6) → council approves with conf=0.6."""
    eid = uuid4()
    existing = str(uuid4())
    novel = str(uuid4())
    repo = _FakeRepo(nodes_by_kind={"Host": [{"id": existing, "ip": "10.0.1.1"}]})
    council = build_regression_council(_router())
    request = _request(
        target_stage=Stage.ENUMERATION,
        trigger_kind=RegressionTriggerKind.NEW_SERVICE_CLASS_DISCOVERED,
        delta={"host_ids": [existing, novel]},
    )
    state = _FakeState(regressions_used=0, total_tokens=100)
    engagement = _engagement(cidrs=["10.0.0.0/8"])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        repo,
        eid,  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.APPROVE
    # Min confidence — Novelty partial = 0.6, Scope no-checkable-entities = 0.6, Budget = 1.0
    assert verdict.confidence == 0.6


@pytest.mark.asyncio
async def test_council_approves_with_full_confidence_when_all_clean():
    council = build_regression_council(_router())
    request = _request(delta={"cidrs": ["10.0.5.0/24"]})
    state = _FakeState()
    engagement = _engagement(cidrs=["10.0.0.0/8"])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        _FakeRepo(),
        uuid4(),  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.APPROVE
    assert verdict.confidence == 1.0


@pytest.mark.asyncio
async def test_council_veto_beats_escalate():
    """If one member vetoes and another escalates, veto wins."""
    council = build_regression_council(_router())
    # Out-of-scope CIDR (veto) + wildcard domain (escalate). Veto takes precedence.
    request = _request(
        delta={"cidrs": ["8.8.8.0/24"], "domains": ["cdn.example.com"]},
        trigger_kind=RegressionTriggerKind.MISCONFIGURATION_OPENS_PATH,
    )
    state = _FakeState()
    engagement = _engagement(cidrs=["10.0.0.0/8"], domains=["*.example.com"])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        _FakeRepo(),
        uuid4(),  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.VETO


@pytest.mark.asyncio
async def test_council_has_three_named_members():
    council = build_regression_council(_router())
    assert isinstance(council, StageRegressionCouncil)
    names = {m.name for m in council.members}
    assert names == {"novelty_critic", "scope_critic", "budget_critic"}


@pytest.mark.asyncio
async def test_council_vetoes_when_budget_exhausted():
    council = build_regression_council(_router())
    request = _request(delta={"cidrs": ["10.0.5.0/24"]})
    state = _FakeState(regressions_used=REGRESSION_BUDGET_MAX_DEFAULT)
    engagement = _engagement(cidrs=["10.0.0.0/8"])
    verdict = await council.evaluate(
        request,
        state,
        engagement,
        _FakeRepo(),
        uuid4(),  # type: ignore[arg-type]
    )
    assert verdict.decision == Verdict.VETO
    assert "budget" in verdict.rationale.lower() or "cap" in verdict.rationale.lower()
