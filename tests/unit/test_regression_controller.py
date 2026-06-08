"""R2 #D — stage-controller regression wiring.

Covers the ``StepDecision.REGRESS_TO`` branch in ``_step_agent``:

- Council APPROVE → state advances to ``target_stage``, counter increments,
  delta + origin stage stashed, audit row + event bus fire.
- Council VETO → state stays put, audit row stamped with veto.
- Council ESCALATE → state stays put, ``interrupt()`` raised.
- Delta clearing on origin-stage re-entry.
- ``_applied_pairs()`` returns the right tuples from a populated table.
- Event-bus + audit-log emissions ordered correctly.

The tests construct a tiny ``_FakeAgent`` whose ``step()`` returns a
scripted ``StepResult``. The controller's node closure is then driven
synchronously through ``_step_agent(agent, repo=..., council=...)``.

Mocking strategy:
- ``record_regression`` / ``update_regression_with_verdict`` / ``audit``
  are patched on the ``stage_controller`` module to record calls in
  order.
- ``get_bus`` is patched to return a ``MagicMock`` whose ``publish`` is an
  ``AsyncMock``.
- The council is a stand-in object whose ``evaluate()`` returns a
  pre-baked ``CouncilVerdict``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from agents.base import StageContext, StepDecision, StepResult
from agents.council import CouncilVerdict
from agents.critic import Critique, Verdict
from agents.regression import RegressionRequest, RegressionTriggerKind
from orchestrator import stage_controller
from orchestrator.stage_controller import (
    EngagementState,
    _handle_regress_to,
    _step_agent,
)
from worldstate.schema import Stage

# --------------------------------------------------------------------------- #
# Shared fakes                                                                #
# --------------------------------------------------------------------------- #


class _FakeAgent:
    """Minimal BaseAgent stand-in. Returns whatever StepResult the test
    scripts in. ``stage`` is configurable so the controller can route on it.
    """

    def __init__(self, stage: Stage, result: StepResult) -> None:
        self.stage = stage
        self._result = result
        self.seen_ctx: StageContext | None = None

    async def step(self, ctx: StageContext) -> StepResult:
        self.seen_ctx = ctx
        return self._result


class _FakeCouncil:
    """Stand-in for StageRegressionCouncil. Returns a scripted
    ``CouncilVerdict`` and records the call args."""

    def __init__(self, verdict: CouncilVerdict) -> None:
        self._verdict = verdict
        self.calls: list[dict[str, Any]] = []

    async def evaluate(
        self,
        request: RegressionRequest,
        engagement_state: Any,
        engagement: dict[str, Any],
        repo: Any,
        engagement_id: UUID,
    ) -> CouncilVerdict:
        # Snapshot the BudgetCritic-relevant fields at decision time so a
        # test can assert ``regressions_used`` was observed at the
        # pre-increment value.
        self.calls.append(
            {
                "request": request,
                "regressions_used_at_decision": getattr(
                    engagement_state, "regressions_used", None
                ),
                "engagement_id": engagement_id,
            }
        )
        return self._verdict


def _verdict(decision: Verdict, *, confidence: float = 0.9, rationale: str = "") -> CouncilVerdict:
    return CouncilVerdict(
        decision=decision,
        confidence=confidence,
        rationale=rationale or f"{decision.value} verdict",
        member_critiques=[
            Critique(
                verdict=decision,
                confidence=confidence,
                primary_concern=rationale or f"{decision.value} verdict",
                issues=[],
                critic_name="test",
            )
        ],
        dissent_summary=None,
        total_cost_usd=0.0,
    )


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


def _state(*, current_stage: Stage = Stage.LATERAL, **kwargs: Any) -> EngagementState:
    base: dict[str, Any] = dict(
        engagement_id=uuid4(),
        engagement={
            "token_budget": 1_000_000,
            "scope": {"cidrs": ["10.0.0.0/8"]},
        },
        current_stage=current_stage,
    )
    base.update(kwargs)
    return EngagementState(**base)


def _regress_to_result(req: RegressionRequest) -> StepResult:
    return StepResult(
        decision=StepDecision.REGRESS_TO,
        tokens_used=100,
        cost_usd=0.0,
        tool_runs=0,
        nodes_written=0,
        regression_request=req,
    )


# --------------------------------------------------------------------------- #
# Mocking helpers                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def regression_mocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the ledger/bus surfaces the controller talks to and record
    every call so tests can assert ordering and arguments."""
    recorded_rows: list[dict[str, Any]] = []
    row_id_seq = uuid4()

    async def _record_regression(
        *,
        engagement_id: UUID,
        target_stage: str,
        trigger_kind: str,
        delta: dict[str, Any],
        evidence: list[str],
        rationale: str,
        delta_fingerprint: str | None = None,
    ) -> UUID:
        recorded_rows.append(
            {
                "engagement_id": engagement_id,
                "target_stage": target_stage,
                "trigger_kind": trigger_kind,
                "delta": delta,
                "evidence": evidence,
                "rationale": rationale,
                "delta_fingerprint": delta_fingerprint,
            }
        )
        return row_id_seq

    updates: list[dict[str, Any]] = []

    async def _update_regression_with_verdict(
        *,
        row_id: UUID,
        council_decision: str,
        council_rationale: str,
        applied_at: Any,
    ) -> None:
        updates.append(
            {
                "row_id": row_id,
                "council_decision": council_decision,
                "council_rationale": council_rationale,
                "applied_at": applied_at,
            }
        )

    audits: list[dict[str, Any]] = []

    async def _audit(
        *,
        engagement_id: UUID | None,
        actor: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        audits.append(
            {
                "engagement_id": engagement_id,
                "actor": actor,
                "action": action,
                "payload": payload,
            }
        )

    bus = MagicMock()
    bus.publish = AsyncMock(return_value="")

    monkeypatch.setattr(stage_controller, "record_regression", _record_regression)
    monkeypatch.setattr(
        stage_controller, "update_regression_with_verdict", _update_regression_with_verdict
    )
    monkeypatch.setattr(stage_controller, "audit", _audit)
    monkeypatch.setattr(stage_controller, "get_bus", lambda: bus)

    return {
        "rows": recorded_rows,
        "updates": updates,
        "audits": audits,
        "bus": bus,
        "row_id": row_id_seq,
    }


# --------------------------------------------------------------------------- #
# 1. REGRESS_TO with no council wired                                          #
# --------------------------------------------------------------------------- #


async def test_regress_to_without_council_does_not_advance_state(
    regression_mocks: dict[str, Any],
) -> None:
    """When the AgentBundle is built without a council (e.g. unit-test
    constructions), REGRESS_TO must not write a backwards transition.
    The row is stamped with ``council_decision="veto"`` /
    ``council_rationale="council_not_wired"`` so the audit trail is
    explicit."""
    state = _state(current_stage=Stage.LATERAL)
    initial_stage = state.current_stage

    req = _request(target_stage=Stage.RECON)
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    node = _step_agent(agent, repo=None, regression_council=None)

    await node(state)

    # Stage unchanged.
    assert state.current_stage == initial_stage
    assert state.regression_delta is None
    assert state.regressions_used == 0
    # Row written, then patched as veto/not_wired.
    assert len(regression_mocks["rows"]) == 1
    assert len(regression_mocks["updates"]) == 1
    assert regression_mocks["updates"][0]["council_decision"] == "veto"
    assert regression_mocks["updates"][0]["council_rationale"] == "council_not_wired"
    assert regression_mocks["updates"][0]["applied_at"] is None


# --------------------------------------------------------------------------- #
# 2. APPROVE branch                                                            #
# --------------------------------------------------------------------------- #


async def test_council_approve_advances_state_and_stashes_delta(
    regression_mocks: dict[str, Any],
) -> None:
    """The canonical happy path. APPROVE → state.current_stage =
    target_stage, regressions_used++, regression_delta + origin_stage set,
    row stamped with applied_at, event bus + audit log fire in order."""
    state = _state(current_stage=Stage.LATERAL, regressions_used=1)
    council = _FakeCouncil(_verdict(Verdict.APPROVE, rationale="all clean"))
    repo = MagicMock()

    req = _request(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta={"cidrs": ["10.0.5.0/24"]},
    )
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    node = _step_agent(agent, repo=repo, regression_council=council)

    await node(state)

    # Stage flipped backwards to the target.
    assert state.current_stage == Stage.RECON
    # Delta stashed for the re-entered sub-pipeline; origin stamped so we
    # know when to clear.
    assert state.regression_delta == {"cidrs": ["10.0.5.0/24"]}
    assert state.regression_origin_stage == Stage.LATERAL.value
    # Counter incremented exactly once.
    assert state.regressions_used == 2

    # The BudgetCritic saw the PRE-increment count.
    assert council.calls[0]["regressions_used_at_decision"] == 1

    # Row written then updated with approve + applied_at.
    assert len(regression_mocks["updates"]) == 1
    upd = regression_mocks["updates"][0]
    assert upd["council_decision"] == "approve"
    assert upd["applied_at"] is not None

    # Bus published requested AND applied.
    bus_kinds = [c.kwargs["kind"] for c in regression_mocks["bus"].publish.await_args_list]
    assert "regression.requested" in bus_kinds
    assert "regression.applied" in bus_kinds

    # Audit log entries: requested, council_decided, applied.
    actions = [a["action"] for a in regression_mocks["audits"]]
    assert actions == [
        "regression.requested",
        "regression.council_decided",
        "regression.applied",
    ]


async def test_council_approve_threads_delta_into_stage_context(
    regression_mocks: dict[str, Any],
) -> None:
    """After APPROVE writes ``state.regression_delta``, the NEXT
    ``_step_agent`` invocation (the re-entered sub-pipeline stage)
    must see the delta on ``ctx.regression_delta``."""
    state = _state(current_stage=Stage.LATERAL)
    council = _FakeCouncil(_verdict(Verdict.APPROVE))
    repo = MagicMock()

    req = _request(delta={"cidrs": ["10.0.7.0/24"]})
    lateral = _FakeAgent(
        Stage.LATERAL, _regress_to_result(req)
    )
    await _step_agent(lateral, repo=repo, regression_council=council)(state)

    # State should now be RECON with the delta stashed.
    assert state.current_stage == Stage.RECON
    assert state.regression_delta == {"cidrs": ["10.0.7.0/24"]}

    # The next agent's StageContext must carry the delta.
    recon = _FakeAgent(
        Stage.RECON,
        StepResult(
            decision=StepDecision.CONTINUE,
            tokens_used=10,
            cost_usd=0.0,
            tool_runs=0,
            nodes_written=0,
        ),
    )
    await _step_agent(recon, repo=repo, regression_council=council)(state)

    assert recon.seen_ctx is not None
    assert recon.seen_ctx.regression_delta == {"cidrs": ["10.0.7.0/24"]}
    assert recon.seen_ctx.regression_origin_stage == Stage.LATERAL.value


# --------------------------------------------------------------------------- #
# 3. VETO branch                                                               #
# --------------------------------------------------------------------------- #


async def test_council_veto_does_not_advance(
    regression_mocks: dict[str, Any],
) -> None:
    """VETO leaves the engagement where it was; row records the verdict
    without an ``applied_at`` stamp."""
    state = _state(current_stage=Stage.LATERAL, regressions_used=2)
    council = _FakeCouncil(_verdict(Verdict.VETO, rationale="out of scope"))
    repo = MagicMock()

    req = _request(target_stage=Stage.RECON)
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    node = _step_agent(agent, repo=repo, regression_council=council)

    await node(state)

    # Stage unchanged.
    assert state.current_stage == Stage.LATERAL
    # No delta written, no counter bump.
    assert state.regression_delta is None
    assert state.regression_origin_stage is None
    assert state.regressions_used == 2

    upd = regression_mocks["updates"][0]
    assert upd["council_decision"] == "veto"
    assert upd["council_rationale"] == "out of scope"
    assert upd["applied_at"] is None

    # Bus: requested + vetoed (not applied).
    bus_kinds = [c.kwargs["kind"] for c in regression_mocks["bus"].publish.await_args_list]
    assert "regression.requested" in bus_kinds
    assert "regression.vetoed" in bus_kinds
    assert "regression.applied" not in bus_kinds


# --------------------------------------------------------------------------- #
# 4. ESCALATE branch                                                           #
# --------------------------------------------------------------------------- #


async def test_repeated_veto_forces_advance_to_break_loop(
    regression_mocks: dict[str, Any],
) -> None:
    """ws-02 regression: a council that VETOes the SAME regression every
    step (because the structural trigger persists and the agent re-emits
    the identical request) must NOT loop forever. After
    ``_REGRESSION_VETO_STAGE_CAP`` consecutive vetoes from the same stage
    the controller forces an ADVANCE so the engagement makes forward
    progress. ``regressions_used`` never advances on VETO, so without this
    bound no budget cap could ever fire."""
    cap = stage_controller._REGRESSION_VETO_STAGE_CAP
    state = _state(current_stage=Stage.LATERAL)
    council = _FakeCouncil(_verdict(Verdict.VETO, rationale="replay_loop"))
    repo = MagicMock()

    req = _request(target_stage=Stage.RECON)

    forced_at: int | None = None
    # Drive the node repeatedly, exactly as the LangGraph router would when
    # current_stage keeps coming back to LATERAL after each VETO.
    for step in range(1, cap + 5):
        # Fresh REGRESS_TO result each step (the agent re-emits identically).
        agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
        node = _step_agent(agent, repo=repo, regression_council=council)
        await node(state)
        if state.current_stage != Stage.LATERAL:
            forced_at = step
            break

    # The loop was broken within the bound — not left to run forever.
    assert forced_at is not None, "VETO loop never broke — unbounded"
    assert forced_at <= cap
    # Forced forward, never APPROVE'd, so the regression budget is untouched.
    assert state.regressions_used == 0
    # Forward progress: ENUMERATION is _next_stage(LATERAL)'s direction
    # (lateral -> loot in canonical order).
    assert state.current_stage == stage_controller._next_stage(Stage.LATERAL)
    # The loop-breaker audited its forced advance.
    actions = [a["action"] for a in regression_mocks["audits"]]
    assert "regression.veto_loop_forced_advance" in actions


async def test_approved_regression_resets_veto_loop_counter(
    regression_mocks: dict[str, Any],
) -> None:
    """ws-02: a couple of vetoes followed by an APPROVE must reset the
    per-stage consecutive-veto tally, so an unrelated future regression
    from the same stage isn't penalised by stale veto counts."""
    state = _state(current_stage=Stage.LATERAL)
    repo = MagicMock()
    req = _request(target_stage=Stage.RECON)

    veto_council = _FakeCouncil(_verdict(Verdict.VETO, rationale="nope"))
    for _ in range(stage_controller._REGRESSION_VETO_STAGE_CAP - 1):
        agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
        await _step_agent(agent, repo=repo, regression_council=veto_council)(state)
    assert state.regression_vetoes_by_stage.get(Stage.LATERAL.value, 0) == (
        stage_controller._REGRESSION_VETO_STAGE_CAP - 1
    )

    # Now an APPROVE — the tally for LATERAL must reset to zero.
    approve_council = _FakeCouncil(_verdict(Verdict.APPROVE))
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    await _step_agent(agent, repo=repo, regression_council=approve_council)(state)
    assert state.regression_vetoes_by_stage.get(Stage.LATERAL.value, 0) == 0
    assert state.current_stage == Stage.RECON


async def test_council_escalate_pauses_graph(
    regression_mocks: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ESCALATE leaves the stage put AND raises ``interrupt()`` so the
    operator decides. LangGraph translates ``interrupt()`` into a
    ``GraphInterrupt`` exception. We monkeypatch ``interrupt`` to a
    sentinel exception so the assertion doesn't depend on running inside
    a LangGraph runnable context."""

    class _TestInterruptError(Exception):
        pass

    def _fake_interrupt(payload: Any) -> Any:
        raise _TestInterruptError(payload)

    monkeypatch.setattr(stage_controller, "interrupt", _fake_interrupt)

    state = _state(current_stage=Stage.LATERAL)
    council = _FakeCouncil(_verdict(Verdict.ESCALATE, rationale="wildcard ambiguous"))
    repo = MagicMock()

    req = _request(target_stage=Stage.RECON)
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    node = _step_agent(agent, repo=repo, regression_council=council)

    with pytest.raises(_TestInterruptError) as exc_info:
        await node(state)

    # The interrupt payload identifies the escalation kind so the operator
    # UI can distinguish a regression-escalate from a tool-veto-escalate.
    payload = exc_info.value.args[0]
    assert payload["kind"] == "regression_council_escalated"
    assert payload["target_stage"] == Stage.RECON.value

    # Stage unchanged because the council escalated.
    assert state.current_stage == Stage.LATERAL
    assert state.regression_delta is None

    # Row stamped as escalate (no applied_at).
    upd = regression_mocks["updates"][0]
    assert upd["council_decision"] == "escalate"
    assert upd["applied_at"] is None

    bus_kinds = [c.kwargs["kind"] for c in regression_mocks["bus"].publish.await_args_list]
    assert "regression.escalated" in bus_kinds


# --------------------------------------------------------------------------- #
# 5. Delta clearing on origin-stage re-entry                                   #
# --------------------------------------------------------------------------- #


async def test_delta_clears_when_origin_stage_re_enters(
    regression_mocks: dict[str, Any],
) -> None:
    """Once the sub-pipeline loops back to the stage that emitted the
    REGRESS_TO, the controller clears the delta + origin stage so the
    engagement resumes its normal forward flow."""
    state = _state(
        current_stage=Stage.LATERAL,
        regression_delta={"cidrs": ["10.0.5.0/24"]},
        regression_origin_stage=Stage.LATERAL.value,
    )

    # An ordinary CONTINUE step on the LATERAL agent should clear the
    # delta at the top of _step_agent (before building the ctx).
    agent = _FakeAgent(
        Stage.LATERAL,
        StepResult(
            decision=StepDecision.CONTINUE,
            tokens_used=10,
            cost_usd=0.0,
            tool_runs=1,  # any progress so the watchdog doesn't trip
            nodes_written=1,
        ),
    )
    await _step_agent(agent, repo=MagicMock(), regression_council=_FakeCouncil(
        _verdict(Verdict.APPROVE)
    ))(state)

    assert state.regression_delta is None
    assert state.regression_origin_stage is None
    # The agent's StageContext should ALREADY see the cleared state
    # (clearing happens before ctx is built).
    assert agent.seen_ctx is not None
    assert agent.seen_ctx.regression_delta is None
    assert agent.seen_ctx.regression_origin_stage is None


async def test_delta_survives_non_origin_stage_re_entry(
    regression_mocks: dict[str, Any],
) -> None:
    """Re-entering an intermediate stage of the sub-pipeline (e.g. RECON
    after LATERAL emitted the regression) MUST NOT clear the delta —
    only the origin stage does."""
    state = _state(
        current_stage=Stage.RECON,
        regression_delta={"cidrs": ["10.0.5.0/24"]},
        regression_origin_stage=Stage.LATERAL.value,  # origin is LATERAL, not RECON
    )

    agent = _FakeAgent(
        Stage.RECON,
        StepResult(
            decision=StepDecision.CONTINUE,
            tokens_used=10,
            cost_usd=0.0,
            tool_runs=1,
            nodes_written=1,
        ),
    )
    await _step_agent(agent, repo=MagicMock(), regression_council=_FakeCouncil(
        _verdict(Verdict.APPROVE)
    ))(state)

    # Delta MUST survive — we're not back at the origin yet.
    assert state.regression_delta == {"cidrs": ["10.0.5.0/24"]}
    assert state.regression_origin_stage == Stage.LATERAL.value


# --------------------------------------------------------------------------- #
# 6. _applied_keys() query                                                     #
# --------------------------------------------------------------------------- #


async def test_applied_pairs_returns_applied_rows_from_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NoveltyCritic's replay-loop guard reads the set of
    ``(target_stage, trigger_kind, delta_fingerprint)`` triples that
    have already been approved-and-applied this engagement. The triple
    key (not the pair key from the original Phase D design) is what
    makes a real multi-segment engagement work — two genuinely-novel
    CIDRs no longer collide.
    """

    class _FakeResult:
        def __init__(self, rows: list[tuple[str, str, str]]) -> None:
            self._rows = rows

        def all(self) -> list[tuple[str, str, str]]:
            return list(self._rows)

    class _FakeSession:
        def __init__(self, rows: list[tuple[str, str, str]]) -> None:
            self._rows = rows

        async def execute(self, _stmt: Any) -> _FakeResult:
            return _FakeResult(self._rows)

    class _FakeSessionCtx:
        def __init__(self, rows: list[tuple[str, str, str]]) -> None:
            self._rows = rows

        async def __aenter__(self) -> _FakeSession:
            return _FakeSession(self._rows)

        async def __aexit__(self, *_: Any) -> None:
            return None

    from orchestrator import db as _db

    rows = [
        ("recon", "new_subnet_discovered", "a" * 64),
        ("enumeration", "new_service_class_discovered", "b" * 64),
    ]
    monkeypatch.setattr(_db, "session_scope", lambda: _FakeSessionCtx(rows))

    from agents.regression_council import _applied_keys

    keys = await _applied_keys(uuid4())
    assert keys == set(rows)


async def test_applied_pairs_returns_empty_on_db_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort: a Postgres outage logs and returns ``set()``. Matches
    the audit-trail tolerance pattern from ``record_regression``."""

    class _Broken:
        async def __aenter__(self) -> Any:
            raise RuntimeError("postgres down")

        async def __aexit__(self, *_: Any) -> None:
            return None

    from orchestrator import db as _db

    # Lambda matches the pattern in test_regression_models.py — a fresh
    # ``_Broken()`` per call mirrors how the real ``session_scope`` returns
    # a fresh ctx; inlining the call would share one instance.
    monkeypatch.setattr(_db, "session_scope", lambda: _Broken())  # noqa: PLW0108

    from agents.regression_council import _applied_keys

    keys = await _applied_keys(uuid4())
    assert keys == set()


# --------------------------------------------------------------------------- #
# 7. Event-bus + audit-log ordering                                            #
# --------------------------------------------------------------------------- #


async def test_event_and_audit_ordering_on_approve(
    regression_mocks: dict[str, Any],
) -> None:
    """``regression.requested`` MUST fire before ``regression.applied`` on
    the bus AND in the audit log so the UI sees the request before it
    sees the outcome (the SSE listener needs the interim state)."""
    state = _state(current_stage=Stage.LATERAL)
    council = _FakeCouncil(_verdict(Verdict.APPROVE))
    repo = MagicMock()

    req = _request(target_stage=Stage.RECON)
    agent = _FakeAgent(Stage.LATERAL, _regress_to_result(req))
    await _step_agent(agent, repo=repo, regression_council=council)(state)

    bus_kinds = [c.kwargs["kind"] for c in regression_mocks["bus"].publish.await_args_list]
    assert bus_kinds.index("regression.requested") < bus_kinds.index("regression.applied")
    actions = [a["action"] for a in regression_mocks["audits"]]
    # Requested first, then council_decided, then applied.
    assert actions.index("regression.requested") == 0
    assert actions.index("regression.council_decided") < actions.index("regression.applied")


async def test_request_with_missing_regression_request_is_logged_but_no_crash(
    regression_mocks: dict[str, Any],
) -> None:
    """An agent that returns REGRESS_TO with no ``regression_request``
    is a programmer error. The controller logs and falls through to
    CONTINUE-shape behaviour rather than crashing the engagement."""
    state = _state(current_stage=Stage.LATERAL)
    council = _FakeCouncil(_verdict(Verdict.APPROVE))
    repo = MagicMock()

    bad_result = StepResult(
        decision=StepDecision.REGRESS_TO,
        tokens_used=10,
        cost_usd=0.0,
        tool_runs=0,
        nodes_written=0,
        regression_request=None,
    )
    agent = _FakeAgent(Stage.LATERAL, bad_result)
    await _step_agent(agent, repo=repo, regression_council=council)(state)

    # Stage unchanged, no rows written.
    assert state.current_stage == Stage.LATERAL
    assert regression_mocks["rows"] == []
    assert regression_mocks["updates"] == []


# --------------------------------------------------------------------------- #
# 8. _handle_regress_to direct unit test (helper isolation)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # model_construct with a bogus enum string emits a serializer warning
async def test_handle_regress_to_invalid_target_stage_silently_aborts(
    regression_mocks: dict[str, Any],
) -> None:
    """If the council somehow approves a target_stage string that doesn't
    map to a known ``Stage`` enum value, the controller refuses to write
    a bogus transition. Defensive — the council vetoes unknown stages."""

    council = _FakeCouncil(_verdict(Verdict.APPROVE))

    # Force a bad target_stage by patching the request post-construction:
    # the model uses ``validate_assignment``, so we have to bypass it via
    # ``model_construct`` for this defensive path.
    req = RegressionRequest.model_construct(
        target_stage="not_a_real_stage",  # type: ignore[arg-type]
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
        delta={"cidrs": ["10.0.5.0/24"]},
        evidence=[],
        rationale="bogus target",
    )

    state = _state(current_stage=Stage.LATERAL, regressions_used=1)
    initial_stage = state.current_stage
    initial_count = state.regressions_used

    await _handle_regress_to(
        agent_stage=Stage.LATERAL,
        request=req,
        state=state,
        repo=MagicMock(),
        council=council,
    )

    # State left as-is; counter NOT bumped because the transition aborted
    # before any state mutation (Stage() validated up front).
    assert state.current_stage == initial_stage
    assert state.regressions_used == initial_count
    assert state.regression_delta is None
