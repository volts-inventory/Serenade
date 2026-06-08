"""R3 #J1 Refactor 1 — StageContext / StageOutbox split.

Phase J1 separated the per-turn READ-ONLY inputs (StageContext) from
the per-turn WRITABLE outputs (StageOutbox). These tests pin:

1. StageOutbox carries the writable fields and starts empty.
2. StageContext.outbox is the canonical writable channel.
3. The legacy ``ctx.<writable>`` proxy properties still work for the
   migration window.
4. StepResult carries the outbox through to the stage controller.
"""

from __future__ import annotations

from uuid import uuid4


from agents.base import StageContext, StageOutbox, StepDecision, StepResult
from worldstate.schema import Stage


def test_stage_outbox_defaults_are_empty() -> None:
    """A fresh StageOutbox has empty containers everywhere."""
    o = StageOutbox()
    assert o.pending_approvals == []
    assert o.tokens_used == 0
    assert o.cost_usd == 0.0
    assert o.tokens_by_tool == {}
    assert o.suppressed_tools == {}
    assert o.consecutive_zero_host_turns == 0
    assert o.pending_regression_request is None


def test_stage_context_carries_outbox() -> None:
    """A StageContext has a default empty outbox if none is passed."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    assert isinstance(ctx.outbox, StageOutbox)
    assert ctx.outbox.pending_approvals == []


def test_proxy_property_pending_approvals_round_trips() -> None:
    """``ctx.pending_approvals.append(x)`` writes through to outbox."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    ctx.pending_approvals.append({"reason": "test"})
    assert ctx.outbox.pending_approvals == [{"reason": "test"}]


def test_proxy_property_tokens_used_round_trips() -> None:
    """``ctx.tokens_used += N`` writes through to outbox."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    ctx.tokens_used += 123
    assert ctx.outbox.tokens_used == 123


def test_proxy_property_tokens_by_tool_round_trips() -> None:
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    ctx.tokens_by_tool["nmap"] = 50
    assert ctx.outbox.tokens_by_tool == {"nmap": 50}


def test_proxy_property_suppressed_tools_round_trips() -> None:
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    ctx.suppressed_tools["mimikatz"] = "worker_disabled"
    assert ctx.outbox.suppressed_tools == {"mimikatz": "worker_disabled"}


def test_proxy_property_consecutive_zero_host_turns_round_trips() -> None:
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
    )
    ctx.consecutive_zero_host_turns = 3
    assert ctx.outbox.consecutive_zero_host_turns == 3


def test_legacy_kwargs_redirect_to_outbox() -> None:
    """A StageContext constructed with the legacy writable kwargs (e.g.
    ``tokens_by_tool={...}``, ``suppressed_tools={...}``,
    ``consecutive_zero_host_turns=N``) seeds the outbox correctly."""
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
        tokens_by_tool={"nmap": 1, "httpx": 2},
        suppressed_tools={"mimikatz": "missing_signing_key"},
        consecutive_zero_host_turns=2,
    )
    assert ctx.outbox.tokens_by_tool == {"nmap": 1, "httpx": 2}
    assert ctx.outbox.suppressed_tools == {"mimikatz": "missing_signing_key"}
    assert ctx.outbox.consecutive_zero_host_turns == 2


def test_step_result_has_outbox_field() -> None:
    """StepResult carries a StageOutbox so the controller can apply
    per-turn writes without reaching back into the StageContext."""
    r = StepResult(
        decision=StepDecision.ADVANCE,
        tokens_used=10,
        cost_usd=0.01,
        tool_runs=1,
        nodes_written=2,
    )
    assert isinstance(r.outbox, StageOutbox)


def test_explicit_outbox_kwarg_wins_over_legacy_kwargs() -> None:
    """If both an explicit ``outbox=`` and legacy writable kwargs are
    passed, the explicit outbox wins (legacy kwargs silently dropped)."""
    explicit = StageOutbox(tokens_by_tool={"a": 100})
    ctx = StageContext(
        engagement_id=uuid4(),
        engagement={},
        stage=Stage.RECON,
        outbox=explicit,
        # These should be ignored because outbox is explicit:
        tokens_by_tool={"b": 200},
    )
    assert ctx.outbox is explicit
    assert ctx.outbox.tokens_by_tool == {"a": 100}
