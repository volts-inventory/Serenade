"""ws-06 regression — ``_enumeration_node`` must NOT run net_enum when
web_enum just regressed BACKWARD to an earlier stage.

The enumeration node fans web_enum → net_enum sequentially. The
save/restore logic keys only on ``current_stage != ENUMERATION``, which is
true both when web_enum advanced FORWARD (the intended net_enum case) and
when a council-APPROVED REGRESS_TO moved the stage BACKWARD (e.g. to
RECON). Running net_enum on a backward regression executes a full
enumeration step against stale state with a ``regression_delta`` it never
requested — wasted work that re-does what the regression is unwinding.

These tests drive the REAL ``_make_enumeration_node`` closure (extracted
to a module-level factory for exactly this testability) with stand-in node
callables that mutate ``current_stage`` the way ``_step_agent`` would.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from orchestrator.stage_controller import EngagementState, _make_enumeration_node
from worldstate.schema import Stage


def _state() -> EngagementState:
    return EngagementState(
        engagement_id=uuid4(),
        engagement={"approval_policy": "manual"},
        current_stage=Stage.ENUMERATION,
    )


def _node_that_sets_stage(target: Stage | None, calls: list[str], label: str) -> Any:
    """Return a stand-in node callable that records it ran and optionally
    moves ``current_stage`` (mimicking ADVANCE / REGRESS_TO)."""

    async def _n(state: EngagementState) -> EngagementState:
        calls.append(label)
        if target is not None:
            state.current_stage = target
        return state

    return _n


@pytest.mark.asyncio
async def test_net_enum_skipped_when_web_enum_regressed_backward() -> None:
    """web_enum regresses to RECON → net_enum must NOT run, and the
    backward target survives so the router carries the regression."""
    calls: list[str] = []
    state = _state()
    web = _node_that_sets_stage(Stage.RECON, calls, "web")  # backward regress
    net = _node_that_sets_stage(None, calls, "net")

    node = _make_enumeration_node(web_node=web, net_node=net)
    out = await node(state)

    assert calls == ["web"], "net_enum must be skipped on backward regression"
    # Backward target preserved for the router.
    assert out.current_stage == Stage.RECON
    assert out.enumeration_substage is None


@pytest.mark.asyncio
async def test_net_enum_runs_when_web_enum_advanced_forward() -> None:
    """web_enum advances to VULN_TRIAGE → net_enum still runs (restored to
    ENUMERATION), then the forward target is re-applied."""
    calls: list[str] = []
    state = _state()
    web = _node_that_sets_stage(Stage.VULN_TRIAGE, calls, "web")  # forward advance
    net = _node_that_sets_stage(None, calls, "net")

    node = _make_enumeration_node(web_node=web, net_node=net)
    out = await node(state)

    assert calls == ["web", "net"], "net_enum must still run after a forward advance"
    # Forward target re-applied after net_enum.
    assert out.current_stage == Stage.VULN_TRIAGE
    assert out.enumeration_substage is None


@pytest.mark.asyncio
async def test_net_enum_runs_when_web_enum_stayed_enumeration() -> None:
    """web_enum CONTINUEs (stays ENUMERATION) → net_enum runs normally."""
    calls: list[str] = []
    state = _state()
    web = _node_that_sets_stage(None, calls, "web")  # stays ENUMERATION
    net = _node_that_sets_stage(None, calls, "net")

    node = _make_enumeration_node(web_node=web, net_node=net)
    out = await node(state)

    assert calls == ["web", "net"]
    assert out.current_stage == Stage.ENUMERATION
    assert out.enumeration_substage is None
