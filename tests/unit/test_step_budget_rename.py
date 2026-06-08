"""R3 #J1 Refactor 2 — StepBudget / BaseAgent turn-counter rename.

Two distinct concepts, same units (turns), so reviewers conflated them:

- ``StepBudget.max_turns`` → ``StepBudget.stage_step_cap``: how many
  times the stage controller may call ``agent.step()`` before forcing
  ADVANCE.
- ``BaseAgent.max_loop_turns`` → ``BaseAgent.interactive_llm_turn_cap``:
  how many LLM turns may fire inside one ``agent.step()`` invocation
  before the interactive loop bails.

These tests pin the renames + the alias migration path.
"""

from __future__ import annotations

import pytest

from agents.base import BaseAgent, StepBudget
from agents.exploit import ExploitAgent
from agents.lateral import LateralAgent
from agents.recon import ReconAgent
from agents.report import ReportAgent


def test_step_budget_canonical_field_is_stage_step_cap() -> None:
    """The renamed field is reachable by its new name."""
    b = StepBudget(stage_step_cap=12)
    assert b.stage_step_cap == 12


def test_step_budget_canonical_field_is_interactive_llm_turn_cap() -> None:
    b = StepBudget(interactive_llm_turn_cap=20)
    assert b.interactive_llm_turn_cap == 20


def test_step_budget_legacy_max_turns_kwarg_aliases_stage_step_cap() -> None:
    """``StepBudget(max_turns=N)`` still works during the migration."""
    b = StepBudget(max_turns=12)
    assert b.stage_step_cap == 12
    # Read alias works too:
    assert b.max_turns == 12


def test_step_budget_legacy_interactive_max_turns_kwarg_aliases() -> None:
    b = StepBudget(interactive_max_turns=20)
    assert b.interactive_llm_turn_cap == 20
    assert b.interactive_max_turns == 20


def test_base_agent_interactive_llm_turn_cap_default() -> None:
    """The base class default mirrors the old ``max_loop_turns`` 15."""
    assert BaseAgent.interactive_llm_turn_cap == 15
    # Alias still works for legacy reads.
    assert BaseAgent.max_loop_turns == 15


def test_init_subclass_mirrors_legacy_attr_to_new_attr() -> None:
    """A subclass written before the rename (sets ``max_loop_turns``)
    should automatically get ``interactive_llm_turn_cap`` populated."""

    class _LegacyAgent(BaseAgent):
        stage = type("FakeStage", (), {"value": "x"})()  # type: ignore[assignment]
        max_loop_turns = 17

        async def _build_world_state_slice(self, ctx):  # type: ignore[no-untyped-def]
            return {}

        def _objective_for_turn(self, ctx, slice_):  # type: ignore[no-untyped-def]
            return ""

        async def _decide_advance(self, ctx):  # type: ignore[no-untyped-def]
            ...

        async def _parse_and_persist(self, ctx, tool, ti, to, trid):  # type: ignore[no-untyped-def]
            return []

    assert _LegacyAgent.max_loop_turns == 17
    assert _LegacyAgent.interactive_llm_turn_cap == 17


def test_init_subclass_mirrors_new_attr_to_legacy_attr() -> None:
    """A new-style subclass (sets ``interactive_llm_turn_cap``) gets
    ``max_loop_turns`` populated for tests still reading the old name."""

    class _NewAgent(BaseAgent):
        stage = type("FakeStage", (), {"value": "x"})()  # type: ignore[assignment]
        interactive_llm_turn_cap = 22

        async def _build_world_state_slice(self, ctx):  # type: ignore[no-untyped-def]
            return {}

        def _objective_for_turn(self, ctx, slice_):  # type: ignore[no-untyped-def]
            return ""

        async def _decide_advance(self, ctx):  # type: ignore[no-untyped-def]
            ...

        async def _parse_and_persist(self, ctx, tool, ti, to, trid):  # type: ignore[no-untyped-def]
            return []

    assert _NewAgent.interactive_llm_turn_cap == 22
    assert _NewAgent.max_loop_turns == 22


@pytest.mark.parametrize(
    "cls,expected",
    [(ReconAgent, 20), (ExploitAgent, 8), (LateralAgent, 10), (ReportAgent, 6)],
)
def test_per_stage_caps_unchanged_by_rename(cls: type, expected: int) -> None:
    """The rename is purely cosmetic — the per-stage caps are the same."""
    assert cls.max_loop_turns == expected
    assert cls.interactive_llm_turn_cap == expected
