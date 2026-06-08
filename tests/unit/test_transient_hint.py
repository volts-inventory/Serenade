"""R3 #H3 Bug 7: transient I/O escalation note carries an operator hint.

Covers:
- ``_TRANSIENT_HINTS`` has an entry for each known ``where`` string used
  in ``base.py`` (``tool_run_pre_persist``, ``tool_exec``,
  ``parse_and_persist``, ``worldstate_upsert``, ``llm_request``).
- ``_hint_for_where`` returns the actionable string for known wheres and
  a non-empty fallback for unknown ones (so the escalation note still
  concatenates cleanly).
- The full escalation note that ``BaseAgent.step`` builds when it
  catches a ``_TransientFailure`` contains the matching hint.
"""

from __future__ import annotations


from agents.base import (
    _TRANSIENT_HINTS,
    _hint_for_where,
    _TransientFailure,
)


def test_known_wheres_have_actionable_hints() -> None:
    """Every ``where`` string the base agent passes to ``_TransientFailure``
    has a non-empty hint mapped. The hint must mention something the
    operator can actually do (a command, a service name) — we assert
    that with a coarse "contains check or backtick" rule."""
    expected = {
        "tool_run_pre_persist",
        "tool_exec",
        "parse_and_persist",
        "worldstate_upsert",
        "llm_request",
    }
    assert expected <= set(_TRANSIENT_HINTS), (
        f"missing hint(s) for: {expected - set(_TRANSIENT_HINTS)}"
    )
    for where, hint in _TRANSIENT_HINTS.items():
        assert hint, f"hint for {where!r} must not be empty"
        # Actionable: mentions a service / command / config the operator can check.
        assert any(
            kw in hint.lower()
            for kw in (
                "check",
                "docker",
                "database",
                "graph",
                "llm",
                "worker",
                "endpoint",
                "logs",
            )
        ), f"hint for {where!r} is not actionable: {hint!r}"


def test_hint_for_where_returns_known_mapping() -> None:
    assert "Database" in _hint_for_where("tool_run_pre_persist")
    assert "graph" in _hint_for_where("parse_and_persist").lower()
    assert "graph" in _hint_for_where("worldstate_upsert").lower()
    assert "LLM backend" in _hint_for_where("llm_request")


def test_hint_for_where_unknown_returns_fallback() -> None:
    """Unknown ``where`` strings must still produce a non-empty hint so
    the escalation-note concatenation in ``step()`` doesn't render a
    trailing ' — Hint: '."""
    out = _hint_for_where("totally_made_up_where")
    assert out
    assert "no specific hint" in out.lower() or "logs" in out.lower()


def test_escalation_note_carries_hint() -> None:
    """Replicate the ``except _TransientFailure`` branch in ``step()``
    and assert the resulting ``notes`` string contains the hint.

    R3 #I1 Bug 6: hint is now PREFIXED to the message (was suffixed and
    got clipped by the 500-char cap when ``repr(e.original)`` was long).
    """
    err = ConnectionError("postgres connection refused")
    exc = _TransientFailure("tool_run_pre_persist", err)
    # Mimic the in-source builder.
    hint = _hint_for_where(exc.where)
    base_msg = f"transient I/O failure: {exc.where}: {exc.original!r}"
    notes = f"Hint: {hint} | {base_msg}"[:500]
    assert "Database unreachable" in notes
    assert "transient I/O failure" in notes
    assert exc.where in notes
    # Hint must be at the very front so it survives even when
    # base_msg's repr inflates past the cap.
    assert notes.startswith("Hint:")


def test_escalation_note_for_graph_path() -> None:
    err = TimeoutError("graph error")
    exc = _TransientFailure("worldstate_upsert", err)
    hint = _hint_for_where(exc.where)
    notes = f"Hint: {hint} | transient I/O failure: {exc.where}: {exc.original!r}"
    assert "graph" in notes.lower()


def test_escalation_note_for_llm_path() -> None:
    err = RuntimeError("anthropic API 503")
    exc = _TransientFailure("llm_request", err)
    hint = _hint_for_where(exc.where)
    notes = f"Hint: {hint} | transient I/O failure: {exc.where}: {exc.original!r}"
    assert "LLM backend" in notes
    assert "serenade llm status" in notes


def test_long_repr_does_not_clip_hint() -> None:
    """R3 #I1 Bug 6: when ``repr(e.original)`` is long enough that the
    full ``Hint: ... | base_msg`` would exceed 500 chars, the hint MUST
    survive (it goes first). Previously the hint was suffixed and got
    truncated off the end."""

    class _BigRepr(Exception):
        def __repr__(self) -> str:
            return "BigRepr(" + ("X" * 400) + ")"

    err = _BigRepr()
    exc = _TransientFailure("tool_run_pre_persist", err)
    hint = _hint_for_where(exc.where)
    base_msg = f"transient I/O failure: {exc.where}: {exc.original!r}"

    # Mirror the in-source builder exactly.
    notes = f"Hint: {hint} | {base_msg}"[:500]
    assert len(notes) <= 500
    # Hint MUST be present in the final 500 chars.
    assert "Database unreachable" in notes
    # The note starts with the operator-actionable hint, not the noisy repr.
    assert notes.startswith("Hint:")


def test_step_escalation_uses_new_ordering() -> None:
    """End-to-end-ish: drive ``BaseAgent.step`` through the
    ``_TransientFailure`` path with a real-shaped exception and confirm
    the returned StepResult.notes starts with ``Hint:`` (the new
    ordering)."""
    import asyncio
    from typing import Any
    from unittest.mock import MagicMock

    from agents.base import BaseAgent, StageContext, StepDecision
    from worldstate.schema import Stage

    class _Tiny(BaseAgent):
        stage = Stage.RECON

        async def _build_world_state_slice(self, ctx: StageContext) -> dict[str, Any]:
            raise _TransientFailure("tool_run_pre_persist", ConnectionError("pg down"))

        def _objective_for_turn(self, ctx: StageContext, slice_: dict[str, Any]) -> str:  # pragma: no cover
            return "n/a"

        async def _decide_advance(self, ctx: StageContext) -> StepDecision:  # pragma: no cover
            return StepDecision.ADVANCE

        async def _parse_and_persist(self, *args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover
            return []

    agent = _Tiny(
        router=MagicMock(),
        repo=MagicMock(),
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )
    ctx = StageContext(
        engagement_id=__import__("uuid").uuid4(),
        engagement={"approval_policy": "manual"},
        stage=Stage.RECON,
    )

    result = asyncio.run(agent.step(ctx))
    assert result.decision == StepDecision.ESCALATE_HUMAN
    assert result.notes.startswith("Hint:")
    assert "Database unreachable" in result.notes
