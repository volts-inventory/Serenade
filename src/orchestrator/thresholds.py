"""Solo-operator threshold-tripped warnings.

Lightweight observability hook: when an engagement crosses a "you probably
want to look at this" boundary (token-budget at 80%, wallclock at 80%, a
sustained critic-veto rate, or a stage where new tools keep getting
suppressed), this module publishes a single warning event on the per-
engagement bus AND writes an audit-log row. Debounced so each threshold
fires at most once per engagement (or, for suppression-spread, once per
stage). No external metrics stack — just events + the audit log.

Hook point: called from ``stage_controller._step_agent`` after every
agent step, so the most recently updated counters are visible.

Wire shape:

    state.warnings_fired: set[str]         # e.g. {"budget:token", "veto_rate"}
    state.recent_critic_verdicts: list[bool]  # rolling window of last N
    state.last_known_suppression_stage_count: dict[str, int]

These live on ``EngagementState`` and survive checkpointing because
``EngagementState`` is the LangGraph state object.

Event kinds published:

    engagement.budget_warning      {usage: 0.82, kind: "token"|"wallclock"}
    engagement.veto_rate_warning   {window: 10, vetoes: 6, rate: 0.6}
    engagement.suppression_spread  {stage, added: ["tool1", "tool2"]}

Each event also writes an audit-log row with action ``engagement.warning``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from orchestrator.events.bus import get_bus
from orchestrator.ledger import audit

if TYPE_CHECKING:
    from orchestrator.stage_controller import EngagementState

log = structlog.get_logger(__name__)


# Per-engagement guard for the threshold check-then-act critical section
# (async-08). The four threshold blocks below each do a non-atomic
# read-membership-then-``fired.add`` against ``state.warnings_fired`` and
# rebuild ``recent_critic_verdicts``. Today nodes run serially per
# engagement (LangGraph), so single-writer is emergent, not enforced — a
# future concurrent fan-out over one ``state`` would double-fire warnings
# or lose verdict entries. This lock makes the section atomic cheaply
# without relying on that invariant.
#
# Bounded + loop-safe: locks are created lazily on first use for an
# engagement and evicted via ``release_engagement_lock`` on engagement
# completion (called from the worker teardown), so the dict can't grow
# without limit (cf. async-03/async-10 footguns with ``defaultdict``).
_THRESHOLD_LOCKS: dict[str, asyncio.Lock] = {}


def _threshold_lock(engagement_id: UUID) -> asyncio.Lock:
    key = str(engagement_id)
    lock = _THRESHOLD_LOCKS.get(key)
    if lock is None:
        # Created inside the running loop on first access; bounded by the
        # eviction in ``release_engagement_lock``.
        lock = asyncio.Lock()
        _THRESHOLD_LOCKS[key] = lock
    return lock


def release_engagement_lock(engagement_id: UUID) -> None:
    """Drop the per-engagement threshold lock for a completed engagement.

    Called from the worker runtime's engagement teardown so
    ``_THRESHOLD_LOCKS`` doesn't leak one ``asyncio.Lock`` per engagement
    on a long-lived worker (async-03/async-10). Best-effort / idempotent.
    """
    _THRESHOLD_LOCKS.pop(str(engagement_id), None)


# Thresholds. Module-level so tests can monkey-patch them.
TOKEN_BUDGET_WARN_RATIO = 0.80
WALLCLOCK_WARN_RATIO = 0.80
VETO_RATE_WINDOW = 10
VETO_RATE_WARN_RATIO = 0.50


async def check_thresholds(state: EngagementState) -> None:
    """Evaluate the four thresholds against the current state.

    Best-effort — bus and audit-log writes both catch their own errors.
    Caller (``stage_controller._step_agent``) invokes this after every
    step. Debounce keyed by ``state.warnings_fired``:

      - ``budget:token``     — fires once per engagement
      - ``budget:wallclock`` — fires once per engagement
      - ``veto_rate``        — fires once per engagement
      - ``suppression_spread:<stage>`` — fires once per stage

    The whole evaluation runs under the per-engagement threshold lock so
    the check-then-``fired.add`` debounce is atomic even if a future
    concurrent fan-out drives two steps against the same ``state`` (the
    single-writer property is otherwise only emergent — async-08).
    """
    async with _threshold_lock(state.engagement_id):
        await _check_thresholds_locked(state)


async def _check_thresholds_locked(state: EngagementState) -> None:
    eng = state.engagement
    fired = state.warnings_fired

    # 1. Token-budget breach of TOKEN_BUDGET_WARN_RATIO.
    # token_budget=0 means unlimited (local backends); skip the warning.
    token_budget = int(eng.get("token_budget", 0) or 0)
    if token_budget > 0 and "budget:token" not in fired:
        ratio = state.total_tokens / token_budget
        if ratio >= TOKEN_BUDGET_WARN_RATIO:
            fired.add("budget:token")
            await _emit(
                state,
                kind="engagement.budget_warning",
                payload={
                    "usage": round(ratio, 4),
                    "kind": "token",
                    "used": state.total_tokens,
                    "budget": token_budget,
                },
            )

    # 2. Wallclock-window breach of WALLCLOCK_WARN_RATIO.
    # The engagement window carries ``start_ns`` + ``end_ns`` (set by
    # ``build_engagement_context``); we compute "fraction elapsed" against it.
    window = eng.get("window") or {}
    start_ns = window.get("start_ns")
    end_ns = window.get("end_ns")
    if start_ns and end_ns and "budget:wallclock" not in fired:
        import time as _time

        total = int(end_ns) - int(start_ns)
        if total > 0:
            elapsed = max(0, _time.time_ns() - int(start_ns))
            ratio = elapsed / total
            if ratio >= WALLCLOCK_WARN_RATIO:
                fired.add("budget:wallclock")
                await _emit(
                    state,
                    kind="engagement.budget_warning",
                    payload={
                        "usage": round(ratio, 4),
                        "kind": "wallclock",
                        "elapsed_ns": elapsed,
                        "window_ns": total,
                    },
                )

    # 3. Critic-veto rate over the last VETO_RATE_WINDOW tool calls.
    # ``recent_critic_verdicts`` is a rolling bool buffer: True = veto,
    # False = approve/no-veto. Once the window is full and the veto
    # fraction crosses VETO_RATE_WARN_RATIO, emit (once).
    window_buf = state.recent_critic_verdicts
    if "veto_rate" not in fired and len(window_buf) >= VETO_RATE_WINDOW:
        # Look at most recent VETO_RATE_WINDOW entries
        tail = window_buf[-VETO_RATE_WINDOW:]
        vetoes = sum(1 for v in tail if v)
        rate = vetoes / VETO_RATE_WINDOW
        if rate > VETO_RATE_WARN_RATIO:
            fired.add("veto_rate")
            await _emit(
                state,
                kind="engagement.veto_rate_warning",
                payload={
                    "window": VETO_RATE_WINDOW,
                    "vetoes": vetoes,
                    "rate": round(rate, 4),
                },
            )

    # 4. Suppression spread within a single stage. Fire the first time
    # the suppressed-tools count for the current stage grows since we
    # last sampled it. ``last_known_suppression_stage_count`` is a
    # ``dict[stage_name, count]`` so cross-stage transitions (which
    # clear suppressions) don't re-trip the warning on every step.
    stage = state.suppressed_tools_stage or state.current_stage.value
    cur_count = len(state.suppressed_tools)
    last_count = state.last_known_suppression_stage_count.get(stage, 0)
    debounce_key = f"suppression_spread:{stage}"
    if cur_count > last_count and debounce_key not in fired:
        fired.add(debounce_key)
        added = sorted(state.suppressed_tools.keys())
        await _emit(
            state,
            kind="engagement.suppression_spread",
            payload={
                "stage": stage,
                "added": added,
                "count": cur_count,
                "previous_count": last_count,
            },
        )
    # Always update the per-stage tracker so the next step compares
    # against the new baseline.
    state.last_known_suppression_stage_count[stage] = cur_count


def record_critic_verdict(state: EngagementState, *, vetoed: bool) -> None:
    """Append the most recent critic verdict to the rolling window.

    Called from ``BaseAgent._handle_one_tool_call`` immediately after
    the critic / council renders its verdict on a tool call. The
    rolling buffer is capped at ``VETO_RATE_WINDOW * 2`` to keep
    state small while leaving room for the threshold check to read
    the tail.
    """
    state.recent_critic_verdicts.append(vetoed)
    if len(state.recent_critic_verdicts) > VETO_RATE_WINDOW * 2:
        state.recent_critic_verdicts = state.recent_critic_verdicts[-VETO_RATE_WINDOW * 2:]


async def _emit(state: EngagementState, *, kind: str, payload: dict[str, Any]) -> None:
    """Publish to the event bus + write an audit-log row. Both writes
    are best-effort; failures log and continue so the agent loop never
    blocks on observability."""
    try:
        await get_bus().publish(
            engagement_id=state.engagement_id,
            kind=kind,
            payload=payload,
        )
    except Exception as e:
        log.warning("thresholds.bus_publish_failed", err=str(e), kind=kind)
    try:
        await audit(
            engagement_id=state.engagement_id,
            actor="threshold_monitor",
            action=kind,
            payload=payload,
        )
    except Exception as e:
        log.warning("thresholds.audit_write_failed", err=str(e), kind=kind)
