"""Stage controller — top-level LangGraph state machine.

Drives RECON → ENUMERATION → VULN_TRIAGE → EXPLOIT → POST_EXPLOIT → LATERAL →
LOOT → REPORT, with rule-based transitions on world-state coverage.

Approval handling: when a step returns `StepDecision.ESCALATE_HUMAN`, the
graph node calls `interrupt()` so the run pauses and the state is durably
checkpointed. On operator approve/reject (via the engagement API), the
worker re-`invoke`s the graph against the same `thread_id`; LangGraph resumes
from the checkpoint with the approval payload routed back to the paused node.

The controller also handles
``StepDecision.REGRESS_TO`` — a deterministic post-step signal from an agent
that an earlier stage must be re-entered with a delta-scoped contract. The
``StageRegressionCouncil`` verifies novelty + scope + budget; on
approve the controller writes the backwards stage transition and stashes the
delta on ``EngagementState`` so the re-entered agent can narrow its slice.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.base import BaseAgent, StageContext, StepDecision
from agents.critic import Verdict
from agents.regression import RegressionRequest
from agents.regression_council import StageRegressionCouncil
from orchestrator.events.bus import get_bus
from orchestrator.ledger import (
    audit,
    clear_tool_suppressions,
    list_approved_regressions,
    record_regression,
    update_regression_with_verdict,
)
from orchestrator.world_state_store import save_world_state
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import Stage

log = structlog.get_logger(__name__)


class EngagementState(BaseModel):
    """LangGraph state carried between nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engagement_id: UUID
    engagement: dict[str, Any]
    current_stage: Stage = Stage.RECON
    turn_counts: dict[str, int] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    # Per-tool running token totals for the per-tool budget gate.
    tokens_by_tool: dict[str, int] = Field(default_factory=dict)
    # Tools that returned terminal_error=True on a prior turn — the base
    # agent filters these out of next turn's candidate list so the LLM
    # stops re-proposing structurally-broken tools (worker disabled, signing
    # key missing, etc.). Stage-scoped: when the engagement transitions to
    # a new stage we clear suppression and re-probe, so a tool whose
    # underlying issue was fixed mid-engagement gets another chance.
    # `suppressed_tools_stage` stamps which stage owns the current list —
    # different stage on entry to `_node` ⇒ reset.
    # Shape is ``dict[tool_name -> reason]`` so the operator UX can surface
    # WHY a tool was muted (mimikatz failing because the Windows worker is
    # misconfigured, etc.). The validator below rehydrates legacy
    # ``list[str]`` checkpoints for the one-release migration window — old
    # rows get ``reason="legacy"`` so the UI still works.
    suppressed_tools: dict[str, str] = Field(default_factory=dict)
    suppressed_tools_stage: str | None = None

    @field_validator("suppressed_tools", mode="before")
    @classmethod
    def _coerce_suppressed_tools(cls, v: Any) -> dict[str, str]:
        """Accept both shapes on rehydrate.

        Legacy state rows (one release ago) stored a ``list[str]`` of tool
        names. New rows store a ``dict[name, reason]``. Coerce to the dict
        shape on read so the controller always operates on the same type.

        Logs loudly on any unrecognised shape — a state-corruption bug
        (Pydantic v2 sometimes hands us a stringified dict from a
        round-trip; a future planner returning a new shape) would
        otherwise be eaten without a trace. The safe default is still
        ``{}`` so the engagement doesn't crash on a checkpoint we can't
        parse.
        """
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(reason) for k, reason in v.items()}
        if isinstance(v, (list, set, tuple)):
            log.warning(
                "suppressed_tools.legacy_list_shape",
                value_type=type(v).__name__,
                sample=list(v)[:5],
            )
            return {str(name): "legacy" for name in v}
        log.warning(
            "suppressed_tools.invalid_shape",
            value_type=type(v).__name__,
            value_repr=repr(v)[:200],
        )
        return {}
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    completed: bool = False
    # Recon zero-progress watchdog state. Survives across turns so
    # the count of consecutive zero-host turns is meaningful even if the
    # agent loop re-enters the recon node from CONTINUE.
    consecutive_zero_host_turns: int = 0
    # Global progress watchdog. Counts consecutive agent steps that
    # produced zero tool_runs AND zero nodes_written, regardless of stage.
    # Reset whenever an agent actually makes progress. `_check_budgets`
    # halts the engagement once this exceeds the threshold so unattended
    # runs fail loud instead of burning tokens on no-progress loops.
    stalled_turn_count: int = 0
    # Lifetime count of approved-and-applied regressions for this
    # engagement. Read by the BudgetCritic to enforce the per-engagement
    # cap (default ``REGRESSION_BUDGET_MAX_DEFAULT`` = 3). Incremented
    # when the controller commits the backwards transition. Old threads
    # without this field will pick up ``0`` via pydantic's default.
    regressions_used: int = 0
    # The active regression delta. Set when the StageRegressionCouncil
    # approves a regression request; cleared when the controller re-enters
    # the stage that emitted the request
    # (``regression_origin_stage``). While set it is threaded onto every
    # ``StageContext.regression_delta`` so agents can narrow their
    # world-state slice — see the contract docstring on
    # ``BaseAgent._build_world_state_slice``. ``None`` outside an active
    # sub-pipeline.
    regression_delta: dict[str, Any] | None = None
    # The stage that produced the most recent ``REGRESS_TO`` signal
    # (e.g. ``"lateral"``). Used by the controller to detect when the
    # delta-scoped sub-pipeline has completed the loop. Cleared together
    # with ``regression_delta`` on origin-stage re-entry.
    regression_origin_stage: str | None = None
    # World-state node IDs that motivated the regression request.
    # Set alongside ``regression_delta`` when the council approves;
    # cleared together with the delta on origin-stage re-entry. Agents
    # may use these to attach provenance context to their delta-scoped
    # work (e.g. the recon agent can reference why this CIDR was added).
    regression_evidence_ids: list[str] | None = None
    # Consecutive council-VETO'd regression requests, keyed by the
    # ORIGINATING stage. A VETO leaves ``current_stage`` unchanged, so the
    # router re-enters the same agent node and the agent — if the structural
    # condition that motivated the regression persists — re-emits the
    # identical request every step. The council then VETOes it again
    # (``replay_loop`` once the fingerprint is in ``_applied_keys``), and
    # nothing advances ``regressions_used`` (that only bumps on APPROVE), so
    # neither the BudgetCritic cap nor the producer-side cap is ever reached
    # → an unbounded no-progress loop that burns the whole token budget
    # (ws-02). This counter bounds it: each VETO increments the count for the
    # emitting stage; once it crosses ``_REGRESSION_VETO_STAGE_CAP`` the
    # controller forces the stage to ADVANCE so the engagement makes forward
    # progress instead of looping. Reset when the stage advances (forward or
    # via an APPROVED regression) so a later, genuinely-novel regression from
    # the same stage on a future visit isn't penalised by stale counts.
    regression_vetoes_by_stage: dict[str, int] = Field(default_factory=dict)
    # ``_enumeration_node`` fan-out checkpoint. The node
    # runs web_enum then net_enum sequentially. If web_enum returns
    # ESCALATE_HUMAN, ``interrupt()`` is thrown and net_enum is skipped;
    # on resume LangGraph re-enters the node from scratch and would
    # re-run web_enum (re-doing any persistence it already did). We
    # checkpoint the per-substage phase here so re-entry skips already-
    # completed substages. ``None`` outside an enumeration step.
    # Lifecycle:
    #   ``None`` -> ``"web_enum_pending"`` -> ``"web_enum_done"``
    #         -> ``"net_enum_pending"`` -> ``"net_enum_done"`` -> ``None``
    enumeration_substage: str | None = None
    # Customer-facing abort flag. The orchestrator's
    # ``POST /engagements/{id}/abort`` flips the engagement row to
    # ``status="aborted"`` and publishes ``engagement.abort_requested``
    # on the event bus. The stage controller checks this flag at every
    # entry point of ``_step_agent`` (and before each tool exec via
    # ``StageContext``) and short-circuits with a final ESCALATE_HUMAN
    # carrying the abort reason. The worker runtime polls the engagement
    # row at the top of every step (best-effort — Postgres unreachable
    # logs and continues, letting the next step pick up the abort), so a
    # bus outage doesn't strand an abort request. ``abort_reason`` is the
    # operator's typed string from the abort endpoint.
    should_abort: bool = False
    abort_reason: str | None = None
    # Solo-operator observability: threshold-tripped warnings.
    # ``warnings_fired`` debounces each warning kind so the bus + audit
    # log don't get spammed across long unattended runs. Keys:
    #   ``budget:token`` / ``budget:wallclock`` (once per engagement)
    #   ``veto_rate`` (once per engagement)
    #   ``suppression_spread:<stage>`` (once per stage)
    warnings_fired: set[str] = Field(default_factory=set)
    # Rolling window of the last critic verdicts (True == VETO). Read
    # by ``thresholds.check_thresholds`` to compute the veto-rate over
    # the most recent N tool calls. Capped to a small multiple of the
    # window size to keep checkpoint payloads compact.
    recent_critic_verdicts: list[bool] = Field(default_factory=list)
    # Per-stage tally of the last-known suppressed-tools count.
    # Drives the "suppression spread" warning, which fires the first
    # time a tool gets suppressed in the current stage.
    last_known_suppression_stage_count: dict[str, int] = Field(default_factory=dict)


class AgentBundle(BaseModel):
    """Concrete agent instances injected into the graph at build time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    recon: BaseAgent
    web_enum: BaseAgent
    net_enum: BaseAgent
    vuln_triage: BaseAgent
    exploit: BaseAgent
    privesc: BaseAgent
    lateral: BaseAgent
    loot: BaseAgent
    report: BaseAgent
    # Shared world-state repo (the council's NoveltyCritic queries it)
    # and the StageRegressionCouncil itself. Both optional so existing test
    # constructions (which never trigger REGRESS_TO) don't need to wire them.
    # The runtime builder in ``worker/runtime.py`` always supplies both.
    repo: InMemoryGraphRepo | None = None
    regression_council: StageRegressionCouncil | None = None
    regression_reflector: Any | None = None


_STALLED_TURN_THRESHOLD = 4

# Max consecutive council-VETO'd regression requests we tolerate from a
# single stage before forcing it to ADVANCE. A VETO never advances
# ``regressions_used`` (only APPROVE does), so without this bound a stage
# whose structural regression-trigger persists re-emits the same VETO'd
# request every step forever (ws-02). After this many consecutive vetoes
# the controller breaks the loop by forcing forward progress.
_REGRESSION_VETO_STAGE_CAP = 3


async def _abort_requested(state: EngagementState) -> bool:
    """Check whether the operator has fired the kill switch.

    The in-memory ``should_abort`` flag is the fast path (set by an
    abort-event consumer that mutates state on the next ``ainvoke``).
    The DB row poll is the durable fallback for cases where the event
    bus dropped its publish or the worker is running detached from a
    long-lived consumer. Either signal flips ``should_abort=True`` and
    seeds ``abort_reason`` for the operator-visible note. Best-effort:
    a Postgres outage during the poll logs and treats the engagement as
    "not aborted" so a database blip doesn't accidentally kill an
    engagement.
    """
    if state.should_abort:
        return True
    try:
        from orchestrator.db import session_scope  # noqa: PLC0415 - local import to avoid cycle
        from orchestrator.models import EngagementRow  # noqa: PLC0415

        async with session_scope() as s:
            row = await s.get(EngagementRow, state.engagement_id)
            if row is not None and row.status == "aborted":
                state.should_abort = True
                state.abort_reason = (
                    getattr(row, "abort_reason", None)
                    or state.abort_reason
                    or "operator-requested abort"
                )
                return True
    except Exception as e:
        log.warning("stage.abort_poll_failed", err=str(e))
    return False


def _check_budgets(state: EngagementState) -> str | None:
    """Return a violation reason if any budget is exceeded; None otherwise.

    Checked: token_budget, wallclock_hours, global stalled-turn watchdog.
    """
    eng = state.engagement
    token_budget = eng.get("token_budget", 0)
    if token_budget and state.total_tokens > token_budget:
        return f"token_budget_exceeded: used {state.total_tokens} > budget {token_budget}"
    # Wallclock: compare current time to window.end_ns (set by build_engagement_context).
    window = eng.get("window") or {}
    end_ns = window.get("end_ns")
    if end_ns:
        import time as _time

        now_ns = _time.time_ns()
        if now_ns > int(end_ns):
            return "wallclock_window_exceeded: current time past engagement end"
    # Same budget shape as the mid-step helper below — keep
    # the wording compatible so the operator-visible note doesn't shift
    # depending on which boundary caught the breach.
    # Global zero-progress watchdog. Halts the engagement when the
    # agent has produced no tool runs AND no graph writes for several
    # consecutive turns. Prevents an unattended run from burning the rest
    # of its token budget on a loop the agent can't escape from.
    if state.stalled_turn_count >= _STALLED_TURN_THRESHOLD:
        return (
            f"stalled_turns_exceeded: {state.stalled_turn_count} consecutive "
            f"agent turns produced zero tool runs and zero graph writes "
            f"(threshold={_STALLED_TURN_THRESHOLD})"
        )
    return None


def _check_budget_mid_step(
    *,
    total_tokens_at_step_start: int,
    accumulated_step_tokens: int,
    token_budget: int,
    wallclock_end_ns: int | None,
) -> str | None:
    """Pure mid-step budget check usable from inside an
    agent's interactive loop.

    The top-of-node ``_check_budgets`` only fires when the LangGraph
    state machine re-enters a node. A single ``step()`` with a large
    ``interactive_llm_turn_cap`` (recon is 20) can chew through many
    per-turn cost units AFTER the engagement is already over budget
    without ever returning to the controller. This helper lets the
    agent check the SAME budgets on every LLM-turn boundary.

    Returns a violation reason (matching the shape of the top-of-step
    notes so audit history stays uniform) or ``None`` when both budgets
    still have headroom.
    """
    if token_budget and (
        total_tokens_at_step_start + accumulated_step_tokens > token_budget
    ):
        return (
            f"token_budget_exceeded_mid_step: used "
            f"{total_tokens_at_step_start + accumulated_step_tokens} "
            f"> budget {token_budget}"
        )
    if wallclock_end_ns is not None:
        import time as _time

        if _time.time_ns() > int(wallclock_end_ns):
            return (
                "wallclock_window_exceeded_mid_step: current time past "
                "engagement end"
            )
    return None


def _step_agent(
    agent: BaseAgent,
    *,
    repo: InMemoryGraphRepo | None = None,
    regression_council: StageRegressionCouncil | None = None,
    regression_reflector: Any | None = None,
):
    """Build the LangGraph node closure for one agent."""

    async def _node(state: EngagementState) -> EngagementState:
        log.info("stage.step", stage=agent.stage.value, engagement_id=str(state.engagement_id))

        # Customer-facing kill switch. The orchestrator's
        # ``POST /engagements/{id}/abort`` flips an engagement-row column
        # AND the in-memory ``should_abort`` flag on this state. Check
        # FIRST so an abort wins against budget-stop / regression-loop /
        # everything else; the operator's manual decision always
        # outranks autonomous progression. Best-effort row poll keeps
        # the abort responsive even when the bus dropped its publish.
        if await _abort_requested(state):
            log.warning(
                "stage.abort_requested",
                stage=agent.stage.value,
                engagement_id=str(state.engagement_id),
                reason=state.abort_reason,
            )
            state.notes.append(
                f"[{agent.stage.value}] ABORT: "
                f"{state.abort_reason or 'operator-requested abort'}"
            )
            state.completed = True
            return state

        # Budget enforcement BEFORE the step runs. Token-budget and
        # wallclock-window violations hard-stop the engagement here; the
        # values accumulate during the step but only halt at this gate.
        budget_violation = _check_budgets(state)
        if budget_violation is not None:
            log.warning(
                "stage.budget_exceeded",
                stage=agent.stage.value,
                reason=budget_violation,
                tokens=state.total_tokens,
                cost_usd=state.total_cost_usd,
            )
            state.notes.append(f"[{agent.stage.value}] BUDGET STOP: {budget_violation}")
            state.completed = True
            return state

        # Stage-boundary reset. If we crossed into a new stage since
        # the last suppression-emitting step, clear the list — a tool that
        # failed terminally in (say) RECON may work fine in LATERAL with
        # different inputs, and the operator may have fixed the underlying
        # config in between.
        if state.suppressed_tools_stage != agent.stage.value:
            if state.suppressed_tools:
                log.info(
                    "stage.suppressed_tools_reset",
                    prev_stage=state.suppressed_tools_stage,
                    new_stage=agent.stage.value,
                    cleared=list(state.suppressed_tools),
                )
            # Shape is dict[name, reason].
            state.suppressed_tools = {}
            state.suppressed_tools_stage = agent.stage.value
            # Soft-delete the prior stage's active
            # suppression rows in the durable table. Stamps
            # ``cleared_at = now()`` so audit history is preserved while
            # the API's active-set query stops returning the cleared
            # rows. A future suppression in the new stage inserts a
            # fresh row with ``cleared_at = NULL``. Best-effort —
            # Postgres unreachable logs and continues.
            try:
                await clear_tool_suppressions(engagement_id=state.engagement_id)
            except Exception as e:
                log.warning("stage.suppressed_tools_clear_failed", err=str(e))

        # The delta-scoped sub-pipeline has looped back to the stage that
        # emitted the original ``REGRESS_TO``. The delta has done its job —
        # clear it so this stage runs with its full scope again and
        # downstream stages don't see leftover narrowing. Must happen
        # BEFORE we build the ``StageContext`` so the agent doesn't see a
        # stale delta on its origin re-entry.
        if (
            state.regression_origin_stage == agent.stage.value
            and state.regression_delta is not None
        ):
            log.info(
                "stage.regression_delta_cleared",
                stage=agent.stage.value,
                regressions_used=state.regressions_used,
            )
            state.regression_delta = None
            state.regression_origin_stage = None
            state.regression_evidence_ids = None

        # Best-effort fetch of approved regression history for the LLM
        # prompt digest. Non-critical; an empty list just omits the
        # "Prior regression activity" section from the prompt.
        try:
            regression_history = await list_approved_regressions(
                engagement_id=state.engagement_id, limit=50
            )
        except Exception as e:
            log.warning("stage.regression_history_fetch_failed", err=str(e))
            regression_history = []

        reflection_rows: list[dict[str, Any]] = []
        try:
            from agents.reflection_memory import list_reflections  # noqa: PLC0415

            reflection_rows = await list_reflections(
                engagement_id=state.engagement_id,
                stage=agent.stage.value,
                limit=10,
            )
        except Exception as e:
            log.warning("stage.reflection_memory_fetch_failed", err=str(e))

        # Per-turn writable state lives on a fresh ``StageOutbox``
        # threaded into the StageContext. We seed the outbox from
        # ``EngagementState`` (per-tool token rolling sum + previous
        # suppression list + recon watchdog counter) so the agent sees
        # the right state on entry; after step() the controller reads
        # ``result.outbox`` to apply the writes.
        from agents.base import StageOutbox  # noqa: PLC0415 - local import to avoid cycle

        outbox = StageOutbox(
            tokens_by_tool=dict(state.tokens_by_tool),
            suppressed_tools=dict(state.suppressed_tools),
            consecutive_zero_host_turns=state.consecutive_zero_host_turns,
        )
        ctx = StageContext(
            engagement_id=state.engagement_id,
            engagement=state.engagement,
            stage=agent.stage,
            turn=state.turn_counts.get(agent.stage.value, 0),
            outbox=outbox,
            # Thread the active regression delta + origin stage so
            # delta-aware agents can narrow their slice. Until
            # an agent opts in, this is informational only.
            regression_delta=(
                dict(state.regression_delta) if state.regression_delta is not None else None
            ),
            regression_origin_stage=state.regression_origin_stage,
            regression_evidence_ids=(
                list(state.regression_evidence_ids)
                if state.regression_evidence_ids is not None
                else None
            ),
            # Thread the live budget counters so an agent
            # that wants to emit ``REGRESS_TO`` can cheap-check the cap
            # before constructing the request. The council's BudgetCritic
            # remains the canonical enforcer; this is producer-side
            # filtering only.
            regressions_used=state.regressions_used,
            # Runaway-loop circuit breaker; 0 = unlimited (the default).
            # Loop safety lives in NoveltyCritic per-fingerprint replay
            # rejection, not this counter — see
            # ``REGRESSION_BUDGET_MAX_DEFAULT`` docstring in models.py.
            regression_budget_max=int(
                state.engagement.get("regression_budget_max", 0) or 0
            ),
            # Snapshot the per-step LLM-turn cap into the ctx
            # so the agent (and tests inspecting ctx) sees the canonical
            # value. ``getattr`` fallback covers test fixtures that
            # define a stand-in agent without inheriting from BaseAgent.
            max_loop_turns=getattr(agent, "interactive_llm_turn_cap", 15),
            # Thread mid-step budget inputs so the
            # interactive loop can break out before burning N more LLM
            # turns of an engagement that's already over budget. The
            # snapshot is taken HERE (before agent.step) so the agent
            # always compares its running step-tokens against the same
            # baseline ``_check_budgets`` used at step entry.
            total_tokens_at_step_start=state.total_tokens,
            token_budget=int(state.engagement.get("token_budget", 0) or 0),
            wallclock_end_ns=(
                int(state.engagement.get("window", {}).get("end_ns"))
                if (state.engagement.get("window") or {}).get("end_ns") is not None
                else None
            ),
            # Thread the abort flag into the ctx so the
            # interactive loop's mid-step check sees the current state.
            # The top-of-step ``_abort_requested`` above already
            # short-circuits when this is set on entry; this thread-
            # through covers the case where an abort lands AFTER the
            # ctx was built (the loop's DB poll then hydrates the
            # ctx-local flag for any further turns).
            should_abort=state.should_abort,
            abort_reason=state.abort_reason,
            regression_history_rows=regression_history,
            reflection_memory=reflection_rows,
        )
        result = await agent.step(ctx)
        # Read writes back from ``result.outbox`` rather than
        # ``ctx``. Both are the same object in practice, but routing
        # through ``result.outbox`` is the new contract: the outbox is
        # the authoritative record of "what happened this turn".
        result_outbox = result.outbox
        state.turn_counts[agent.stage.value] = ctx.turn
        state.total_cost_usd += result.cost_usd
        state.total_tokens += result.tokens_used
        # Persist the watchdog counter the recon agent maintained.
        state.consecutive_zero_host_turns = result_outbox.consecutive_zero_host_turns
        # Global progress watchdog. A turn with no tool runs AND no
        # graph writes is "no progress"; increment the engagement-level
        # stalled counter. Any real work resets it. `_check_budgets` halts
        # the engagement once this passes the threshold.
        if result.tool_runs == 0 and result.nodes_written == 0:
            state.stalled_turn_count += 1
        else:
            state.stalled_turn_count = 0
        # Merge any new per-tool tokens back into engagement state
        for tn, used in result_outbox.tokens_by_tool.items():
            state.tokens_by_tool[tn] = max(state.tokens_by_tool.get(tn, 0), used)
        # Merge newly-suppressed tools back to engagement state.
        # ``outbox.suppressed_tools`` is ``dict[name, reason]``
        # so the reason flows through to ``state`` (and hence to the API +
        # the Overview tab). First-seen reason wins on collision.
        for tname, reason in result_outbox.suppressed_tools.items():
            if tname not in state.suppressed_tools:
                state.suppressed_tools[tname] = reason
        # Solo-operator threshold warnings. First merge the per-turn
        # critic-verdict tape from the outbox into the engagement-level
        # rolling window, then evaluate every threshold. Best-effort —
        # the module catches its own publish/audit errors so a Redis
        # blip never blocks the agent loop.
        try:
            from orchestrator.thresholds import (  # noqa: PLC0415
                VETO_RATE_WINDOW,
                check_thresholds,
            )

            for vetoed in getattr(result_outbox, "critic_verdicts", []):
                state.recent_critic_verdicts.append(bool(vetoed))
            # Cap the window so checkpoint payloads stay small.
            cap = VETO_RATE_WINDOW * 2
            if len(state.recent_critic_verdicts) > cap:
                state.recent_critic_verdicts = state.recent_critic_verdicts[-cap:]
            await check_thresholds(state)
        except Exception as e:
            log.warning("stage.threshold_check_failed", err=str(e))
        state.pending_approvals.extend(result_outbox.pending_approvals)
        if result_outbox.step_promise_score is not None:
            try:
                await audit(
                    engagement_id=state.engagement_id,
                    actor="step_scorer",
                    action="step_score.recorded",
                    payload={
                        "stage": agent.stage.value,
                        "step": ctx.turn,
                        "promise": result_outbox.step_promise_score,
                        "progress": result_outbox.step_progress_score,
                    },
                )
            except Exception as e:
                log.warning("stage.step_score_audit_failed", err=str(e))
        state.notes.append(
            f"[{agent.stage.value} t{ctx.turn}] decision={result.decision.value} "
            f"tools={result.tool_runs} nodes={result.nodes_written}"
        )
        # Snapshot the world-state graph after a step that wrote nodes so the
        # discovered hosts/services/findings/credentials survive a
        # pause/resume. ``save_world_state`` is best-effort (it swallows its
        # own errors), so the agent loop never blocks on the snapshot. We
        # skip the write when ``nodes_written == 0`` to avoid redundant
        # snapshots — the prior step's snapshot already persists.
        #
        # Atomicity: this is a SEPARATE write from the LangGraph checkpoint
        # below (interrupt() / forward transitions); they are not one
        # transaction. Deterministic uuid5 node ids make a slightly-stale
        # snapshot + agent re-run re-converge — eventual consistency, not
        # corruption.
        if repo is not None and result.nodes_written > 0:
            await save_world_state(state.engagement_id, repo.to_json())
        if result.decision == StepDecision.ADVANCE:
            _did_regress = False
            if regression_reflector is not None and regression_council is not None:
                try:
                    _ws_slice = getattr(ctx, "last_world_state_slice", None) or {}
                    _verbal_memory = getattr(ctx, "reflection_memory", []) or []
                    _scope = state.engagement.get("scope", {})
                    _refl_request = await regression_reflector.reflect(
                        engagement_id=state.engagement_id,
                        current_stage=agent.stage.value,
                        world_state_slice=_ws_slice,
                        engagement_scope=_scope,
                        verbal_memory=_verbal_memory,
                    )
                    if _refl_request is not None:
                        log.info(
                            "stage.llm_regression_suggested",
                            stage=agent.stage.value,
                            target=_refl_request.target_stage.value
                            if hasattr(_refl_request.target_stage, "value")
                            else str(_refl_request.target_stage),
                        )
                        _saved_stage = state.current_stage
                        await _handle_regress_to(
                            agent_stage=agent.stage,
                            request=_refl_request,
                            state=state,
                            repo=repo,
                            council=regression_council,
                        )
                        # If the council approved, state.current_stage was
                        # changed to the target. Don't also advance forward.
                        if state.current_stage != _saved_stage:
                            _did_regress = True
                except Exception as e:
                    log.warning("stage.llm_regression_reflection_failed", err=str(e))

            if not _did_regress:
                # Forward progress: clear this stage's consecutive-VETO
                # tally so a regression on a future re-visit starts fresh.
                state.regression_vetoes_by_stage.pop(agent.stage.value, None)
                state.current_stage = _next_stage(agent.stage)
        elif result.decision == StepDecision.ESCALATE_HUMAN:
            state.current_stage = agent.stage
            # Actually pause the graph here. interrupt() throws
            # GraphInterrupt; LangGraph catches it and checkpoints state.
            # The orchestrator's `POST /engagements/{id}/resume` route
            # re-invokes the graph against the same thread_id to resume.
            interrupt(
                {
                    "kind": "approvals_pending",
                    "stage": agent.stage.value,
                    "count": len(result_outbox.pending_approvals),
                    "approvals": result_outbox.pending_approvals,
                }
            )
        elif result.decision == StepDecision.REGRESS_TO:
            # The agent has emitted a structural regression request.
            # Route it through the StageRegressionCouncil, apply the
            # verdict, and write audit + event-bus rows. The helper
            # mutates ``state`` in place and (on ESCALATE) raises
            # ``interrupt()`` to pause the graph.
            _stage_before = state.current_stage
            await _handle_regress_to(
                agent_stage=agent.stage,
                request=result.regression_request,
                state=state,
                repo=repo,
                council=regression_council,
            )
            # ws-02 loop-breaker. If the council VETOed (state.current_stage
            # unchanged) and this stage has now accumulated
            # ``_REGRESSION_VETO_STAGE_CAP`` consecutive vetoes, the agent is
            # stuck re-emitting the same VETO'd request every step (the
            # structural trigger persists and ``regressions_used`` never
            # advances, so no budget cap can fire). Force forward progress
            # so the engagement can't loop indefinitely.
            if (
                state.current_stage == _stage_before
                and state.regression_vetoes_by_stage.get(agent.stage.value, 0)
                >= _REGRESSION_VETO_STAGE_CAP
            ):
                forced_next = _next_stage(agent.stage)
                log.warning(
                    "stage.regression_veto_loop_forced_advance",
                    stage=agent.stage.value,
                    consecutive_vetoes=state.regression_vetoes_by_stage.get(
                        agent.stage.value, 0
                    ),
                    forced_next=forced_next.value,
                    engagement_id=str(state.engagement_id),
                )
                state.notes.append(
                    f"[{agent.stage.value}] regression VETO loop "
                    f"({state.regression_vetoes_by_stage.get(agent.stage.value, 0)} "
                    f"consecutive) — forcing ADVANCE → {forced_next.value} to "
                    f"break no-progress loop"
                )
                try:
                    await audit(
                        engagement_id=state.engagement_id,
                        actor="stage_controller",
                        action="regression.veto_loop_forced_advance",
                        payload={
                            "stage": agent.stage.value,
                            "consecutive_vetoes": state.regression_vetoes_by_stage.get(
                                agent.stage.value, 0
                            ),
                            "forced_next": forced_next.value,
                        },
                    )
                except Exception as e:
                    log.warning("stage.veto_loop_audit_failed", err=str(e))
                # Clear the tally now that we've broken the loop so the
                # destination stage starts clean.
                state.regression_vetoes_by_stage.pop(agent.stage.value, None)
                state.current_stage = forced_next
        # CONTINUE / REGRESS_TO-veto / REGRESS_TO-missing-deps all keep
        # state.current_stage as-is so the router re-enters this node.
        return state

    return _node


async def _handle_regress_to(
    *,
    agent_stage: Stage,
    request: RegressionRequest | None,
    state: EngagementState,
    repo: InMemoryGraphRepo | None,
    council: StageRegressionCouncil | None,
) -> None:
    """Apply the StageRegressionCouncil's verdict to engagement state.

    Extracted from ``_step_agent`` to keep the node closure readable.
    Mutates ``state`` in place; on ESCALATE the function calls
    ``interrupt()`` (LangGraph will propagate it past the caller).

    The flow:

      1. Persist a ``RegressionRow`` (best-effort).
      2. Publish ``regression.requested`` on the event bus + audit log.
      3. If the council / repo isn't wired (unit-test path or missing
         dependency) treat it as VETO with a "not_wired" rationale so
         the run continues without a backwards transition.
      4. Otherwise ``await council.evaluate(...)`` and dispatch on the
         verdict: APPROVE writes the transition + stashes delta;
         VETO leaves state alone; ESCALATE leaves state alone AND
         calls ``interrupt()`` so the operator decides.
      5. Patch the ``RegressionRow`` with the verdict + ``applied_at``
         (APPROVE only).

    The order matters. The audit row exists first so a council-side
    crash leaves a "requested but never decided" trail. The bus
    publishes mirror the audit-log writes for SSE/UI surfaces.
    """
    if request is None:
        # Programmer error: the agent returned REGRESS_TO without a
        # request. Log loudly and keep moving — same shape as CONTINUE
        # so the engagement doesn't deadlock on a broken agent.
        log.warning(
            "stage.regress_to_missing_request",
            stage=agent_stage.value,
            engagement_id=str(state.engagement_id),
        )
        return

    target_stage_str = (
        request.target_stage.value
        if hasattr(request.target_stage, "value")
        else str(request.target_stage)
    )
    trigger_kind_str = (
        request.trigger_kind.value
        if hasattr(request.trigger_kind, "value")
        else str(request.trigger_kind)
    )

    # 1. Persist the requested row first; capture the id for the later
    # verdict patch. ``record_regression`` is best-effort — a Postgres
    # outage returns ``None`` so the in-memory flow can still proceed.
    # The fingerprint is computed here so the audit row carries the
    # exact key the NoveltyCritic will use for replay detection; the
    # council recomputes it during evaluate but the values are
    # deterministic and equal by construction.
    from agents.regression_council import fingerprint_delta  # noqa: PLC0415

    delta_dict = dict(request.delta or {})
    delta_fp = fingerprint_delta(delta_dict)
    row_id = await record_regression(
        engagement_id=state.engagement_id,
        target_stage=target_stage_str,
        trigger_kind=trigger_kind_str,
        delta=delta_dict,
        evidence=list(request.evidence or []),
        rationale=request.rationale,
        delta_fingerprint=delta_fp,
    )

    # 2. Bus publish + audit log: visible to operators immediately even if
    # the council later vetoes or the row write failed.
    req_payload = request.model_dump(mode="json")
    try:
        await get_bus().publish(
            engagement_id=state.engagement_id,
            kind="regression.requested",
            payload=req_payload,
        )
    except Exception as e: 
        log.warning("stage.regression_bus_publish_failed", err=str(e), kind="requested")
    await audit(
        engagement_id=state.engagement_id,
        actor="stage_controller",
        action="regression.requested",
        payload={
            "from_stage": agent_stage.value,
            "target_stage": target_stage_str,
            "trigger_kind": trigger_kind_str,
            "regressions_used": state.regressions_used,
        },
    )

    # 3. Council missing → treat as a no-op VETO so the agent doesn't
    # silently regress without verification. This is the defensive
    # path for unit tests that construct a bundle without a council;
    # production runtime always wires both.
    if council is None or repo is None:
        log.warning(
            "stage.regress_to_council_unavailable",
            stage=agent_stage.value,
            council_wired=council is not None,
            repo_wired=repo is not None,
        )
        if row_id is not None:
            await update_regression_with_verdict(
                row_id=row_id,
                council_decision="veto",
                council_rationale="council_not_wired",
                applied_at=None,
            )
        return

    # 4. Council deliberation. Pass the FULL engagement state — the
    # BudgetCritic reads ``regressions_used`` BEFORE we increment it,
    # so the cap is evaluated at decision time (a council that votes
    # "approve" on the 3rd request would then push the count to 4 only
    # if approved; the budget critic sees 2 going in).
    verdict = await council.evaluate(
        request,
        state,
        state.engagement,
        repo,
        state.engagement_id,
    )

    council_decision_str = (
        verdict.decision.value if hasattr(verdict.decision, "value") else str(verdict.decision)
    )

    # Audit-log the decision regardless of outcome.
    # Stash a per-critic breakdown so the report renderer
    # can show "2 approve / 1 veto" in the Pipeline Regressions section.
    # Each member critique is projected to a small dict — we deliberately
    # drop ``revised_proposal`` to keep the row light + scrub anything that
    # might contain a tool result string.
    member_breakdown = []
    for c in getattr(verdict, "member_critiques", None) or []:
        member_breakdown.append(
            {
                "critic_name": getattr(c, "critic_name", "") or "",
                "verdict": (
                    c.verdict.value if hasattr(c.verdict, "value") else str(c.verdict)
                ),
                "confidence": getattr(c, "confidence", 0.0),
                "primary_concern": getattr(c, "primary_concern", "") or "",
            }
        )
    await audit(
        engagement_id=state.engagement_id,
        actor="stage_regression_council",
        action="regression.council_decided",
        payload={
            "from_stage": agent_stage.value,
            "target_stage": target_stage_str,
            "trigger_kind": trigger_kind_str,
            "decision": council_decision_str,
            "confidence": verdict.confidence,
            "rationale": verdict.rationale,
            # Per-critic breakdown for the report's
            # Pipeline Regressions section.
            "regression_row_id": str(row_id) if row_id is not None else None,
            "member_critiques": member_breakdown,
        },
    )

    if verdict.decision == Verdict.APPROVE:
        # 5a. Approve → commit the backwards transition. Increment the
        # counter AFTER the budget critic has voted; stash the delta and
        # origin stage so the re-entered sub-pipeline scopes correctly.
        # Validate the target stage BEFORE mutating any state — if the
        # string somehow doesn't map to a known ``Stage`` enum value,
        # refuse the whole transition rather than leave a half-committed
        # bump on the counter.

        # Refuse ``post_exploit`` as a target stage.
        # ``post_exploit`` is a no-op pivot — it has no agent and no
        # ``_decide_advance`` predicate that could consult the delta.
        # Landing a regression on it would drop the delta scope on the
        # floor (the next stage would either ignore it or see stale
        # narrowing). Reject at the controller layer so the audit row
        # carries a clear reason and the in-memory flow never bumps
        # the regression counter for a transition that wouldn't fire.
        if target_stage_str == Stage.POST_EXPLOIT.value:
            log.warning(
                "stage.regress_to_post_exploit_rejected",
                from_stage=agent_stage.value,
                target_stage=target_stage_str,
                trigger_kind=trigger_kind_str,
            )
            await audit(
                engagement_id=state.engagement_id,
                actor="stage_controller",
                action="regression.target_stage_invalid",
                payload={
                    "from_stage": agent_stage.value,
                    "target_stage": target_stage_str,
                    "trigger_kind": trigger_kind_str,
                    "reason": "post_exploit_is_not_a_delta_routable_stage",
                },
            )
            if row_id is not None:
                await update_regression_with_verdict(
                    row_id=row_id,
                    council_decision="veto",
                    council_rationale="post_exploit_is_not_a_delta_routable_stage",
                    applied_at=None,
                )
            return

        try:
            target_stage_enum = Stage(target_stage_str)
        except ValueError:
            # Should never happen — the council vetoes unknown stages.
            log.warning(
                "stage.regress_to_invalid_target",
                target_stage=target_stage_str,
            )
            return
        state.regressions_used += 1
        # An approved regression is forward progress for the loop-breaker:
        # reset the consecutive-VETO tally for this stage so a future,
        # genuinely-novel regression from the same stage isn't penalised
        # by vetoes that preceded a successful one.
        state.regression_vetoes_by_stage.pop(agent_stage.value, None)
        state.regression_delta = dict(request.delta or {})
        state.regression_origin_stage = agent_stage.value
        state.regression_evidence_ids = list(request.evidence or [])
        state.current_stage = target_stage_enum
        applied_at = datetime.now(UTC)
        if row_id is not None:
            await update_regression_with_verdict(
                row_id=row_id,
                council_decision=council_decision_str,
                council_rationale=verdict.rationale,
                applied_at=applied_at,
            )
        try:
            await get_bus().publish(
                engagement_id=state.engagement_id,
                kind="regression.applied",
                payload={
                    "request": req_payload,
                    "council_decision": council_decision_str,
                    "regressions_used": state.regressions_used,
                },
            )
        except Exception as e: 
            log.warning("stage.regression_bus_publish_failed", err=str(e), kind="applied")
        await audit(
            engagement_id=state.engagement_id,
            actor="stage_controller",
            action="regression.applied",
            payload={
                "from_stage": agent_stage.value,
                "target_stage": target_stage_str,
                "trigger_kind": trigger_kind_str,
                "regressions_used": state.regressions_used,
            },
        )
        state.notes.append(
            f"[{agent_stage.value}] regression APPROVED → {target_stage_str} "
            f"(trigger={trigger_kind_str}, used={state.regressions_used})"
        )
        return

    if verdict.decision == Verdict.ESCALATE:
        # 5b. Escalate → leave the stage where it is and pause the graph
        # so the operator decides. Matches the ESCALATE_HUMAN shape from
        # the forward path. The row keeps ``applied_at=None``.
        if row_id is not None:
            await update_regression_with_verdict(
                row_id=row_id,
                council_decision=council_decision_str,
                council_rationale=verdict.rationale,
                applied_at=None,
            )
        try:
            await get_bus().publish(
                engagement_id=state.engagement_id,
                kind="regression.escalated",
                payload={
                    "request": req_payload,
                    "council_decision": council_decision_str,
                    "rationale": verdict.rationale,
                },
            )
        except Exception as e: 
            log.warning("stage.regression_bus_publish_failed", err=str(e), kind="escalated")
        state.notes.append(
            f"[{agent_stage.value}] regression ESCALATED ({target_stage_str}): {verdict.rationale}"
        )
        interrupt(
            {
                "kind": "regression_council_escalated",
                "stage": agent_stage.value,
                "target_stage": target_stage_str,
                "trigger_kind": trigger_kind_str,
                "rationale": verdict.rationale,
            }
        )
        return

    # 5c. VETO (or any other non-approve non-escalate). State stays
    # where it is; the agent re-enters this node next routing tick
    # (CONTINUE-equivalent behaviour).
    if row_id is not None:
        await update_regression_with_verdict(
            row_id=row_id,
            council_decision=council_decision_str,
            council_rationale=verdict.rationale,
            applied_at=None,
        )
    try:
        await get_bus().publish(
            engagement_id=state.engagement_id,
            kind="regression.vetoed",
            payload={
                "request": req_payload,
                "council_decision": council_decision_str,
                "rationale": verdict.rationale,
            },
        )
    except Exception as e:
        log.warning("stage.regression_bus_publish_failed", err=str(e), kind="vetoed")
    # Count this VETO against the per-stage loop-breaker. A VETO never
    # bumps ``regressions_used`` and leaves ``current_stage`` unchanged, so
    # the agent re-enters and (while the structural trigger persists)
    # re-emits the same request indefinitely. The caller (``_node``) reads
    # this tally after we return and forces an ADVANCE once it crosses
    # ``_REGRESSION_VETO_STAGE_CAP`` (ws-02).
    state.regression_vetoes_by_stage[agent_stage.value] = (
        state.regression_vetoes_by_stage.get(agent_stage.value, 0) + 1
    )
    state.notes.append(
        f"[{agent_stage.value}] regression VETOED ({target_stage_str}): {verdict.rationale} "
        f"[consecutive vetoes from {agent_stage.value}: "
        f"{state.regression_vetoes_by_stage[agent_stage.value]}]"
    )


def _route(state: EngagementState) -> str:
    if state.current_stage == Stage.REPORT:
        return "report"
    if state.completed:
        return END
    return state.current_stage.value


# Canonical forward stage ordering. Single source of truth so
# ``_next_stage`` and the forward/backward direction check (ws-06) agree.
_STAGE_ORDER = [
    Stage.RECON,
    Stage.ENUMERATION,
    Stage.VULN_TRIAGE,
    Stage.EXPLOIT,
    Stage.POST_EXPLOIT,
    Stage.PRIVESC,
    Stage.LATERAL,
    Stage.LOOT,
    Stage.REPORT,
]


def _stage_index(stage: Stage) -> int:
    """Position of ``stage`` in the canonical forward order."""
    return _STAGE_ORDER.index(stage)


def _next_stage(stage: Stage) -> Stage:
    idx = _STAGE_ORDER.index(stage)
    if idx + 1 < len(_STAGE_ORDER):
        return _STAGE_ORDER[idx + 1]
    return Stage.REPORT


def _make_enumeration_node(*, web_node: Any, net_node: Any) -> Any:
    """Build the ``enumeration`` graph node closure.

    ``web_node`` / ``net_node`` are the already-wired ``_step_agent``
    closures for the web- and net-enumeration agents. Extracted to a
    module-level factory (rather than an inline closure in ``build_graph``)
    so it can be unit-tested directly with stand-in node callables.

    Substage-level checkpoint so a mid-node interrupt+resume (e.g. web_enum
    returns ESCALATE_HUMAN and the operator approves later) doesn't re-run
    web_enum and double-persist its nodes. The substage marker on
    ``EngagementState`` lets us skip an already-``_done`` substage on
    re-entry. The Neo4j ``MERGE``s the repo issues are idempotent on node
    id, but Endpoint / Finding / Credential nodes ship fresh UUIDs every
    parse pass — without this gate a re-run would write duplicates with
    different ids.
    """

    async def _enumeration_node(state: EngagementState) -> EngagementState:
        # On the first entry of a fresh enumeration step the marker is
        # ``None``; flip it to ``web_enum_pending``. On a resume after
        # interrupt the marker may already be ``web_enum_pending``,
        # ``web_enum_done``, ``net_enum_pending`` — we honour whatever
        # it says.
        if state.enumeration_substage is None:
            state.enumeration_substage = "web_enum_pending"

        if state.enumeration_substage == "web_enum_pending":
            await web_node(state)
            # If web_enum interrupts via ESCALATE_HUMAN, we never reach
            # the next line; on resume LangGraph re-enters the node and
            # we run web_enum again. That's safe — the marker is still
            # ``web_enum_pending`` so its writes were never marked done.
            state.enumeration_substage = "web_enum_done"

        if state.enumeration_substage == "web_enum_done":
            # web_enum may have moved ``current_stage`` in one of three
            # ways:
            #   * stayed ENUMERATION (CONTINUE / ESCALATE-resumed) — run
            #     net_enum normally;
            #   * ADVANCED forward (e.g. to vuln_triage) — restore to
            #     ENUMERATION, run net_enum, then re-apply the forward target;
            #   * REGRESSED backward (council APPROVED a REGRESS_TO to an
            #     earlier stage like RECON) — net_enum must NOT run. Re-running
            #     net_enum here would execute a full enumeration step against
            #     stale state with a ``regression_delta`` it never requested,
            #     wasting work and re-doing what the regression is unwinding
            #     (ws-06). Skip the net_enum substage and let the router carry
            #     the regression target through.
            regressed_backward = (
                state.current_stage != Stage.ENUMERATION
                and _stage_index(state.current_stage) < _stage_index(Stage.ENUMERATION)
            )
            if regressed_backward:
                log.info(
                    "stage.enumeration.net_enum_skipped_on_regress",
                    target_stage=state.current_stage.value,
                    engagement_id=str(state.engagement_id),
                    note="web_enum regressed backward; skipping net_enum substage",
                )
            else:
                # If web_enum advanced FORWARD past enumeration, restore so
                # net_enum can run, then re-apply the forward target.
                saved_next: Stage | None = None
                if state.current_stage != Stage.ENUMERATION:
                    saved_next = state.current_stage
                    state.current_stage = Stage.ENUMERATION
                state.enumeration_substage = "net_enum_pending"
                await net_node(state)
                state.enumeration_substage = "net_enum_done"
                if saved_next is not None and state.current_stage == Stage.ENUMERATION:
                    state.current_stage = saved_next

        # Step complete — clear the marker so the next entry (a fresh
        # enumeration step driven by ADVANCE/CONTINUE routing) starts
        # over from web_enum_pending.
        state.enumeration_substage = None
        return state

    return _enumeration_node


def build_graph(bundle: AgentBundle, checkpointer: Any | None = None):
    """Compose the LangGraph for the stage state machine.

    Each stage is a node; conditional routing reads `current_stage` from state.
    Stages with no agent yet (EXPLOIT, POST_EXPLOIT, PRIVESC, LATERAL, LOOT,
    REPORT) route to a placeholder that marks the engagement completed so the
    loop terminates after triage.

    `checkpointer`: optional LangGraph checkpoint saver. When set,
    `interrupt()` calls durably pause the graph and `ainvoke(None, ...)`
    resumes from the pause.
    """
    g: StateGraph = StateGraph(EngagementState)
    # Thread the shared repo + StageRegressionCouncil into every
    # node so REGRESS_TO can be verified anywhere in the pipeline.
    repo = bundle.repo
    council = bundle.regression_council
    reflector = bundle.regression_reflector

    def _wire(a: BaseAgent) -> Any:
        return _step_agent(
            a,
            repo=repo,
            regression_council=council,
            regression_reflector=reflector,
        )

    g.add_node("recon", _wire(bundle.recon))

    # Run WebEnumAgent + NetEnumAgent sequentially inside the `enumeration`
    # node — both share Stage.ENUMERATION. Web runs first because it's faster;
    # net runs after so AD-host metadata is in place.
    g.add_node(
        "enumeration",
        _make_enumeration_node(
            web_node=_wire(bundle.web_enum),
            net_node=_wire(bundle.net_enum),
        ),
    )
    g.add_node("vuln_triage", _wire(bundle.vuln_triage))
    g.add_node("exploit", _wire(bundle.exploit))
    g.add_node("privesc", _wire(bundle.privesc))
    g.add_node("lateral", _wire(bundle.lateral))
    g.add_node("loot", _wire(bundle.loot))

    async def _report_node(state: EngagementState) -> EngagementState:
        log.info("stage.report.start", engagement_id=str(state.engagement_id))
        # ReportAgent's build_draft does the actual work; we still call _step
        # to keep telemetry consistent.
        await _wire(bundle.report)(state)
        state.completed = True
        return state

    g.add_node("report", _report_node)

    # POST_EXPLOIT is a no-op pivot: a straight-line edge to privesc. The
    # canonical-order helper stays intact without an extra hop.
    #
    # ``_validate_target_stage_for_regression`` refuses post_exploit as a
    # delta-routable target at the council layer (reason
    # ``"post_exploit_is_not_a_delta_routable_stage"``). Belt-and-braces here:
    # if a delta + origin_stage still reach this node, clear them and log
    # loudly so the privesc agent doesn't act on stale narrowing.
    async def _post_exploit_node(state: EngagementState) -> EngagementState:
        if state.regression_delta is not None or state.regression_origin_stage is not None:
            log.warning(
                "stage.post_exploit_regression_delta_dropped",
                engagement_id=str(state.engagement_id),
                regression_delta=state.regression_delta,
                regression_origin_stage=state.regression_origin_stage,
                note=(
                    "delta + origin reached post_exploit — should have been "
                    "rejected by council. Dropping to avoid stale narrowing in privesc."
                ),
            )
            state.regression_delta = None
            state.regression_origin_stage = None
            state.regression_evidence_ids = None
        state.current_stage = Stage.PRIVESC
        state.notes.append("[post_exploit] pivot -> privesc")
        return state

    g.add_node("post_exploit", _post_exploit_node)
    g.add_edge("post_exploit", "privesc")

    g.add_conditional_edges(
        START,
        lambda s: s.current_stage.value,
        {
            "recon": "recon",
            "enumeration": "enumeration",
            "vuln_triage": "vuln_triage",
            "exploit": "exploit",
            "post_exploit": "post_exploit",
            "privesc": "privesc",
            "lateral": "lateral",
            "loot": "loot",
            "report": "report",
        },
    )

    # post_exploit has a straight edge to privesc (above), so it's
    # excluded from the conditional-edge wiring here.
    for node in ("recon", "enumeration", "vuln_triage", "exploit", "privesc", "lateral", "loot"):
        g.add_conditional_edges(
            node,
            _route,
            {
                "recon": "recon",
                "enumeration": "enumeration",
                "vuln_triage": "vuln_triage",
                "exploit": "exploit",
                "post_exploit": "post_exploit",
                "privesc": "privesc",
                "lateral": "lateral",
                "loot": "loot",
                "report": "report",
                END: END,
            },
        )
    g.add_edge("report", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
