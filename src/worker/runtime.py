"""Worker runtime — picks engagements from the queue and runs the stage graph."""

from __future__ import annotations

import asyncio
import contextlib
from uuid import UUID

import structlog

from agents.council import triage_council
from agents.critic import PreExploitCritic, PreLateralCritic, ReportCritic
from agents.exploit import ExploitAgent
from agents.lateral import LateralAgent
from agents.loot import LootAgent
from agents.net_enum import NetEnumAgent
from agents.poc_runtime import PoCDispatcher, PoCLibrary
from agents.privesc import PrivEscAgent
from agents.recon import ReconAgent
from agents.regression_council import build_regression_council
from agents.report import ReportAgent
from agents.vuln_triage import VulnTriageAgent
from agents.web_enum import WebEnumAgent
from orchestrator.settings import Settings, get_settings
from orchestrator.stage_controller import AgentBundle, EngagementState, build_graph
from policy.engine import PolicyEngine
from sandbox.runner import SandboxRunner
from tools import poc_exec as poc_exec_tool
from tools.base import build_default_registry
from worldstate.neo4j_repo import InMemoryGraphRepo
from worldstate.schema import Stage

log = structlog.get_logger(__name__)

# Upper bound on the in-loop checkpoint existence probe (async3-01 follow-up).
# Keeps a slow/locked checkpointer from wedging the worker dispatch loop.
_HAS_CHECKPOINT_TIMEOUT_S = 5.0


def state_flag(state: object, name: str) -> bool:
    """Read a boolean EngagementState flag from whatever ``graph.ainvoke``
    returned (async4-01).

    LangGraph's ``ainvoke`` returns the final state as a plain ``dict`` of
    channel values, NOT an ``EngagementState`` instance — so ``getattr(state,
    "completed", False)`` silently reads ``False`` for EVERY run, making the
    worker classify completed runs as ``paused`` (network leak + endless
    resume). Read by key when the state is a mapping, by attribute otherwise,
    so terminal detection is correct regardless of the concrete return type.
    """
    if state is None:
        return False
    if isinstance(state, dict):
        return bool(state.get(name, False))
    return bool(getattr(state, name, False))


async def build_bundle(
    settings: Settings | None = None,
    *,
    snapshot: dict | None = None,
) -> tuple[AgentBundle, InMemoryGraphRepo, PolicyEngine]:
    s = settings or get_settings()
    # Prefer the engagement's snapshotted LLM config when one
    # was provided; otherwise fall back to env defaults (Settings).
    from llm.router import build_router_async

    router = await build_router_async(s, snapshot=snapshot)
    # Cross-check graph-write provenance against Postgres.
    from orchestrator.provenance import postgres_provenance_validator

    repo = InMemoryGraphRepo(
        provenance_validator=postgres_provenance_validator,
    )
    policy = PolicyEngine(s)
    sandbox = SandboxRunner()
    registry = build_default_registry()
    library = PoCLibrary()
    dispatcher = PoCDispatcher(library, sandbox)
    # Wire the singletons that CuratedPoCExec reaches via module-level state.
    poc_exec_tool.configure(library, dispatcher)

    council = triage_council(router)
    pre_exploit = PreExploitCritic(router)
    pre_lateral = PreLateralCritic(router)
    report_critic = ReportCritic(router)

    # Feature-flagged step scoring and reflection memory.
    step_scorer = None
    reflection_memory = None
    if s.enable_step_scoring:
        from agents.step_scorer import StepScorer  # noqa: PLC0415

        step_scorer = StepScorer(router)
    if s.enable_reflection_memory:
        from agents.reflection_memory import ReflectionMemory  # noqa: PLC0415

        reflection_memory = ReflectionMemory(router)

    # Phase 4 feature flags.
    regression_reflector = None
    if s.enable_regression_reflection:
        from agents.regression_reflection import RegressionReflector  # noqa: PLC0415

        regression_reflector = RegressionReflector(router)

    shared = {
        "router": router,
        "repo": repo,
        "registry": registry,
        "policy": policy,
        "sandbox": sandbox,
        "step_scorer": step_scorer,
        "reflection_memory": reflection_memory,
    }
    # One regression
    # council per worker. Reused across engagements -- per-call context
    # is threaded through ``StageRegressionCouncil.evaluate``. The stage
    # controller pulls it off the bundle in ``build_graph`` and threads
    # it into every node's ``_step_agent`` closure so any REGRESS_TO
    # signal gets verified.
    regression_council = build_regression_council(router)

    bundle = AgentBundle(
        recon=ReconAgent(**shared),
        web_enum=WebEnumAgent(**shared),
        net_enum=NetEnumAgent(**shared),
        vuln_triage=VulnTriageAgent(triage_council=council, **shared),
        exploit=ExploitAgent(
            library=library,
            pre_exploit_critic=pre_exploit,
            enable_debate_gates=s.enable_debate_gates,
            **shared,
        ),
        privesc=PrivEscAgent(**shared),
        lateral=LateralAgent(
            pre_lateral_critic=pre_lateral,
            enable_debate_gates=s.enable_debate_gates,
            **shared,
        ),
        loot=LootAgent(**shared),
        report=ReportAgent(report_critic=report_critic, **shared),
        repo=repo,
        regression_council=regression_council,
        regression_reflector=regression_reflector,
    )
    return bundle, repo, policy


async def has_checkpoint(engagement_id: UUID) -> bool:
    """Return True if a LangGraph checkpoint exists for this engagement.

    Used by the worker dispatch loop to decide whether a re-dispatch should
    ``resume=True`` (continue past a prior interrupt/pause) instead of
    restarting the graph from a fresh ``EngagementState``. A checkpoint is
    written whenever the graph runs with the durable checkpointer wired, so
    an engagement that previously paused on ``interrupt()`` has one.

    Best-effort and BOUNDED: if the checkpoint store can't initialize, the
    lookup errors, or it does not answer within ``_HAS_CHECKPOINT_TIMEOUT_S``,
    return ``False`` so dispatch falls back to a fresh run rather than
    crashing — or *wedging* — the poll loop. The timeout matters: this runs
    inside the worker's dispatch loop, so a slow/locked checkpointer must
    never be able to stall dispatch indefinitely (it would freeze the whole
    worker). ``resume=False`` is the safe fallback.
    """
    try:
        from orchestrator.checkpoint import get_checkpointer, thread_id_for  # noqa: PLC0415

        async def _lookup() -> bool:
            checkpointer = await get_checkpointer()
            config = {"configurable": {"thread_id": thread_id_for(engagement_id)}}
            tup = await checkpointer.aget_tuple(config)
            return tup is not None

        return await asyncio.wait_for(_lookup(), timeout=_HAS_CHECKPOINT_TIMEOUT_S)
    except Exception as e:
        log.warning("checkpoint.lookup_failed", engagement_id=str(engagement_id), err=str(e))
        return False


async def run_engagement(engagement_id: UUID, resume: bool = False) -> EngagementState:
    """Drive (or resume) the stage graph for an engagement.

    First invocation: build engagement-dict, set up the per-engagement sandbox
    network, invoke the graph from a fresh initial state.

    Resume invocation: rehydrate from the LangGraph SQLite checkpoint;
    LangGraph routes back to the paused interrupt point and continues.
    """
    from orchestrator.checkpoint import get_checkpointer, thread_id_for
    from orchestrator.db import session_scope
    from orchestrator.models import EngagementRow
    from policy.engagement_dict import build_engagement_context
    from sandbox.network import ensure_engagement_network, teardown_engagement_network

    async with session_scope() as s:
        row = await s.get(EngagementRow, engagement_id)
        if row is None:
            raise ValueError(f"engagement not found: {engagement_id}")
        engagement = build_engagement_context(row)
        # Pull the engagement's LLM-config snapshot so the
        # router is built against the *creator's* config, not whatever
        # the global env happens to be right now.
        snapshot = getattr(row, "llm_config_snapshot", {}) or {}

    # Spin up the abort-event consumer BEFORE
    # ``ensure_engagement_network`` + ``ensure_templates_fresh``. That
    # setup work can take seconds (docker network create, nuclei +
    # searchsploit DB updates), and a ``from_id="$"`` subscriber that
    # started AFTER would silently miss any abort published in the
    # window. The consumer only needs the bus client + the registry
    # singleton -- both available immediately. Combined with the one-time
    # DB poll inside ``run_abort_consumer`` at startup, this closes the
    # gap end-to-end: starting early collapses it from seconds to the ms
    # it takes the subscriber to issue its first xread, and the DB poll
    # catches any abort that landed in that residual ms-window.
    from orchestrator.abort_consumer import start_abort_consumer  # noqa: PLC0415
    from sandbox.runner import SandboxRunner as _Runner  # noqa: PLC0415

    # Build a runner instance bound to this engagement so the consumer's
    # ``terminate`` calls operate against the same docker context as the
    # agents (network attachment + nftables ruleset). Cheap to construct
    # -- the runner is a thin wrapper over subprocess invocations.
    abort_runner = _Runner(engagement_id=str(engagement_id))
    abort_task = start_abort_consumer(
        engagement_id, runner=abort_runner
    )

    final: EngagementState | None = None
    interrupted = False
    failed = False
    repo = None
    policy = None
    try:
        network = await ensure_engagement_network(engagement_id)
        log.info("engagement.network.ready", network=network, engagement_id=str(engagement_id))

        # Refresh nuclei templates + searchsploit DB at
        # engagement start, throttled to 6h. Best-effort; failure logged.
        from sandbox.template_update import ensure_templates_fresh

        if not resume:
            await ensure_templates_fresh()

        bundle, repo, policy = await build_bundle(snapshot=snapshot)
        # Rehydrate the world-state snapshot on resume. ``build_bundle``
        # returns a FRESH empty repo; on resume we load the last-persisted
        # graph and populate that repo IN PLACE so the bundle's agents (which
        # share this exact repo object) see the rehydrated hosts/services/
        # findings/credentials instead of starting empty. ``load_json``
        # decrypts the credential secret fields back to plaintext.
        #
        # This snapshot and the LangGraph checkpoint loaded below are
        # SEPARATE reads (not one transaction). Deterministic uuid5 node ids
        # make a slightly-stale snapshot + agent re-run re-converge —
        # eventual consistency, not corruption.
        if resume and repo is not None:
            from orchestrator.world_state_store import load_world_state  # noqa: PLC0415

            data = await load_world_state(engagement_id)
            if data:
                repo.load_json(data)
        # Wire the durable checkpointer if available; fall back to default
        # in-memory if the checkpoint extra can't initialize (e.g. unit tests).
        checkpointer = None
        try:
            checkpointer = await get_checkpointer()
        except Exception as e:
            log.warning("checkpoint.disabled", err=str(e))
        graph = build_graph(bundle, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id_for(engagement_id)}}

        if resume:
            final = await graph.ainvoke(None, config=config)
        else:
            initial = EngagementState(
                engagement_id=engagement_id,
                engagement=engagement,
                current_stage=Stage.RECON,
            )
            final = await graph.ainvoke(initial, config=config)
        # If the graph paused due to interrupt(), `final` may be `None` or
        # the partial state -- LangGraph signals interrupt via its own
        # mechanism; we detect by checking that the final state isn't
        # marked completed.
        if final is None or not state_flag(final, "completed"):
            interrupted = True
        return final  # type: ignore[return-value]
    except asyncio.CancelledError:
        # Worker shutdown cancelled this run mid-flight. Treat it like an
        # interrupt-paused run (the engagement stays resumable; keep the
        # network/abort signal) and let cancellation propagate.
        interrupted = True
        raise
    except Exception:
        # The graph (or setup) raised. Record a terminal ``failed`` status so
        # the dispatch loop does NOT re-select and re-run this engagement from
        # scratch on the next poll (asyncN-01), then re-raise so the loop's
        # reaper still surfaces the exception.
        failed = True
        raise
    finally:
        # Cancel the abort consumer task before tearing down the rest
        # of the runtime; otherwise the subscribe loop holds the Redis
        # connection open and the worker shutdown stalls.
        abort_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await abort_task
        # Drop the cooperative abort-signal entry and the per-engagement
        # threshold lock for this engagement so ``_abort_events`` /
        # ``thresholds._THRESHOLD_LOCKS`` don't leak one entry per engagement
        # on a long-lived worker (async-03/async-08/async3-03). The bus
        # consumer is already cancelled above, so nothing is waiting on the
        # event anymore.
        #
        # This runs on the interrupt/pause path TOO (async3-03): a paused
        # engagement that lingers for the worker's lifetime must not pin its
        # abort event + threshold lock forever. Eviction is safe across a
        # resume — both primitives are rebuilt lazily on next access
        # (``request_abort``/``is_abort_requested`` create the Event on
        # demand; ``_threshold_lock`` creates the Lock on demand), and a
        # paused run by definition saw no abort (an abort would have made the
        # terminal status ``aborted``, not ``paused``), so there is no set
        # abort flag to lose. Best-effort.
        with contextlib.suppress(Exception):
            from orchestrator.abort_consumer import clear_abort  # noqa: PLC0415

            clear_abort(engagement_id)
        with contextlib.suppress(Exception):
            from orchestrator.thresholds import (  # noqa: PLC0415
                release_engagement_lock,
            )

            release_engagement_lock(engagement_id)
        # ``repo`` / ``policy`` may be unset if ``build_bundle`` raised
        # before assigning them. The abort consumer spawns above the
        # setup work, so the try-block covers
        # ``ensure_engagement_network`` and ``build_bundle`` too -- guard
        # the cleanup so a startup failure still tears down cleanly.
        if repo is not None:
            await repo.close()
        if policy is not None:
            await policy.aclose()
        # Only tear down the per-engagement network when the engagement is
        # genuinely done; an interrupt-paused run keeps it for resume.
        if not interrupted:
            await teardown_engagement_network(engagement_id)

        # Persist the terminal/parked status so the worker dispatch loop stops
        # re-selecting this ``running`` row and re-dispatching it from a fresh
        # state every poll (asyncN-01). Determine the outcome:
        #   * exception  -> ``failed``
        #   * abort seen -> ``aborted``
        #   * interrupt  -> ``paused`` (resumable; keeps network/abort signal)
        #   * otherwise  -> ``completed``
        # ``record_terminal_status`` never clobbers an already-terminal row
        # (e.g. an operator abort that landed mid-run wins over a late
        # ``completed``/``paused``). Best-effort: a write failure must not mask
        # the run's own exception, so it's suppressed.
        with contextlib.suppress(Exception):
            from orchestrator.services.engagements import (  # noqa: PLC0415
                record_terminal_status,
            )

            if failed:
                terminal = "failed"
            elif final is not None and state_flag(final, "should_abort"):
                terminal = "aborted"
            elif interrupted:
                terminal = "paused"
            else:
                terminal = "completed"
            await record_terminal_status(engagement_id, terminal)
