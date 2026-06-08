# Architecture

Runtime reference for the internals. Updated in the same PR as any
architectural change.

For day-to-day use see [TUI.md](./TUI.md) and the project
[README](../README.md).

---

## Runtime topology

```
serenade (single Python process)
  |
  |-- TUI (Textual, in-process)
  |     '-- calls service functions directly (no HTTP)
  |
  |-- Stage controller (LangGraph state machine, in-process)
  |     '-- drives 9-stage pipeline: recon -> report
  |
  |-- Agents (per-stage, in-process)
  |     '-- canonical loop: read -> plan -> policy -> critic -> execute -> parse -> write
  |
  |-- Policy engine (Python, in-process)
  |     '-- scope + risk + time + destructive checks
  |
  |-- Event bus (asyncio pub/sub, in-process)
  |     '-- TUI subscribes for live updates
  |
  |-- LLM router (in-process)
  |     '-- Ollama (local, default) or Anthropic API
  |
  |-- Data layer (all local)
  |     |-- SQLite via aiosqlite (~/.serenade/serenade.db)
  |     |-- networkx DiGraph (in-memory; snapshotted to world_state_snapshots, rehydrated on resume)
  |     '-- filesystem (~/.serenade/evidence/)
  |
  '-- Sandbox (external, ephemeral)
        '-- docker run --rm serenade/sandbox:dev <tool> <args>
```

### Components

| Component | Implementation | Where |
|---|---|---|
| TUI | Textual app | in-process |
| Stage controller | LangGraph FSM | in-process |
| Agents | Python classes (BaseAgent subclasses) | in-process |
| Policy engine | Python function (`src/policy/scope.py`) | in-process |
| Event bus | asyncio queues | in-process |
| LLM router | httpx to Ollama or Anthropic | in-process |
| DB | SQLite via aiosqlite | `~/.serenade/serenade.db` |
| World-state graph | networkx DiGraph | in-memory during a run; snapshotted to `world_state_snapshots` (SQLite) after steps, rehydrated on resume |
| Evidence store | filesystem (pathlib) | `~/.serenade/evidence/` |
| Checkpoint | SQLiteCheckpointer (LangGraph BaseCheckpointSaver) | same SQLite DB |
| Sandbox containers | `docker run --rm` per tool call | ephemeral Docker containers |
| Local LLM | Ollama (`ollama serve`) | separate host process |
| Windows worker | Optional PowerShell agent (Redis IPC) | external VM |

---

## Layered architecture

```
+-----------------------------------------------------------+
|  L5  TUI / Approvals                                      |
+-----------------------------------------------------------+
|  L4  Stage controller (LangGraph state machine)           |
|      +- rule-based forward transitions on world-state     |
|      '- council-verified backwards transitions (regress)  |
+-----------------------------------------------------------+
|  L3  Agents (per stage)                                   |
|      '- canonical interactive loop: read -> plan -> policy|
|         -> critic -> execute -> parse -> write -> maybe   |
|         regress                                           |
+-----------------------------------------------------------+
|  L2  Gates                                                |
|      +- Scope/risk policy (deterministic Python)          |
|      +- Critic / Council (devil's advocate)               |
|      +- StageRegressionCouncil (backwards transitions)    |
|      '- Human approval queue (operator TUI)               |
+-----------------------------------------------------------+
|  L1  Capabilities                                         |
|      +- LLMRouter (anthropic | ollama)                    |
|      +- Tool registry + typed adapters                    |
|      '- Sandbox runner (docker run -> tool binary)        |
+-----------------------------------------------------------+
|  L0  State                                                |
|      +- World-state graph (networkx, deterministic UUIDv5)|
|      +- Evidence blobs (filesystem, content-addressed)    |
|      +- Engagement / audit / approvals / suppressed_tools |
|      |   / regressions (SQLite)                           |
|      '- Event bus + checkpoints (asyncio + SQLite)        |
+-----------------------------------------------------------+
```

A layer may depend only on layers below it. Agents (L3) import
`LLMRouter` (L1) and `Tool` (L1) but never import other agents. The
stage controller (L4) imports agents; agents never import the stage
controller.

---

## The canonical agent loop

`src/agents/base.py::BaseAgent.step()`. Every stage agent inherits this.
The default is the **interactive multi-turn loop**
(`_step_interactive`): one call to `step()` runs up to
`interactive_llm_turn_cap` LLM turns (default 15), feeding tool results
back to the model between turns so it can chain decisions. The legacy
single-shot path (`_step_batched`) is preserved behind
`SERENADE_BATCHED_AGENT_LOOP=1` as a rollback switch -- both share
identical gate logic, audit rows, and `_decide_advance` semantics.

```
                    +------------------+
                    | read world-state |   queries.py (deterministic)
                    |      slice       |   delta-aware on regression
                    +--------+---------+
                             |
                             v
                    +--------v---------+
                    |  LLM PLANNER     |   CallSite.PLANNER
                    |  emit tool call  |   structured (tool-use or
                    |                  |    local constrained decoding)
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  schema validate |   Pydantic Input.model_validate
                    +--------+---------+
                             |
                             v
                    +--------v---------+
                    |  scope/risk      |   Python policy engine
                    |  policy gate     |   scope + risk + time
                    +--------+---------+
            DENY <-----------+
                             | ALLOW or REQUIRE_APPROVAL
                             v
                    +------------------+
                    |  Critic/Council  |   CallSite.CRITIC
                    |   gate (only     |   only for high-stakes
                    |   when risk-     |   risk classes
                    |   class flagged) |
                    +--------+---------+
            VETO <-----------+--->  approvals queue (auditable, never silent)
            ESCALATE/REVISE -+--->  approvals queue
                             | APPROVE
                             v
                    +------------------+
                    |   sandbox exec   |   docker run --rm -> tool binary
                    +--------+---------+
                             | stdout/stderr -> evidence blob
                             v
                    +------------------+
                    |   parse output   |   typed parser -> Pydantic Output
                    +--------+---------+
                             |
                             v
                    +--------v---------+
                    |  write world-    |   nodes with provenance=tool_run_id
                    |  state (networkx)|   id = uuid5(ns, kind:eng:nat_key)
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  step scoring    |   (opt-in) StepScorer: promise +
                    |  (AgentPRM)      |   progress scores via E4B.
                    |                  |   Backtrack if both < threshold.
                    +--------+---------+
                             |
                             v
                    +--------v---------+  CONTINUE | ADVANCE | ESCALATE_HUMAN
                    |  decide advance  |  | REGRESS_TO (deterministic post-
                    +------------------+    step trigger, council-verified)
```

Properties:

- **Single-writer per engagement.** One engagement per process;
  avoids write races on the in-memory graph.
- **Provenance is mandatory.** `InMemoryGraphRepo.upsert_node()` refuses
  any node without a `tool_run_id` in `provenance`. Hallucinated findings
  cannot land in the graph by construction.
- **Deterministic node IDs.** For `Endpoint`, `Finding`, `Credential`,
  `ExploitCandidate`, `AttackPath`, `Host`, `Service`, `WebApp`, and
  `ExploitAttempt`, the node `id` is
  `uuid5(zero_namespace, "<kind>:<engagement_id>:<natural_key>")` -- see
  `src/worldstate/schema.py::deterministic_node_id`. Same natural key ->
  same UUID, so upsert-on-id deduplicates idempotent re-parses across
  retry, resume, and re-runs.
- **Three independent gates.** Policy (declarative, deterministic), critic
  (LLM judgment, structured), human (operator click). Each can block
  independently. Critic VETO is *not* silently swallowed: it routes
  through `_resolve_approval_gate` and lands an `approvals` row plus an
  event.
- **Schema-constrained LLM I/O.** Agents never run free-form shell.
  Tool calls are JSON-Schema validated; invalid -> retried with the
  validation error fed back, then escalated.
- **Tool suppression.** A tool that returns `terminal_error=True` is
  added to the engagement's per-stage suppression set; the planner does
  not see it again until the stage boundary clears the set. Suppression
  is a first-class row in SQLite (`suppressed_tools` table) so both
  the agent (write) and the TUI (read) speak the same source of truth.

### StageContext + StageOutbox

`BaseAgent.step()` takes a `StageContext` -- the read-only view of
engagement state for this step -- and writes back through a separate
`StageOutbox` carried on the result. The split (see `src/agents/base.py`)
disambiguates inputs (frozen on entry) from per-turn deltas that the
controller must merge back into `EngagementState`. Outbox fields cover
per-tool token totals, newly-suppressed tools (with reasons), the recon
zero-host watchdog counter, any pending approvals queued during the
step, and `critic_verdicts: list[bool]` -- the per-turn verdict tape from
`BaseAgent`'s pre-action critic gate (True = VETO). The controller folds
this into `EngagementState.recent_critic_verdicts` so the threshold
monitor can compute the veto-rate over a rolling window (see [Threshold monitor](#threshold-monitor)).

### Stalled-turn and zero-host watchdogs

Three operator-facing guards live in the controller:

- **Recon zero-host watchdog**: after 3 consecutive recon turns with
  zero hosts + zero services, the agent emits `ESCALATE_HUMAN` so a
  firewalled scope fails loud instead of looping to `max_turns`.
- **Global stalled-turn watchdog**: 4 consecutive agent steps with zero
  `tool_runs` and zero `nodes_written` flips `EngagementState.completed`
  and lands a `BUDGET STOP` note. Stops an unattended run from burning
  the token cap on a no-progress loop.
- **Threshold monitor** (`src/orchestrator/thresholds.py`) -- fires
  `engagement.{budget_warning,veto_rate_warning,suppression_spread}` once
  per threshold per engagement; visible in event bus, audit log, and TUI
  event stream. See [Threshold monitor](#threshold-monitor) for the full mechanism.

---

## Data layer

### SQLite -- relational, lifecycle

All relational state lives in `~/.serenade/serenade.db` (WAL mode
enabled for concurrent reads during TUI polling).

- `engagements` -- scope, `mode` (single axis; allowed risk classes +
  approval behaviour are derived from it), time window, authorization
  artifact hash, status, `llm_config_snapshot` (JSON), `abort_reason`
- `tool_runs` -- every sandbox execution: input, sandbox id, blob refs,
  exit code, timing
- `llm_calls` -- every LLM call: model, site, token counts, cost, latency
- `evidence` -- index of evidence blobs: kind, sha256, key, redacted flag
- `audit_log` -- append-only, tamper-evident via a per-engagement HMAC
  hash chain. Every gate decision + state transition: `actor`, `action`,
  `payload`, `ts`, plus the chain columns `seq` / `prev_hash` / `row_hash`
  -- see [Audit log integrity](#audit-log-integrity).
- `approvals` -- operator-pending decisions: kind, payload, decision,
  `decided_by`, `decided_at`
- `suppressed_tools` -- durable per-engagement tool-mute rows; active
  rows have `cleared_at IS NULL`. Read by the Overview tab.
- `reflection_memory` -- strategy-level lessons (opt-in Reflexion memory).
- `regressions` -- one row per `StepDecision.REGRESS_TO` request, with
  council verdict and `applied_at` patched in by `_handle_regress_to`.
- `world_state_snapshots` -- one row per engagement (PK = `engagement_id`)
  holding the node-link serialization of the world-state graph
  (`InMemoryGraphRepo.to_json()`, Credential secret fields Fernet-encrypted
  at rest). UPSERTed after every agent step that wrote nodes and rehydrated
  into the bundle's repo on resume (see [networkx
  persistence](#networkx----graph-world-state)).
- `checkpoints`, `checkpoint_writes` -- LangGraph durable state
  (`EngagementState`) so `interrupt()` survives process restarts. Created
  as raw tables by `src/orchestrator/checkpoint.py` (not ORM models). The
  networkx world-state graph is **not** in this payload -- it has its own
  `world_state_snapshots` table (above). The snapshot write and the
  checkpoint write are separate (non-transactional); deterministic `uuid5`
  node ids make a slightly-stale snapshot + agent re-run re-converge.

(The ORM models in `src/orchestrator/models.py` are exactly: `engagements`,
`world_state_snapshots`, `tool_runs`, `evidence`, `llm_calls`, `audit_log`,
`suppressed_tools`, `approvals`, `reflection_memory`, `regressions`. There is
no `step_scores`, `findings_review`, or `credentials` table.)

Schema lifecycle: **alembic** is canonical. `serenade --migrate` applies
pending revisions; `alembic revision --autogenerate -m "..."` authors
from ORM diffs.

### networkx -- graph, world-state

In-memory `DiGraph` wrapped by `InMemoryGraphRepo` with secondary
indices for O(1) lookups by kind and indexed fields.

Every node carries `engagement_id`, `provenance` (list of
`tool_run_id`), `confidence`, `first_seen`, `last_seen`. Node types:
`Target, Host, Service, WebApp, Endpoint, Credential, Identity, Domain,
Finding, PotentialFinding, ExploitCandidate, ExploitAttempt, Session,
AttackPath`. Edge types: `RUNS_ON, OF_WEBAPP, BELONGS_TO, MEMBER_OF,
HAS_SESSION_ON, LOCAL_ADMIN_ON, LEADS_TO, PIVOTED_VIA, AFFECTS, EXPLOITS,
YIELDED, EVIDENCED_BY, DISCOVERED_BY`.

Node `id` is `uuid5(zero_namespace, "<kind>:<engagement>:<natural_key>")`
for kinds with a stable natural key -- see
`src/worldstate/schema.py::natural_key_for_*`. This is what lets
re-parses and resume-after-interrupt collapse onto the same node instead
of fanning out duplicates.

**Persistence**: The world-state graph is in-memory *during a run* (held on
`AgentBundle.repo`, a live object shared by every agent), and it is **also
snapshotted** to the `world_state_snapshots` table so it survives a
pause/resume. After every agent step that wrote nodes, the stage controller
calls `save_world_state(engagement_id, repo.to_json())` (best-effort UPSERT,
one row per engagement). On resume, `run_engagement` builds a fresh empty
repo via `build_bundle` and then calls `repo.load_json(load_world_state(...))`
to rehydrate that exact repo **in place** -- so the bundle's agents (which
share the reference) see the restored hosts/services/findings/credentials
instead of starting empty.

This snapshot is *separate* from the LangGraph `EngagementState` checkpoint
and the two writes are not one transaction. Because node `id`s are
deterministic `uuid5` (above), a slightly-stale snapshot plus an agent re-run
re-converges onto the same nodes -- eventual consistency, not corruption.
`EngagementState` (control-flow, budgets, suppression, the last redacted
world-state slice) is checkpointed separately so a resumed run keeps its
bookkeeping; the graph facts are restored from the snapshot.

### Filesystem -- evidence blobs

Content-addressed at `~/.serenade/evidence/{engagement_id}/{blob_id}`.
Every tool run's stdout / stderr lives here. Findings reference blobs
via `EvidenceRef(tool_run_id, blob_id, byte_offset?)`.

SHA256 integrity checking is preserved. No locking needed (single writer).

### Credential encryption at rest

`Credential` node secret fields (marked `secret=True` on the schema; today
`password`, `nt_hash`, `aes256_key`, `ssh_private_key_ref` -- the canonical
list is derived from the model as `CREDENTIAL_SECRET_FIELDS`) are protected
in two layers, both **live** at-rest controls now that the graph is
snapshotted to `world_state_snapshots` (see Persistence above). The separate
LLM exposure -- secrets reaching the model -- is closed by slice redaction.

**Whole-snapshot encryption (the primary at-rest boundary).**
`world_state_store.save_world_state` Fernet-encrypts the *entire*
serialized graph before the UPSERT, so the on-disk blob is ciphertext
regardless of which node/field a secret landed in -- not just `Credential`
fields, but also secrets that leak into `Finding` evidence/description,
`Session` fields, or raw tool-output excerpts. `load_world_state` decrypts;
if the blob can't be decrypted (master-key mismatch) it returns `None` and
logs `world_state.decrypt_failed`, so resume fails safe (empty graph,
re-derive) rather than feeding ciphertext to tools.

**Field-level encryption (belt-and-suspenders).** Independently, whenever
the graph is serialized (`InMemoryGraphRepo.to_json`), each non-None
`Credential` secret field is replaced *in the serialized copy* with a Fernet
token (the live in-memory node is never mutated); `from_json`/`load_json`
reverse this. This keeps `to_json()` output credential-safe on any path even
without the whole-blob layer, and round-trips losslessly (a legacy/unreadable
value passes through and logs `world_state.decrypt_failed` rather than
crashing). There is no separate `credentials` table -- secrets are encrypted
in place. Crypto lives in `src/orchestrator/crypto.py`.

Key: a per-install Fernet master key at `<data_dir>/.master_key`
(default `~/.serenade/.master_key`), generated on first use with mode
0600 (owner read/write only).

**In LLM-facing slices (redaction).** Whenever a credential node would be
placed into a world-state slice or LLM prompt, its secret fields are
replaced with the literal string `[ENCRYPTED]` (see
`redact_credential_secrets` in `src/worldstate/queries.py`, applied in
`credentials_with_known_hosts`). Agent slices that only need
boolean/branching signals already reduce secrets to `has_password` /
`has_hash` flags or carry only non-secret fields
(username/domain/source), so no plaintext secret reaches the model.

---

## Safety gates

### Policy engine

`src/policy/scope.py` -- a small, pure module whose `evaluate()` function
checks scope, risk class, time window, and the destructive flag on every
tool call. A tool whose `destructive` ClassVar is `True` is DENIED with
reason `destructive_not_authorized` **unless** the engagement set
`destructive_opt_in` (a separate axis from mode, fail-closed `False`). When
opted in, the call is allowed but always `require_approval=True` -- a
destructive action is never executed silently (human gate in guided; critic
+ auto-approve record in autonomous). No shipped tool is destructive by
default: the EDB tool `edb_exploit_exec` is **not** destructive (it runs
non-destructive scripts; DoS modules are gated per-module inside the tool,
also behind `destructive_opt_in`).

```python
def evaluate(targets, engagement, tool) -> PolicyResult:
    # CIDR + domain scope check
    # Risk class check
    # Time window check
    # Destructive: tool.destructive and not destructive_opt_in
    #   -> DENY destructive_not_authorized; else -> require_approval
    # -> PolicyResult(allow=bool, require_approval=bool, reason=str)
```

### The critic / council layer

`src/agents/critic.py`, `src/agents/council.py`,
`src/agents/regression_council.py`. Four gate types, in increasing
weight:

| Gate | Where | Cost | Mechanism |
|---|---|---|---|
| **Policy** | Every tool call | ~ms, free | Python scope/risk/time checks |
| **Single critic** | High-stakes risk classes | 1 Haiku call | `BaseCritic.review()` -> `Critique` |
| **Council** (triage / report) | VULN_TRIAGE consolidation, REPORT finalization | 3-4 Haiku calls + 1 synth | `Council.deliberate()` -> `CouncilVerdict` |
| **StageRegressionCouncil** | Every `REGRESS_TO` signal | 3 critics (Novelty + Scope + Budget) + synth | `StageRegressionCouncil.evaluate()` -> backwards transition gate |

Policy fires on every tool call (rounding-error cost). The single critic
fires only if `tool.risk_class in BaseAgent.critic_required_for_risk_classes`.
Subclasses opt in: `ExploitAgent` sets `(EXPLOIT, DESTRUCTIVE)`,
`LateralAgent` sets `(POST_EXPLOIT, EXPLOIT)`. Council runs at stage
transition points, not inside the per-tool loop.

Critic verdicts:

| Verdict | Behaviour |
|---|---|
| `APPROVE` | Proceed to sandbox execution |
| `VETO` | Queue an `approval` row (`critic_veto:<tool>`); never silently skipped |
| `REVISE` | Queue with `critic_revise: {primary_concern}` |
| `ESCALATE` | Queue with `critic_escalate: {primary_concern}` |

Critic output -- structured feedback fields:

| Field | Type | Purpose |
|---|---|---|
| `reasoning_trace` | `str` | Chain-of-thought analysis |
| `suggested_fix` | `str \| None` | Concrete, actionable guidance for the planner |
| `flaws` | `list[str]` | Specific reasoning flaws identified |

These fields are always present on `Critique` (defaults to empty). When a
critic returns REVISE or VETO, the planner's next turn prompt includes the
`suggested_fix` and `flaws` so it can adjust its proposal without blind
guessing.

Council two-tier short-circuit (cost optimization):

- **Unanimous high-confidence approve** -> skip synthesizer, return approve.
- **Any veto** -> skip synthesizer, return veto with the strongest concern.
- Otherwise -> synthesizer call resolves the split.

The expensive synthesizer fires only when members disagree.

### Multi-agent debate mode

`Council.deliberate(mode="debate")` -- opt-in via
`enable_debate_gates` setting (default off). When active, exploit and
lateral stage gates use iterative multi-round debate instead of the
default single-round council:

1. Two perspective critics -- `AggressivePentester` (favours
   exploitation) and `CautiousDefender` (highlights risks, scope
   concerns, detection likelihood) -- evaluate the proposed action.
2. Each round, members see all prior responses. Default: a single round
   (`debate_rounds = 1`); raise `debate_rounds` above 1 for iterative debate.
3. The synthesizer resolves the final verdict from the full debate
   transcript.

The same unanimous-approve / any-veto short-circuits apply. Debate
mode only fires at exploit/lateral gates; triage and report councils
are unaffected.

### StageRegressionCouncil

When an agent emits `StepDecision.REGRESS_TO` (a deterministic
post-step trigger, never an LLM hunch), the controller runs the
regression council before committing any backwards transition. Three
critics:

- **NoveltyCritic** -- queries the graph to confirm the proposed delta
  is not already covered.
- **ScopeCritic** -- refuses any delta outside the engagement's declared
  scope. We do not auto-expand scope.
- **BudgetCritic** -- enforces the `regression_budget_max` cap
  (`REGRESSION_BUDGET_MAX_DEFAULT = 50`; new engagements may set `0` for
  unlimited) plus a token-reserve check. A separate per-`(target_stage,
  trigger)` storm guard lives in **NoveltyCritic**
  (`REGRESSION_STORM_THRESHOLD = 5` distinct fingerprints within a
  10-minute window).

Verdicts: `APPROVE` writes the backwards stage transition and stashes
the delta on `EngagementState`; `VETO` leaves state alone; `ESCALATE`
calls `interrupt()` so the operator decides. Every outcome lands in the
`regressions` table and on the event bus
(`regression.requested|applied|vetoed|escalated`).

### Three-layer scope enforcement

| Layer | Status | Mechanism |
|---|---|---|
| L1 Application | implemented | Each tool implements `extract_targets()` so scope checks can't be silently bypassed |
| L2 Policy | implemented | `src/policy/scope.py`; engagement dict built by `src/policy/engagement_dict.py::build_engagement_context()` (single source of truth) |
| L3 Network | Docker `--internal` + optional nftables | Per-engagement Docker network so sandbox containers can't reach the host's default gateway. For production, `src/sandbox/nftables.py::compile_ruleset(scope)` emits an nftables ruleset compiled from the same `Scope` spec the policy uses |

If L1 or L2 says no, the call doesn't happen. L3 is defense-in-depth.

**L3 fail-closed switch.** `Settings.sandbox_l3_required` (env
`SANDBOX_L3_REQUIRED`, default `False`) controls the failure mode when the
per-engagement nftables ruleset can't be applied. When set `True`, apply
failure raises `RuntimeError` and the engagement refuses to start; the
default (`False`) logs a warning and falls back to L2-only isolation.
Production deployments that depend on per-engagement L3 ACLs should set it
`True`.

---

## Event bus

In-process asyncio pub/sub. Agents, the controller, and the approval
system publish events. The TUI subscribes and redraws on each event.

```
BaseAgent.step()    stage controller    approvals + audit    regression council
        |                  |                  |                       |
        v                  v                  v                       v
        src/orchestrator/events/bus.py::EventBus
                asyncio.Queue per subscriber
                          |
                          v
                TUI widgets (direct subscription)
```

- **Topic**: per-engagement event stream.
- **Best-effort publish**: bus errors log WARN and continue. The
  agent loop never blocks on the bus.
- **Replay**: `from_id="0"` replays the whole history; `"$"` (default)
  live-tails. Late-joining TUI tabs replay missed events automatically.
- **Abort signal**: `asyncio.Event` replaces the former Redis
  subscription. Abort sets the event; the controller checks it at each
  step boundary.

Event kinds: `stage.step.start`, `agent.llm_turn`, `tool.completed`,
`approval.queued`, `engagement.abort_requested`, `regression.requested`,
`regression.applied`, `regression.vetoed`, `regression.escalated`,
`engagement.budget_warning`, `engagement.veto_rate_warning`,
`engagement.suppression_spread`, `_heartbeat`, plus critic events.

### Threshold monitor

`src/orchestrator/thresholds.py::check_thresholds` -- lightweight
solo-operator observability. Called from
`stage_controller._step_agent` after every agent step. Three event
kinds emitted on the bus, each accompanied by an `audit_log` row with
`actor="threshold_monitor"`:

| Kind | Trips when |
|---|---|
| `engagement.budget_warning` (`kind=token`) | `total_tokens / token_budget` >= 0.80 |
| `engagement.budget_warning` (`kind=wallclock`) | elapsed-fraction of engagement window >= 0.80 |
| `engagement.veto_rate_warning` | > 50% VETOs over the last 10 critic verdicts |
| `engagement.suppression_spread` | The current stage's suppressed-tools count grew since last sample |

Debounced via `EngagementState.warnings_fired` so each
threshold fires **at most once per engagement** -- except
`suppression_spread`, which is keyed `suppression_spread:<stage>` and
fires once per stage.

### EngagementSummary digest

`serenade engagement summary <id>` returns an operator-facing decision
digest. Every counter is SQL over `audit_log` / `tool_runs` /
`llm_calls` / `regressions`. Fields:

| Field | Source |
|---|---|
| `tool_calls_total` | `tool_runs` count |
| `tool_calls_auto_approved` | `audit_log` rows tagged `decided_by: auto-policy, decision: approve` |
| `tool_calls_critic_vetoed` | `audit_log` rows tagged `critic_veto:*` |
| `tool_calls_human_approved_gated` | gated tools with an explicit human approve |
| `tool_calls_ungated` | `total - auto - critic - human-gated` (low-risk recon, etc.) |
| `suppressed_tools_count` | active rows in `suppressed_tools` (`cleared_at IS NULL`) |
| `llm_cost_usd` | sum over `llm_calls.cost_usd` |
| `wallclock_minutes_used` | engagement clock delta |
| `regressions_used` | regressions table count |
| `findings_count` | from graph coverage query |
| `warnings_fired: list[str]` | projection of `audit_log` threshold rows |

---

## LLM routing

`src/llm/router.py::LLMRouter` is the single entry point. Backend
selected per **engagement** (snapshot at create-time). Within a backend,
the model is selected per **call site**:

| `CallSite` | Anthropic default | Local (Ollama) | Used by |
|---|---|---|---|
| `PLANNER` | `claude-sonnet-4-6` | `gemma4:26b` | Stage agents picking next tool |
| `SYNTHESIZER` | `claude-opus-4-7` | `gemma4` (E4B) | Report narrative, exec summary |
| `PARSER` | `claude-haiku-4-5-20251001` | `gemma4:26b` | Tool output -> typed facts |
| `TRIAGE` | `claude-haiku-4-5-20251001` | `gemma4` (E4B) | Exploitability / severity |
| `CRITIC` | `claude-haiku-4-5-20251001` | `gemma4` (E4B) | Critics + council members |

**Prompt caching** (Anthropic only): engagement-stable prefix (system +
tool registry + world-state schema) cached via `cache_control: ephemeral`
on the last system block. Per-turn user block sits *after* the cache
breakpoint.

**Local path**: Gemma 4 has no native tool-use API. We inject the tool
catalog into the system prompt as JSON-Schema and parse
`<tool_call>{...}</tool_call>` tags. Same Pydantic schemas as the
Anthropic path; agents are backend-agnostic. Ollama is the recommended
runtime; any OpenAI-compatible endpoint works.

**Backend selection (explicit opt-in).** Local is the default. Config
lives in `.env` (see `.env.example`). `ANTHROPIC_API_KEY` in the shell is
**not** sufficient on its own. When the key is present but no explicit
backend is chosen, serenade stays on local and emits
`llm.anthropic_key_ignored` so the gap is visible in logs. Operators
who want Anthropic must set `SERENADE_LLM_BACKEND=anthropic` in `.env`.

**Token budgets and local backends**: local backends don't bill per
token, so `token_budget` becomes a kill-switch rather than a cost cap.
The new-engagement form marks the budget optional with a "local LLM"
badge when the backend is local; `0` ("unlimited") is allowed
only on a local backend. With a hosted backend, `token_budget: 0` is
refused at create-time.

### Model lifecycle (local stack)

The agent loop naturally batches calls by CallSite:

1. `_step_interactive` inner loop: N consecutive PLANNER calls (Gemma 4
   26B stays loaded in VRAM).
2. Pre-action critic gate: 1-3 CRITIC calls (Gemma 4 E4B loads, 26B
   idles).
3. Post-step scorer (opt-in): 1 CRITIC call (E4B already loaded).
4. Reflection memory write (opt-in): 1 CRITIC call (E4B already loaded).
5. Next step: PLANNER resumes (26B reloads if idle > `keep_alive`).

Ollama's `keep_alive` default (5m) means the 26B model survives across
back-to-back agent steps (typically <30s apart). The E4B model loads
for the critic burst and unloads after. On a 32 GB Apple Silicon
machine, both models fit in memory simultaneously; on smaller machines
the swap is ~2s cold-load for the 4B model.

**Mid-step budget enforcement.** Because one `step()` is now N LLM
turns, the budget check fires both at the top of `_step_agent` *and*
between LLM turns inside `_step_interactive` (see
`_check_budget_mid_step` in `stage_controller.py`). An engagement that
crosses its cap mid-step bails out at the next turn boundary instead of
finishing all 15 turns.

---

## Sandbox + egress

**Per-call container model** -- every `SandboxRunner.run()` spawns a
fresh `serenade/sandbox:dev` container via `docker run --rm`. No
long-lived shared sandbox; cross-engagement leakage through `/tmp`,
`/proc`, tool caches is structurally impossible.

- Container name: `serenade-sb-{engagement_prefix}-{run_prefix}`.
- No host bind-mounts.
- Optional mitmproxy egress logging (install separately with
  `pip install mitmproxy` if desired).

Tool output parsing runs **inside** the sandbox container. The container
runs the tool, parses the output, and emits structured JSON on stdout.
The serenade process only deserializes trusted JSON via Pydantic. Raw
XML/JSONL from tool output never touches the host process.

---

## Exploit sourcing + curation

Two tiers:

- **Discovery tier**: `searchsploit`, ExploitDB mirror, Metasploit
  module index, nuclei templates, NVD. Output: `ExploitCandidate`
  nodes. Read-only, runs automatically.
- **Curated tier**: `content/pocs/<id>/manifest.yaml` + `exploit.py`.
  Manifest specifies provenance, version range, parameter schema, risk
  class, `auto_run` rules (structured Pydantic predicates only -- no
  free-form CEL), rollback hooks, SPDX license, CI test.

**Execution path**:

```
LLM emits: curated_poc_exec{poc_id, target_host, params, ...}
        | BaseAgent.step canonical loop
        | policy gate (L2)
        | PreExploitCritic gate
        v
CuratedPoCExec.run() --> PoCDispatcher.dispatch()
                            +-- can_auto_run()
                            +-- PreconditionEvaluator
                            +-- JSON Schema validate
                            +-- build_invocation()  <- pure, in-process
                            +-- SandboxRunner.run() <- exploit runs HERE
                            '-- parse_result()      <- pure, in-process
```

`ExploitAgent` proposes `curated_poc_exec` (preferred) and, for candidates
that only have an ExploitDB reference, `edb_exploit_exec`. PoC modules define
two pure functions; neither performs I/O. Both paths run the exploit body
entirely inside the sandbox container -- the serenade process never executes
exploit code, so a compromised script cannot reach `/var/run/docker.sock`.

**EDB execution tier** (`tools/edb_exec.py`, `edb_exploit_exec`): fetches an
ExploitDB script from the local mirror and runs it in the sandbox. The tool
is **not** blanket-destructive — non-destructive scripts
(remote/local/privesc/webapp/bypass) execute, gated by scope + the
`PreExploitCritic` + the `edb_auto_exploit` toggle (auto under autonomous;
surfaced-for-approval in guided). Destructiveness is enforced **per-module**
inside `run()`: a DoS module (`type="dos"`) is reported as a finding and not
executed unless the engagement set `destructive_opt_in`. EDB candidates also
surface in the TUI for operator promotion to a curated `content/pocs/<id>/`
entry via PR review. See [POCS.md](./POCS.md).

**Chain-first planning**: at the start of the exploit stage the agent asks
the LLM to plan goal-oriented attack *chains* -- an objective with ordered
steps, each carrying ranked alternative candidates. A per-run cursor
(`ctx.chain_plan`, in-memory only) drives execution: try the active step's
best alternative, mark outcomes, and on failure swap to the next alternative
or advance to the next step; a chain is `succeeded` when every step lands and
`exhausted` when a step runs out of untried alternatives.

**PotentialFindings**: every `ExploitCandidate` is promoted to a
`PotentialFinding` node (unconfirmed) after the exploit loop -- whether or not
it matched a curated PoC -- so the report distinguishes confirmed
(PoC-validated) findings from plausible-but-unconfirmed ones. `assess` mode
produces these without ever executing an exploit.

---

## Stage controller (LangGraph)

`src/orchestrator/stage_controller.py`. Each stage is a graph node
holding one agent. `EngagementState` (Pydantic) flows between nodes:

```python
class EngagementState(BaseModel):
    engagement_id: UUID
    engagement: dict                              # frozen scope/mode/budgets
    current_stage: Stage = Stage.RECON
    turn_counts: dict[str, int]
    total_cost_usd: float
    total_tokens: int
    tokens_by_tool: dict[str, int]                # per-tool budget gate
    suppressed_tools: dict[str, str]              # name -> reason
    suppressed_tools_stage: str | None            # stage-boundary reset
    pending_approvals: list[dict]
    notes: list[str]
    completed: bool

    # Watchdogs
    consecutive_zero_host_turns: int = 0          # recon zero-host
    stalled_turn_count: int = 0                   # global no-progress

    # Bounded regression
    regressions_used: int = 0
    regression_delta: dict | None = None
    regression_origin_stage: str | None = None

    # Mid-node fan-out checkpoint (web_enum -> net_enum)
    enumeration_substage: str | None = None

    # Operator kill switch
    should_abort: bool = False
    abort_reason: str | None = None

    # Threshold-monitor state
    warnings_fired: list[str]                      # debounce list
    recent_critic_verdicts: list[bool]             # rolling window; True = VETO
    last_known_suppression_stage_count: dict[str, int]
```

Canonical forward order:

```
RECON -> ENUMERATION -> VULN_TRIAGE -> EXPLOIT -> POST_EXPLOIT
      -> PRIVESC -> LATERAL -> LOOT -> REPORT
```

Per-node routing:

- `CONTINUE` -> re-enter the same node.
- `ADVANCE` -> `_next_stage(current)` writes the forward step.
- `ESCALATE_HUMAN` -> `langgraph.types.interrupt({...})`. LangGraph
  durably checkpoints the entire `EngagementState` to SQLite via
  `SQLiteCheckpointer`. The process can crash or restart without losing
  state.
- `REGRESS_TO` -> `_handle_regress_to` runs the
  `StageRegressionCouncil`; on approve writes the backwards transition
  and stashes the delta.

**POST_EXPLOIT.** A no-op pivot node with a static `add_edge` to
`privesc`. Exists as a logical pivot point and as a guard that drops
any stale regression delta.

**Per-node composition.** ENUMERATION runs two agents sequentially:
`WebEnumAgent` first (faster; depends only on WebApp nodes from recon),
then `NetEnumAgent` (AD / SMB / LDAP; depends on Host + Service nodes).
A substage checkpoint (`enumeration_substage`) means a mid-node
interrupt + resume does not double-run `web_enum`.

**Resume.** `serenade` on restart rehydrates from the latest checkpoint
and continues past the `interrupt()`. The per-engagement Docker network
is preserved across interrupt pauses; only torn down on genuine
completion.

**Abort.** The TUI abort action flips
`EngagementRow.status = 'aborted'` and publishes
`engagement.abort_requested` on the bus. A per-engagement subscriber
(`src/orchestrator/abort_consumer.py`) terminates in-flight sandbox
containers. The controller also polls the engagement row at the top
of each `_step_agent` so a missed event doesn't strand the abort.

### Concurrency model

Textual runs its own asyncio event loop. The agent loop runs as an
`asyncio.Task` spawned from the TUI's event loop:

```
TUI event loop (Textual)
  |-- asyncio.Task: agent_loop(engagement_id)
  |     |-- LangGraph ainvoke() runs tool calls, publishes events
  |     |-- On interrupt(): publishes "approval_needed" event, awaits
  |     '-- On cancel: sets abort flag, cleans up sandbox
  |-- EventBus.subscribe() -> redraws widgets on new events
  '-- User input -> approve/reject/abort -> signals agent task
```

The agent never blocks the TUI because it's a separate task. The TUI
never blocks the agent because Textual's rendering is non-blocking.

### Per-stage playbooks

Each agent provides `_pentester_playbook(slice_)` -- a per-stage natural
language guidance block injected into the planner's user prompt. The
playbook tells the model what a senior pentester would attempt next
given the current world-state slice. Playbooks live next to each agent
in `src/agents/<stage>.py` so they evolve with the agent's tool roster.

---

## Modern agent patterns

Four implemented opt-in features (all default off) that move the agent
loop from a linear tool chain toward bounded search with reflection, plus
one planned pattern (Tree-of-Thoughts) not yet written. Enable via
settings -- each has a boolean toggle.

### Per-step scoring (AgentPRM)

`src/agents/step_scorer.py::StepScorer`. Setting:
`enable_step_scoring` (default `false`).

After each agent step, the scorer makes one cheap E4B call to rate the
step on two dimensions:

| Score | Meaning |
|---|---|
| `promise` (0-1) | How likely is this line of investigation to yield a finding? |
| `progress` (0-1) | How much new information did this step produce? |

When both scores fall below the backtrack threshold (default 0.3), the
agent emits `ESCALATE_HUMAN` to avoid wasting turns on a dead-end path.
Scores are surfaced in the audit log (there is no dedicated `step_scores`
table).

### Reflexion verbal memory

`src/agents/reflection_memory.py::ReflectionMemory`. Setting:
`enable_reflection_memory` (default `false`).

Strategy-level episodic memory stored in SQLite
(`reflection_memory` table). After a failed or low-scoring step, the
E4B model generates a one-paragraph lesson. These lessons are injected
into the planner's prompt as a "Lessons learned" section.

Key constraints: max 10 entries per stage, 500 chars per entry.
Prompt-size impact at cap: ~1,500 tokens (negligible on 128K context).

### Regression reflection (LLM-suggested)

`src/agents/regression_reflection.py::StageReflector`. Setting:
`enable_regression_reflection` (default `false`).

At each stage ADVANCE boundary, the E4B model reviews the world-state
and decides whether the engagement should regress. If it suggests a
regression, the proposal goes through the full
`StageRegressionCouncil` -- the LLM proposes, the deterministic
council disposes.

### Multi-agent debate

See [Multi-agent debate mode](#multi-agent-debate-mode) above. Setting:
`enable_debate_gates` (default `false`).

### Tree-of-Thoughts exploration (planned -- not yet implemented)

> **Status: not implemented.** There is no `src/agents/thought_tree.py`,
> no `ThoughtTree` class, and no `enable_thought_tree` setting in the
> current build. A bounded search tree for multi-path exploration
> (`max_depth=4`, `max_breadth=3`) is on the roadmap but not yet written.

### Feature-flag summary

| Setting | Env var | Default | What it enables |
|---|---|---|---|
| `enable_step_scoring` | `ENABLE_STEP_SCORING` | `false` | Per-step promise/progress scoring |
| `enable_reflection_memory` | `ENABLE_REFLECTION_MEMORY` | `false` | Verbal memory (Reflexion) |
| `enable_debate_gates` | `ENABLE_DEBATE_GATES` | `false` | Multi-agent debate at exploit/lateral |
| `enable_regression_reflection` | `ENABLE_REGRESSION_REFLECTION` | `false` | LLM-suggested regressions at ADVANCE |

(The `Settings` model has no env prefix, so the env-var name is the bare
field name in upper-case -- e.g. `ENABLE_STEP_SCORING`, not
`SERENADE_ENABLE_STEP_SCORING`.)

---

## Engagement mode

Mode is the single configuration axis. Each engagement carries one `mode`;
the allowed risk classes and the approval behaviour are *derived* from it in
`policy/engagement_dict.py` — there is no separate `approval_policy` column.
(There is one separate flag: `destructive_opt_in`, a fail-closed `False`
column that governs the curated-PoC destructive path only — see below.)

| Mode | Pipeline | Exploitation | Gating |
|---|---|---|---|
| `recon_only` | recon + enum only | none | auto-transitions |
| `assess` | full pipeline | none (chain planning + PotentialFindings only) | auto-transitions |
| `guided` *(default)* | full pipeline | curated + EDB, **operator-approved** | phase gate (`require_approval`) |
| `autonomous` | full pipeline | curated + EDB, auto-executed | none |

Derivation: `recon_only`/`assess` → `["recon","enum"]`, `auto_approve`;
`guided` → `["recon","enum","exploit","post-exploit"]`, `phase_gate`;
`autonomous` → same risk classes, `auto_approve`. Legacy values
(`passive`/`normal`/`aggressive`/`lab`) are normalized at read time via
`normalize_mode()`.

The derived `approval_policy` decides what happens when a gate would
otherwise queue for a human:

| Derived policy | Effect |
|---|---|
| `phase_gate` (guided) | Exploit/post-exploit calls queue for the operator. |
| `auto_approve` (recon_only/assess/autonomous) | Records `decision: approve, decided_by: auto-policy` and proceeds. |

`critic_veto:*` is auto-rejected regardless of mode. Per-tool token-budget
overruns always require manual override.

**Destructive actions are gated behind an explicit opt-in — never executed
silently.** "Destructive" means irreversible harm to the *target* (DoS,
wipe, account lockout); the sandbox isolates our host, not the client's box,
so this is gated by authorization rather than isolation. Destructiveness is a
**separate axis from mode**: the per-engagement `destructive_opt_in` flag (a
fail-closed `False` column, surfaced by `build_engagement_context`). The gate
is live, not aspirational:

- **Default (`destructive_opt_in: false`):** any tool whose `destructive`
  ClassVar is `True` is DENIED by the policy gate (`policy/scope.py`) with
  reason `destructive_not_authorized`; a curated PoC with
  `risk.destructive: true` is gated by `PoCDispatcher.can_auto_run`
  (`destructive_without_optin`, reported-not-run); and an EDB script is
  reported-not-run if it is a DoS module **or** its body statically scans as
  destructive — file/data removal or overwrite, disk wipe, fork bomb (see
  `tools/destructive_scan.py`, applied in `edb_exec.run()` and enforced on
  curated PoCs at CI). "Destructive" explicitly includes **deleting or
  overwriting target files**, not just DoS.
- **Opted in (`destructive_opt_in: true`):** the same actions are *allowed*,
  but always reviewed and recorded — never silent:
  - **Destructive tools** → the policy gate returns `require_approval=True`,
    so the call goes through the approval gate (human click in guided; a
    recorded `auto-policy` `ApprovalRow` in autonomous). The pre-action critic
    also fires for any destructive tool (the critic gate keys on
    `tool.destructive`, not just the agent's risk-class set).
  - **Destructive curated PoCs** → `can_auto_run` permits them; they run per
    the manifest's `auto_run.requires_approval` (set it `true` to force an
    operator click) and are critic-reviewed on the `curated_poc_exec` call.
  - **DoS EDB modules** → execute in the sandbox like other EDB scripts.
  Every destructive action lands an audit-log row regardless of path.

No recon/enum/detection tool (nmap, nuclei, httpx, ffuf, …) is destructive,
and the EDB dispatcher `edb_exploit_exec` is **not** blanket-destructive — it
runs non-destructive scripts and only gates DoS modules per-module.

Pair `mode: autonomous` + `auto_start: true` for an unattended overnight run:
curated PoCs and non-destructive EDB scripts auto-execute when their
preconditions pass; nothing destructive fires unless the operator explicitly
set `destructive_opt_in`.

---

## Security model

### Audit log integrity

`AuditLogRow` is tamper-evident at rest via a **per-engagement HMAC hash
chain** (tracking issue #112, P0 #2). Beyond `id, engagement_id, actor,
action, payload, ts` each row carries three chain columns:

- `seq` -- a 1-based monotonic counter *per chain* (genesis row = 1).
- `prev_hash` -- the predecessor row's `row_hash` (`NULL` only for the
  genesis row); folding it into the hashed bytes is what links the rows
  into a chain rather than a bag of independently-signed rows.
- `row_hash` -- `HMAC-SHA256(master_key, canonical)` over the row's
  canonical serialization (`seq, engagement_id, actor, action, payload,
  ts, prev_hash`, sorted keys, tight separators). The key is the raw
  per-install master key at `<data_dir>/.master_key` (same key as the
  credential-encryption Fernet, used here as raw HMAC bytes, not
  Fernet-wrapped).

A `UNIQUE(engagement_id, seq)` constraint is the race backstop: two
appenders can't claim the same chain position. Chains are scoped by
`engagement_id`; rows with `engagement_id IS NULL` share one global chain
under the `__global__` sentinel scope.

Writes go through `ledger.append_audit_row` (and `ledger.audit`), which
reads the chain tail (`SELECT ... ORDER BY seq DESC LIMIT 1`, with
`with_for_update()` on Postgres -- a no-op on SQLite, whose single-writer
lock + the UNIQUE constraint give the same no-fork guarantee), computes
the next `seq` / `prev_hash` / `row_hash` inside one transaction, and
retries on `IntegrityError` if it lost the seq slot. No code path inserts
audit rows with a NULL/wrong `prev_hash` when a predecessor exists.

`ledger.verify_audit_chain(engagement_id)` recomputes the chain and
returns `(ok, first_bad_seq, reason)`, distinguishing `hash-mismatch`
(a row was edited), `broken-link` (a row was inserted or reordered),
`seq-gap` (a row was deleted), and `seq-dup`. The PDF audit-log export
runs this verifier first and stamps **VERIFIED** / **TAMPERED (at seq N)**
onto the document, surfacing each row's stored `seq` / `prev_hash` /
`row_hash` for independent re-verification. The export-time
`SERENADE_AUDIT_EXPORT_SECRET` HMAC is retained as a transport MAC layered
on top; the at-rest master-key chain is the integrity source of truth.

**Threat-model boundary (important).** The chain is keyed off the symmetric
master key at `<data_dir>/.master_key`, which by default lives in the *same*
directory as `serenade.db` (`~/.serenade/`). This makes the log fully
tamper-evident against a **DB-only attacker** (someone who can write
`audit_log` rows but cannot read the key): any edit/delete/reorder is
detected, because they cannot recompute a valid `row_hash` and cannot
re-stamp the dependent tail. It is **not** tamper-*proof* against an
attacker who also reads the master key (host/filesystem compromise): with
the key, the HMAC that verifies is the same one that forges, so a full-tail
rewrite is possible and would verify clean. If a stronger boundary is
required, move the key out of `data_dir` (a separate secrets mount / KMS)
or switch to an asymmetric signature so the signing key never sits beside
the DB and a DB-write attacker holds only a public verify key.

### Prompt injection guards

Critic system prompts are prefixed with a `_PROMPT_INJECTION_GUARD`
constant defined in `src/agents/critic.py`, so parser / tool output that
lands in critic context cannot override system instructions. (Note: the
guard is applied to critic prompts; planner prompts do not currently
carry it.)

### Operator-exec escape hatch

The TUI's Op-Exec tab lets the operator drive a single tool call
manually. Three guards apply:

1. Required `justification` field (>=20 chars), audit-logged.
2. The same policy gate fires as for autonomous calls -- never
   bypassable.
3. `PreExploitCritic` still runs unless the caller sets
   `bypass_critic: true` *and* supplies a separate `bypass_justification`
   (>=20 chars). Both reasons land in `audit_log`.

### Windows worker isolation

When enabled, the Windows worker VM runs in operator-controlled infra.
During an engagement it is **inside** the threat boundary -- it
executes payloads against target hosts.

- Worker stdout is parsed (mimikatz, shell, assembly results).
  Adversary-controlled stdout is in scope.
- `auth_token` and `signing_key` are pre-shared secrets.
- `assembly_b64` jobs carry an `assembly_sig` (HMAC-SHA256). The worker
  verifies before invoking the entry point.
- `parse_mimikatz_output` is bounded: 5 MB input cap, 50,000 line cap,
  200 credential cap.

Redis transport between the serenade process and the worker is an
optional dependency (`[windows]` extra), only loaded when
`SERENADE_WINDOWS_WORKER_ENABLED=1`.

---

## Doc update policy

Any PR that changes a component boundary, adds / removes a gate,
changes an LLM call site, adds a node / edge type in world-state,
or adjusts the threat model must update this file in the same PR.
