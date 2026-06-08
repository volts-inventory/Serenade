# Curated PoC Library

`ExploitAgent` prefers curated PoCs from `content/pocs/<id>/` — hand-reviewed, CI-tested against a pinned lab, merged via PR. This is the design that lets the agent fire unattended without inviting a supply-chain breach.

For candidates with no curated match, the agent can fall back to the **EDB execution tier** (`edb_exploit_exec`): an ExploitDB script fetched from the local mirror and run **inside the sandbox** (the real safety boundary — isolated network, no host access, ephemeral filesystem). Non-destructive scripts (remote/local/privesc/webapp/bypass) execute — automatically under `autonomous` (via `edb_auto_exploit`), surfaced for approval in `guided` — gated by scope + the `PreExploitCritic`. Destructive modules — DoS, or any script whose body statically scans as destructive (file/data **removal** or overwrite, disk wipe, fork bomb; see `tools/destructive_scan.py`) — are reported as a finding (not run) unless the engagement set `destructive_opt_in`. Promoting a useful EDB script to a PR-reviewed curated entry remains the higher-confidence path. See [ARCHITECTURE.md](./ARCHITECTURE.md#exploit-sourcing--curation).

For runtime / gate / sandbox internals see [ARCHITECTURE.md - Exploit sourcing + curation](./ARCHITECTURE.md#exploit-sourcing--curation). Full manifest field reference: [content/pocs/SCHEMA.md](../content/pocs/SCHEMA.md).

---

## What ships today

| PoC | Status |
|---|---|
| `example-banner-check` | Benign TCP-connect reference. Read this first to see the shape. |
| `log4shell-CVE-2021-44228` | nuclei-backed detection + trigger. `auto_run.allowed: false` (operator must approve). |
| `spring4shell-CVE-2022-22965` | nuclei-backed detection. `auto_run.allowed: false`. |

Both real CVE entries ship gated. Promote them to auto-run after you've validated against your own lab.

---

## When to add a new one

The triage stage emits `ExploitCandidate` nodes for every CVE / EDB match it discovers. Candidates without a curated entry surface in the **Approvals** queue with:

> Detected: CVE-X (EDB-Y). Not in curated library.

That's the signal to promote.

---

## Add a new PoC

```bash
cp -r content/pocs/_template content/pocs/CVE-YYYY-NNNNN-shortname
cd content/pocs/CVE-YYYY-NNNNN-shortname
```

### 1. Edit `manifest.yaml`

Required fields (full reference in [SCHEMA.md](../content/pocs/SCHEMA.md)):

| Field | Notes |
|---|---|
| `id` | Must match the directory name |
| `provenance` | At least one entry. Sources: `exploitdb`, `metasploit`, `nuclei`, `cve_writeup`, `original_research` |
| `target.service` / `target.version_range` | semver-like (`>=2.0,<2.15.0`) |
| `target.platforms` | `linux`, `windows`, or `multiple` |
| `parameters.schema_ref` | Path to a Pydantic-compatible JSON schema |
| `risk.class` | Highest applicable `RiskClass`: `recon` / `enum` / `exploit` / `post-exploit` / `destructive` |
| `risk.destructive` | `true` PoCs are reported-not-run by default: `PoCDispatcher.can_auto_run` returns `(False, "destructive_without_optin")` unless the engagement sets the operator `destructive_opt_in` flag (PoC path only) |
| `risk.modifies_target` / `modifies_filesystem` / `network_egress_from_target` | Behaviour disclosure |
| `risk.detection_likelihood` | `low` / `medium` / `high` |
| `auto_run.allowed` | `false` until CI is green and the PR is reviewed |
| `auto_run.requires_approval` | If `true`, operator must click approve even in `autonomous` mode |
| `auto_run.preconditions` | Structured Pydantic predicates over `{engagement, target, world_state}` — no free-form strings |
| `rollback.available` + `rollback.notes` | What persists after the run |
| `license.spdx` | Real SPDX identifier; `UNKNOWN` is a hard CI fail |
| `license.attribution` | Original author / source citation |
| `ci.test_command` | Default: `pytest test/test_exploit.py` |

### 2. Implement `exploit.py`

Two **pure functions** — no I/O, no orchestrator imports:

```python
from agents.poc_runtime import PoCExecutionContext, PoCResult, SandboxInvocation
from sandbox.runner import SandboxResult

def build_invocation(params, ctx: PoCExecutionContext) -> SandboxInvocation:
    """Return the argv + env to run in the sandbox container."""
    ...

def parse_result(sandbox_result: SandboxResult, params, ctx) -> PoCResult:
    """Parse stdout/stderr/exit_code into a typed result."""
    ...
```

The worker process **never executes your exploit body** — the sandbox container does. This is structural: a compromised PoC cannot reach `/var/run/docker.sock` because `build_invocation` and `parse_result` are pure and the actual exploit runs inside `docker run --rm`.

### 3. Write `test/test_exploit.py`

Pytest against a pinned vulnerable lab (vulhub or `test/target_compose.yaml`). The test must:

- Stand up the lab.
- Call the dispatcher.
- Assert the exploit fires correctly AND that `parse_result` returns the `PoCResult` shape the agent expects.

The test **must pass** before merge.

### 4. PR review

CI runs:

| Check | What it enforces |
|---|---|
| AST lint (`src/agents/poc_lint.py::lint_module`) | No top-level statements other than imports and `def`. PoC modules can't run code on import. |
| Manifest schema validation | Required fields present, types right, SPDX not `UNKNOWN`. |
| Your pytest against the pinned lab | The exploit actually works against the vulnerable version. |
| Registry load test (`tests/unit/test_poc_library_loads_shipped.py`) | The new entry loads cleanly into the curated registry. |

Once merged, `ExploitAgent` will start proposing it for matching candidates.

---

## Execution path

When the agent fires a curated PoC, it goes through the canonical loop like any other tool call:

```
LLM emits: curated_poc_exec{poc_id, target_host, params, ...}
        │
        ▼ schema validate (Pydantic Input)
        ▼ scope/risk policy gate (scope, risk class, time window)
        ▼ PreExploitCritic gate
        ▼ CuratedPoCExec.run()
                ├── can_auto_run()         — checks auto_run.allowed + requires_approval
                ├── PreconditionEvaluator  — structured predicates over world-state
                ├── JSON Schema validate parameters
                ├── build_invocation()     ← pure, on worker
                ├── SandboxRunner.run()    ← exploit runs HERE (sandbox container)
                └── parse_result()         ← pure, on worker
        ▼ write to world-state with manifest sha256 in provenance
```

Every execution records the manifest sha256 in the `ExploitAttempt` evidence, so the report can prove exactly which version of the PoC was used.

---

## Hard rules

1. Only entries with `auto_run.allowed: true` AND `auto_run.requires_approval: false` AND all preconditions true run without operator click.
2. `risk.destructive: true` PoCs are **reported, not run** by default — `PoCDispatcher.can_auto_run` gates them with `destructive_without_optin`. The operator's per-engagement `destructive_opt_in` flag (a fail-closed `False` engagement column) clears that gate; the same flag also un-blocks destructive *tools* at the policy gate (which then return `require_approval` rather than `destructive_not_authorized`) and DoS EDB modules. `destructive_opt_in` is a separate axis from mode and routes every destructive action through approval. The shipped `content/pocs/destructive-gate-fixture/` is a non-functional placeholder that exists only to exercise this gate.
3. Every execution records the manifest sha256 in the `ExploitAttempt` evidence.
4. `license.spdx == "UNKNOWN"` is a hard fail in CI.
5. `exploit.py` must pass `src/agents/poc_lint.py::lint_module` — only imports and `def` at module top level.

---

## Directory layout

```
content/pocs/<id>/
├── manifest.yaml             # required; see SCHEMA.md
├── exploit.py                # required; the parameterized executable
├── parameters.schema.json    # required; JSON Schema for params
├── README.md                 # required; ops notes (intended behaviour, side effects, what success looks like)
└── test/
    ├── target_compose.yaml   # optional; vulhub or self-built lab
    └── test_exploit.py       # required; pytest assertion against the lab
```

The `content/pocs/_template/` directory is a working scaffold — copy it, don't mutate it.

---

## Promotion workflow (EDB candidate → curated entry)

1. `VulnTriageAgent` emits an `ExploitCandidate` with `source=exploitdb` and `auto_runnable=false`.
2. UI surfaces it in Approvals with "Not in curated library."
3. Operator copies `content/pocs/_template/` to `content/pocs/<new-id>/`.
4. Operator implements `exploit.py`, parameter schema, CI test, and sets risk flags.
5. PR review gate before merge.
6. CI runs `pytest content/pocs/<new-id>/test/` against the pinned lab — must pass.
7. After merge, the entry is available in the curated tier on the next engagement.
