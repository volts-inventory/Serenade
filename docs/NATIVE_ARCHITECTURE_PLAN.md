# Serenade -- Native Architecture Plan

## What this is

Serenade is an autonomous penetration testing tool. You point it at a
scope, it runs the full pipeline (recon through reporting), and three
safety gates fire on every tool call. It runs on your laptop.

Today the codebase is shaped like a SaaS backend: 9 Docker containers,
a FastAPI REST API, and an HTTP client in the TUI. This document is
the plan to collapse it into a single native Python process with Docker
used only for sandbox isolation.

---

## Current state

```
Host
  serenade CLI
    -> docker compose up (9 containers)
    -> serenade-tui (Textual app, separate process)
        -> HTTP to localhost:8000

Containers
  orchestrator   FastAPI REST API (uvicorn)
  worker         LangGraph agent loop
  postgres       engagement state, audit log, checkpoints (9 tables)
  neo4j          world-state graph (hosts, services, findings, edges)
  bloodhound     second Neo4j for AD graph data
  redis          event bus, abort signals, Windows worker queue
  minio          evidence blob storage (tool stdout/stderr)
  opa            policy engine (102 lines of Rego)
  proxy-logger   mitmproxy egress logging
  + ephemeral sandbox containers (docker run --rm per tool call)
```

Codebase: ~51K LOC Python across src/, clients/, tests/.

Dependencies that exist only for infra: asyncpg, neo4j, redis, minio,
fastapi, uvicorn, sse-starlette, python-multipart,
langgraph-checkpoint-postgres. Nine packages that go away.

---

## Target state

```
serenade (single process)
  |-- TUI -- Textual, in-process, calls Python functions directly
  |-- Agents -- LangGraph stage controller, in-process
  |-- Policy -- Python function, ~60 LOC (no OPA container)
  |-- Events -- asyncio pub/sub, in-process (no Redis)
  |-- DB -- SQLite via aiosqlite (~/.serenade/serenade.db)
  |-- Checkpoint -- SQLiteCheckpointer (custom, ~80 LOC)
  |-- Evidence -- local filesystem (~/.serenade/evidence/)
  |-- Graph -- networkx DiGraph, persisted to SQLite per engagement
  |-- LLM -- Ollama (local) or Anthropic API (httpx)
  '-- Sandbox -- docker run --rm serenade/sandbox:dev <tool> <args>
```

Install: `pip install serenade` (Docker must be on the machine).
Run: `serenade`.
Data: `~/.serenade/` (sqlite db, evidence blobs, config).

---

## Replacement plan by service

### 1. Postgres -> SQLite (aiosqlite)

Models: Keep SQLAlchemy ORM, switch dialect to `sqlite+aiosqlite`.
db.py: DSN becomes `sqlite+aiosqlite:///{data_dir}/serenade.db`.
Alembic: Switch to SQLite dialect; squash 8 migrations into one.
Engagement state: 9 tables, all simple CRUD. SQLite handles this.
Audit log: Append-only inserts. SQLite's strength.
JSONB columns: Use SQLAlchemy `JSON` type (SQLite stores as TEXT).
WAL mode: Enable `PRAGMA journal_mode=WAL` on connection for
concurrent reads during TUI polling.

LOC touched: ~500 (db.py, models.py, alembic, checkpoint.py, settings.py)

#### LangGraph checkpoint (critical: approval interrupts)

The agent loop calls `interrupt()` when it needs operator approval.
LangGraph checkpoints state; the TUI shows the approval queue; the
operator decides; the graph resumes from checkpoint. This flow MUST
persist across the interrupt.

MemorySaver is NOT safe because if the process crashes between
interrupt and approval, the engagement state is lost. The operator
would have to restart the entire engagement.

Solution: Write a thin `SQLiteCheckpointer` (~80 LOC) implementing
LangGraph's `BaseCheckpointSaver` interface. On `interrupt()`, state
is serialized (JSON) to a `checkpoints` table in the same SQLite DB.
On resume, it's loaded back. Normal operation stays in-memory (fast);
SQLite is the crash-safety fallback.

This replaces `langgraph-checkpoint-postgres` without adding a
dependency. Same DB, same process, same file.

### 2. Redis -> in-process asyncio

Event bus (events/bus.py, 120L): Replace Redis Streams with an
in-process `EventBus` class:

```python
class EventBus:
    _topics: dict[str, list[asyncio.Queue]]
    _history: dict[str, list[dict]]  # for replay/backfill

    async def publish(self, topic: str, event: dict) -> str:
        # Assign monotonic ID, append to history, put to all queues
    async def subscribe(self, topic: str, from_id: str = "$") -> AsyncIterator[dict]:
        # from_id="$" -> new events only
        # from_id="0" -> replay full history first, then live
```

Abort consumer: Replace Redis subscription with `asyncio.Event`.
Abort sets the event; worker checks it at each step boundary.

SIEM/event forwarding: Uses the in-process bus directly.

Windows worker queue: Keep Redis ONLY when
`SERENADE_WINDOWS_WORKER_ENABLED=1`. The `redis` package moves to
an optional `[windows]` extra. Core code never imports redis at
module load time -- the import is inside `WindowsWorkerDispatcher`
methods, guarded by the enabled flag.

LOC touched: ~200 (bus.py, abort_consumer.py, siem.py)

### 3. MinIO -> local filesystem

evidence.py (93L): Replace MinIO client with pathlib writes:

```python
EVIDENCE_DIR = Path(data_dir) / "evidence"

async def put_blob(engagement_id, blob_id, data: bytes) -> tuple[str, str]:
    path = EVIDENCE_DIR / str(engagement_id) / blob_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()
    return blob_id, sha256

async def get_blob(engagement_id, blob_id) -> bytes:
    return (EVIDENCE_DIR / str(engagement_id) / blob_id).read_bytes()
```

SHA256 content-addressing stays. No locking needed (single writer).

LOC touched: ~80 (evidence.py, settings.py)

Concurrency note: Only one engagement runs at a time in a single
serenade process. Multiple engagements in parallel requires multiple
serenade instances with different `--data-dir` values.

### 4. OPA -> Python function

scope.rego (102L) becomes ~60 lines of Python in `src/policy/scope.py`:

```python
import ipaddress
from datetime import datetime, UTC

def evaluate(targets: list[str], engagement: dict, tool: dict) -> PolicyResult:
    for t in targets:
        if not _target_in_scope(t, engagement["scope"]):
            return PolicyResult(allow=False, reason="out_of_scope")
    if not _risk_allowed(tool["risk_class"], engagement["allowed_risk_classes"]):
        return PolicyResult(allow=False, reason="risk_class_not_allowed")
    if not _in_time_window(engagement["window"]):
        return PolicyResult(allow=False, reason="outside_time_window")
    if tool.get("destructive"):  # never executed in any mode — finding-only
        return PolicyResult(allow=False, reason="destructive_not_authorized")
    require_approval = (
        tool["risk_class"] in ("exploit", "post-exploit")
        and engagement["approval_policy"] == "phase_gate"  # derived from mode
    )
    return PolicyResult(allow=True, require_approval=require_approval, reason="ok")

def _target_in_scope(target: str, scope: dict) -> bool:
    # CIDR check
    for cidr in scope.get("cidrs", []):
        try:
            if ipaddress.ip_address(target) in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            pass
    # Domain check (exact or dotted-suffix)
    hostname = _extract_hostname(target)
    for domain in scope.get("domains", []):
        if hostname.lower() == domain.lower():
            return True
        if hostname.lower().endswith("." + domain.lower()):
            return True
    return False
```

engine.py (81L): Replace HTTP call to OPA with direct call to
`scope.evaluate()`. Delete `src/policy/rego/` directory.

engagement_dict.py (70L): Rename `build_opa_engagement()` to
`build_engagement_context()`. The dict structure stays (agents
consume `allowed_risk_classes`, `per_tool_token_budget`, etc.) --
it's not OPA-specific, just misleadingly named.

LOC touched: ~140 (engine.py rewrite, scope.py new, rego deleted)

### 5. Neo4j -> networkx (in-process graph)

neo4j_repo.py (281L): Rewrite to `InMemoryGraphRepo` wrapping
networkx `DiGraph`:

```python
class InMemoryGraphRepo:
    def __init__(self, graph: nx.DiGraph | None = None):
        self.G = graph or nx.DiGraph()
        # Secondary indices for fast lookups
        self._by_kind: dict[str, set[str]] = defaultdict(set)
        self._by_kind_field: dict[tuple[str, str, Any], set[str]] = defaultdict(set)

    async def upsert_node(self, node: WorldStateNode) -> None:
        if not node.provenance:
            raise ProvenanceRequiredError(...)
        existing = self.G.nodes.get(str(node.id))
        if existing:
            # Merge: update props, union provenance
            existing["provenance"] = list(set(existing["provenance"]) | set(map(str, node.provenance)))
            existing.update(node.model_dump(exclude={"id", "provenance"}, exclude_none=True))
        else:
            self.G.add_node(str(node.id), **node.model_dump())
            self._by_kind[node.kind].add(str(node.id))
        # Update secondary indices
        ...

    async def upsert_edge(self, edge: Edge) -> None:
        self.G.add_edge(str(edge.src_id), str(edge.dst_id),
                        type=edge.type, **edge.props)

    async def get_nodes_by_kind(self, engagement_id, kind) -> list[dict]:
        return [self.G.nodes[nid] for nid in self._by_kind.get(kind, set())
                if self.G.nodes[nid].get("engagement_id") == str(engagement_id)]
```

Provenance enforcement and deterministic UUIDv5 IDs stay unchanged
(they're Python checks).

queries.py (78L): 6 Cypher read queries become networkx traversals.
Example conversions:

| Cypher | networkx |
|---|---|
| `MATCH (s:Service)-[:RUNS_ON]->(h:Host) RETURN...` | `[(G.nodes[s], G.nodes[e[1]]) for s in by_kind["Service"] for e in G.out_edges(s, data=True) if e[2]["type"] == "RUNS_ON"]` |
| `MATCH (c:Credential {username: $u}) RETURN c LIMIT 1` | `_by_kind_field[("Credential", "username", u)]` (secondary index, O(1)) |
| `MATCH (n {id: $x})-[:EDGE_TYPE]->(m) RETURN m` | `[G.nodes[n] for n in G.successors(x) if G.edges[x, n].get("type") == edge_type]` |

Agent inline Cypher (~25 refs across 7 agent files): Convert each
to networkx API calls. These are tight lookup loops -- secondary
indices (`_by_kind_field`) keep them O(1) instead of naive O(N)
scans that would degrade agent latency.

BloodHound Neo4j: Drop entirely. BloodHound tool adapter writes
to the in-process graph (same node types, just imported data).

#### Graph persistence (serialization contract)

When: After each agent step completes (inside `_step_agent` in the
stage controller), serialize the graph.

How: `networkx.node_link_data(G)` -> JSON blob -> UPSERT into
`world_state_snapshots` table keyed by `(engagement_id)`.

Rehydration: On engagement start or resume, load the latest snapshot:
`G = networkx.node_link_graph(json.loads(blob))`. Rebuild secondary
indices from the loaded graph.

On interrupt + resume: The SQLiteCheckpointer saves EngagementState
(which has engagement_id + current_stage). The resuming code loads
the matching WorldStateSnapshot and initializes InMemoryGraphRepo(G).

Size: Engagements produce hundreds to low thousands of nodes. JSON
serialization of 5K nodes is <1MB and takes <50ms. Not a concern.

LOC touched: ~400 (neo4j_repo.py, queries.py, agent graph reads)

Dependencies: Remove `neo4j`. Add `networkx`.

### 6. API layer -> direct Python calls

Extract business logic from `api/*.py` into `services/*.py` as plain
async functions. The TUI calls these directly.

Key design decisions:

**Error handling**: API routes currently raise `HTTPException(status_code, detail)`.
Service functions must raise domain exceptions instead:

```python
# services/errors.py
class ServiceError(Exception):
    def __init__(self, message: str): ...

class NotFoundError(ServiceError): ...
class ValidationError(ServiceError): ...
class PolicyViolation(ServiceError): ...
class PermanentFailure(ServiceError): ...  # e.g., WeasyPrint unavailable
```

TUI catches `ServiceError` subtypes instead of checking HTTP status codes.
Example: `api/reports.py` raises `HTTPException(503)` when WeasyPrint fails.
After: `services/reports.py` raises `PermanentFailure("PDF rendering unavailable")`.
TUI: `except PermanentFailure as e: self.notify(str(e), severity="error")`.

**Depends() injection removal**: API files use FastAPI's `Depends()` for
lazy singletons (DB session, Neo4j repo). Extract into module-level
factories:

```python
# services/deps.py
_repo: InMemoryGraphRepo | None = None
async def get_repo() -> InMemoryGraphRepo:
    global _repo
    if _repo is None:
        _repo = InMemoryGraphRepo()
    return _repo

async def get_session() -> AsyncSession:
    async with session_scope() as s:
        yield s
```

Service functions call these directly. No dependency injection framework.

**Request object removal**: `api/engagements.py` accepts `request: Request`
but only uses it for client IP (rate limiting -- removed). The
`created_by` field is hardcoded `"local"`. After extraction, service
functions take explicit parameters: `async def create_engagement(payload: EngagementCreate, created_by: str = "local")`.

**File uploads (imports.py)**: Currently uses FastAPI `File()`/`Form()`.
After: service function takes raw bytes directly:
`async def import_scan(engagement_id, kind: str, data: bytes, filename: str = "")`.
TUI reads the file and passes bytes. Simpler.

**SSE events -> in-process subscription**: The TUI event stream widget
currently does HTTP SSE via `client.stream_events()`. After: it calls
`await bus.subscribe(engagement_id)` and iterates the async generator
directly. Same widget code structure, no HTTP.

LOC: ~2,400 deleted (api/), ~241 deleted (client.py), ~600 new (services/)

### 7. Docker Compose -> sandbox only

Delete: `docker-compose.yml`, `Dockerfile.orchestrator`,
`orchestrator-entrypoint.sh`.

Keep: `Dockerfile.sandbox` (the pentest tool image).

Sandbox runner (sandbox/runner.py, 467L): No change. Already calls
`docker run --rm` directly with no compose dependency.

Setup wizard: Check for `docker` on PATH, offer to install:
- Mac: `brew install --cask docker`
- Linux: `curl -fsSL https://get.docker.com | sh`

Sandbox image: `docker build` on first run if image doesn't exist,
or pull from a registry.

proxy-logger (mitmproxy): Make optional. Operator can run natively
with `pip install mitmproxy` if egress logging is desired.

### 8. pyproject.toml dependency cleanup

Remove (9 packages):
```
asyncpg, langgraph-checkpoint-postgres, neo4j, redis,
minio, fastapi, uvicorn, sse-starlette, python-multipart
```

Add (2 packages):
```
aiosqlite, networkx
```

Optional extras:
```
[windows] = ["redis>=5.1"]  # only for Windows worker VM IPC
```

Keep:
```
pydantic, pydantic-settings, structlog, anyio, httpx
anthropic, langgraph, openai
sqlalchemy[asyncio], alembic
jinja2, weasyprint, markdown-it-py
defusedxml, pyyaml, jsonschema
textual, httpx-sse, rich  (tui extra)
```

---

## What stays unchanged

| Component | LOC | Why it stays |
|---|---|---|
| Agent pipeline (src/agents/) | 8,421 | The product. Recon through report |
| Stage controller (stage_controller.py) | 1,186 | LangGraph FSM driving the 9-stage pipeline |
| Tool adapters (src/tools/) | 3,932 | nmap, nuclei, ffuf, netexec, impacket, etc |
| Sandbox runner (src/sandbox/) | 1,308 | Per-call Docker container isolation |
| LLM router (src/llm/) | 947 | Ollama + Anthropic dispatch. No infra deps |
| Report renderer (src/reports/) | 293 | HTML/PDF/Markdown/JSON/SARIF templates |
| World-state schema (worldstate/schema.py) | 533 | Pydantic node types, UUIDv5 IDs, edges |
| PoC library (content/pocs/) | ~23 files | Curated exploits with manifests |
| TUI (clients/tui/) | 2,411 | Screens and widgets. Only client.py changes |
| Safety gates | -- | Three-gate model unchanged (policy + critic + approval) |
| Windows worker (clients/windows_worker/) | ~800 | Optional, external VM, PowerShell |
| Tests (tests/) | 24,572 | Infra-coupled tests already pruned |

---

## File-level change map

### Delete entirely
```
docker-compose.yml
infra/docker/Dockerfile.orchestrator
infra/docker/orchestrator-entrypoint.sh
src/orchestrator/main.py                 FastAPI app
src/orchestrator/api/                    all 10 route files (2,400L)
src/orchestrator/checkpoint.py           Postgres checkpointer (replaced)
src/policy/rego/                         OPA Rego bundle
clients/tui/client.py                    HTTP client
```

### Rewrite
```
src/orchestrator/db.py                   Postgres DSN -> SQLite DSN
src/orchestrator/models.py               JSONB -> JSON, drop removed tables
src/orchestrator/evidence.py             MinIO -> filesystem
src/orchestrator/events/bus.py           Redis Streams -> asyncio queues
src/orchestrator/abort_consumer.py       Redis sub -> asyncio.Event
src/orchestrator/siem.py                 Redis -> in-process bus
src/orchestrator/settings.py             remove infra settings, add data_dir
src/policy/engine.py                     OPA HTTP -> direct Python call
src/policy/engagement_dict.py            rename build_opa_engagement()
src/worldstate/neo4j_repo.py             Neo4j driver -> networkx + indices
src/worldstate/queries.py                Cypher -> networkx traversals
src/orchestrator/importers/nmap_xml.py   Neo4j repo -> InMemoryGraphRepo
src/orchestrator/importers/nuclei_jsonl.py  same
src/cli/main.py                          remove docker compose, inline startup
src/agents/recon.py                      Cypher -> networkx (9 refs)
src/agents/lateral.py                    Cypher -> networkx (6 refs)
src/agents/loot.py                       Cypher -> networkx (3 refs)
src/agents/privesc.py                    Cypher -> networkx (3 refs)
src/agents/exploit.py                    Cypher -> networkx (1 ref)
src/agents/web_enum.py                   Cypher -> networkx (1 ref)
```

### New
```
src/orchestrator/services/               business logic from api/ (plain async fns)
src/orchestrator/services/errors.py      domain exception hierarchy
src/orchestrator/services/deps.py        singleton factories (repo, session)
src/orchestrator/services/engagements.py CRUD + lifecycle
src/orchestrator/services/findings.py    graph queries for findings
src/orchestrator/services/reports.py     report rendering
src/orchestrator/services/events.py      in-process event subscription
src/orchestrator/services/cost.py        cost tracking
src/orchestrator/services/imports.py     scan file import
src/orchestrator/services/operator_exec.py  one-off command execution
src/policy/scope.py                      Python port of scope.rego
src/orchestrator/sqlite_checkpoint.py    LangGraph SQLiteCheckpointer
```

### Modify (light touch)
```
clients/tui/widgets/*.py                 client.foo() -> services.foo()
clients/tui/screens/*.py                 same
pyproject.toml                           dependency swap
infra/alembic/                           squash to single SQLite migration
```

### Delete or keep as-is (no change needed)
```
src/orchestrator/ledger.py               uses SQLAlchemy, not Postgres-specific
src/orchestrator/thresholds.py           pure Python constants, no infra deps
src/orchestrator/cost.py                 pure computation
src/orchestrator/evidence_scrub.py       regex patterns, no infra deps
src/orchestrator/provenance.py           validation logic, no infra deps
src/orchestrator/windows_worker.py       keeps Redis internally (optional)
src/orchestrator/scripts/seed_demo.py    keep for demo mode
```

---

## Test impact

Tests that need changes after the migration:

| Test pattern | Count | Change needed |
|---|---|---|
| Tests importing `HTTPException` and asserting status codes | 6 files | Assert domain exceptions instead |
| Tests that mock Neo4j driver | ~5 files | Mock InMemoryGraphRepo instead |
| Tests that mock Redis | ~3 files | Mock in-process EventBus |
| Tests that mock MinIO | ~2 files | Mock filesystem (tmp_path fixture) |
| Tests calling `from orchestrator.api.auth import ...` | 0 (already pruned) | -- |
| Integration tests needing live services | 3 files | Rewrite to use SQLite + networkx |

Most unit tests already mock at the Python interface level (they mock
`Neo4jRepo`, not the Neo4j wire protocol). The interface stays the
same (`upsert_node`, `get_nodes_by_kind`), just backed by networkx.
These tests need zero changes.

Specific files needing exception-type updates:
- test_engagement_update_consent_gate.py
- test_engagement_update_lab_mode_gate.py
- test_engagement_abort.py
- test_engagement_clone_consent.py
- test_engagement_summary_endpoint.py
- test_regression_api.py

---

## Execution order

### Phase 1 -- Drop-in replacements (no architecture change)

These swap one backend for another behind the same interface. Nothing
else in the codebase notices.

1. MinIO -> filesystem. Rewrite evidence.py (93L).
2. OPA -> Python. Rewrite engine.py (81L), write scope.py (~60L),
   delete src/policy/rego/.
3. Delete docker-compose infra. Remove docker-compose.yml,
   Dockerfile.orchestrator, orchestrator-entrypoint.sh.

### Phase 2 -- Data layer

4. Postgres -> SQLite. Rewrite db.py, update models.py, squash Alembic,
   write SQLiteCheckpointer.
5. Redis -> asyncio. Rewrite events/bus.py, abort_consumer.py, siem.py.
   Move redis to optional [windows] extra.
6. Neo4j -> networkx. Rewrite neo4j_repo.py, queries.py, update 7
   agent files + 2 importer files. Add secondary indices. Add graph
   persistence (serialize to SQLite blob after each step).

### Phase 3 -- Collapse the process boundary

7. Write services/errors.py (domain exception hierarchy).
8. Extract services from api/ endpoints into services/*.py as plain
   async functions. Replace HTTPException with domain exceptions.
   Replace Depends() with direct factory calls.
9. TUI direct calls. Replace client.foo() with services.foo() in every
   TUI widget/screen. Delete client.py.
10. Delete API layer. Remove api/, main.py, FastAPI/uvicorn deps.

### Phase 4 -- Single entry point

11. Unified startup. `serenade` initializes SQLite, checks for Docker,
    builds sandbox image if needed, starts the TUI in-process. No
    docker compose, no separate worker process.
12. Setup wizard. Auto-install Docker if missing, auto-build sandbox
    image, create ~/.serenade/.

### Phase 5 -- Rewrite all docs

Every doc except POCS.md references removed infrastructure.

13. README.md (162L) -- complete rewrite. "Autonomous pentest tool,
    runs on your laptop." Quick start: pip install + serenade.
14. ARCHITECTURE.md (1,119L) -- heavy rewrite. Single-process topology,
    networkx graph, asyncio events, SQLite, no containers.
15. LAUNCH_FIRST_TIME.md (138L) -- rewrite. pip install -> serenade
    --setup -> serenade.
16. RUNBOOK.md (217L) -- rewrite. Where data lives (~/.serenade/),
    how to back up (copy dir), troubleshooting Docker/sandbox.
17. TUI.md (174L) -- update. Remove login/auth/MFA references.
18. POCS.md (166L) -- no changes needed.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| SQLite concurrent access | Single writer enforced. Multiple reads during TUI polling | WAL mode (`PRAGMA journal_mode=WAL`) |
| networkx graph too large | Engagements produce hundreds of nodes, not millions | Log warning at 100K nodes. Has never happened |
| Approval checkpoint lost on crash | Process crash between interrupt and approval loses state | SQLiteCheckpointer persists to disk. Crash = reload from checkpoint, not restart from scratch |
| Inline Cypher -> networkx performance | Naive O(N) scans on 10K nodes in tight loops | Secondary indices (`_by_kind_field`) for O(1) lookups |
| Tests coupled to removed backends | ~16 test files need updates | Most mock at Python interface (unchanged). 6 files need exception-type swaps |
| WeasyPrint system deps | Hard to pip install on some systems | PDF is optional. HTML/Markdown work without it |
| Parallel engagement isolation | Two serenade processes on same laptop could conflict | Single process constraint. Multiple instances use `--data-dir` for isolation |
| Windows worker Redis | Optional dep only loaded when enabled | Guarded import inside WindowsWorkerDispatcher. Core never touches redis |

---

## Critic review findings (incorporated)

Three expert critics reviewed this plan. Their findings are addressed
below as amendments to the design.

### Concurrency model (TUI + agent in same process)

Problem: Textual runs its own event loop. LangGraph's `ainvoke()` is a
long-running coroutine. If both share one loop, the TUI freezes during
agent steps.

Solution: The agent loop runs as an `asyncio.Task` spawned from the
TUI's event loop. Textual is already asyncio-native (uses `trio` or
`asyncio` depending on backend). The agent task publishes events to
the in-process EventBus; TUI widgets subscribe and redraw on each
event. Ctrl+C cancels the agent task (same as today's abort flow but
without Redis).

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

### Graph + checkpoint atomicity

Problem: If the process crashes between graph serialization and
checkpoint save, they're inconsistent on restart.

Solution: Both happen inside a single SQLite transaction. The
SQLiteCheckpointer's `put()` method:
1. Opens a transaction
2. UPSERTs the graph JSON blob to `world_state_snapshots`
3. UPSERTs the LangGraph state to `checkpoints`
4. Commits

If the process crashes mid-transaction, SQLite rolls back both. On
resume, graph and checkpoint are always consistent (either both at
step N or both at step N-1).

### EngagementState serialization

Problem: `warnings_fired: set[str]` is not JSON-serializable.

Fix: Change to `list[str]` in the Pydantic model. The validator
already coerces incoming data; outgoing serialization uses
`model_dump(mode="json")` which handles lists natively. Apply the
same fix to any other set-typed fields discovered during implementation.

### Secondary index consistency on node updates

Problem: When a node's indexed field changes, the old index entry is
stale.

Fix: On upsert with existing node, compare old vs new values for
indexed fields. Remove from old index entries before adding new ones:

```python
if existing:
    for field in INDEXED_FIELDS:
        old_val = existing.get(field)
        new_val = props.get(field)
        if old_val != new_val and old_val is not None:
            self._by_kind_field[(kind, field, old_val)].discard(node_id)
        if new_val is not None:
            self._by_kind_field[(kind, field, new_val)].add(node_id)
```

### SQLiteCheckpointer protocol

Problem: LangGraph's checkpointer may require async context manager
protocol (`__aenter__`/`__aexit__`).

Fix: Implement both. The `__aenter__` opens the SQLite connection; the
`__aexit__` closes it. This is trivial (~5 lines) and matches the
current `AsyncPostgresSaver.from_conn_string()` pattern.

### Service layer LOC estimate

Problem: 600 LOC is optimistic. engagements.py alone is 1,507L.

Fix: Revised estimate is ~1,500 LOC for services/. The 1,507L in
engagements.py breaks down as: ~400L DTOs/validators (keep as-is),
~300L FastAPI glue (delete), ~800L business logic (extract). Other
API files add ~700L of extractable logic. Total services/: ~1,500L.
Net reduction still significant (2,400 API + 241 client deleted,
1,500 services added = -1,141 LOC net).

### SQLite Alembic limitations (DROP COLUMN)

Problem: SQLite < 3.35.5 can't do ALTER TABLE DROP COLUMN. Several
migrations use it.

Fix: Since we're squashing all migrations into a single CREATE TABLE
migration for the native architecture, this is a non-issue. The
squashed migration creates all tables with their final schema. No
ALTER TABLE needed. Alembic batch mode is available as a fallback for
any future migrations that need column changes.

### Parser sandbox retention (security critical)

Problem: After migration, XML/JSON parsers (nmap, nuclei) run
in-process. A parser exploit gives full access to the serenade
process (LLM keys, credentials, everything).

Decision: Keep tool OUTPUT PARSING in the sandbox container. The tool
runs in the container AND its output is parsed there too. The
container returns structured JSON (Pydantic-serialized) on stdout.
The serenade process only deserializes trusted JSON, never raw XML.

Implementation: Each tool adapter's `parse_output()` runs inside the
sandbox container (not in the host process). The container entry point
is: run tool -> parse output -> emit JSON on stdout. The host reads
JSON and validates via Pydantic. This is already close to the current
design (parsers run in the orchestrator container); we just keep them
in the sandbox container instead of moving them to the host.

This means `defusedxml`, nmap parsing logic, and nuclei JSONL parsing
stay inside `Dockerfile.sandbox`, not in the pip-installed package.

### Credential encryption at rest

Problem: Credential nodes (passwords, NT hashes, SSH keys) stored in
plaintext in SQLite and in-memory networkx graph.

Decision: Encrypt sensitive fields before SQLite persistence. The
graph serialization step redacts credential fields to `[ENCRYPTED]`
in the JSON blob. A separate `credentials` table stores encrypted
values keyed by (engagement_id, credential_node_id). Decryption
happens on-demand when the agent needs the actual value for a tool
call.

Key derivation: From a per-install master key stored at
`~/.serenade/.master_key` (chmod 0600, generated on first run).
Fernet (AES-128-CBC + HMAC-SHA256) via the existing `cryptography`
package — actually, we removed `cryptography`. Use `hashlib` +
`hmac` from stdlib for HMAC, and add back `cryptography` as a
dependency (it was removed as an auth dep, but it's needed for
credential protection).

### Audit log integrity

Problem: SQLite audit log is trivially tamperable by the operator.

Decision: Merkle hash chain. Each `AuditLogRow` gets a `prev_hash`
column. New row hash = SHA256(prev_hash + row_content). On export,
verify the chain. Tampering (delete or modify a row) breaks the chain
and is detectable by any downstream consumer (e.g., client receiving
the report).

This doesn't prevent the operator from tampering (it's their laptop),
but it makes tampering DETECTABLE by the client receiving the report.

### Prompt injection guards in base agent

Problem: Critic prompts have injection guards but the main planner
prompts don't. Parser output lands in planner context.

Fix: Prepend the same `_PROMPT_INJECTION_GUARD` text to all planner
system prompts (not just critic). Already exists in `agents/critic.py`;
factor it into a shared constant in `agents/prompts.py` and reference
from both `BaseAgent._build_planner_messages()` and critic prompts.

### L3 sandbox isolation default

Problem: `sandbox_l3_required` defaults to False. Operators may think
they have network-layer scope enforcement when they don't.

Fix: Default to True in the native architecture. A laptop tool should
have maximum isolation by default. Operators can set
`sandbox_l3_required=false` for lab environments where nftables isn't
available.

### First-run experience (DX)

The setup wizard (`serenade --setup`) must handle:

1. Check Docker installed + daemon running. If not: print
   platform-specific install command and exit.
2. Build sandbox image (5-10 min, show progress). Offer to skip
   if image already exists.
3. Pull Ollama model (8-12 GB, show progress). Warn if RAM < 32 GB
   and offer Anthropic as alternative.
4. Write config to `~/.serenade/config.toml` (not .env in repo root).
5. Print data location and basic commands.

Config format: TOML at `~/.serenade/config.toml`, not .env. Simpler,
discoverable, survives repo deletion. Example:

```toml
[llm]
backend = "ollama"  # or "anthropic"
# anthropic_api_key = "sk-ant-..."  # only if backend = anthropic

[sandbox]
l3_required = true

[budget]
token_budget = 2000000
wallclock_hours = 8
```

### Engagement quick-start

Problem: Creating an engagement asks 10 questions. A pentester with
30 minutes wants: name, scope, go.

Fix: TUI offers two paths:
- Quick start (3 fields): name, CIDRs/domains, Go. Defaults: guided
  mode (exploits pause for approval), 2M token budget, 8h wallclock.
- Advanced (full form): all fields visible for tuning.

Default is quick start. Press `Tab` or `[Advanced]` for full form.

### Error messages

All pre-flight checks produce actionable errors:
- "Docker daemon not running. Start Docker Desktop (macOS: open -a Docker)"
- "Sandbox image not found. Run: serenade --setup"
- "Ollama not responding on localhost:11434. Start it: ollama serve"
- "RAM: 16 GB detected. Gemma 4 26B needs ~20 GB. Consider: serenade --setup to switch to Anthropic"
