<div align="center">

# Serenade

[Install](#install) &nbsp;•&nbsp; [Documentation](#documentation) &nbsp;•&nbsp; [Features](#features) &nbsp;•&nbsp; [Getting help](#getting-help) &nbsp;•&nbsp; [Contribute](#contribute)

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Backend](https://img.shields.io/badge/llm-local%20first%20(ollama%20%2B%20gemma%204)-brightgreen)
![Sandbox](https://img.shields.io/badge/sandbox-docker%20run%20--rm-informational)
![Gates](https://img.shields.io/badge/safety%20gates-3%20per%20call-orange)
![License](https://img.shields.io/badge/license-proprietary-lightgrey)

```
recon → enum → triage → exploit → post-exploit → privesc → lateral → loot → report
        └──────────────── three safety gates per tool call ────────────────┘
              scope/risk policy  +  LLM critic  +  optional human approval
```

</div>

> *It is a story about an operator who had run one engagement too many — too many half-remembered playbooks, too many tools left running against the wrong subnet, too many findings written up at 3am. They wanted an agent that would run the whole pipeline on its own, but one that could not be talked into stepping out of scope, and one that wrote down every decision it made. So they built Serenade: autonomous from recon to report, but never trusted further than three independent gates will allow.*

<div align="center">

**⚠️ Authorized use only.** Serenade is a pentesting tool — use it solely on systems you own or are
authorized in writing to administer. [Full disclaimer ↓](#authorized-use--disclaimer)

</div>

---

**Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [CLI reference](#cli-reference)
- [Documentation](#documentation)
- [Getting help](#getting-help)
- [Authorized use & disclaimer](#authorized-use--disclaimer)
- [Contribute](#contribute)

---

## Introduction

Serenade is an autonomous penetration testing tool tailored for operators who want a real agent driving the engagement — recon through reporting — without giving up scope discipline, auditability, or the ability to run it all on their own laptop. Point it at a scope, and stage agents work the pipeline; on every single tool call, three independent safety gates have to agree before anything touches the target. It runs as one native Python process. No cluster, no compose stack, no cloud.

It is built around a few mantras:

- **Autonomous, but never unsupervised.** The agent plans and acts on its own, but every tool call passes a deterministic scope/risk policy, a structured LLM critic, and — when you want it — a human approval queue. Any one of them can veto.
- **Local-first.** Ollama + Gemma 4 is the default backend. No API key, no data leaving the box. The Anthropic backend exists, but it is strictly opt-in — a key in your shell is never enough.
- **Write everything down.** An append-only audit log records every gate decision and state transition. The world-state graph uses deterministic IDs and mandatory provenance. The report ships with the full evidence chain.
- **Sandbox by default.** Serenade runs natively; the *tools* it invokes do not. Every `nmap`, `nuclei`, or `ffuf` call executes in a throwaway `docker run --rm` container — no host bind-mounts, no persistent state.
- **Opinionated, but not stubborn.** Sensible playbooks per stage out of the box, with the spicier agent behaviors (step scoring, reflection memory, debate gates) behind opt-in flags that all default to off.

## Features

- **Full pipeline.** RECON → ENUM → TRIAGE → EXPLOIT → POST-EXPLOIT → PRIVESC → LATERAL → LOOT → REPORT, driven by stage agents with per-stage playbooks and orchestrated as a LangGraph stage controller.
- **Three safety gates per tool call.** A deterministic Python scope/risk policy, a structured LLM critic, and an optional human approval queue — independent, and any one can veto.
- **Curated PoC library.** Reviewed exploits live in `content/pocs/`, AST-linted and CI-tested against pinned lab targets.
- **Sandbox isolation.** Per-call ephemeral `docker run --rm` containers with no host bind-mounts and no shared state between calls.
- **World-state graph.** A networkx graph with deterministic UUIDv5 node IDs and mandatory provenance on every node and edge.
- **Append-only audit log.** Every gate decision and stage transition is recorded and exported alongside the report.
- **Bounded regression.** Council-verified backwards stage transitions, capped by token and wall-clock budgets so the agent can't loop forever.
- **Multi-format reports.** HTML, PDF, Markdown, JSON, and SARIF, each carrying the full evidence chain.
- **Config snapshotting.** Engagements snapshot their LLM config at create-time, so editing `.env` never disturbs an in-flight run.
- **Optional Windows worker.** A PowerShell agent for mimikatz / .NET assemblies, off unless you wire up a VM and enable it.
- **Opt-in agent patterns** (all default off):
  - `enable_step_scoring` — per-step promise/progress scores; backtrack on dead-end paths.
  - `enable_reflection_memory` — strategy-level verbal memory injected into planner prompts.
  - `enable_debate_gates` — attacker + defender debate at exploit/lateral gates.
  - `enable_regression_reflection` — LLM-suggested regressions at stage ADVANCE boundaries.

## Prerequisites

Serenade requires the following:

- **Python** 3.11 or newer.
- **Docker**, with the daemon running. This is *not* infrastructure — it is used only for per-call sandbox isolation of pentest tools.
- An **LLM backend**, one of:
  - **Ollama + Gemma 4** (default, recommended). Minimum 32 GB RAM (Apple Silicon) or 32 GB VRAM (Linux/CUDA). Serenade manages the Ollama lifecycle automatically — starts it on launch, stops it on exit.
  - **Anthropic API** (opt-in). Requires `ANTHROPIC_API_KEY` *and* `SERENADE_LLM_BACKEND=anthropic` set explicitly in `.env`.

Optional:

- **WeasyPrint** system libraries, if you want PDF report export.
- A **Windows VM** running `clients/windows_worker/agent/agent.ps1`, only if you enable the Windows worker.

> [!WARNING]
> Serenade is an offensive security tool. Only run it against systems you are explicitly authorized to test. The scope/risk policy is a guardrail, not a substitute for written authorization.

> [!IMPORTANT]
> Having `ANTHROPIC_API_KEY` in your shell is **not** enough to enable the hosted backend. Serenade will not silently spend against a key that was set for another tool — you must also set `SERENADE_LLM_BACKEND=anthropic`.

## Install

```bash
pip install serenade
serenade --setup            # one-time wizard: picks LLM backend, writes .env
serenade                    # launches the TUI
```

For the local backend, pull the models once (or run `make ollama-pull`):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:26b       # planner + parser (~16 GB)
ollama pull gemma4           # critic / synthesizer / triage (E4B, ~3 GB)
```

To kick the tires, seed a DVWA demo engagement:

```bash
python -m orchestrator.scripts.seed_demo
```

Model routing per call site:

| Call site         | Anthropic default            | Local (Ollama) |
|-------------------|------------------------------|----------------|
| `PLANNER`         | `claude-sonnet-4-6`          | `gemma4:26b`   |
| `SYNTHESIZER`     | `claude-opus-4-7`            | `gemma4` (E4B) |
| `PARSER`          | `claude-haiku-4-5-20251001`  | `gemma4:26b`   |
| `TRIAGE`/`CRITIC` | `claude-haiku-4-5-20251001`  | `gemma4` (E4B) |

See [`.env.example`](./.env.example) for the full configuration surface.

## CLI reference

| Command | Effect |
|---------|--------|
| `serenade` | Launch the TUI |
| `serenade --setup` | First-run wizard, writes `.env` |
| `serenade --migrate` | Apply pending Alembic migrations |
| `serenade --test [unit\|integration\|e2e\|llm\|all]` | Run test suites |
| `serenade engagement watch <id>` | Tail the live event stream |
| `serenade engagement summary <id>` | Operator digest (spend, vetoes, findings) |

Shell completions: `source scripts/completions/serenade.bash` (or `.zsh`).

## Documentation

| Doc | Purpose |
|-----|---------|
| [LAUNCH_FIRST_TIME.md](./docs/LAUNCH_FIRST_TIME.md) | First-time setup: install to first engagement in ~5 min |
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Internals for contributors: topology, agent loop, gates, state |
| [RUNBOOK.md](./docs/RUNBOOK.md) | Operational playbook: backup, reset, troubleshooting |
| [TUI.md](./docs/TUI.md) | Terminal client reference: layout, keyboard shortcuts, workflow |
| [POCS.md](./docs/POCS.md) | Adding a curated exploit: manifest, implementation, CI, review |

Repo layout:

```
src/                    production Python packages
    cli/                    CLI: setup wizard, startup, TUI launch
    agents/                 stage agents, critic/council, PoC runtime/lint
    llm/                    router, Anthropic + local (Ollama) backends, prompts
    orchestrator/           DB models, evidence store, event bus, stage controller (LangGraph)
    policy/                 Python scope/risk policy engine
    sandbox/                per-call SandboxRunner, Docker container isolation
    tools/                  typed adapters: nmap, httpx, nuclei, ffuf, netexec, ...
    worldstate/             Pydantic schemas, networkx graph repo, queries
    worker/                 queue worker that runs the stage graph for an engagement
    reports/                Jinja2 templates, HTML/PDF/MD/JSON/SARIF renderer
clients/
    tui/                    Textual terminal client
    windows_worker/         PowerShell agent for mimikatz / .NET assemblies (optional)
content/pocs/           curated exploit library
infra/                  Dockerfiles, Alembic migrations, scripts
tests/                  unit/, integration/, fixtures/
docs/                   documentation set
```

## Getting help

- **Start with the docs.** [LAUNCH_FIRST_TIME.md](./docs/LAUNCH_FIRST_TIME.md) gets you from install to first engagement; [RUNBOOK.md](./docs/RUNBOOK.md) covers backup, reset, and troubleshooting.
- **Inspect a running engagement.** `serenade engagement watch <id>` tails the live event stream; `serenade engagement summary <id>` prints the operator digest (spend, vetoes, findings).
- **Read the audit log.** Every gate decision and stage transition is recorded and exported with the report — the fastest way to understand *why* the agent did or didn't do something.
- **Understand the internals.** [ARCHITECTURE.md](./docs/ARCHITECTURE.md) walks the topology, the agent loop, the three gates, and the world-state model.

# Authorized use & disclaimer

> [!CAUTION]
> Use it **only** on systems you own or are explicitly authorized in
> writing to administer.

Usage of Serenade for pentesting or acting on systems without prior
mutual consent is illegal. It is the end user's responsibility to obey all
applicable local, state, national, and international laws, and to obtain any
authorization required before operating against a host. The authors and
contributors assume no liability and are not responsible for any misuse or
damage caused by this program.

Serenade is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement. In no event shall
the authors or contributors be liable for any claim, damages, or other
liability arising from, out of, or in connection with the software or its
use.

## Contribute

Contributions are welcome — especially curated PoCs, tool adapters, and policy rules.

- **Add a PoC.** Follow [POCS.md](./docs/POCS.md): write the manifest, implement it, get it AST-linted and CI-tested against a pinned lab target, then submit for review.
- **Add a tool adapter.** Typed adapters live in `src/tools/`; match the existing `nmap` / `nuclei` / `ffuf` patterns.
- **Run the checks.** `make lint`, `make type`, and `make test` before opening a PR; `make help` lists every dev target.
- **File security issues privately.** Open a private advisory with a `security:` prefix — do not open a public issue.
