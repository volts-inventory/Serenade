# Launch -- first time

A quickstart for the solo operator. ~5 minutes from install to your first engagement.

## Prerequisites

- **Python 3.11+**
- **Docker** (daemon must be running -- used for sandbox isolation only)
- **32 GB RAM** recommended (for local LLM). Works with less but models swap instead of co-residing.

## 1. Install

```bash
pip install serenade
```

For development:

```bash
git clone https://github.com/volts-inventory/serenade && cd serenade
pip install -e ".[dev,tui]"
```

## 2. Setup

```bash
serenade --setup
```

The wizard:

1. Checks Docker is installed and the daemon is running.
2. Builds the sandbox image (5-10 min first time; skipped if it already exists).
3. Asks which LLM backend you want (Ollama local or Anthropic API).
4. If Ollama: pulls the required models (~19 GB total). Warns if RAM < 32 GB and offers Anthropic as an alternative.
5. If Anthropic: asks for your API key.
6. Writes config to `.env` (project root; see `.env.example`).

## 3. Set up the local LLM (if you chose Ollama)

If the setup wizard handled Ollama for you, skip this. Otherwise:

```bash
# Install Ollama (macOS + Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models (one-time, multi-GB download)
ollama pull gemma4:26b    # planner + parser (~16 GB)
ollama pull gemma4        # critic / synthesizer / triage (E4B, ~3 GB)
```

Minimum hardware: 32 GB RAM (Apple Silicon) or 32 GB VRAM (Linux/CUDA). On machines with less memory, both models still work but will swap between calls instead of co-residing.

## 4. Launch

```bash
serenade
```

This starts a single process: initializes SQLite, checks Docker, starts the local LLM (if Ollama), and opens the TUI. No containers to bring up, no stack to manage.

Data lives at `~/.serenade/`:

| Path | Contents |
|---|---|
| `~/.serenade/serenade.db` | SQLite database (engagements, audit log, checkpoints) |
| `~/.serenade/evidence/` | Tool output blobs (content-addressed) |
| `.env` (project root) | Your configuration (LLM backend, sandbox, budgets) |
| `~/.config/serenade/config.toml` | TUI preferences (URL, report download dir) |

## 5. Demo

```bash
python -m orchestrator.scripts.seed_demo
```

Loads a DVWA demo engagement so you can explore the TUI and see what the pipeline looks like without pointing at a real target.

## 6. Your first engagement

Use a target you own -- a local DVWA, a HackTheBox box, your own homelab. **Not** real customer infrastructure without an authorization letter.

**Recommended first-engagement settings:**

| Field | Value | Why |
|---|---|---|
| Mode | `guided` *(default)* | Exploit/post-exploit calls pause for your approval. Don't jump to `autonomous` until you've watched the agent behave. Use `recon_only`/`assess` for no-exploitation runs. |
| Wallclock | 2 hours | Plenty for a single-box test. |
| Token budget | 1,000,000 | Generous for local; constraining for hosted. |

In the TUI: press `n` to start the new-engagement wizard (quick start: just name + scope + go).

## 7. Watch it run

```bash
serenade engagement watch <id>     # tail the event stream, colorized
serenade engagement summary <id>   # snapshot -- spend, vetoes, regressions, findings
```

You can also stay in the TUI -- the Events tab (`2`) tails the same stream with color coding.

**Threshold warnings** fire automatically when:
- Token spend exceeds 80% of budget
- Critic-veto rate exceeds 50% over the last 10 verdicts
- Tools are being terminally suppressed during a stage

These appear as colored rows in the TUI event stream and as glows on the Overview tab tiles.

## 8. Read the report

When the engagement reaches the `report` stage, open the TUI's **Report**
tab and pick a format (HTML / PDF / Markdown / JSON / SARIF). The selected
file is written to your `download_dir` (default: the current directory) as
`report-<engagement-id>.<ext>`:

- HTML -- customer-facing report
- Markdown -- editable source
- JSON / SARIF -- machine-readable for pipelines

The audit trail (every gate decision) is recorded in the database and
rendered alongside the report. It is tamper-evident at rest via a
per-engagement master-key HMAC hash chain, verified on export — see
[ARCHITECTURE.md](./ARCHITECTURE.md#audit-log-integrity).

## 9. Back up your data

Your data is a single directory. Copy it:

```bash
cp -r ~/.serenade/ /path/to/backup/
```

Or automate with rsync/restic/whatever you prefer. See [RUNBOOK.md](./RUNBOOK.md) for details.

## 10. (Optional) Enable advanced agent patterns

Four opt-in features ship disabled by default (scoring, memory, debate, reflection). Enable them in `.env` when ready to experiment. Start with `ENABLE_STEP_SCORING=1` -- it adds one cheap E4B call per step and surfaces promise/progress scores in the audit log.

Full list and architecture: [ARCHITECTURE.md - Modern agent patterns](./ARCHITECTURE.md#modern-agent-patterns).

## Common follow-ups

| Want to | Do |
|---|---|
| Pause a running engagement | TUI: Overview tab -> Pause button |
| Abort hard (kills in-flight containers) | TUI: red Abort button + typed-CONFIRM modal |
| Approve all pending gates from a stage | TUI: Approvals tab -> "Approve all in {stage}" |
| Reset everything | Delete `~/.serenade/serenade.db` and restart |
| Change LLM backend | `serenade --setup` or edit `.env` |
| Get help | `serenade --help`, `serenade engagement --help` |

## Heads-ups

- **Auto mode** (the toggle on the new-engagement form, default off) sets `mode=autonomous` + `auto_start=true`: the full pipeline runs and curated/non-destructive EDB exploits auto-execute without per-gate approval. **Destructive** actions (DoS/wipe/lockout) are denied by default and only run if you set the separate `destructive_opt_in` flag — and even then they are critic-reviewed and audit-logged (and require a human click in `guided`). Use `guided` if you want to approve each exploit.
- **`ANTHROPIC_API_KEY` in your shell is ignored** unless `SERENADE_LLM_BACKEND=anthropic` is also set in `.env`. A one-time warning at startup tells you when this happens.
- **Local LLM minimum**: Gemma 4 26B for planning/parsing, Gemma 4 E4B for critic/synthesizer/triage. Ollama is the recommended runtime. See step 3 above.

That's it. Ship it.
