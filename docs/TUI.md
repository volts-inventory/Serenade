# Terminal UI

Serenade's primary interface. Built with Textual, runs in-process.

```
+-- engagements ---------------+-- Overview Events Approvals Findings Hosts Op-Exec Report -+
| * acme-q3-external           |  acme-q3-external   id=eng_4a9c...                         |
|   running - exploit          |  status: running    stage: exploit                          |
| * lab-dvwa-demo              |  recon -> enum -> triage -> exploit -> ... -> report        |
|   completed                  |  mode: autonomous (exploits auto-execute) (p to cycle)      |
|                              |  cost so far: $0.43  tokens: 91,200  calls: 14              |
|                              |                                                             |
|                              |  [Start]  [Pause]  [Resume]                                 |
|                              |                                                             |
|  status bar: local - eng_4a9c - $0.43 - 0 pending                                         |
+------------------------------+-------------------------------------------------------------+
```

---

## Launch

```bash
serenade                    # default -- start LLM, open the TUI
```

`serenade` handles Docker checks, the local LLM lifecycle, and the TUI
in one step. Closing the TUI tears the LLM back down.

---

## Layout

| Region | Content |
|---|---|
| Sidebar (left) | All engagements; click or arrow-key to select |
| Tabs (top right) | Overview, Events, Approvals, Findings, Hosts, Op-Exec, Report |
| Status bar (bottom) | Selected engagement, live cost, pending approvals |
| Footer | Active key bindings for the current screen |

The widget files map to the panels: `clients/tui/widgets/overview_panel.py`, `event_stream.py`, `approvals_panel.py`, `findings_panel.py`, `operator_exec_panel.py`, `report_panel.py`, `engagement_list.py`, `status_bar.py`. (The Hosts panel shares `findings_panel.py` rather than having its own file.)

---

## Keyboard

### Global (any screen)

| Key | Action |
|---|---|
| `q` | Quit |
| `n` | Create a new engagement (quick start: name + scope + go; or [Advanced] for full form) |
| `r` | Refresh engagements + selected engagement detail |
| `p` | Cycle the selected engagement's mode `recon_only -> assess -> guided -> autonomous -> recon_only` |
| `1`-`7` | Switch to tab 1..7 (Overview, Events, Approvals, Findings, Hosts, Op-Exec, Report) |

### Approvals tab

| Key | Action |
|---|---|
| `a` | Approve the focused row |
| `r` | Reject the focused row |
| `Shift+a` | Bulk-approve every pending row of the focused row's kind |
| `Shift+r` | Bulk-reject every pending row of the focused row's kind |

### Events tab

| Key | Action |
|---|---|
| `space` | Pause / resume the live event tail |

Threshold-warning events get distinct rendering -- yellow for
`engagement.budget_warning` and `engagement.suppression_spread`,
red for `engagement.veto_rate_warning`.

### Findings tab

| Key | Action |
|---|---|
| `Enter` | Open the focused finding (full evidence + critic verdict + tool_run_id) |

---

## Daily-driver flow

1. **Launch** -- `serenade`.
2. **Pick or create an engagement** -- sidebar shows recent ones; press `n` for a new one. Quick start asks for name + CIDRs/domains and uses safe defaults.
3. **Pre-flight** -- paste the 64-char SHA-256 of your signed authorization letter.
4. **Start it** -- Overview tab -> `[Start]` button, or pre-set `Auto-start: yes` on the new-engagement form.
5. **Monitor** -- Events tab (`2`) tails the event stream; Overview shows the stage strip + live cost; Approvals (`3`) badges pending count. Regression events are color-coded in the stream; the Overview tab shows regressions used. Threshold warnings are also color-coded with their glyphs.
6. **Handle approvals as they arrive** -- Approvals tab, `a` / `r` per row, or `Shift+a` to bulk-approve. Approvals only queue in `guided` mode; cycle the engagement `mode` with `p` (recon_only → assess → guided → autonomous) and pick `autonomous` for an unattended overnight run.
7. **Drill into findings** -- Findings tab, `Enter` on a row -> full detail. Critic verdict + evidence excerpt + `tool_run_id` traceable to evidence blobs.
8. **Download the report** -- Report tab -> pick format (HTML / PDF / Markdown / JSON / SARIF). Files land in `$PWD` by default; override with `download_dir` in config.

---

## Configuration

Core engine settings (LLM backend, models, sandbox, budgets) live in a
`.env` file, loaded as environment variables. The setup wizard writes it;
see `.env.example` for the full key list.

```bash
SERENADE_LLM_BACKEND=ollama
# ANTHROPIC_API_KEY=sk-ant-...
SANDBOX_L3_REQUIRED=false
ENGAGEMENT_DEFAULT_TOKEN_BUDGET=2000000
ENGAGEMENT_DEFAULT_WALLCLOCK_HOURS=8
```

TUI-specific preferences (server URL, last-selected engagement, report
`download_dir`) live separately in `~/.config/serenade/config.toml`,
written on graceful exit.

Engagements snapshot the LLM config at create-time, so editing `.env`
doesn't disturb in-flight runs.

---

## Op-Exec (escape hatch)

When the AI is wrong and you want to fire a tool yourself, the **Op-Exec** tab is the escape hatch. Three guards still apply -- see [ARCHITECTURE.md - Operator-exec escape hatch](./ARCHITECTURE.md#operator-exec-escape-hatch):

1. A `justification` field -- minimum 20 characters, audit-logged. Be specific.
2. Scope / risk / time policy -- never bypassable.
3. PreExploitCritic -- bypassable via `bypass_critic: true` plus a separate `bypass_justification` (also >=20 chars).

Use it sparingly. Every invocation lands in `audit_log` with
`actor="local"`.

---

## Opt-in agent patterns

Four opt-in features (all default off) add scoring, memory, debate, and
reflection. The TUI surfaces these through the Events tab (live stream)
and Approvals tab (debate verdicts). Step scores and reflection entries
are also in the audit log export and the engagement summary.

Architecture and config keys: [ARCHITECTURE.md - Modern agent patterns](./ARCHITECTURE.md#modern-agent-patterns). Operational monitoring: [RUNBOOK.md](./RUNBOOK.md#operating-the-opt-in-agent-patterns).
