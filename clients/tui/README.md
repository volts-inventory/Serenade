# Serenade Terminal UI

A pure-terminal Textual client for driving a Serenade engagement without a
browser.

## Install

```bash
pip install -e .[tui]
```

This adds three deps (`textual`, `httpx-sse`, `rich`) under the optional
`tui` extra. No DB or local-state dependencies — the TUI is a pure REST/SSE
client against the orchestrator.

## Run

```bash
# Username/password (interactive login):
serenade-tui --url http://localhost:8000

# API-key (no prompts):
serenade-tui --url https://orch.example.com --api-key serenade_pk_...

# Or via env:
export SERENADE_URL=https://orch.example.com
export SERENADE_API_KEY=serenade_pk_...
serenade-tui
```

Equivalent:

```bash
python -m tui
```

## Config

State is read from (in order): CLI flags → env (`SERENADE_URL`,
`SERENADE_API_KEY`) → `~/.config/serenade/config.toml` → defaults.

On exit, the URL and last-selected engagement id are persisted back to
`~/.config/serenade/config.toml` (mode `0600`).

## Keys

| Key   | Action                            |
|-------|-----------------------------------|
| `q`   | Quit                              |
| `l`   | Logout                            |
| `n`   | New engagement                    |
| `r`   | Refresh                           |
| `1-7` | Switch tab (Overview…Report)      |
| `space` (Events tab) | Pause/resume the SSE tail |
| `a`/`r` (Approvals tab) | Approve / reject focused row |
| `A`/`R` (Approvals tab) | Bulk-decide all of the same kind |

## What's in here

| File | Purpose |
|------|---------|
| `app.py` | `SerenadeApp` + CLI entry point |
| `client.py` | Async HTTP/SSE client (CSRF auto-injection for cookie auth) |
| `config.py` | Load/save `~/.config/serenade/config.toml` |
| `screens/login.py` | Username/password OR API-key login |
| `screens/main_screen.py` | Sidebar + tabbed right pane |
| `screens/new_engagement.py` | Create-engagement modal form |
| `screens/finding_detail.py` | Per-finding detail modal |
| `widgets/engagement_list.py` | Sidebar DataTable |
| `widgets/overview_panel.py` | Overview tab + lifecycle buttons |
| `widgets/event_stream.py` | Live SSE tail (pauseable) |
| `widgets/approvals_panel.py` | Pending approvals + decide actions |
| `widgets/findings_panel.py` | Findings + Hosts DataTables |
| `widgets/operator_exec_panel.py` | Manual one-off tool runs |
| `widgets/report_panel.py` | Report counts + download buttons |
| `widgets/status_bar.py` | Bottom bar (URL, cost, pending count) |
| `styles.tcss` | Textual CSS |

## Test

```bash
pytest tests/unit/test_tui_client.py -x -q
```

## Smoke test offline

The client accepts an `httpx.AsyncBaseTransport`, so you can drive the UI
against an in-memory mock orchestrator using `httpx.MockTransport`. See
`tests/unit/test_tui_client.py` for examples.
