"""Overview tab: engagement metadata + start/pause/resume buttons."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

def _mode_gating(mode: str) -> str:
    """One-line description of what a mode does, for the overview header."""
    return {
        "recon_only": "recon + enum only",
        "assess": "full pipeline, no exploitation",
        "guided": "exploits pause for approval",
        "autonomous": "exploits auto-execute",
    }.get(mode, "exploits pause for approval")


# Mirror of ui/components/StageProgress.tsx — the canonical pipeline order.
_PIPELINE_STAGES: tuple[str, ...] = (
    "recon",
    "enum",
    "triage",
    "exploit",
    "post-exploit",
    "privesc",
    "lateral",
    "loot",
    "report",
)


def _render_stage_strip(current_stage: str, status: str) -> str:
    """Build a single-line process tracker.

    Active stage in [bright_green], completed stages in [white], pending
    in [dim]. Status `completed` paints every stage green; `aborted` /
    `failed` leaves the tail dim.
    """
    norm = (current_stage or "").lower().replace("_", "-")
    try:
        idx = _PIPELINE_STAGES.index(norm)
    except ValueError:
        idx = -1
    parts: list[str] = []
    for i, stage in enumerate(_PIPELINE_STAGES):
        if status == "completed":
            parts.append(f"[white]{stage}[/white]")
        elif idx < 0:
            parts.append(f"[dim]{stage}[/dim]")
        elif i < idx:
            parts.append(f"[white]{stage}[/white]")
        elif i == idx:
            parts.append(f"[b bright_green]{stage}[/b bright_green]")
        else:
            parts.append(f"[dim]{stage}[/dim]")
    return " → ".join(parts)


class EngagementLifecycleAction(Message):
    """Posted by the Overview panel when a lifecycle button is pressed."""

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action  # "start" | "pause" | "resume"


class OverviewPanel(Vertical):
    """Right-pane Overview tab."""

    def __init__(self) -> None:
        super().__init__(classes="tab-pane")
        self._engagement: dict[str, Any] | None = None
        self._cost: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Static("(no engagement selected)", id="overview-meta")
        yield Static("", id="overview-stages")
        # Regression-budget line. Hidden when none used.
        yield Static("", id="overview-regressions")
        yield Static("", id="overview-cost")
        with Horizontal():
            yield Button("Start", id="ov-start", variant="primary")
            yield Button("Pause", id="ov-pause")
            yield Button("Resume", id="ov-resume", variant="primary")

    def set_engagement(self, row: dict[str, Any] | None) -> None:
        self._engagement = row
        self._refresh()

    def set_cost(self, cost: dict[str, Any] | None) -> None:
        self._cost = cost
        self._refresh()

    def _refresh(self) -> None:
        meta = self.query_one("#overview-meta", Static)
        stages_widget = self.query_one("#overview-stages", Static)
        regressions_widget = self.query_one("#overview-regressions", Static)
        if self._engagement is None:
            meta.update("(no engagement selected)")
            stages_widget.update("")
            regressions_widget.update("")
            self.query_one("#overview-cost", Static).update("")
            return
        e = self._engagement
        scope = e.get("scope", {})
        lines = [
            f"[b]{e.get('name', '?')}[/b]   id={e.get('id', '?')}",
            f"status: [yellow]{e.get('status', '?')}[/yellow]"
            f"   stage: {e.get('current_stage', '?')}",
            f"mode: [magenta]{e.get('mode', '?')}[/magenta]"
            f"   ({_mode_gating(str(e.get('mode', '')))})"
            f"   (press [b]p[/b] to cycle)",
            f"budget: {e.get('token_budget', 0)} tokens,"
            f" {e.get('wallclock_hours', 0)}h wallclock",
            f"scope.cidrs: {json.dumps(scope.get('cidrs', []))}",
            f"scope.domains: {json.dumps(scope.get('domains', []))}",
        ]
        meta.update("\n".join(lines))
        stages_widget.update(
            _render_stage_strip(
                str(e.get("current_stage", "")),
                str(e.get("status", "")),
            )
        )
        # Show the regression-budget line only when at least one has
        # actually fired. Keeps the panel clean for the common case of
        # zero loops. Cap defaults to 3 (REGRESSION_BUDGET_MAX_DEFAULT)
        # so older orchestrator builds that don't surface the field
        # render a sensible label.
        used = int(e.get("regressions_used", 0) or 0)
        cap = int(e.get("regression_budget_max", 3) or 3)
        if used > 0:
            regressions_widget.update(
                f"[b magenta]↺ regressions: {used}/{cap} used[/b magenta]"
            )
        else:
            regressions_widget.update("")
        cost_widget = self.query_one("#overview-cost", Static)
        if self._cost is None:
            cost_widget.update("[dim]cost: (loading...)[/dim]")
        else:
            c = self._cost
            cost_widget.update(
                f"cost so far: ${c.get('cost_usd', 0):.4f}  "
                f"tokens: {c.get('tokens_used', 0):,}  "
                f"calls: {c.get('calls', 0)}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {"ov-start": "start", "ov-pause": "pause", "ov-resume": "resume"}
        action = mapping.get(event.button.id or "")
        if action is not None:
            self.post_message(EngagementLifecycleAction(action))
