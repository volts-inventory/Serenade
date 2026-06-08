"""Events tab: live in-process event stream.

Subscribes to the in-process EventBus directly via
``orchestrator.services.events.subscribe()``.  Each event is appended
to a :class:`RichLog` widget.

Pause/resume is just a flag — when paused, incoming events queue but
aren't written until resumed (so we don't lose them).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from collections import deque
from typing import Any, ClassVar
from uuid import UUID

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static

_MAX_QUEUE = 1000


class EventStreamPanel(Vertical):
    """Renders an event log; press ``space`` while focused to pause."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("space", "toggle_pause", "Pause/Resume"),
    ]

    def __init__(self) -> None:
        super().__init__(classes="tab-pane")
        self._paused = False
        self._queue: deque[dict[str, Any]] = deque(maxlen=_MAX_QUEUE)
        self._engagement_id: str | None = None
        self._worker_handle = None

    def compose(self) -> ComposeResult:
        yield Static("[dim]waiting for events...[/dim]", id="event-status")
        log = RichLog(highlight=True, markup=True, classes="event-log", id="event-log")
        log.can_focus = True
        yield log

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = self.query_one("#event-status", Static)
        if self._paused:
            status.update("[yellow]paused (space to resume)[/yellow]")
        else:
            status.update("[green]streaming[/green]")
            self._drain_queue()

    def attach(self, engagement_id: str) -> None:
        """(Re)connect the event stream for a given engagement.

        Safe to call multiple times — the previous worker is cancelled.
        """
        self.detach()
        self._engagement_id = engagement_id
        self.query_one("#event-status", Static).update("[green]streaming[/green]")
        self.query_one("#event-log", RichLog).clear()
        self._worker_handle = self.app.run_worker(
            self._consume(engagement_id),
            exclusive=False,
            group="event-stream",
        )

    def detach(self) -> None:
        if self._worker_handle is not None:
            with contextlib.suppress(Exception):
                self._worker_handle.cancel()
            self._worker_handle = None
        self._engagement_id = None

    async def _consume(self, engagement_id: str) -> None:
        try:
            from orchestrator.services.events import subscribe

            async for ev in subscribe(UUID(engagement_id)):
                self._handle_event(ev)
        except Exception as e:
            self.query_one("#event-status", Static).update(
                f"[red]stream error: {e}[/red]"
            )

    def _handle_event(self, ev: dict[str, Any]) -> None:
        if self._paused:
            self._queue.append(ev)
            return
        self._write(ev)

    def _drain_queue(self) -> None:
        while self._queue:
            self._write(self._queue.popleft())

    def _write(self, ev: dict[str, Any]) -> None:
        log = self.query_one("#event-log", RichLog)
        ts = _dt.datetime.now(_dt.UTC).strftime("%H:%M:%S")
        kind = ev.get("kind", "message")
        payload = ev.get("payload", {})
        summary = (
            _summarize_threshold(kind, payload)
            if _is_threshold_kind(kind)
            else _summarize(payload)
        )
        style = _kind_style(kind)
        if kind.startswith("regression."):
            log.write(f"[dim]{ts}[/dim] [{style}]↺ {kind}[/{style}] {summary}")
        elif _is_threshold_kind(kind):
            glyph = _threshold_glyph(kind)
            log.write(
                f"[dim]{ts}[/dim] [{style}]{glyph} {kind}[/{style}] {summary}"
            )
        else:
            log.write(f"[dim]{ts}[/dim] [{style}]{kind}[/{style}] {summary}")


def _kind_style(kind: str) -> str:
    if kind.startswith("stage."):
        return "b cyan"
    if kind.startswith("tool."):
        return "b green"
    if kind.startswith("approval.") or kind.startswith("policy_"):
        return "b yellow"
    if kind.startswith("critic"):
        return "b magenta"
    if kind.startswith("finding"):
        return "b red"
    if kind == "regression.requested":
        return "b blue"
    if kind == "regression.applied":
        return "b magenta"
    if kind == "regression.vetoed":
        return "b red"
    if kind == "regression.escalated":
        return "b orange3"
    if kind in ("engagement.budget_warning", "engagement.suppression_spread"):
        return "b yellow1"
    if kind == "engagement.veto_rate_warning":
        return "b red3"
    return "b white"


_THRESHOLD_KINDS: frozenset[str] = frozenset(
    {
        "engagement.budget_warning",
        "engagement.veto_rate_warning",
        "engagement.suppression_spread",
    }
)


def _is_threshold_kind(kind: str) -> bool:
    return kind in _THRESHOLD_KINDS


def _threshold_glyph(kind: str) -> str:
    if kind == "engagement.veto_rate_warning":
        return "🚫"
    if kind == "engagement.suppression_spread":
        return "🔇"
    return "⚠️"


def _summarize_threshold(kind: str, payload: dict[str, Any]) -> str:
    if kind == "engagement.budget_warning":
        usage = float(payload.get("usage", 0.0) or 0.0)
        pct = int(round(usage * 100))
        budget_kind = str(payload.get("kind") or "token")
        if budget_kind == "wallclock":
            elapsed_ns = int(payload.get("elapsed_ns") or 0)
            window_ns = int(payload.get("window_ns") or 0)
            if elapsed_ns > 0 and window_ns > 0:
                hours_used = elapsed_ns / 3.6e12
                hours_total = window_ns / 3.6e12
                return (
                    f"Wallclock at {pct}% used "
                    f"({hours_used:.1f} of {hours_total:.1f} hours)"
                )
            return f"Wallclock at {pct}% used"
        return f"Token budget at {pct}% used"
    if kind == "engagement.veto_rate_warning":
        rate = float(payload.get("rate", 0.0) or 0.0)
        window = int(payload.get("window") or payload.get("window_size") or 10)
        pct = int(round(rate * 100))
        return f"Critic veto rate {pct}% over last {window} verdicts"
    if kind == "engagement.suppression_spread":
        count = int(
            payload.get("new_count")
            if payload.get("new_count") is not None
            else payload.get("count", 0) or 0
        )
        stage = str(payload.get("stage") or "")
        plural = "" if count == 1 else "s"
        return f"{count} tool{plural} suppressed in stage {stage}"
    return _summarize(payload)


def _summarize(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    for key in ("message", "summary", "action", "title"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    trimmed = {k: v for k, v in payload.items() if not _is_big(v)}
    out = ", ".join(f"{k}={v}" for k, v in list(trimmed.items())[:5])
    return out or "(no payload)"


def _is_big(v: Any) -> bool:
    if isinstance(v, list | dict):
        return True
    return bool(isinstance(v, str) and len(v) > 80)
