"""Report tab: counts + download buttons.

Files land in ``$PWD/report-{engagement_id}.{ext}`` by default; the user
can override via config (``download_dir``).
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, Static


class ReportDownloadRequested(Message):
    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.fmt = fmt  # html | markdown | json | pdf | sarif


class ReportPanel(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="tab-pane")
        self._findings: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Label("Report", classes="form-label")
        yield Static("(no engagement selected)", id="report-status")
        yield Static("", id="report-counts")
        with Horizontal():
            yield Button("Download HTML", variant="primary", id="rp-html")
            yield Button("Download Markdown", id="rp-md")
            yield Button("Download JSON", id="rp-json")
            yield Button("Download PDF", id="rp-pdf")
            yield Button("Download SARIF", id="rp-sarif")
        yield Static("", id="report-result")

    def update_findings(self, by_severity: dict[str, Any]) -> None:
        self._findings = by_severity
        counts = []
        for sev in ("critical", "high", "medium", "low", "info"):
            items = by_severity.get(sev) or []
            counts.append(f"{sev}: {len(items)}")
        self.query_one("#report-counts", Static).update("   ".join(counts))
        self.query_one("#report-status", Static).update("[green]ready[/green]")

    def show_result(self, msg: str) -> None:
        self.query_one("#report-result", Static).update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "rp-html": "html",
            "rp-md": "markdown",
            "rp-json": "json",
            "rp-pdf": "pdf",
            "rp-sarif": "sarif",
        }
        fmt = mapping.get(event.button.id or "")
        if fmt is not None:
            self.post_message(ReportDownloadRequested(fmt))
