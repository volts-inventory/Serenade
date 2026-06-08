"""Modal screen showing a single finding's full detail.

Loaded when the user hits ``Enter`` on a row in the findings table.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class FindingDetailScreen(ModalScreen[None]):
    """Read-only pretty-printed dump of one finding."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, finding: dict[str, Any]) -> None:
        super().__init__()
        self._finding = finding

    def compose(self) -> ComposeResult:
        title = str(self._finding.get("title", "(untitled finding)"))
        severity = str(self._finding.get("severity", "?"))
        body = json.dumps(self._finding, indent=2, default=str)
        with Container(), Vertical(classes="login-box"):
            yield Static(f"[b]{title}[/b]  ({severity})", classes="login-title")
            yield Static(body)
            yield Button("Close", id="fd-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fd-close":
            self.dismiss(None)
