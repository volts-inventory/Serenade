"""Operator-exec tab: the manual escape hatch.

Submits to ``POST /engagements/{id}/operator_exec`` with:
  - ``argv``: list of strings (parsed from the text field, simple split)
  - ``target``: a single target string
  - ``justification``: free-text reason (required server-side)
  - ``bypass_critic`` + ``bypass_justification``: optional

The form does *not* try to be a shell parser — operators paste argv as
space-separated tokens. Quoting is left to the orchestrator (server-side
``OperatorExecInput`` validates).
"""

from __future__ import annotations

import shlex
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, TextArea


class OperatorExecRequested(Message):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.payload = payload


class OperatorExecPanel(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="tab-pane")

    def compose(self) -> ComposeResult:
        yield Label("Operator one-off exec (escape hatch)", classes="form-label")
        with Vertical(id="operator-exec-form"):
            yield Label("argv (shell-quoted, e.g. `nmap -sV 10.0.0.5`)", classes="form-label")
            yield Input(placeholder="nmap -sV 10.0.0.5", id="oe-argv")
            yield Label("target (single ip/host/url)", classes="form-label")
            yield Input(placeholder="10.0.0.5", id="oe-target")
            yield Label("justification (required)", classes="form-label")
            yield Input(placeholder="manual verify of CVE-2024-XXXX", id="oe-just")
            yield Label(
                "bypass critic? (yes/no — needs bypass justification)",
                classes="form-label",
            )
            yield Input(value="no", id="oe-bypass")
            yield Label("bypass justification (only if bypass=yes)", classes="form-label")
            yield Input(
                placeholder="critic LLM unreachable; manual review done",
                id="oe-bypass-just",
            )
            with Horizontal():
                yield Button("Submit", variant="primary", id="oe-submit")
                yield Button("Clear", id="oe-clear")
        yield Label("Output", classes="form-label")
        out = TextArea(id="operator-exec-output", read_only=True)
        yield out

    def show_output(self, text: str) -> None:
        widget = self.query_one("#operator-exec-output", TextArea)
        widget.load_text(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "oe-clear":
            for f in ("oe-argv", "oe-target", "oe-just", "oe-bypass-just"):
                self.query_one(f"#{f}", Input).value = ""
            self.query_one("#oe-bypass", Input).value = "no"
            self.show_output("")
            return
        if bid == "oe-submit":
            argv_str = self.query_one("#oe-argv", Input).value.strip()
            target = self.query_one("#oe-target", Input).value.strip()
            just = self.query_one("#oe-just", Input).value.strip()
            bypass_s = self.query_one("#oe-bypass", Input).value.strip().lower()
            bypass_just = self.query_one("#oe-bypass-just", Input).value.strip()
            if not argv_str or not target or not just:
                self.show_output("argv, target, and justification are required")
                return
            try:
                argv = shlex.split(argv_str)
            except ValueError as e:
                self.show_output(f"argv parse error: {e}")
                return
            payload: dict[str, Any] = {
                "argv": argv,
                "target": target,
                "justification": just,
                "bypass_critic": bypass_s in ("yes", "y", "true", "1"),
            }
            if payload["bypass_critic"]:
                payload["bypass_justification"] = bypass_just or None
            self.post_message(OperatorExecRequested(payload))
