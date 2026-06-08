"""Approvals tab: pending approvals + decide actions.

Bindings (in focus):
  ``a`` — approve the focused row
  ``r`` — reject the focused row
  ``A`` — bulk-approve all of the focused row's kind
  ``R`` — bulk-reject all of the focused row's kind
  ``enter`` — same as ``a`` (the common case)

Notes are entered in a small ``Input`` above the table. Empty notes are fine.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Label, Static


class ApprovalDecisionRequested(Message):
    """Posted when the user wants the app to issue a decide call.

    ``bulk`` controls single vs. ``bulk_decide``. When bulk, ``approval_id``
    is the kind (we re-derive it in the panel before posting).
    """

    def __init__(
        self,
        *,
        approval_id: str,
        kind: str,
        decision: str,
        notes: str,
        bulk: bool,
    ) -> None:
        super().__init__()
        self.approval_id = approval_id
        self.kind = kind
        self.decision = decision
        self.notes = notes
        self.bulk = bulk


class ApprovalsPanel(Vertical):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
        ("shift+a", "bulk_approve", "Bulk approve same kind"),
        ("shift+r", "bulk_reject", "Bulk reject same kind"),
    ]

    def __init__(self) -> None:
        super().__init__(classes="tab-pane")
        self._rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Label("Pending approvals", classes="form-label")
        yield Static("(no engagement selected)", id="approvals-status")
        # Surface auto-policy activity so unattended runs are
        # visible. Counts auto-decided rows in the most recent batch.
        yield Static("", id="approvals-auto-status")
        with Horizontal():
            yield Input(placeholder="notes (optional)", id="approval-notes")
            yield Button("Approve", id="ap-approve", variant="primary")
            yield Button("Reject", id="ap-reject", variant="error")
            yield Button("Bulk approve (same kind)", id="ap-bulk-approve")
            yield Button("Bulk reject (same kind)", id="ap-bulk-reject", variant="error")
        table = DataTable(id="approvals-table")
        table.cursor_type = "row"
        table.add_columns("kind", "id (short)", "decision", "created")
        yield table

    def update_approvals(self, rows: list[dict[str, Any]]) -> None:
        # Pending only — orchestrator returns all; we filter for the panel.
        self._rows = [r for r in rows if not r.get("decision")]
        status = self.query_one("#approvals-status", Static)
        auto_status = self.query_one("#approvals-auto-status", Static)
        table = self.query_one("#approvals-table", DataTable)
        table.clear()

        # Auto-policy summary: count auto-decided rows in the full batch.
        auto_rows = [r for r in rows if r.get("decided_by") == "auto-policy"]
        if auto_rows:
            approved = sum(1 for r in auto_rows if r.get("decision") == "approve")
            rejected = sum(1 for r in auto_rows if r.get("decision") == "reject")
            last_kind = str(auto_rows[-1].get("kind", "?"))
            auto_status.update(
                f"[magenta]auto-policy: {approved} approved, {rejected} rejected"
                f"   (latest: {last_kind})[/magenta]"
            )
        else:
            auto_status.update("")

        if not self._rows:
            status.update("[dim]no pending approvals[/dim]")
            return
        status.update(f"[yellow]{len(self._rows)} pending[/yellow]")
        for r in self._rows:
            ap_id = str(r.get("id", ""))
            short = ap_id[:8]
            table.add_row(
                str(r.get("kind", "?")),
                short,
                str(r.get("decision") or "pending"),
                str(r.get("created_at", ""))[:19],
                key=ap_id,
            )

    def _focused_row(self) -> dict[str, Any] | None:
        table = self.query_one("#approvals-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            cur = table.cursor_row
        except Exception:
            return None
        if cur is None or cur < 0 or cur >= len(self._rows):
            return None
        return self._rows[cur]

    def _post(self, decision: str, *, bulk: bool) -> None:
        row = self._focused_row()
        if row is None:
            return
        notes = self.query_one("#approval-notes", Input).value
        self.post_message(
            ApprovalDecisionRequested(
                approval_id=str(row.get("id", "")),
                kind=str(row.get("kind", "")),
                decision=decision,
                notes=notes,
                bulk=bulk,
            )
        )

    def action_approve(self) -> None:
        self._post("approve", bulk=False)

    def action_reject(self) -> None:
        self._post("reject", bulk=False)

    def action_bulk_approve(self) -> None:
        self._post("approve", bulk=True)

    def action_bulk_reject(self) -> None:
        self._post("reject", bulk=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "ap-approve":
                self.action_approve()
            case "ap-reject":
                self.action_reject()
            case "ap-bulk-approve":
                self.action_bulk_approve()
            case "ap-bulk-reject":
                self.action_bulk_reject()
