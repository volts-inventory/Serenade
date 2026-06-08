"""Sidebar widget: list of engagements with a refresh action."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Label, Static


class EngagementSelected(Message):
    """Emitted when the user selects an engagement row."""

    def __init__(self, engagement_id: str) -> None:
        super().__init__()
        self.engagement_id = engagement_id


class EngagementList(Vertical):
    """Sidebar DataTable of engagements.

    Keys: ``r`` refresh, ``n`` new (handled by MainScreen),
    ``enter`` select. We surface selection via :class:`EngagementSelected`.
    """

    def __init__(self) -> None:
        super().__init__(id="sidebar")
        self._engagements: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Label("Engagements", id="sidebar-title")
        yield Static("(none yet — press [b]n[/b] to create)", id="sidebar-empty")
        table = DataTable(id="engagement-table")
        table.cursor_type = "row"
        table.add_columns("name", "status", "mode")
        yield table

    def update_engagements(self, rows: list[dict[str, Any]]) -> None:
        self._engagements = rows
        table = self.query_one("#engagement-table", DataTable)
        empty = self.query_one("#sidebar-empty", Static)
        table.clear()
        if not rows:
            empty.display = True
            return
        empty.display = False
        for row in rows:
            name = str(row.get("name", "?"))[:20]
            status = str(row.get("status", "?"))
            mode = str(row.get("mode", "?"))
            # Row key is the engagement uuid so we can find it on selection.
            table.add_row(name, status, mode, key=str(row.get("id")))

    def select_first(self) -> str | None:
        """Programmatically select the first row, return its id, or None."""
        table = self.query_one("#engagement-table", DataTable)
        if table.row_count == 0:
            return None
        table.move_cursor(row=0)
        key = table.coordinate_to_cell_key((0, 0)).row_key.value
        return str(key) if key is not None else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        self.post_message(EngagementSelected(str(event.row_key.value)))
