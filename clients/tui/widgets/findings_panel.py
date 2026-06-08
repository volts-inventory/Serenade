"""Findings + Hosts panels. Two related tables; one widget per tab.

The findings endpoint returns ``{severity: [finding, ...]}``. We flatten
into a single sortable table with a leading severity column.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Label, Static


class FindingActivated(Message):
    """Emitted when the user hits Enter on a finding row."""

    def __init__(self, finding: dict[str, Any]) -> None:
        super().__init__()
        self.finding = finding


class FindingsPanel(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="tab-pane")
        self._flat: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Label("Findings", classes="form-label")
        yield Static("(no engagement selected)", id="findings-status")
        table = DataTable(id="findings-table")
        table.cursor_type = "row"
        table.add_columns("severity", "title", "host", "cve")
        yield table

    def update_findings(self, by_severity: dict[str, Any]) -> None:
        self._flat = []
        order = ["critical", "high", "medium", "low", "info"]
        for sev in order:
            items = by_severity.get(sev) or []
            for f in items:
                f2 = dict(f)
                f2["_severity"] = sev
                self._flat.append(f2)
        # any unknown severities at the end
        for sev, items in by_severity.items():
            if sev in order:
                continue
            for f in items or []:
                f2 = dict(f)
                f2["_severity"] = sev
                self._flat.append(f2)

        status = self.query_one("#findings-status", Static)
        status.update(f"{len(self._flat)} findings")
        table = self.query_one("#findings-table", DataTable)
        table.clear()
        for i, f in enumerate(self._flat):
            cve_list = f.get("cve") or []
            cve = ",".join(cve_list[:2]) if isinstance(cve_list, list) else str(cve_list)
            table.add_row(
                str(f.get("_severity", "?")),
                str(f.get("title", "?"))[:50],
                str(f.get("host") or f.get("target") or "")[:32],
                cve[:24],
                key=str(i),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        try:
            idx = int(str(event.row_key.value))
        except ValueError:
            return
        if 0 <= idx < len(self._flat):
            self.post_message(FindingActivated(self._flat[idx]))


class HostsPanel(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="tab-pane")

    def compose(self) -> ComposeResult:
        yield Label("Hosts", classes="form-label")
        yield Static("(no engagement selected)", id="hosts-status")
        table = DataTable(id="hosts-table")
        table.cursor_type = "row"
        table.add_columns("ip", "hostname", "os", "open ports")
        yield table

    def update_hosts(self, rows: list[dict[str, Any]]) -> None:
        self.query_one("#hosts-status", Static).update(f"{len(rows)} hosts")
        table = self.query_one("#hosts-table", DataTable)
        table.clear()
        for h in rows:
            ports = h.get("open_ports") or h.get("ports") or []
            if isinstance(ports, list):
                ports_str = ",".join(str(p) for p in ports[:8])
            else:
                ports_str = str(ports)
            table.add_row(
                str(h.get("ip") or h.get("address") or ""),
                str(h.get("hostname", ""))[:32],
                str(h.get("os", ""))[:24],
                ports_str[:40],
                key=str(h.get("id") or h.get("ip") or ""),
            )
