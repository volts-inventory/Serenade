"""Bottom status bar: URL, auth method, selected engagement, cost, pending."""

from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    def __init__(self) -> None:
        super().__init__("", id="status-bar")
        self._url: str = ""
        self._auth: str = ""
        self._engagement: str = "(none)"
        self._cost_usd: float | None = None
        self._pending: int = 0
        self._username: str = ""

    def set_connection(self, url: str, auth: str, username: str = "") -> None:
        self._url = url
        self._auth = auth
        self._username = username
        self._render()

    def set_engagement(self, engagement_id: str | None) -> None:
        if engagement_id is None:
            self._engagement = "(none)"
        else:
            self._engagement = engagement_id[:8]
        self._render()

    def set_cost(self, cost_usd: float | None) -> None:
        self._cost_usd = cost_usd
        self._render()

    def set_pending(self, n: int) -> None:
        self._pending = n
        self._render()

    def _render(self) -> None:
        cost = f"${self._cost_usd:.4f}" if self._cost_usd is not None else "(loading)"
        user = f" as {self._username}" if self._username else ""
        msg = (
            f" [b]{self._url}[/b] [{self._auth}{user}]   "
            f"engagement={self._engagement}   "
            f"cost={cost}   "
            f"pending={self._pending}   "
            f"[dim]q=quit  l=logout  n=new  r=refresh  1-7=tabs[/dim]"
        )
        self.update(msg)
