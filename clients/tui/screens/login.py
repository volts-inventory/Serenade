"""Startup screen — initializes the database and enters main screen.

After the native architecture migration, there is no orchestrator to
connect to.  This screen initializes SQLite and transitions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Label, Static

if TYPE_CHECKING:
    from tui.app import SerenadeApp


class LoginScreen(Screen):
    """Initializes the database and enters the main screen."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [("q", "app.quit", "Quit")]

    CSS_PATH = None

    def __init__(self) -> None:
        super().__init__()
        self._status: Static | None = None

    def compose(self) -> ComposeResult:
        with Container(id="login-container"), Vertical(classes="login-box"):
            yield Label("Serenade", classes="login-title")
            self._status = Static(
                "Initializing...",
                classes="login-status",
            )
            yield self._status

    async def on_mount(self) -> None:
        try:
            from orchestrator.db import create_all, init_engine

            init_engine()
            await create_all()

            from tui.screens.main_screen import MainScreen

            await self.app.push_screen(MainScreen())
        except Exception as e:
            if self._status is not None:
                self._status.update(f"Initialization failed: {e}")
