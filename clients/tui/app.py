"""Top-level Textual app + CLI entry point.

``python -m tui`` or ``serenade-tui`` lands here. We parse a small set of
flags, build a :class:`TUIConfig`, and run the Textual app. The actual UI
lives in screens/ and widgets/.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import ClassVar

from textual.app import App

from tui.config import TUIConfig, resolve_config


class SerenadeApp(App):
    """Textual application root.

    Owns the resolved :class:`TUIConfig`.  After the native architecture
    migration, the TUI calls service functions directly — no HTTP client.
    """

    CSS_PATH = "styles.tcss"
    TITLE = "Serenade — Terminal Mode"
    SUB_TITLE = ""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, cfg: TUIConfig) -> None:
        super().__init__()
        self.cfg: TUIConfig = cfg

    async def on_mount(self) -> None:
        # Lazy import to avoid a circular at module load.
        from tui.screens.login import LoginScreen  # noqa: PLC0415

        await self.push_screen(LoginScreen())

    async def on_unmount(self) -> None:
        # Persist last-used engagement on exit. Don't let a write
        # failure crash the app at shutdown.
        with contextlib.suppress(Exception):
            self.cfg.save()


# --------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="serenade-tui",
        description="Serenade terminal client (Textual TUI). Native architecture — no HTTP.",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit (does NOT launch the UI).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from tui import __version__  # noqa: PLC0415

        print(f"serenade-tui {__version__}")
        return 0

    cfg = resolve_config()

    if args.print_config:
        print(f"last_engagement   = {cfg.last_engagement_id or '(none)'}")
        print(f"download_dir      = {cfg.download_dir}")
        return 0

    # Textual takes over the terminal — anything beyond this point should not
    # write to stdout/stderr directly.
    app = SerenadeApp(cfg)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SerenadeApp", "main"]
