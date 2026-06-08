"""Serenade unified CLI.

``serenade`` is the single entry point operators use day to day. It
wraps alembic, the TUI, local LLM server lifecycle, and pytest so
nobody has to remember `make` targets. First run launches a guided
setup wizard; subsequent runs accept subcommands directly.

Docker is used only for sandbox containers (tool isolation), not for
infrastructure services. Internal Make targets still exist for power
users + CI but are no longer the customer-facing entry point.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    from cli.main import main as _main

    return _main()
