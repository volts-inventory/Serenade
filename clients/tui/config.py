"""TUI configuration loading and persistence.

Precedence (highest first):
  1. CLI flags (``--url``, ``--api-key``)
  2. Environment (``SERENADE_URL``, ``SERENADE_API_KEY``)
  3. ``~/.config/serenade/config.toml``
  4. Hard-coded defaults

Config is written back on graceful exit so the next session remembers
the last URL and last-selected engagement id.

We use the stdlib ``tomllib`` for reads (3.11+) and a tiny hand-rolled
writer for writes — we don't need a full TOML library for two keys.
"""

from __future__ import annotations

import os

# tomllib is stdlib in 3.11+, read-only. We write by hand to avoid a dep.
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://localhost:8000"
CONFIG_DIR_ENV = "SERENADE_TUI_CONFIG_DIR"


def _default_config_path() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    base = Path(override) if override else Path.home() / ".config" / "serenade"
    return base / "config.toml"


@dataclass
class TUIConfig:
    """User-facing TUI configuration.

    Persisted under ``~/.config/serenade/config.toml`` so the user gets
    the same URL and engagement back on next launch.
    """

    url: str = DEFAULT_URL
    api_key: str | None = None
    last_engagement_id: str | None = None
    # Where to drop downloaded reports. Default cwd is fine for most users.
    download_dir: str = "."

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, path: Path | None = None) -> TUIConfig:
        """Load from disk; returns defaults on missing/corrupt file."""
        path = path or _default_config_path()
        if not path.exists():
            return cls()
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            # Corrupt config shouldn't break the app; back to defaults.
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TUIConfig:
        # Ignore any api_key key on disk. Older configs may still
        # have one — we treat it as if it weren't there. The user
        # must re-supply via env or CLI.
        return cls(
            url=str(data.get("url", DEFAULT_URL)),
            api_key=None,
            last_engagement_id=data.get("last_engagement_id") or None,
            download_dir=str(data.get("download_dir", ".")),
        )

    # ---------------------------------------------------------------- save
    def save(self, path: Path | None = None) -> None:
        import contextlib  # noqa: PLC0415

        path = path or _default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml(), encoding="utf-8")
        # Best-effort permission tightening — config can contain an api_key.
        with contextlib.suppress(OSError):
            path.chmod(0o600)

    def to_toml(self) -> str:
        # Never persist the API key to disk. The 0600 perms only
        # defend against same-host user separation; not against dotfile-to-
        # github, cloud-sync, or laptop-stolen scenarios. Source the key
        # from env/CLI on every launch instead.
        lines: list[str] = [
            "# Serenade TUI config — auto-generated; safe to edit.",
            "# Note: the API key is NOT persisted here. Set it via",
            "# SERENADE_API_KEY env var or --api-key on each launch.",
        ]
        for key, value in asdict(self).items():
            if value is None or key == "api_key":
                continue
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        return "\n".join(lines) + "\n"

    # --------------------------------------------------------------- merge
    def merged_with(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
    ) -> TUIConfig:
        """Return a copy with CLI/env overrides applied (None means skip)."""
        return TUIConfig(
            url=url or self.url,
            api_key=api_key if api_key is not None else self.api_key,
            last_engagement_id=self.last_engagement_id,
            download_dir=self.download_dir,
        )


def resolve_config(
    *,
    cli_url: str | None = None,
    cli_api_key: str | None = None,
    config_path: Path | None = None,
) -> TUIConfig:
    """Apply the precedence stack and return the final TUIConfig.

    CLI flags > env > on-disk config > defaults.
    """
    cfg = TUIConfig.load(config_path)

    # env layer
    env_url = os.environ.get("SERENADE_URL")
    env_key = os.environ.get("SERENADE_API_KEY")

    # CLI overrides env which overrides file
    final_url = cli_url or env_url or cfg.url
    final_key: str | None
    if cli_api_key is not None:
        final_key = cli_api_key
    elif env_key is not None:
        final_key = env_key
    else:
        final_key = cfg.api_key

    return TUIConfig(
        url=final_url,
        api_key=final_key,
        last_engagement_id=cfg.last_engagement_id,
        download_dir=cfg.download_dir,
    )


# Re-export for tests that want to round-trip a config without touching disk.
__all__ = ["DEFAULT_URL", "TUIConfig", "resolve_config"]
