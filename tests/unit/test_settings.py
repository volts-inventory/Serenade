from __future__ import annotations

from orchestrator.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_backend in ("anthropic", "ollama")
    assert s.data_dir == "~/.serenade"
    assert "sqlite" in s.sqlite_dsn


def test_sqlite_dsn():
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir="/tmp/test-serenade",
    )
    assert s.sqlite_dsn == "sqlite+aiosqlite:////tmp/test-serenade/serenade.db"
