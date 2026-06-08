"""Anthropic backend is **explicit opt-in only** — Phase L4 safety rule.

Background: before this change, ``get_settings()`` auto-promoted to the
hosted Anthropic backend whenever ``ANTHROPIC_API_KEY`` was present in
the environment. That was a real foot-gun: any developer who had the
key exported for Cursor, Claude Code, or another tool would silently
start spending against it the moment Serenade ran its first engagement.

New contract:

* ``SERENADE_LLM_BACKEND`` is the canonical env var (matches the rest of
  the ``SERENADE_*`` namespace).
* ``LLM_BACKEND`` (unprefixed) is honoured as a deprecated fallback;
  using it triggers a one-time ``llm.legacy_env_var_deprecated`` warning.
* ``ANTHROPIC_API_KEY`` alone is NEVER enough — when the key is present
  without an explicit backend choice, Serenade stays on the local
  default and logs ``llm.anthropic_key_ignored`` so the gap is loud.
* ``SERENADE_LLM_BACKEND=anthropic`` with an empty key fails fast at
  startup rather than two LLM calls into an engagement.
"""

from __future__ import annotations

import pytest


def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pull every env var that can influence backend resolution.
    Keeps each test starting from a known-clean slate regardless of
    how the harness was invoked."""
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Core opt-in rules                                                            #
# --------------------------------------------------------------------------- #


def test_anthropic_key_alone_stays_local_and_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ANTHROPIC_API_KEY in env + no SERENADE_LLM_BACKEND → local backend
    selected, ``llm.anthropic_key_ignored`` warning emitted.

    This is the headline safety property of Phase L4 — the previous
    auto-promote was the silent-spend foot-gun this whole change exists
    to close."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-cursor")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_backend == "ollama"
    # structlog's default PrintLoggerFactory writes warnings to stdout,
    # so we grep the captured output for the event name + advice line.
    out = capsys.readouterr().out
    assert "anthropic_key_ignored" in out
    assert "IGNORED" in out


def test_explicit_anthropic_with_key_uses_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SERENADE_LLM_BACKEND=anthropic + ANTHROPIC_API_KEY set → Anthropic.
    Happy path for an operator who actually meant to opt in."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit-optin")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_backend == "anthropic"
    assert s.anthropic_api_key == "sk-ant-explicit-optin"


def test_explicit_anthropic_without_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SERENADE_LLM_BACKEND=anthropic + no key → fail fast at startup
    with a clear message naming both variables. Better to refuse to boot
    than to 401 mid-engagement."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "anthropic")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(RuntimeError) as exc:
        get_settings()
    msg = str(exc.value)
    assert "SERENADE_LLM_BACKEND" in msg
    assert "ANTHROPIC_API_KEY" in msg


def test_legacy_unprefixed_var_still_works_with_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """LLM_BACKEND=anthropic (legacy unprefixed) still routes to anthropic
    but emits a ``llm.legacy_env_var_deprecated`` warning so operators
    know to rename to SERENADE_LLM_BACKEND."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-legacy")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_backend == "anthropic"
    out = capsys.readouterr().out
    assert "legacy_env_var_deprecated" in out


def test_explicit_mlx_with_anthropic_key_no_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SERENADE_LLM_BACKEND=mlx + ANTHROPIC_API_KEY set -> ollama
    (deprecated alias), NO ignored-key warning. The operator made an
    explicit choice; we trust it and shut up."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "mlx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shut-up")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_backend == "ollama"
    out = capsys.readouterr().out
    assert "anthropic_key_ignored" not in out


def test_canonical_var_beats_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both are set, SERENADE_LLM_BACKEND wins. AliasChoices order
    guarantees this; the test locks it in so a future refactor that
    swaps the order trips a red flag."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_BACKEND", "anthropic")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    assert get_settings().llm_backend == "ollama"


def test_ollama_value_accepted_as_local_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SERENADE_LLM_BACKEND=ollama`` validates as a Literal member.
    The router treats it identically to ``mlx`` (both speak OpenAI-compat
    behind MLX_BASE_URL); only the CLI wizard cares about the per-platform
    binary distinction."""
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "ollama")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    assert get_settings().llm_backend == "ollama"
