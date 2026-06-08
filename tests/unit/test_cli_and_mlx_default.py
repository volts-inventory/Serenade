"""Two new things landed together:

1. MLX is now the default LLM backend (``Settings.llm_backend = "mlx"``).
   Anthropic is **explicit opt-in only** — ``ANTHROPIC_API_KEY`` alone
   does NOT promote to the hosted backend (see Phase L4). The operator
   must set ``SERENADE_LLM_BACKEND=anthropic`` (or the legacy unprefixed
   ``LLM_BACKEND=anthropic``) for the hosted backend to engage.
   ``test_llm_backend_optin.py`` covers the safety contract in detail.

2. ``serenade`` CLI replaces direct ``make`` invocation. Exercises here
   are import-safety + .env round-trip + arg-dispatch, not the full
   subprocess execution paths (those are covered by integration once we
   actually drive Docker in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Settings: MLX default + ANTHROPIC_API_KEY override                          #
# --------------------------------------------------------------------------- #


def test_settings_default_backend_is_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env vars -> ollama. The Wireshark-shape laptop default."""
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    assert get_settings().llm_backend == "ollama"


def test_settings_explicit_mlx_maps_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the operator pinned SERENADE_LLM_BACKEND=mlx explicitly, it
    maps to ollama (deprecated alias). The API key presence DOES NOT
    silently flip to anthropic. Explicit beats implicit."""
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "mlx")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-but-i-want-local")

    from orchestrator.settings import get_settings

    get_settings.cache_clear()
    assert get_settings().llm_backend == "ollama"


# --------------------------------------------------------------------------- #
# CLI: import safety + dispatch                                                #
# --------------------------------------------------------------------------- #


def test_cli_module_imports_cleanly() -> None:
    """`python -m cli ...` must import without side effects."""
    import importlib

    mod = importlib.import_module("cli.main")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_cli_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`serenade --help` prints help and exits 0. Catches half the
    regressions a CLI can have."""
    from cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # New shape: primary modes named in the description.
    assert "Serenade" in out
    assert "--setup" in out
    assert "--test" in out


def test_cli_default_routes_to_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare `serenade` (no args) dispatches to ``cmd_default_tui``.
    We don't actually want to bring up Docker in a unit test — patch
    the function out and verify the dispatch landed."""
    from cli import main as cli_main

    called: list[str] = []
    monkeypatch.setattr(cli_main, "cmd_default_tui", lambda: called.append("tui") or 0)
    rc = cli_main.main([])
    assert rc == 0
    assert called == ["tui"]


def test_cli_setup_flag_routes_to_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """`serenade --setup` dispatches to ``cmd_setup``."""
    from cli import main as cli_main

    called: list[str] = []
    monkeypatch.setattr(cli_main, "cmd_setup", lambda: called.append("setup") or 0)
    rc = cli_main.main(["--setup"])
    assert rc == 0
    assert called == ["setup"]


def test_cli_unknown_test_scope_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`serenade --test bogus` is rejected by argparse before reaching
    the handler. The error message lists valid choices."""
    from cli.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--test", "bogus"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    # argparse's default error names the offending value + the choices.
    assert "bogus" in err or "invalid choice" in err


# --------------------------------------------------------------------------- #
# .env round-trip                                                              #
# --------------------------------------------------------------------------- #


def test_env_round_trip(tmp_path: Path) -> None:
    """_read_env + _write_env survives a round-trip without mangling
    values. Tests the wizard's persistence layer in isolation."""
    from cli.main import _read_env, _write_env

    src = tmp_path / ".env"
    src.write_text(
        "# header comment\n"
        "LLM_BACKEND=mlx\n"
        "ANTHROPIC_API_KEY=sk-ant-with-equals=preserved\n"
        "SERENADE_TEAM_MODE=0\n"
    )
    env = _read_env(src)
    assert env["LLM_BACKEND"] == "mlx"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-with-equals=preserved"
    assert env["SERENADE_TEAM_MODE"] == "0"

    env["LLM_BACKEND"] = "anthropic"
    env["SERENADE_SESSION_SECRET"] = "added-by-wizard"
    _write_env(src, env)

    reloaded = _read_env(src)
    assert reloaded["LLM_BACKEND"] == "anthropic"
    assert reloaded["ANTHROPIC_API_KEY"] == "sk-ant-with-equals=preserved"
    assert reloaded["SERENADE_SESSION_SECRET"] == "added-by-wizard"


def test_env_read_handles_quoted_values(tmp_path: Path) -> None:
    """Quoted values in .env (common from copy-paste of templates) are
    unwrapped on read."""
    from cli.main import _read_env

    src = tmp_path / ".env"
    src.write_text(
        'LLM_BACKEND="mlx"\n'
        "ANTHROPIC_API_KEY='sk-ant-quoted'\n"
    )
    env = _read_env(src)
    assert env["LLM_BACKEND"] == "mlx"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-quoted"


# --------------------------------------------------------------------------- #
# MLX lifecycle: backend detection + auto-start/stop                          #
# --------------------------------------------------------------------------- #


def test_configured_backend_env_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SERENADE_LLM_BACKEND env beats whatever's in .env."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("SERENADE_LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.setenv("SERENADE_LLM_BACKEND", "anthropic")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli_main._configured_backend() == "anthropic"


def test_configured_backend_falls_back_to_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No SERENADE_LLM_BACKEND in shell env → read from .env."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("SERENADE_LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli_main._configured_backend() == "mlx"


def test_configured_backend_anthropic_key_does_not_promote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ANTHROPIC_API_KEY alone does NOT promote to anthropic. The
    operator must opt in explicitly via SERENADE_LLM_BACKEND=anthropic
    (Phase L4 safety rule — prevents silent spend against a key set for
    Cursor / Claude Code / other tools)."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("# no backend pinned\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-implicit")

    assert cli_main._configured_backend() == "mlx"


def test_configured_backend_default_is_mlx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing configured anywhere → MLX. The Wireshark-shape default."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("# empty\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli_main._configured_backend() == "mlx"


def test_mlx_lifecycle_noop_when_anthropic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anthropic backend → context manager yields None, never spawns
    a subprocess. Critical: we must NOT shell out to mlx_lm.server
    just because the user happens to have it installed."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=anthropic\nANTHROPIC_API_KEY=sk-ant\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)

    popen_calls: list[list[str]] = []

    def _no_popen(*args, **kwargs):  # noqa: ANN001, ANN003
        popen_calls.append(list(args[0]))
        raise AssertionError("Popen should not be called when backend != mlx")

    monkeypatch.setattr(cli_main.subprocess, "Popen", _no_popen)

    with cli_main._mlx_lifecycle_if_local() as proc:
        assert proc is None
    assert popen_calls == []


def test_mlx_lifecycle_noop_when_already_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If MLX is already on :8080 (operator started it manually) we
    leave it alone — no Popen, no cleanup."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.setattr(cli_main, "_mlx_already_running", lambda port=8080: True)

    def _no_popen(*args, **kwargs):  # noqa: ANN001, ANN003
        raise AssertionError("Popen should not be called when MLX is already up")

    monkeypatch.setattr(cli_main.subprocess, "Popen", _no_popen)

    with cli_main._mlx_lifecycle_if_local() as proc:
        assert proc is None


def test_mlx_lifecycle_starts_and_cleans_up_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: MLX backend + nothing listening on :8080 + binary on
    PATH → Popen is called, the wait helper sees the server come up,
    and the cleanup path kills the process on context exit.

    Forces the macOS platform path so the test is platform-stable —
    the same test runs unchanged whether CI is macOS or Linux."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.setattr(cli_main, "_is_macos", lambda: True)
    monkeypatch.setattr(cli_main, "_is_linux", lambda: False)
    monkeypatch.setattr(cli_main, "_local_llm_already_running", lambda probe_url: False)
    monkeypatch.setattr(cli_main.shutil, "which", lambda _: "/usr/bin/mlx_lm.server")
    monkeypatch.setattr(
        cli_main,
        "_wait_for_local_llm",
        lambda probe_url, timeout_seconds=120: True,
    )

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 99999
            self.returncode: int | None = None
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            # The cleanup path calls SIGTERM → wait(timeout=5). Pretend
            # we exited cleanly.
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    spawned: list[_FakeProc] = []

    def _fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN003
        assert cmd[0] == "mlx_lm.server"
        p = _FakeProc()
        spawned.append(p)
        return p

    monkeypatch.setattr(cli_main.subprocess, "Popen", _fake_popen)
    # killpg target — pretend we don't have permission so we fall back
    # to .terminate(). Simpler than mocking the full os.killpg path.
    monkeypatch.setattr(cli_main.os, "killpg", lambda *_: (_ for _ in ()).throw(PermissionError()))

    with cli_main._mlx_lifecycle_if_local() as proc:
        assert proc is spawned[0]

    # After the `with` block, the cleanup path ran. The fallback used
    # .terminate(); .wait() then returned the cleaned-up returncode.
    assert spawned[0].terminated is True
    assert spawned[0].returncode == 0


def test_mlx_lifecycle_skips_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MLX backend + nothing on :8080 + `mlx_lm.server` not installed:
    surface a warning, yield None, don't crash. The orchestrator will
    eventually 503 on LLM calls — better that than a CLI traceback."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.setattr(cli_main, "_mlx_already_running", lambda port=8080: False)
    monkeypatch.setattr(cli_main.shutil, "which", lambda _: None)

    with cli_main._mlx_lifecycle_if_local() as proc:
        assert proc is None


# --------------------------------------------------------------------------- #
# Platform-aware local-LLM server selection                                   #
# --------------------------------------------------------------------------- #


def test_local_llm_server_spec_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS → MLX server on :8080. The binary the lifecycle will probe
    for, the argv it'll spawn, and the model warmup are all set."""
    from cli import main as cli_main

    monkeypatch.setattr(cli_main, "_is_macos", lambda: True)
    monkeypatch.setattr(cli_main, "_is_linux", lambda: False)

    spec = cli_main._local_llm_server_spec()
    assert spec is not None
    assert spec["name"] == "mlx"
    assert spec["binary"] == "mlx_lm.server"
    assert spec["port"] == 8080
    assert spec["model_warmup"] is None  # MLX downloads on demand
    assert "mlx_lm.server" in spec["argv"]


def test_local_llm_server_spec_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux → Ollama server on :11434, with a pre-serve `ollama pull`
    warmup so the first agent call isn't a multi-GB download with no
    UX."""
    from cli import main as cli_main

    monkeypatch.setattr(cli_main, "_is_macos", lambda: False)
    monkeypatch.setattr(cli_main, "_is_linux", lambda: True)

    spec = cli_main._local_llm_server_spec()
    assert spec is not None
    assert spec["name"] == "ollama"
    assert spec["binary"] == "ollama"
    assert spec["port"] == 11434
    assert spec["argv"] == ["ollama", "serve"]
    # Linux warmup pulls the configured default model.
    warmup = spec["model_warmup"]
    assert warmup[0] == "ollama"
    assert warmup[1] == "pull"


def test_local_llm_server_spec_unknown_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows-without-WSL etc. → no default server. Lifecycle yields
    None and warns; we don't try to invent one."""
    from cli import main as cli_main

    monkeypatch.setattr(cli_main, "_is_macos", lambda: False)
    monkeypatch.setattr(cli_main, "_is_linux", lambda: False)

    assert cli_main._local_llm_server_spec() is None


def test_mlx_lifecycle_on_linux_starts_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Linux, the lifecycle context calls `ollama serve` (not
    `mlx_lm.server`) and probes the Ollama port for readiness. This is
    the test that locks the Linux behaviour in."""
    from cli import main as cli_main

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=mlx\n")
    monkeypatch.setattr(cli_main, "ENV_PATH", env_file)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SERENADE_LLM_BACKEND", raising=False)
    monkeypatch.setattr(cli_main, "_is_macos", lambda: False)
    monkeypatch.setattr(cli_main, "_is_linux", lambda: True)
    monkeypatch.setattr(cli_main, "_local_llm_already_running", lambda probe_url: False)
    monkeypatch.setattr(cli_main.shutil, "which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(
        cli_main,
        "_wait_for_local_llm",
        lambda probe_url, timeout_seconds=120: True,
    )

    spawned_argv: list[list[str]] = []
    warmup_argv: list[list[str]] = []

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 99998
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            pass

    def _fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN003
        spawned_argv.append(list(cmd))
        return _FakeProc()

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        warmup_argv.append(list(cmd))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cli_main, "_run", _fake_run)
    # Fall back to .terminate() in cleanup (no killpg permission).
    monkeypatch.setattr(
        cli_main.os, "killpg", lambda *_: (_ for _ in ()).throw(PermissionError())
    )

    with cli_main._mlx_lifecycle_if_local() as proc:
        assert proc is not None
        assert proc.pid == 99998

    # Warm-up pulled the configured model first.
    assert any(c[:2] == ["ollama", "pull"] for c in warmup_argv), warmup_argv
    # Server invocation was `ollama serve`, not mlx_lm.server.
    assert any(c[:2] == ["ollama", "serve"] for c in spawned_argv), spawned_argv
    # Cleanup ran.
    assert spawned_argv  # at least one Popen

