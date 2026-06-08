"""Single-entry CLI for Serenade.

Two primary modes:

    serenade            Launch the TUI (optionally starts local LLM server).
    serenade --setup    Guided first-run setup. Writes .env.

Secondary flags exist for the days you need them (run tests, migrate,
etc.) — see ``serenade --help``. The Makefile is still the power-user
surface; the CLI is the customer-facing path.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_HEALTH_POLL_INTERVAL = 1.0

# Local LLM endpoint probe — the CLI owns + lifecycle-manages a local
# OpenAI-compatible server when the configured backend is `mlx`
# (orchestrator's name for "local") AND nothing is already serving on
# the platform's default port. Two flavours:
#   - macOS → ``mlx_lm.server`` on :8080
#   - Linux → ``ollama serve`` on :11434
# Either way, the orchestrator's MLX backend speaks OpenAI-compat and
# doesn't care which one is on the other end of MLX_BASE_URL.
_LOCAL_LLM_START_TIMEOUT_SECONDS = 120
_MLX_DEFAULT_MODEL = "mlx-community/gemma-3-27b-it-4bit"
_MLX_DEFAULT_PORT = 8080
_OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_OLLAMA_DEFAULT_PORT = 11434


# --------------------------------------------------------------------------- #
# Output helpers — no external deps                                           #
# --------------------------------------------------------------------------- #


def _info(msg: str) -> None:
    print(f"\033[1;36m▸\033[0m {msg}")


def _ok(msg: str) -> None:
    print(f"\033[1;32m✓\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m!\033[0m {msg}")


def _err(msg: str) -> None:
    print(f"\033[1;31m✗\033[0m {msg}", file=sys.stderr)


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    try:
        return subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=check).returncode
    except FileNotFoundError as e:
        _err(f"{cmd[0]} not found: {e}. Install it and retry.")
        return 127
    except subprocess.CalledProcessError as e:
        return e.returncode


# --------------------------------------------------------------------------- #
# .env management                                                              #
# --------------------------------------------------------------------------- #


def _read_env(path: Path) -> dict[str, str]:
    """Return KEY=VALUE pairs from a dotenv-style file. Ignores comments
    + blank lines. Values keep embedded `=` (split on first only) and
    have surrounding quotes unwrapped."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def _write_env(path: Path, values: dict[str, str]) -> None:
    """Rewrite the env file, preserving comments + ordering from
    .env.example when present."""
    lines: list[str] = []
    seen: set[str] = set()
    template = ENV_EXAMPLE if ENV_EXAMPLE.exists() else None
    if template:
        for raw in template.read_text().splitlines():
            s = raw.strip()
            if not s or s.startswith("#") or "=" not in s:
                lines.append(raw)
                continue
            key = s.split("=", 1)[0].strip()
            if key in values:
                lines.append(f"{key}={values[key]}")
                seen.add(key)
            else:
                lines.append(raw)
    for k, v in values.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# MLX lifecycle (auto-start + cleanup when the configured backend is local)   #
# --------------------------------------------------------------------------- #


def _configured_backend() -> str:
    """Resolve the effective LLM backend — env wins over .env, mirroring
    the orchestrator's :func:`orchestrator.settings.get_settings`.

    Resolution order (first match wins):

      1. ``SERENADE_LLM_BACKEND`` shell env (canonical).
      2. ``LLM_BACKEND`` shell env (legacy fallback).
      3. ``SERENADE_LLM_BACKEND`` in .env.
      4. ``LLM_BACKEND`` in .env (legacy fallback).
      5. Local default — ``mlx``. Anthropic is NEVER picked implicitly
         just because ``ANTHROPIC_API_KEY`` is in the environment; the
         operator must opt in explicitly (set ``SERENADE_LLM_BACKEND=
         anthropic`` or pick Anthropic in the wizard / per-user UI).
    """
    for var in ("SERENADE_LLM_BACKEND", "LLM_BACKEND"):
        explicit = os.environ.get(var, "").strip()
        if explicit:
            return explicit
    env = _read_env(ENV_PATH)
    for var in ("SERENADE_LLM_BACKEND", "LLM_BACKEND"):
        explicit = env.get(var, "").strip()
        if explicit:
            return explicit
    return "mlx"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _local_llm_server_spec() -> dict[str, object] | None:
    """Pick the local-LLM server flavour for this host.

    Returns ``None`` on platforms where neither flavour applies (Windows
    without WSL, etc.) — callers fall back to "trust the operator".
    """
    if _is_macos():
        return {
            "name": "mlx",
            "binary": "mlx_lm.server",
            "argv": [
                "mlx_lm.server",
                "--model",
                _MLX_DEFAULT_MODEL,
                "--port",
                str(_MLX_DEFAULT_PORT),
            ],
            "probe_url": f"http://127.0.0.1:{_MLX_DEFAULT_PORT}/v1/models",
            "port": _MLX_DEFAULT_PORT,
            "install_hint": "pip install mlx-lm",
            "model_warmup": None,
        }
    if _is_linux():
        return {
            "name": "ollama",
            "binary": "ollama",
            "argv": ["ollama", "serve"],
            "probe_url": f"http://127.0.0.1:{_OLLAMA_DEFAULT_PORT}/api/tags",
            "port": _OLLAMA_DEFAULT_PORT,
            "install_hint": "curl -fsSL https://ollama.com/install.sh | sh",
            # Warm-up pulls the model before serve so the first agent
            # call doesn't time out fetching gigabytes from the internet.
            "model_warmup": ["ollama", "pull", _OLLAMA_DEFAULT_MODEL],
        }
    return None


def _local_llm_already_running(probe_url: str) -> bool:
    """Probe any OpenAI/Ollama-compatible endpoint with a short timeout."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(probe_url, timeout=1) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_local_llm(
    probe_url: str,
    timeout_seconds: int = _LOCAL_LLM_START_TIMEOUT_SECONDS,
) -> bool:
    """Poll the endpoint until it answers or the deadline expires.
    Model loading can take a long time on a cold start (especially on
    Linux+Ollama if the model isn't cached), so the default timeout is
    generous."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _local_llm_already_running(probe_url):
            return True
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


# Backwards-compat aliases for tests that targeted the MLX-specific
# helpers before the Linux/Ollama generalisation.
def _mlx_already_running(port: int = _MLX_DEFAULT_PORT) -> bool:
    return _local_llm_already_running(f"http://127.0.0.1:{port}/v1/models")


def _wait_for_mlx(
    port: int = _MLX_DEFAULT_PORT,
    timeout_seconds: int = _LOCAL_LLM_START_TIMEOUT_SECONDS,
) -> bool:
    return _wait_for_local_llm(
        f"http://127.0.0.1:{port}/v1/models", timeout_seconds=timeout_seconds
    )


@contextlib.contextmanager
def _mlx_lifecycle_if_local() -> Iterator[subprocess.Popen[bytes] | None]:
    """Spin up the platform-appropriate local LLM server for the
    duration of the wrapped block when the configured backend is
    ``mlx`` AND nothing is already serving on its default port. The
    subprocess is terminated on context exit so the model is unloaded
    as soon as the user closes the TUI —
    no orphaned process eating RAM.

    Platform → server:
      - macOS → ``mlx_lm.server`` (Apple-Silicon native)
      - Linux → ``ollama serve``

    Yields the Popen handle for tests; callers don't need it.

    The function name is historical — the orchestrator backend is
    still called ``mlx`` because it's the abstraction for "talk to a
    local OpenAI-compat endpoint." On Linux that endpoint just happens
    to be Ollama.
    """
    backend = _configured_backend()
    if backend != "mlx":
        yield None
        return

    spec = _local_llm_server_spec()
    if spec is None:
        _warn(
            "Configured backend is `mlx` (local) but this platform has no "
            "default local-LLM server. Start one yourself and set "
            "MLX_BASE_URL appropriately."
        )
        yield None
        return

    name = spec["name"]
    probe_url = str(spec["probe_url"])
    if _local_llm_already_running(probe_url):
        _info(f"{name} already on :{spec['port']} — leaving it alone.")
        yield None
        return
    binary = str(spec["binary"])
    if not shutil.which(binary):
        _warn(
            f"Configured backend is `mlx` but `{binary}` is not on PATH.\n"
            f"    Install with: {spec['install_hint']}\n"
            "Continuing — the orchestrator will surface 503s when it tries to call the LLM."
        )
        yield None
        return

    # Linux/Ollama: pre-pull the model so the first agent call isn't a
    # multi-GB download with no UX. macOS/MLX downloads on first use too
    # but the mlx_lm.server hosts a progress bar on stderr.
    warmup = spec.get("model_warmup")
    if warmup:
        _info(f"Pulling Ollama model: {_OLLAMA_DEFAULT_MODEL} (one-time, takes a few minutes)…")
        rc = _run(list(warmup), check=False)  # type: ignore[arg-type]
        if rc != 0:
            _warn(
                "Model pull exited non-zero. Continuing — Ollama will pull on first request, "
                "but the orchestrator may time out the first LLM call."
            )

    argv = list(spec["argv"])  # type: ignore[arg-type]
    _info(f"Starting local {name} server on :{spec['port']}…")
    # New session so terminate() sends to the whole group — the local
    # LLM servers spawn worker subprocesses we want to take down with them.
    proc = subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_local_llm(probe_url):
        _err(f"{name} did not come up within the timeout. Killing it.")
        _kill_process_group(proc)
        yield None
        return
    _ok(f"{name} ready on :{spec['port']}.")
    try:
        yield proc
    finally:
        _info(f"Shutting down local {name} server…")
        _kill_process_group(proc)


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the child + any descendants it spawned in the new
    session. SIGTERM first (5s grace), then SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError):
        # Fallback for platforms without killpg (Windows): just kill the
        # process and hope the children are well-behaved.
        proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()


# --------------------------------------------------------------------------- #
# setup wizard                                                                 #
# --------------------------------------------------------------------------- #


def _prompt(question: str, *, default: str | None = None, choices: list[str] | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {question}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if choices and raw not in choices:
            _warn(f"Pick one of {choices}")
            continue
        if raw:
            return raw


def _prompt_password(question: str, *, default: str = "") -> str:
    import getpass

    suffix = " (press Enter to keep current)" if default else ""
    return getpass.getpass(f"  {question}{suffix}: ").strip() or default




def cmd_setup() -> int:
    """Guided first-run setup. Writes .env, checks Docker, builds sandbox."""
    _info("Serenade setup")
    print()

    data_dir = Path(os.environ.get("SERENADE_DATA_DIR", "~/.serenade")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    _ok(f"Data directory: {data_dir}")

    if not ENV_PATH.exists() and ENV_EXAMPLE.exists():
        _info("Copying .env.example to .env")
        ENV_PATH.write_text(ENV_EXAMPLE.read_text())

    env = _read_env(ENV_PATH)

    print()
    if _docker_available():
        _ok("Docker is installed and running.")
    else:
        _warn("Docker is not available.")
        if _is_macos():
            print("  Install: brew install --cask docker")
        elif _is_linux():
            print("  Install: curl -fsSL https://get.docker.com | sh")
        else:
            print("  Install: https://docs.docker.com/get-docker/")
        print("  Docker is required for sandbox isolation (running pentest tools).")

    print()
    has_anthropic = bool(
        env.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )
    print("LLM backend")
    print("  1) Local (default -- Ollama + Gemma 4, no cost)")
    print("  2) Hosted Anthropic API (paid; needs an API key)")
    if has_anthropic:
        _info("ANTHROPIC_API_KEY detected. To use Anthropic, pick (2).")
    choice = _prompt("Pick a backend", default="1", choices=["1", "2"])
    if choice == "1":
        env["SERENADE_LLM_BACKEND"] = "ollama"
        env.pop("LLM_BACKEND", None)
        env["LOCAL_BASE_URL"] = "http://127.0.0.1:11434/v1"
        _ok("Using local LLM via Ollama.")
    else:
        env["SERENADE_LLM_BACKEND"] = "anthropic"
        env.pop("LLM_BACKEND", None)
        key = _prompt_password(
            "Anthropic API key (will be saved to .env)",
            default=env.get("ANTHROPIC_API_KEY", ""),
        )
        if key:
            env["ANTHROPIC_API_KEY"] = key
        _ok("Using Anthropic API.")

    _write_env(ENV_PATH, env)
    _ok(f"Wrote {ENV_PATH.relative_to(REPO_ROOT)}")

    print()
    if _docker_available():
        _info("Checking sandbox image...")
        rc = subprocess.run(
            ["docker", "image", "inspect", "serenade/sandbox:dev"],
            capture_output=True, check=False,
        ).returncode
        if rc != 0:
            _info("Building sandbox image (one-time, 5-10 min)...")
            build_rc = _run(
                ["docker", "build", "-t", "serenade/sandbox:dev",
                 "-f", "infra/docker/Dockerfile.sandbox", "."],
                check=False,
            )
            if build_rc == 0:
                _ok("Sandbox image built.")
            else:
                _warn("Sandbox build failed. Tools will fail until the image exists.")
        else:
            _ok("Sandbox image already exists.")

    if env.get("SERENADE_LLM_BACKEND") == "ollama" and shutil.which("ollama"):
        print()
        _info("Pulling Ollama model (gemma4:26b, ~8 GB)...")
        pull_rc = _run(["ollama", "pull", "gemma4:26b"], check=False)
        if pull_rc == 0:
            _ok("Model ready.")
        else:
            _warn("Model pull failed. Ollama will retry on first use.")

    print()
    _ok("Setup complete.")
    _info(f"Data: {data_dir}")
    _info("Run: serenade")
    return 0


# --------------------------------------------------------------------------- #
# Pre-flight checks                                                           #
# --------------------------------------------------------------------------- #


def _ensure_env_or_setup() -> bool:
    """If .env is missing, run the setup wizard. Returns True if the
    file exists (after possibly creating it), False on user abort."""
    if ENV_PATH.exists():
        return True
    _info(".env missing — running setup wizard.")
    return cmd_setup() == 0 and ENV_PATH.exists()


def _docker_available() -> bool:
    """Check that the Docker CLI is on PATH and the daemon is reachable."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _preflight() -> int:
    """Run pre-flight checks before launching the TUI. Returns 0 on
    success, non-zero on failure."""
    if not _ensure_env_or_setup():
        _err("Aborting — no .env in place.")
        return 1
    if not _docker_available():
        _warn(
            "Docker daemon not reachable. Sandbox isolation requires Docker.\n"
            "    Install: https://docs.docker.com/get-docker/\n"
            "    Then start the daemon and retry."
        )
        # Not fatal — the operator may be doing non-sandbox work, or Docker
        # may come up before the first tool call.
    return 0


# --------------------------------------------------------------------------- #
# Primary modes                                                                #
# --------------------------------------------------------------------------- #


def cmd_default_tui() -> int:
    """`serenade` with no flags: run pre-flight checks, optionally spin
    up a local LLM server, then launch the TUI. The LLM server is
    killed when the TUI exits."""
    rc = _preflight()
    if rc != 0:
        return rc
    if not shutil.which("serenade-tui"):
        _err(
            "`serenade-tui` not on PATH. Install the optional extra:\n"
            "    pip install -e .[tui]"
        )
        return 127
    with _mlx_lifecycle_if_local():
        _info("Launching the TUI…")
        return _run(["serenade-tui"], check=False)



# --------------------------------------------------------------------------- #
# Secondary flags                                                              #
# --------------------------------------------------------------------------- #


def cmd_mlx(model: str | None = None, port: int | None = None) -> int:
    """Run the local LLM server in the foreground.

    Picks ``mlx_lm.server`` on macOS, ``ollama serve`` on Linux. Mainly
    useful for power-user CI scripts — the default ``serenade`` flow
    auto-starts and cleans up the server itself.
    """
    spec = _local_llm_server_spec()
    if spec is None:
        _err(
            "No default local-LLM server known for this platform. "
            "Start one yourself and set MLX_BASE_URL appropriately."
        )
        return 1
    binary = str(spec["binary"])
    if not shutil.which(binary):
        _err(f"`{binary}` not found. Install: {spec['install_hint']}")
        return 127
    # Linux/Ollama ignores --port (it reads OLLAMA_HOST); macOS/MLX
    # accepts --model + --port directly.
    if spec["name"] == "mlx":
        argv = [
            "mlx_lm.server",
            "--model",
            model or _MLX_DEFAULT_MODEL,
            "--port",
            str(port or _MLX_DEFAULT_PORT),
        ]
    else:
        # Ollama: optionally pull a non-default model before serving.
        if model and model != _OLLAMA_DEFAULT_MODEL:
            _info(f"Pulling Ollama model: {model}…")
            _run(["ollama", "pull", model], check=False)
        if port:
            os.environ["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        argv = ["ollama", "serve"]
    _info(f"Starting {spec['name']} server (foreground; Ctrl-C to stop)…")
    return _run(argv, check=False)


def cmd_migrate() -> int:
    return _run(["alembic", "upgrade", "head"], check=False)




def cmd_test(scope: str = "unit") -> int:
    if scope == "unit":
        return _run(
            ["pytest", "-m", "not integration and not sandbox and not llm", "-q"],
            check=False,
        )
    if scope == "integration":
        return _run(["pytest", "-m", "integration", "-q"], check=False)
    if scope == "e2e":
        os.environ["RUN_E2E"] = "1"
        return _run(
            ["pytest", "-m", "integration", "tests/integration/test_dvwa_e2e.py", "-v"],
            check=False,
        )
    if scope == "llm":
        os.environ["RUN_LIVE_LLM"] = "1"
        return _run(
            ["pytest", "-m", "llm", "tests/integration/test_live_llm.py", "-v"],
            check=False,
        )
    if scope == "all":
        return _run(["pytest", "-q"], check=False)
    _err(f"Unknown scope {scope!r}. Pick: unit, integration, e2e, llm, all.")
    return 2


# --------------------------------------------------------------------------- #
# engagement watch / summary                                                  #
# --------------------------------------------------------------------------- #


# High-signal kinds the operator usually wants to see, by default.
_DEFAULT_WATCH_KINDS: tuple[str, ...] = (
    "tool.completed",
    "tool.suppressed",
    "approval.queued",
    "approval.decided",
    "regression.requested",
    "regression.applied",
    "regression.vetoed",
    "stage.changed",
    "engagement.aborted",
    "engagement.abort_requested",
    "agent.escalate_human",
    # Phase-M2 threshold-trip events live in the same surface.
    "engagement.budget_warning",
    "engagement.veto_rate_warning",
    "engagement.suppression_spread",
)

# Terminal-state kinds that cause ``watch`` to stop tailing.
_TERMINAL_KINDS: frozenset[str] = frozenset(
    {"engagement.completed", "engagement.aborted"}
)


def _default_orchestrator_url() -> str:
    """Resolve the orchestrator base URL.

    Priority: explicit env override → .env file → default
    (``http://127.0.0.1:8000``). In the native architecture the
    orchestrator runs in-process, so this is mainly used by the
    engagement watch/summary commands for backward compatibility.
    """
    explicit = os.environ.get("SERENADE_ORCHESTRATOR_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    env = _read_env(ENV_PATH)
    explicit = env.get("SERENADE_ORCHESTRATOR_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return "http://127.0.0.1:8000"


# ANSI fallbacks used when rich isn't available — preserves coloured
# output without a hard dependency on the library.
_ANSI = {
    "reset": "\033[0m",
    "green": "\033[32m",
    "amber": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "red": "\033[1;31m",
    "grey": "\033[90m",
    "bold": "\033[1m",
}


def _color_for_kind(kind: str) -> str:
    """Map an event kind to a colour role.

    Returns one of the keys in ``_ANSI`` (and the matching rich style
    name) so renderers can share the lookup. Default: grey for unknown.
    """
    if kind in ("tool.completed",):
        return "green"
    if kind in ("tool.suppressed", "engagement.suppression_spread"):
        return "amber"
    if kind in ("approval.queued",):
        return "blue"
    if kind in (
        "regression.requested",
        "regression.applied",
        "regression.vetoed",
        "regression.escalated",
    ):
        return "magenta"
    if kind in (
        "agent.escalate_human",
        "engagement.aborted",
        "engagement.abort_requested",
        "engagement.budget_warning",
        "engagement.veto_rate_warning",
    ):
        return "red"
    if kind in ("approval.decided",):
        return "bold"
    return "grey"


def _summarize_event(kind: str, payload: dict[str, Any]) -> str:
    """Human-readable one-line summary of a payload dict.

    Best-effort: anything we don't recognise falls back to a JSON
    compact-dump. Keeps the watch tail readable even for unknown kinds.
    """
    p = payload or {}
    if kind == "tool.completed":
        return (
            f"{p.get('tool', '?')} stage={p.get('stage', '?')} "
            f"exit={p.get('exit_code', '?')} nodes={p.get('nodes_written', 0)}"
        )
    if kind == "tool.suppressed":
        return (
            f"{p.get('tool', '?')} stage={p.get('stage', '?')} "
            f"reason={p.get('reason', '?')}"
        )
    if kind == "approval.queued":
        return f"{p.get('tool', p.get('kind', '?'))} stage={p.get('stage', '?')}"
    if kind == "approval.decided":
        return (
            f"decision={p.get('decision', '?')} "
            f"by={p.get('decided_by', '?')} kind={p.get('kind', '?')}"
        )
    if kind.startswith("regression."):
        req = p.get("request") or p
        return (
            f"target={req.get('target_stage', '?')} "
            f"trigger={req.get('trigger_kind', '?')}"
        )
    if kind == "stage.changed":
        return f"{p.get('from', '?')} -> {p.get('to', '?')}"
    if kind == "engagement.aborted" or kind == "engagement.abort_requested":
        return f"by={p.get('aborted_by', '?')} reason={p.get('reason', '?')}"
    if kind == "agent.escalate_human":
        return str(p.get("reason", "human review required"))
    if kind == "engagement.budget_warning":
        return f"kind={p.get('kind', '?')} usage={p.get('usage', '?')}"
    if kind == "engagement.veto_rate_warning":
        return (
            f"vetoes={p.get('vetoes', '?')}/{p.get('window', '?')} "
            f"rate={p.get('rate', '?')}"
        )
    if kind == "engagement.suppression_spread":
        added = p.get("added") or []
        sample = ",".join(added[:3])
        return f"stage={p.get('stage', '?')} added=[{sample}]"
    # Fallback — compact JSON.
    import json as _json

    try:
        return _json.dumps(p, default=str)[:200]
    except Exception:
        return repr(p)[:200]


def _format_line(ts: str, stage: str, kind: str, summary: str, color_key: str) -> str:
    """Build a single rendered line of watch output (ANSI-coloured)."""
    # ``ts`` is an ISO timestamp from the bus; render the HH:MM:SS portion.
    short_ts = ts[11:19] if len(ts) >= 19 else ts
    color = _ANSI.get(color_key, "")
    reset = _ANSI["reset"] if color else ""
    grey = _ANSI["grey"]
    return (
        f"{grey}[{short_ts}]{_ANSI['reset']} "
        f"{_ANSI['bold']}[{stage}]{_ANSI['reset']} "
        f"{color}{kind}{reset}: {summary}"
    )


def cmd_engagement_watch(
    *,
    engagement_id: str,
    filter_kinds: str | None,
    base_url: str | None,
) -> int:
    """Stream engagement events from the local orchestrator's SSE feed.

    Runs until Ctrl-C or the engagement reaches a terminal state. On
    exit, prints a one-line summary of events seen + duration. Network
    errors raise a single ``_err`` line and return 1.
    """
    import time as _time

    base = (base_url or _default_orchestrator_url()).rstrip("/")
    kinds_filter: set[str]
    if filter_kinds:
        kinds_filter = {k.strip() for k in filter_kinds.split(",") if k.strip()}
    else:
        kinds_filter = set(_DEFAULT_WATCH_KINDS)

    url = f"{base}/engagements/{engagement_id}/events/stream"
    _info(f"Watching {engagement_id} — Ctrl-C to exit.")
    _info(f"Filter: {sorted(kinds_filter)}")

    try:
        import httpx
        from httpx_sse import connect_sse
    except ImportError:
        _err("httpx and httpx-sse are required. Install with `pip install -e .`")
        return 127

    n_events = 0
    n_warnings = 0
    start = _time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
            with connect_sse(client, "GET", url) as event_source:
                for sse in event_source.iter_sse():
                    kind = sse.event or "message"
                    if kind in ("_heartbeat", "message", ""):
                        # The bus heartbeats on idle; skip silently.
                        if kind != "_heartbeat":
                            # Some SSE servers don't set the event name
                            # explicitly. Try the data payload.
                            pass
                        else:
                            continue
                    if kinds_filter and kind not in kinds_filter:
                        continue
                    import json as _json

                    try:
                        payload = _json.loads(sse.data) if sse.data else {}
                    except _json.JSONDecodeError:
                        payload = {"raw": sse.data}
                    n_events += 1
                    if kind.endswith("_warning") or kind in (
                        "engagement.suppression_spread",
                        "engagement.aborted",
                        "engagement.abort_requested",
                        "agent.escalate_human",
                    ):
                        n_warnings += 1
                    stage = (payload or {}).get("stage", "?")
                    ts = (payload or {}).get("ts") or ""
                    summary = _summarize_event(kind, payload or {})
                    line = _format_line(ts, str(stage), kind, summary, _color_for_kind(kind))
                    print(line, flush=True)
                    if kind in _TERMINAL_KINDS:
                        _info(f"Terminal event {kind!r} received — stopping.")
                        break
    except KeyboardInterrupt:
        print()  # break the ^C line
    except httpx.HTTPError as e:
        _err(f"Stream failed: {e}")
        return 1
    duration = _time.time() - start
    _info(
        f"Watched {n_events} events over {duration:.1f}s. "
        f"{n_warnings} warning{'s' if n_warnings != 1 else ''}."
    )
    return 0


def _render_summary_text(s: dict[str, Any]) -> str:
    """Render a compact, colour-coded text table for ``cmd_engagement_summary``.

    Red threshold rules:
      - spend (llm_cost_usd + tool_cost_usd) > 80% of any configured cap
        — we don't have the cap on hand, so colour only when the
        wallclock_minutes_used exceeds 80% of the engagement window or
        regressions_used == regression_budget_max. token-budget thresholds
        already trip in the orchestrator and surface on the bus.
    """
    tools_total = int(s.get("tool_calls_total", 0))
    auto = int(s.get("tool_calls_auto_approved", 0))
    vetoed = int(s.get("tool_calls_critic_vetoed", 0))
    human = int(s.get("tool_calls_human_approved_gated", 0))
    ungated = int(s.get("tool_calls_ungated", 0))
    suppressed = int(s.get("suppressed_tools_count", 0))
    llm_usd = float(s.get("llm_cost_usd", 0.0) or 0.0)
    tool_usd = float(s.get("tool_cost_usd", 0.0) or 0.0)
    wallclock = float(s.get("wallclock_minutes_used", 0.0) or 0.0)
    regr_used = int(s.get("regressions_used", 0))
    findings = int(s.get("findings_count", 0))

    # Red flags. We don't always know the engagement cap from /summary
    # alone; the engagement endpoint carries it. For now: any vetoes >
    # 30% of total tool calls OR suppressed > 0 colours that line amber.
    red = _ANSI["red"]
    amber = _ANSI["amber"]
    reset = _ANSI["reset"]
    bold = _ANSI["bold"]

    def _amber_if(cond: bool, s_: str) -> str:
        return f"{amber}{s_}{reset}" if cond else s_

    def _red_if(cond: bool, s_: str) -> str:
        return f"{red}{s_}{reset}" if cond else s_

    veto_rate = (vetoed / tools_total) if tools_total else 0.0
    lines = [
        f"{bold}Engagement summary{reset}",
        f"  Spend (LLM):     ${llm_usd:.4f}",
        f"  Spend (tools):   ${tool_usd:.4f}",
        f"  Tool calls:      {tools_total} "
        f"(auto={auto}, vetoed={_red_if(veto_rate > 0.5, str(vetoed))}, "
        f"human={human}, ungated={ungated})",
        f"  Suppressed:      {_amber_if(suppressed > 0, str(suppressed))}",
        f"  Regressions:     {regr_used}",
        f"  Wallclock used:  {wallclock:.1f} min",
        f"  Findings:        {findings}",
    ]
    return "\n".join(lines)


def cmd_engagement_summary(
    *,
    engagement_id: str,
    fmt: str,
    base_url: str | None,
) -> int:
    """One-shot ``GET /engagements/{id}/summary`` digest.

    Output format is text/json/yaml as requested. Network errors return 1.
    The endpoint itself returns a partial digest on any internal failure
    (each counter is independent), so this command rarely needs special
    handling for degraded responses.
    """
    base = (base_url or _default_orchestrator_url()).rstrip("/")
    url = f"{base}/engagements/{engagement_id}/summary"
    try:
        import httpx
    except ImportError:
        _err("httpx is required. Install with `pip install -e .`")
        return 127
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        _err(f"Server returned {e.response.status_code}: {e.response.text[:200]}")
        return 1
    except httpx.HTTPError as e:
        _err(f"Request failed: {e}")
        return 1

    if fmt == "json":
        import json as _json

        print(_json.dumps(data, indent=2, default=str))
        return 0
    if fmt == "yaml":
        try:
            import yaml
        except ImportError:
            _err("PyYAML not installed; falling back to JSON.")
            import json as _json

            print(_json.dumps(data, indent=2, default=str))
            return 0
        print(yaml.safe_dump(data, sort_keys=True), end="")
        return 0
    # Default: text/table.
    print(_render_summary_text(data))
    return 0


# --------------------------------------------------------------------------- #
# Argparse                                                                     #
# --------------------------------------------------------------------------- #


_DESCRIPTION = """\
Serenade — autonomous penetration testing tool.

Primary modes:
  serenade               launch the TUI
  serenade --setup       guided first-run setup (writes .env)

Docker is used only for sandbox containers (tool isolation).
The Makefile remains for power users and CI.
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="serenade",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Subcommands (engagement watch/summary). Implemented as an
    # optional subparser so the flag-based default modes still work
    # ``serenade`` (no args) lands on cmd_default_tui;
    # ``serenade engagement watch <id>``
    # lands on the new subcommand handler below.
    sub = p.add_subparsers(dest="subcommand", required=False)
    eng = sub.add_parser(
        "engagement",
        help="engagement lifecycle observability commands",
    )
    eng_sub = eng.add_subparsers(dest="engagement_subcommand", required=False)

    watch = eng_sub.add_parser(
        "watch",
        help="stream high-signal events for an engagement (Ctrl-C to exit)",
    )
    watch.add_argument("engagement_id", help="engagement UUID")
    watch.add_argument(
        "--filter",
        dest="filter_kinds",
        default=None,
        help="comma-separated event-kind filter (overrides the default high-signal set)",
    )
    watch.add_argument(
        "--url",
        default=None,
        help="orchestrator base URL (default: http://127.0.0.1:8000)",
    )

    summary = eng_sub.add_parser(
        "summary",
        help="one-shot engagement digest (spend, tool calls, regressions, findings)",
    )
    summary.add_argument("engagement_id", help="engagement UUID")
    summary.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="output format (default: text)",
    )
    summary.add_argument(
        "--url",
        default=None,
        help="orchestrator base URL (default: http://127.0.0.1:8000)",
    )

    primary = p.add_argument_group("primary modes")
    primary.add_argument(
        "--setup",
        action="store_true",
        help="Guided first-run setup (writes .env)",
    )
    other = p.add_argument_group("other")
    other.add_argument(
        "--mlx",
        action="store_true",
        help="run mlx_lm.server on the host (macOS — separate terminal)",
    )
    other.add_argument(
        "--mlx-model",
        metavar="ID",
        help="MLX model id (default: mlx-community/gemma-3-27b-it-4bit)",
    )
    other.add_argument("--mlx-port", type=int, metavar="N", help="MLX port (default: 8080)")
    other.add_argument("--migrate", action="store_true", help="apply alembic migrations")
    other.add_argument(
        "--test",
        nargs="?",
        const="unit",
        metavar="SCOPE",
        choices=["unit", "integration", "e2e", "llm", "all"],
        help="run pytest at the given scope (default: unit)",
    )
    return p




def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Subcommand dispatch — wins over every flag-based path.
    # ``serenade engagement watch <id>`` and ``serenade engagement
    # summary <id>`` are local-orchestrator observability tools; they
    # don't bring up the stack themselves (operator runs them while the
    # stack is already up).
    if getattr(args, "subcommand", None) == "engagement":
        sub = getattr(args, "engagement_subcommand", None)
        if sub == "watch":
            return cmd_engagement_watch(
                engagement_id=args.engagement_id,
                filter_kinds=args.filter_kinds,
                base_url=args.url,
            )
        if sub == "summary":
            return cmd_engagement_summary(
                engagement_id=args.engagement_id,
                fmt=args.format,
                base_url=args.url,
            )
        # `serenade engagement` with no further word: print help.
        _err("Pick a subcommand: `engagement watch <id>` or `engagement summary <id>`.")
        return 2

    # Single primary-mode dispatch. The order matters: flags that should
    # win over each other go first. We never combine primary modes.
    if args.setup:
        return cmd_setup()
    if args.mlx:
        return cmd_mlx(model=args.mlx_model, port=args.mlx_port)
    if args.migrate:
        return cmd_migrate()
    if args.test is not None:
        return cmd_test(args.test)
    # Default: spin up + launch TUI.
    return cmd_default_tui()
