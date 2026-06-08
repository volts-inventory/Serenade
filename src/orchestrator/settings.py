"""Application settings via pydantic-settings."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    # LLM backend selection. Local (ollama) is the default — most
    # operators start on their own machine without a hosted-LLM bill.
    #
    # SAFETY: Anthropic is **explicit opt-in only**. Having
    # ``ANTHROPIC_API_KEY`` in your shell (e.g. set for Cursor / Claude
    # Code) is NOT enough — that would silently consume the operator's
    # paid key on a fresh engagement. ``get_settings()`` emits a one-time
    # warning when the key is present but the backend isn't explicitly
    # ``anthropic``.
    #
    # Env-var aliases: ``SERENADE_LLM_BACKEND`` is canonical (matches the
    # rest of the ``SERENADE_*`` namespace). The unprefixed ``LLM_BACKEND``
    # is honoured as a deprecated fallback with a startup warning.
    llm_backend: Literal["anthropic", "ollama"] = Field(
        default="ollama",
        validation_alias=AliasChoices("SERENADE_LLM_BACKEND", "LLM_BACKEND", "llm_backend"),
    )

    @field_validator("llm_backend", mode="before")
    @classmethod
    def _map_mlx_to_ollama(cls, v: str) -> str:
        if v == "mlx":
            warnings.warn(
                "llm_backend='mlx' is deprecated; use 'ollama'",
                DeprecationWarning,
                stacklevel=2,
            )
            return "ollama"
        return v

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model_planner: str = "claude-sonnet-4-6"
    anthropic_model_synthesizer: str = "claude-opus-4-7"
    anthropic_model_parser: str = "claude-haiku-4-5-20251001"
    anthropic_model_triage: str = "claude-haiku-4-5-20251001"
    # Critic / council members run cheap-and-fast by default; one council
    # synthesizer in agents/council.py overrides to Sonnet for the verdict.
    anthropic_model_critic: str = "claude-haiku-4-5-20251001"

    # Local LLM (Ollama / OpenAI-compatible endpoint)
    # validation_alias accepts the old MLX_BASE_URL / MLX_MODEL env vars
    # so operators with those in .env don't get silent default overrides.
    local_base_url: str = Field(
        default="http://host.docker.internal:11434/v1",
        validation_alias=AliasChoices("LOCAL_BASE_URL", "MLX_BASE_URL"),
    )
    local_model_planner: str = Field(
        default="gemma4:26b",
        validation_alias=AliasChoices("LOCAL_MODEL_PLANNER", "MLX_MODEL"),
    )
    local_model_critic: str = "gemma4"
    local_model_synthesizer: str = "gemma4"
    local_model_parser: str = "gemma4:26b"
    local_model_triage: str = "gemma4"
    local_keep_alive: str = "5m"

    # Feature flags (opt-in, default off).
    enable_step_scoring: bool = False
    enable_reflection_memory: bool = False
    enable_debate_gates: bool = False
    enable_regression_reflection: bool = False
    # Deprecated aliases for the old mlx_* names. These properties
    # emit DeprecationWarnings so callers can migrate at their own pace.
    @property
    def mlx_base_url(self) -> str:
        warnings.warn(
            "Settings.mlx_base_url is deprecated; use local_base_url",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.local_base_url

    @property
    def mlx_model(self) -> str:
        warnings.warn(
            "Settings.mlx_model is deprecated; use local_model_planner",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.local_model_planner

    # Redis is only needed for the optional Windows worker VM IPC.
    # When SERENADE_WINDOWS_WORKER_ENABLED=1, the WindowsJobQueue reads
    # REDIS_URL directly from the environment (default redis://redis:6379/0).
    # Core code (event bus, abort signals, SIEM) uses in-process asyncio.

    # Data directory (evidence blobs, config, etc.)
    data_dir: str = "~/.serenade"

    log_level: str = "INFO"

    # Sandbox
    sandbox_image: str = "serenade/sandbox:dev"
    # When True, a nftables-ruleset apply failure during sandbox container
    # creation is treated as fatal: the run raises RuntimeError instead of
    # silently falling back to L2-only isolation. Production deployments that
    # depend on per-engagement L3 ACLs (egress allow-lists, per-target deny)
    # should set this to True; dev/test deployments keep it False and accept
    # the documented warning.
    sandbox_l3_required: bool = False

    # Budgets
    engagement_default_token_budget: int = 2_000_000
    engagement_default_wallclock_hours: int = 8

    # Metasploit RPC TLS. The msfrpcd default deployment ships with a
    # self-signed cert, so out-of-the-box ``MSFRPC_TLS_VERIFY=False`` matches
    # historical behaviour. Production deployments with a CA-signed cert
    # (or an internal CA bundle) should set ``MSFRPC_CA_BUNDLE`` to a
    # filesystem path containing the trusted PEM(s); when set, the adapter
    # verifies the cert against that bundle and ``MSFRPC_TLS_VERIFY`` is
    # treated as True. When neither is set, the adapter logs a startup
    # warning ("msfrpc TLS verification disabled") so operators see the
    # gap in their logs.
    msfrpc_tls_verify: bool = False
    msfrpc_ca_bundle: str | None = None

    # Metasploit RPC connection. ``msfrpc_url`` covers the full URL incl. the
    # ``/api/`` REST prefix that recent msfrpcd builds expose. ``msfrpc_user``
    # and ``msfrpc_password`` are optional at the Settings layer so a deployment
    # without msfrpc configured still passes pydantic validation; the adapter
    # raises a clean error when it tries to use them. Reads happen via
    # ``get_settings()`` so the lru_cache is honoured (the previous
    # ``os.environ.get`` path in ``msfrpc_.MetasploitExploit.run`` bypassed it).
    # Backwards compat for operators on the old env-var path is documented in
    # ``msfrpc_._resolve_msfrpc_credentials``.
    msfrpc_url: str = "https://127.0.0.1:55553/api/"
    msfrpc_user: str | None = None
    msfrpc_password: str | None = None

    # BloodHound integration is no longer backed by a separate Neo4j.
    # The BloodHound tool adapter writes directly to the in-process
    # networkx graph (same node types, just imported data).

    @property
    def sqlite_dsn(self) -> str:
        """SQLite DSN for Alembic and other consumers that need the URL."""
        from pathlib import Path

        data_dir = Path(self.data_dir).expanduser()
        return f"sqlite+aiosqlite:///{data_dir}/serenade.db"


@lru_cache
def get_settings() -> Settings:
    """Resolve runtime settings with the Anthropic-opt-in safety rule.

    Resolution order (first match wins):

      1. ``SERENADE_LLM_BACKEND`` env var (canonical).
      2. ``LLM_BACKEND`` env var (legacy, deprecated — emits a warning).
      3. Settings default — ``ollama`` (local).

    The historical "implicit auto-flip to anthropic when
    ``ANTHROPIC_API_KEY`` is set" behaviour has been REMOVED. Anthropic
    is now strictly opt-in: the operator must say so via the env var or
    the per-user LLM config table. Rationale — a developer who exports
    ``ANTHROPIC_API_KEY`` for Cursor / Claude Code would otherwise have
    Serenade silently spend against that key on the first engagement.

    When the key is present but the explicit backend isn't ``anthropic``,
    we emit a single startup warning (``llm.anthropic_key_ignored``) so
    the gap is obvious in logs.
    """
    import os

    import structlog

    s = Settings()  # type: ignore[call-arg]

    log = structlog.get_logger(__name__)
    canonical_set = bool(os.environ.get("SERENADE_LLM_BACKEND", "").strip())
    legacy_set = bool(os.environ.get("LLM_BACKEND", "").strip())

    # Warn when the deprecated LLM_BACKEND env var is the sole selector.
    if legacy_set and not canonical_set:
        log.warning(
            "llm.legacy_env_var_deprecated",
            advice=(
                "LLM_BACKEND is deprecated. Rename to SERENADE_LLM_BACKEND "
                "in your .env for consistency with other SERENADE_* vars."
            ),
        )

    # Warn when ANTHROPIC_API_KEY is present but no backend was chosen
    # (don't warn if the operator explicitly picked a local backend).
    no_explicit_choice = not (canonical_set or legacy_set)
    if (
        s.llm_backend != "anthropic"
        and no_explicit_choice
        and os.environ.get("ANTHROPIC_API_KEY", "").strip()
    ):
        log.warning(
            "llm.anthropic_key_ignored",
            backend=s.llm_backend,
            advice=(
                "ANTHROPIC_API_KEY is set but SERENADE_LLM_BACKEND is not "
                "'anthropic'. The key will be IGNORED. To use Anthropic, "
                "set SERENADE_LLM_BACKEND=anthropic in your .env."
            ),
        )

    # Fail fast: anthropic backend without a key is a boot-time error.
    if s.llm_backend == "anthropic" and not s.anthropic_api_key.strip():
        raise RuntimeError(
            "SERENADE_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is empty. "
            "Set ANTHROPIC_API_KEY in your .env or switch to a local "
            "backend (SERENADE_LLM_BACKEND=ollama)."
        )

    return s


