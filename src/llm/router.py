"""LLMRouter — backend-agnostic interface used by all agents.

Selects backend per-deployment (env `SERENADE_LLM_BACKEND`, legacy
`LLM_BACKEND` honoured with a deprecation warning) and routes per
call-site inside the backend. Agents never import a specific backend.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Protocol

import structlog

from orchestrator.settings import Settings, get_settings

log = structlog.get_logger(__name__)


class CallSite(StrEnum):
    PLANNER = "planner"  # stage agent decides next tools
    SYNTHESIZER = "synthesizer"  # report narrative, exec summary
    PARSER = "parser"  # tool output -> structured facts (no grammar)
    TRIAGE = "triage"  # exploitability/severity scoring
    CRITIC = "critic"  # devil's advocate / council member (bounded review)


class ToolSpec(Protocol):
    """Minimal contract for what a backend needs about a tool."""

    name: str
    description: str

    def input_schema(self) -> dict[str, Any]: ...


class LLMRequest:
    def __init__(
        self,
        *,
        system: str,
        user: str,
        site: CallSite,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        cache_breakpoint_system: bool = True,
        engagement_id: str | None = None,
        # Interactive loop — when set, backends use the messages list
        # directly instead of wrapping ``user`` in a single message. This is
        # how BaseAgent.step() threads a multi-turn conversation (assistant
        # tool_use blocks + user tool_result blocks) through the LLM so the
        # model sees its own tool results inside the same context. Unset is
        # the legacy single-shot path; ``user`` is still required as a
        # backwards-compatible fallback string for telemetry / logs.
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.system = system
        self.user = user
        self.site = site
        self.tools = tools or []
        self.response_schema = response_schema
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_breakpoint_system = cache_breakpoint_system
        self.engagement_id = engagement_id
        self.messages = messages


class ToolCall:
    def __init__(self, name: str, input: dict[str, Any], call_id: str) -> None:
        self.name = name
        self.input = input
        self.call_id = call_id


class LLMResponse:
    def __init__(
        self,
        *,
        text: str,
        tool_calls: list[ToolCall],
        structured: dict[str, Any] | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cost_usd: float,
        latency_ms: int,
        raw: Any,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls
        self.structured = structured
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms
        self.raw = raw


class LLMBackend(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...


class LLMRouter:
    """Routes LLM calls to a backend with bounded retry on transient errors.

    Retryable errors (429, 5xx, connection errors) trigger exponential
    backoff up to `max_retries`. Non-retryable errors (400, 401, 422)
    bubble immediately so the agent escalates to operator.
    """

    def __init__(
        self,
        backend: LLMBackend,
        settings: Settings,
        *,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    async def complete(self, req: LLMRequest, *, agent: str = "unknown") -> LLMResponse:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self.max_retries:
            try:
                resp = await self.backend.complete(req)
                break
            except Exception as e:
                last_exc = e
                if not _is_retryable(e) or attempt >= self.max_retries:
                    log.warning(
                        "llm.call_failed_non_retryable",
                        err=str(e),
                        attempt=attempt,
                        backend=self.backend.name,
                    )
                    raise
                delay = self.base_delay_seconds * (2**attempt)
                log.warning(
                    "llm.retry",
                    err=str(e),
                    attempt=attempt + 1,
                    delay_seconds=delay,
                    backend=self.backend.name,
                )
                await asyncio.sleep(delay)
                attempt += 1
        else:
            # max retries exhausted without break (defensive — shouldn't be reachable)
            raise last_exc or RuntimeError("LLM retry loop exhausted with no exception")

        log.info(
            "llm.call",
            site=req.site,
            backend=self.backend.name,
            model=resp.model,
            in_tokens=resp.input_tokens,
            cached_in=resp.cached_input_tokens,
            out_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=resp.latency_ms,
            engagement_id=req.engagement_id,
            retries=attempt,
        )
        # Best-effort persistence to llm_calls. Skipped silently if Postgres
        # is unreachable or no engagement_id (e.g. unit tests).
        if req.engagement_id:
            try:
                from uuid import UUID as _UUID

                from orchestrator.ledger import record_llm_call

                await record_llm_call(
                    engagement_id=_UUID(req.engagement_id),
                    agent=agent,
                    site=req.site.value if hasattr(req.site, "value") else str(req.site),
                    backend=self.backend.name,
                    model=resp.model,
                    input_tokens=resp.input_tokens,
                    cached_input_tokens=resp.cached_input_tokens,
                    output_tokens=resp.output_tokens,
                    cost_usd=resp.cost_usd,
                    latency_ms=resp.latency_ms,
                )
            except Exception as e:
                log.debug("llm.ledger_skip", err=str(e))
        return resp


def _is_retryable(exc: Exception) -> bool:
    """Decide whether to retry on an exception from a backend call.

    Retryable: HTTP 408/429/5xx, asyncio TimeoutError, connection errors.
    Non-retryable: HTTP 400/401/403/404/422 (request shape problem), KeyError, etc.
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return bool(status in {408, 429} or status >= 500)
    # Anthropic SDK exception class names include "APITimeoutError",
    # "APIConnectionError", "RateLimitError", "InternalServerError".
    name = type(exc).__name__
    return name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }


# Public alias so the base agent's transient-failure detection can
# share the same retryability rules the router uses for its own retry loop.
# Keep the underscore-prefixed name working for internal callers.
is_retryable_exception = _is_retryable


def build_router(settings: Settings | None = None) -> LLMRouter:
    """Build a router from a Settings snapshot (env-only).

    Use :func:`build_router_async` when you want the DB-backed runtime
    config to overlay onto the env-derived defaults.
    """
    s = settings or get_settings()
    backend: LLMBackend
    if s.llm_backend == "anthropic":
        from llm.anthropic_backend import AnthropicBackend

        backend = AnthropicBackend(s)
    elif s.llm_backend == "ollama":
        from llm.local_backend import LocalBackend

        backend = LocalBackend(s)
    else:  # pragma: no cover - guarded by settings literal
        raise ValueError(f"Unknown SERENADE_LLM_BACKEND: {s.llm_backend}")
    return LLMRouter(backend, s)


# Snapshot key names. Kept in lock-step with
# ``orchestrator.services.engagements._build_llm_config_snapshot`` so the write
# and read sides agree (findings llm-04, llm-05/db-06). The backend lives under
# ``llm_backend``; per-call-site models live under ``{backend}_model_{site}``.
_SNAPSHOT_MODEL_FIELDS = (
    "anthropic_model_planner",
    "anthropic_model_synthesizer",
    "anthropic_model_parser",
    "anthropic_model_triage",
    "anthropic_model_critic",
    "local_model_planner",
    "local_model_synthesizer",
    "local_model_parser",
    "local_model_triage",
    "local_model_critic",
)


def snapshot_from_settings(settings: Settings) -> dict:
    """Build the LLM-config snapshot dict from a Settings object.

    The inverse of :func:`apply_snapshot`; round-trips through the same keys
    the engagement read side consumes. Only the active backend's per-call-site
    models are recorded.
    """
    snap: dict = {"llm_backend": settings.llm_backend}
    prefix = "anthropic_model_" if settings.llm_backend == "anthropic" else "local_model_"
    for field in _SNAPSHOT_MODEL_FIELDS:
        if field.startswith(prefix):
            val = getattr(settings, field, None)
            if val is not None:
                snap[field] = val
    return snap


def apply_snapshot(settings: Settings, snapshot: dict | None) -> Settings:
    """Overlay an engagement's ``llm_config_snapshot`` onto env Settings.

    Pins the backend (and per-call-site models) the engagement was created
    with, so a mid-run / resume-time change to ``SERENADE_LLM_BACKEND`` (or the
    model env vars) cannot silently switch backends underneath a running
    engagement (finding llm-04).

    Fails closed: if the snapshot demands the ``anthropic`` backend but no API
    key is configured in the current environment, raises rather than building
    an unusable backend (preserves the opt-in/credentials guarantee, llm-06).
    """
    if not isinstance(snapshot, dict) or not snapshot:
        return settings
    overrides: dict[str, Any] = {}
    # Back-compat: oldest snapshots stored the backend under "backend".
    backend = snapshot.get("llm_backend") or snapshot.get("backend")
    if backend == "mlx":  # deprecated alias
        backend = "ollama"
    if backend in ("anthropic", "ollama"):
        overrides["llm_backend"] = backend
    for field in _SNAPSHOT_MODEL_FIELDS:
        if field in snapshot and snapshot[field] is not None:
            overrides[field] = snapshot[field]
    if not overrides:
        return settings
    pinned = settings.model_copy(update=overrides)
    if pinned.llm_backend == "anthropic" and not (pinned.anthropic_api_key or "").strip():
        raise RuntimeError(
            "engagement llm_config_snapshot pins backend=anthropic but "
            "ANTHROPIC_API_KEY is empty in this environment; refusing to "
            "switch backends. Configure the key or resume on the original host."
        )
    log.info(
        "llm.snapshot_pinned",
        backend=pinned.llm_backend,
        env_backend=settings.llm_backend,
    )
    return pinned


async def build_router_async(
    settings: Settings | None = None,
    *,
    snapshot: dict | None = None,
) -> LLMRouter:
    """Build a router, pinning the engagement's snapshotted LLM config.

    When ``snapshot`` is provided (an engagement's ``llm_config_snapshot``),
    its backend and per-call-site models override the env-derived defaults so
    the engagement stays on the backend it was created with for its entire
    lifetime — including resume after an env change (finding llm-04).
    """
    s = settings or get_settings()
    s = apply_snapshot(s, snapshot)
    return build_router(s)
