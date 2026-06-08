"""Tests for LLM router retry/backoff (Phase 16, Tier A #3)."""

from __future__ import annotations

import pytest

from llm.router import LLMRequest, LLMResponse, LLMRouter, _is_retryable
from orchestrator.settings import Settings


class _MockHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _CountingBackend:
    """Fails N times with a retryable error, then succeeds."""

    name = "counting"

    def __init__(self, fail_with: Exception, success_after: int) -> None:
        self.fail_with = fail_with
        self.success_after = success_after
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        if self.calls <= self.success_after:
            raise self.fail_with
        return LLMResponse(
            text="ok",
            tool_calls=[],
            structured=None,
            model="counting",
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=0,
            cost_usd=0.0001,
            latency_ms=5,
            raw=None,
        )


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_429():
    backend = _CountingBackend(_MockHTTPError(429), success_after=2)
    router = LLMRouter(backend, Settings(_env_file=None), max_retries=5, base_delay_seconds=0)  # type: ignore[call-arg]
    resp = await router.complete(LLMRequest(system="s", user="u", site="parser"))  # type: ignore[arg-type]
    assert resp.text == "ok"
    assert backend.calls == 3


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_503():
    backend = _CountingBackend(_MockHTTPError(503), success_after=1)
    router = LLMRouter(backend, Settings(_env_file=None), max_retries=3, base_delay_seconds=0)  # type: ignore[call-arg]
    resp = await router.complete(LLMRequest(system="s", user="u", site="parser"))  # type: ignore[arg-type]
    assert resp.text == "ok"


@pytest.mark.asyncio
async def test_does_not_retry_400():
    backend = _CountingBackend(_MockHTTPError(400), success_after=100)
    router = LLMRouter(backend, Settings(_env_file=None), max_retries=5, base_delay_seconds=0)  # type: ignore[call-arg]
    with pytest.raises(_MockHTTPError):
        await router.complete(LLMRequest(system="s", user="u", site="parser"))  # type: ignore[arg-type]
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_does_not_retry_401():
    backend = _CountingBackend(_MockHTTPError(401), success_after=100)
    router = LLMRouter(backend, Settings(_env_file=None), max_retries=5, base_delay_seconds=0)  # type: ignore[call-arg]
    with pytest.raises(_MockHTTPError):
        await router.complete(LLMRequest(system="s", user="u", site="parser"))  # type: ignore[arg-type]
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    backend = _CountingBackend(_MockHTTPError(503), success_after=100)
    router = LLMRouter(backend, Settings(_env_file=None), max_retries=2, base_delay_seconds=0)  # type: ignore[call-arg]
    with pytest.raises(_MockHTTPError):
        await router.complete(LLMRequest(system="s", user="u", site="parser"))  # type: ignore[arg-type]
    # 1 initial + 2 retries = 3 calls
    assert backend.calls == 3


def test_is_retryable_classifies_correctly():
    assert _is_retryable(_MockHTTPError(429))
    assert _is_retryable(_MockHTTPError(500))
    assert _is_retryable(_MockHTTPError(502))
    assert _is_retryable(_MockHTTPError(503))
    assert _is_retryable(_MockHTTPError(408))
    assert _is_retryable(ConnectionError("boom"))
    assert _is_retryable(TimeoutError("slow"))
    assert not _is_retryable(_MockHTTPError(400))
    assert not _is_retryable(_MockHTTPError(401))
    assert not _is_retryable(_MockHTTPError(403))
    assert not _is_retryable(_MockHTTPError(404))
    assert not _is_retryable(_MockHTTPError(422))
    assert not _is_retryable(ValueError("bad input"))
