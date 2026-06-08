"""Regression tests for engagement LLM-config snapshot pinning and
Anthropic cache-aware token/cost accounting (findings llm-04, llm-07,
llm-05/db-06).

These lock in three guarantees that were previously broken:

* llm-04: ``build_router_async(snapshot=...)`` must pin the engagement's
  backend for its lifetime, so a mid-run / resume-time change to
  ``SERENADE_LLM_BACKEND`` cannot silently switch backends.
* llm-07: with prompt caching on (the default), the Anthropic backend
  must count cache-read tokens for budget purposes and must NOT
  double-subtract ``cache_creation`` in the cost formula.
* llm-05/db-06: the snapshot must round-trip through the SAME key names
  on the write side, the read side, and the router builder.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm.router import (
    apply_snapshot,
    build_router_async,
    snapshot_from_settings,
)
from orchestrator.services.engagements import (
    _build_llm_config_snapshot,
    _row_to_read,
)
from orchestrator.settings import Settings


# --------------------------------------------------------------------------- #
# llm-04 — snapshot pins the backend even if env changes                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_snapshot_pins_ollama_even_when_env_is_anthropic() -> None:
    """Engagement created on ollama must stay on ollama even though the
    live env (Settings passed in) now resolves to anthropic."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="anthropic",
        anthropic_api_key="sk-ant-present",
    )
    snapshot = {"llm_backend": "ollama", "local_model_planner": "gemma4"}

    with patch("llm.router.build_router") as mk:
        await build_router_async(env, snapshot=snapshot)

    built_settings = mk.call_args.args[0]
    assert built_settings.llm_backend == "ollama"
    # The pinned model overlay also applied.
    assert built_settings.local_model_planner == "gemma4"


@pytest.mark.asyncio
async def test_snapshot_pins_anthropic_even_when_env_is_ollama() -> None:
    """Engagement created on anthropic stays on anthropic on resume even
    though the live env resolves to ollama — provided credentials exist."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="ollama",
        anthropic_api_key="sk-ant-present",
    )
    snapshot = {
        "llm_backend": "anthropic",
        "anthropic_model_planner": "claude-sonnet-4-6",
    }

    with patch("llm.router.build_router") as mk:
        await build_router_async(env, snapshot=snapshot)

    built_settings = mk.call_args.args[0]
    assert built_settings.llm_backend == "anthropic"
    assert built_settings.anthropic_model_planner == "claude-sonnet-4-6"


def test_apply_snapshot_fails_closed_when_anthropic_without_key() -> None:
    """A snapshot pinning anthropic must refuse to build if the current
    env has no API key — preserves the opt-in/credentials guarantee
    (llm-06) rather than producing an unusable backend."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="ollama",
        anthropic_api_key="",
    )
    with pytest.raises(RuntimeError) as exc:
        apply_snapshot(env, {"llm_backend": "anthropic"})
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_apply_snapshot_noop_for_empty_or_missing() -> None:
    """No snapshot (or empty dict) leaves env settings untouched."""
    env = Settings(_env_file=None, llm_backend="ollama")  # type: ignore[call-arg]
    assert apply_snapshot(env, None) is env
    assert apply_snapshot(env, {}) is env


def test_apply_snapshot_backcompat_backend_key() -> None:
    """Oldest snapshots stored the backend under ``backend`` — still honoured."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="anthropic",
        anthropic_api_key="sk-ant-present",
    )
    pinned = apply_snapshot(env, {"backend": "ollama"})
    assert pinned.llm_backend == "ollama"


@pytest.mark.asyncio
async def test_build_router_async_without_snapshot_uses_env() -> None:
    """No snapshot → behave exactly like the env-derived defaults."""
    env = Settings(_env_file=None, llm_backend="ollama")  # type: ignore[call-arg]
    with patch("llm.router.build_router") as mk:
        await build_router_async(env, snapshot=None)
    assert mk.call_args.args[0] is env


def _fake_engagement_row(
    snapshot: dict, *, id: str = "11111111-1111-1111-1111-111111111111"
) -> SimpleNamespace:
    """Minimal stand-in for ``EngagementRow`` that satisfies ``_row_to_read``."""
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=id,
        name="e",
        scope={"cidrs": [], "domains": []},
        objectives=["o"],
        mode="guided",
        time_window_start=now,
        time_window_end=now + timedelta(hours=1),
        token_budget=1000,
        wallclock_hours=1,
        authorization_artifact_sha256="0" * 64,
        status="created",
        current_stage="recon",
        created_by=None,
        created_at=now,
        updated_at=now,
        regression_budget_max=0,
        llm_config_snapshot=snapshot,
    )


# --------------------------------------------------------------------------- #
# llm-05 / db-06 — snapshot round-trips through the SAME keys                   #
# --------------------------------------------------------------------------- #


def test_snapshot_write_read_round_trip_anthropic() -> None:
    """Write side and read side agree on key names: an engagement created
    on anthropic surfaces a non-null backend/model in the read model."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="anthropic",
        anthropic_api_key="sk-ant-present",
        anthropic_model_planner="claude-sonnet-4-6",
    )
    snap = _build_llm_config_snapshot(env)
    assert snap["llm_backend"] == "anthropic"
    assert snap["anthropic_model_planner"] == "claude-sonnet-4-6"

    read = _row_to_read(_fake_engagement_row(snap))
    assert read.llm_backend_used == "anthropic"
    assert read.llm_model_used == "claude-sonnet-4-6"


def test_snapshot_write_read_round_trip_ollama() -> None:
    """Same round-trip on the local (ollama) path — must not regress."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="ollama",
        local_model_planner="planner-model",
    )
    snap = _build_llm_config_snapshot(env)
    assert snap["llm_backend"] == "ollama"
    assert snap["local_model_planner"] == "planner-model"

    read = _row_to_read(
        _fake_engagement_row(snap, id="22222222-2222-2222-2222-222222222222")
    )
    assert read.llm_backend_used == "ollama"
    assert read.llm_model_used == "planner-model"


def test_snapshot_from_settings_matches_engagement_writer() -> None:
    """The router's ``snapshot_from_settings`` and the engagement service's
    ``_build_llm_config_snapshot`` must agree on key names so a snapshot
    written by one is readable by the other (single canonical contract)."""
    env = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="anthropic",
        anthropic_api_key="sk-ant-present",
        anthropic_model_planner="claude-sonnet-4-6",
    )
    router_snap = snapshot_from_settings(env)
    engagement_snap = _build_llm_config_snapshot(env)
    assert router_snap["llm_backend"] == engagement_snap["llm_backend"]
    # Both write the planner model under the same key.
    assert (
        router_snap["anthropic_model_planner"]
        == engagement_snap["anthropic_model_planner"]
    )
    # And the snapshot a router emits is re-applyable, pinning the same backend.
    pinned = apply_snapshot(
        Settings(_env_file=None, llm_backend="ollama", anthropic_api_key="k"),  # type: ignore[call-arg]
        router_snap,
    )
    assert pinned.llm_backend == "anthropic"


# --------------------------------------------------------------------------- #
# llm-07 — Anthropic cache-aware token/cost accounting                         #
# --------------------------------------------------------------------------- #


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens, cache_read, cache_creation):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


class _FakeMessage:
    def __init__(self, usage):
        self.usage = usage
        self.content = []


@pytest.mark.asyncio
async def test_anthropic_input_tokens_include_cache_read() -> None:
    """Budget enforcement reads ``input_tokens`` — it must include the
    cache-read prefix re-read every turn, otherwise well-cached runs
    under-count and budgets trip late (llm-07)."""
    from llm.anthropic_backend import AnthropicBackend
    from llm.router import CallSite, LLMRequest

    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_backend="anthropic",
        anthropic_api_key="sk-ant-present",
        anthropic_model_planner="claude-sonnet-4-6",
    )
    with patch("llm.anthropic_backend.AsyncAnthropic"):
        backend = AnthropicBackend(s)

    usage = _FakeUsage(
        input_tokens=100,  # uncached, non-cache-creation
        output_tokens=50,
        cache_read=900,  # cached prefix re-read this turn
        cache_creation=200,  # written to cache this turn
    )

    async def _fake_create(**kwargs):
        return _FakeMessage(usage)

    backend._client.messages.create = _fake_create  # type: ignore[attr-defined]

    resp = await backend.complete(
        LLMRequest(system="s", user="u", site=CallSite.PLANNER)
    )

    # All three input buckets counted for budget: 100 + 200 + 900 = 1200.
    assert resp.input_tokens == 1200
    assert resp.cached_input_tokens == 900
    assert resp.output_tokens == 50


def test_anthropic_cost_no_double_subtraction() -> None:
    """``input_tok`` already excludes cache tokens, so it must be priced
    directly — the previous formula double-subtracted ``cache_creation``
    from a number that never included it (llm-07)."""
    from llm.anthropic_backend import _PRICES, _cost

    model = "claude-sonnet-4-6"
    price_in, price_out, price_cache = _PRICES[model]
    in_tok, out_tok, cached, cache_creation = 100, 50, 900, 200

    got = _cost(model, in_tok, out_tok, cached, cache_creation)
    expected = (
        in_tok * price_in / 1_000_000  # priced directly, no subtraction
        + cache_creation * price_in * 1.25 / 1_000_000
        + cached * price_cache / 1_000_000
        + out_tok * price_out / 1_000_000
    )
    assert got == pytest.approx(expected)

    # And it must be strictly greater than a (buggy) formula that subtracted
    # cache_creation from the uncached-input line, confirming no under-charge.
    undercharged = (
        max(0, in_tok - cache_creation) * price_in / 1_000_000
        + cache_creation * price_in * 1.25 / 1_000_000
        + cached * price_cache / 1_000_000
        + out_tok * price_out / 1_000_000
    )
    assert got > undercharged


def test_anthropic_cost_zero_when_no_cache() -> None:
    """With no cache activity the cost equals the plain in/out pricing."""
    from llm.anthropic_backend import _PRICES, _cost

    model = "claude-sonnet-4-6"
    price_in, price_out, _ = _PRICES[model]
    got = _cost(model, 100, 50, 0, 0)
    expected = 100 * price_in / 1_000_000 + 50 * price_out / 1_000_000
    assert got == pytest.approx(expected)
