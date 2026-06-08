"""Anthropic SDK backend with model routing per call site + prompt caching."""

from __future__ import annotations

import time
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock

from llm.router import CallSite, LLMRequest, LLMResponse, ToolCall
from orchestrator.settings import Settings

# Published per-million pricing as of plan date. Update via env when prices shift.
_PRICES: dict[str, tuple[float, float, float]] = {
    # model -> (input/1M, output/1M, cached_input/1M)
    "claude-opus-4-7": (15.0, 75.0, 1.5),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5-20251001": (0.80, 4.0, 0.08),
}


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model_for: dict[CallSite, str] = {
            CallSite.PLANNER: settings.anthropic_model_planner,
            CallSite.SYNTHESIZER: settings.anthropic_model_synthesizer,
            CallSite.PARSER: settings.anthropic_model_parser,
            CallSite.TRIAGE: settings.anthropic_model_triage,
            CallSite.CRITIC: settings.anthropic_model_critic,
        }

    async def complete(self, req: LLMRequest) -> LLMResponse:
        model = self._model_for[req.site]

        # System as a list of blocks so we can attach cache_control to the prefix
        # for engagement-stable context.
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": req.system}]
        if req.cache_breakpoint_system:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        # Interactive loop: when the caller provides ``messages``, pass
        # it through so the model sees the full conversation (assistant
        # tool_use turns + user tool_result blocks). Otherwise wrap the
        # single ``user`` string. Cache breakpoints stay on the system
        # block; the growing conversation suffix is uncached. We strip
        # private ``_text_fallback`` hints the BaseAgent attaches for
        # local-LLM backends; the Anthropic API rejects unknown keys.
        messages: list[dict[str, Any]]
        if req.messages is not None:
            messages = [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in req.messages
            ]
        else:
            messages = [{"role": "user", "content": req.user}]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": system_blocks,
            "messages": messages,
        }

        if req.tools:
            anthropic_tools: list[dict[str, Any]] = []
            for t in req.tools:
                anthropic_tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema(),
                    }
                )
            # Cache the tool registry — same for an engagement's lifetime.
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = anthropic_tools
            kwargs["tool_choice"] = {"type": "auto"}

        t0 = time.monotonic()
        msg = await self._client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCall(name=block.name, input=dict(block.input or {}), call_id=block.id)
                )
            elif getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")

        usage = msg.usage
        in_tok = usage.input_tokens or 0
        out_tok = usage.output_tokens or 0
        cached_in = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        cost = _cost(model, in_tok, out_tok, cached_in, cache_creation)

        # Anthropic reports input usage as THREE disjoint counters:
        #   - usage.input_tokens          -> uncached, non-cache-creation input
        #   - cache_creation_input_tokens -> tokens written to the cache
        #   - cache_read_input_tokens     -> tokens served from the cache
        # For budget enforcement (BaseAgent reads ``input_tokens``) we must
        # count ALL input the engagement actually consumed, including the
        # cache-read prefix re-read every turn — otherwise well-cached runs
        # systematically under-count and budgets trip late (finding llm-07).
        total_input = in_tok + cache_creation + cached_in

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            structured=None,
            model=model,
            input_tokens=total_input,
            output_tokens=out_tok,
            cached_input_tokens=cached_in,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw=msg,
        )


def _cost(model: str, input_tok: int, output_tok: int, cached: int, cache_creation: int) -> float:
    """Cost for a turn given Anthropic's three disjoint input counters.

    ``input_tok`` is already the uncached, non-cache-creation input count, so it
    is priced directly (no subtraction — the previous formula double-subtracted
    ``cache_creation`` from a number that never included it; finding llm-07).
    Cache creation is billed at ~1.25x the base input rate; cache reads at the
    discounted cached rate.
    """
    price_in, price_out, price_cache = _PRICES.get(model, (0.0, 0.0, 0.0))
    return (
        (input_tok * price_in / 1_000_000)
        + (cache_creation * price_in * 1.25 / 1_000_000)
        + (cached * price_cache / 1_000_000)
        + (output_tok * price_out / 1_000_000)
    )
