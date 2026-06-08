"""Local backend -- talks to Ollama (or any OpenAI-compatible endpoint).

Gemma 4 does not have a native tool-use API, so we use prompt-encoded tool
descriptions and JSON-schema-validated structured outputs. The same Pydantic
schemas as the Anthropic backend are used, so agents are backend-agnostic.

Per-CallSite model routing: the backend holds a ``_model_map`` that maps
each :class:`CallSite` value to a model name. Planner and parser default
to the larger model (e.g. ``gemma4:26b``); critic, synthesizer, and triage
default to the smaller model (e.g. ``gemma4`` / E4B).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import structlog
from openai import AsyncOpenAI

from llm.router import LLMRequest, LLMResponse, ToolCall
from orchestrator.settings import Settings

log = structlog.get_logger(__name__)


_TOOL_INSTRUCTION_TEMPLATE = """
You may call exactly one of the following tools by emitting a JSON object on a
single line wrapped in <tool_call></tool_call> tags. Do not call a tool that is
not listed here.

Available tools:
{tool_specs}

Output format when calling a tool:
<tool_call>{{"name": "<tool_name>", "input": {{...}}}}</tool_call>

If no tool is appropriate, respond with a JSON object wrapped in
<final></final> tags describing why and what should happen next.
""".strip()


class LocalBackend:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.local_base_url,
            api_key="ollama-local",
        )
        self._model_map: dict[str, str] = {
            "planner": settings.local_model_planner,
            "parser": settings.local_model_parser,
            "critic": settings.local_model_critic,
            "synthesizer": settings.local_model_synthesizer,
            "triage": settings.local_model_triage,
        }
        self._default_model = settings.local_model_planner

    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Send a chat completion to the Ollama OpenAI-compat endpoint.

        ``keep_alive`` is forwarded via ``extra_body`` so Ollama keeps the
        model loaded in VRAM for the configured duration after the last
        request.  The agent loop naturally batches by call site: N
        consecutive PLANNER calls keep the 26B model warm, then the critic
        gate loads E4B.  The ``keep_alive`` window (default 5m) covers the
        gap between back-to-back agent steps (~30s) so the model swap only
        happens at PLANNER-to-CRITIC transitions, not between steps.
        """
        model = self._model_map.get(req.site.value, self._default_model)

        system = req.system
        if req.tools:
            specs = "\n".join(
                f"- {t.name}: {t.description}\n  input_schema: {json.dumps(t.input_schema())}"
                for t in req.tools
            )
            system = system + "\n\n" + _TOOL_INSTRUCTION_TEMPLATE.format(tool_specs=specs)

        messages: list[dict[str, Any]]
        if req.messages is not None:
            messages = [
                {"role": "system", "content": system},
                *_normalise_messages_for_text_backend(req.messages),
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": req.user},
            ]

        t0 = time.monotonic()
        completion = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            extra_body={"keep_alive": self.settings.local_keep_alive},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        text = completion.choices[0].message.content or ""
        tool_calls = _parse_tool_calls(text)
        text_clean = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

        usage = completion.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=text_clean,
            tool_calls=tool_calls,
            structured=None,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            raw=completion,
        )


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _normalise_messages_for_text_backend(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse Anthropic-shaped tool_use/tool_result blocks into plain
    text turns the OpenAI-compatible chat endpoint can consume.

    User turns with a ``_text_fallback`` key use that value directly;
    assistant turns with list-of-blocks content are serialised to text.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        text_fallback = msg.get("_text_fallback")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if text_fallback is not None and isinstance(text_fallback, str):
            out.append({"role": role, "content": text_fallback})
            continue
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", "") or "")
                elif btype == "tool_use":
                    payload = json.dumps(
                        {"name": block.get("name"), "input": block.get("input") or {}}
                    )
                    parts.append(f"<tool_call>{payload}</tool_call>")
                elif btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                    else:
                        parts.append(json.dumps(inner, default=str))
                else:
                    parts.append(json.dumps(block, default=str))
            out.append({"role": role, "content": "\n".join(parts)})
            continue
        # Fallback: stringify whatever we got.
        out.append({"role": role, "content": str(content)})
    return out


def _parse_tool_calls(text: str) -> list[ToolCall]:
    out: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("local.tool_call.bad_json", raw=raw[:200])
            continue
        name = data.get("name")
        if not isinstance(name, str):
            continue
        out.append(
            ToolCall(
                name=name,
                input=dict(data.get("input") or {}),
                call_id=str(uuid.uuid4()),
            )
        )
    return out
