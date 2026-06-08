"""Per-step scoring (AgentPRM-style).

Evaluates each agent step on two axes (promise and progress, both
0.0-1.0). When both fall below ``BACKTRACK_THRESHOLD`` the scorer flags
``should_backtrack=True`` so the stage controller can override CONTINUE
with ESCALATE_HUMAN. Routes via ``CallSite.CRITIC`` (cheap E4B model).
Opt-in via ``enable_step_scoring`` on Settings.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from llm.router import CallSite, LLMRequest, LLMRouter

log = structlog.get_logger(__name__)


class StepScore(BaseModel):
    """Result of a per-step evaluation."""

    promise: float = Field(ge=0.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    should_backtrack: bool = False


_SCORER_SYSTEM = (
    "You are a step evaluator for an autonomous penetration testing agent. "
    "Given the engagement goal, the current world-state slice, and the "
    "tool results from the latest step, evaluate two axes:\n\n"
    "1. **promise** (0.0-1.0): how likely is this step moving toward the "
    "engagement's overall goal? 1.0 = directly advancing, 0.0 = completely "
    "irrelevant or counterproductive.\n"
    "2. **progress** (0.0-1.0): did this step make tangible forward progress "
    "compared to the prior state? 1.0 = significant new information or "
    "capability gained, 0.0 = no change or regression.\n\n"
    "Respond with a single JSON object:\n"
    '{"promise": <float>, "progress": <float>, "reasoning": "<one sentence>"}\n\n'
    "Return only the JSON object, no prose."
)

_STEP_SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["promise", "progress", "reasoning"],
    "properties": {
        "promise": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "progress": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": 300},
    },
}


def _format_scorer_input(
    *,
    engagement_goal: str,
    stage: str,
    step: int,
    world_state_slice: dict[str, Any],
    last_tool_results: str,
) -> str:
    """Build the user-turn content for the scoring prompt."""
    ws_str = json.dumps(world_state_slice, indent=2, default=str)[:2000]
    tool_str = last_tool_results[:1000] if last_tool_results else "(no tool results)"
    return (
        f"## Engagement goal\n{engagement_goal or 'General penetration test'}\n\n"
        f"## Current stage\n{stage} (step {step})\n\n"
        f"## World-state slice (summary)\n```json\n{ws_str}\n```\n\n"
        f"## Latest tool results\n{tool_str}\n\n"
        "Evaluate and respond with JSON."
    )


_NEUTRAL_SCORE = StepScore(
    promise=0.5,
    progress=0.5,
    reasoning="scorer produced unparseable output — defaulting to neutral",
)


def _parse_score(text: str) -> StepScore:
    """Tolerant parser for the scorer's JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("step_scorer.parse_fail", text_excerpt=text[:300])
        return _NEUTRAL_SCORE

    try:
        promise = max(0.0, min(1.0, float(data.get("promise", 0.5))))
    except (ValueError, TypeError):
        log.warning("step_scorer.non_numeric_promise", value=data.get("promise"))
        promise = 0.5
    try:
        progress = max(0.0, min(1.0, float(data.get("progress", 0.5))))
    except (ValueError, TypeError):
        log.warning("step_scorer.non_numeric_progress", value=data.get("progress"))
        progress = 0.5
    return StepScore(
        promise=promise,
        progress=progress,
        reasoning=str(data.get("reasoning", ""))[:300],
        should_backtrack=(
            promise < StepScorer.BACKTRACK_THRESHOLD
            and progress < StepScorer.BACKTRACK_THRESHOLD
        ),
    )


class StepScorer:
    """LLM-based step scorer. Injected into ``BaseAgent`` via an optional
    kwarg; ``None`` (the default) skips scoring entirely."""

    BACKTRACK_THRESHOLD: float = 0.3

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    async def score(
        self,
        *,
        engagement_id: UUID,
        stage: str,
        step: int,
        world_state_slice: dict[str, Any],
        last_tool_results: str,
        engagement_goal: str,
    ) -> StepScore:
        """Score the latest agent step for promise and progress."""
        user_content = _format_scorer_input(
            engagement_goal=engagement_goal,
            stage=stage,
            step=step,
            world_state_slice=world_state_slice,
            last_tool_results=last_tool_results,
        )
        try:
            resp = await self.router.complete(LLMRequest(
                system=_SCORER_SYSTEM,
                user=user_content,
                site=CallSite.CRITIC,
                max_tokens=200,
                temperature=0.0,
                engagement_id=str(engagement_id),
            ))
            return _parse_score(resp.text)
        except Exception as e:
            log.warning("step_scorer.score_failed", err=str(e))
            return StepScore(
                promise=0.5,
                progress=0.5,
                reasoning=f"scoring failed: {type(e).__name__}",
            )

    def should_backtrack(self, score: StepScore) -> bool:
        """Return True when both scores are below the backtrack threshold."""
        return score.promise < self.BACKTRACK_THRESHOLD and score.progress < self.BACKTRACK_THRESHOLD
