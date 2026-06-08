"""LLM-driven regression reflection at stage ADVANCE boundaries.

At each ADVANCE boundary the ``RegressionReflector`` asks the LLM whether
there is uncovered in-scope surface that the deterministic triggers missed.
If so, it returns a ``RegressionRequest`` with
``trigger_kind=LLM_SUGGESTED_REGRESSION`` which flows through the full
``StageRegressionCouncil`` for verification.

Opt-in via ``enable_regression_reflection: bool = False`` on Settings.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from agents.regression import RegressionRequest, RegressionTriggerKind
from llm.router import CallSite, LLMRequest, LLMRouter
from worldstate.schema import Stage

log = structlog.get_logger(__name__)


class RegressionSuggestion(BaseModel):
    """Parsed output from the reflector's LLM call."""

    should_regress: bool = False
    target_stage: str | None = None
    delta: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


_REFLECTOR_SYSTEM = (
    "You are a stage-boundary reflection agent for an autonomous "
    "penetration testing system. The agent has just completed a stage "
    "and is about to advance to the next. Your job is to determine "
    "whether there is uncovered, in-scope surface that the deterministic "
    "regression triggers may have missed.\n\n"
    "Consider:\n"
    "- Are there CIDRs or domains in scope that have not been enumerated?\n"
    "- Are there services discovered but not vulnerability-scanned?\n"
    "- Are there credential-implied pivot targets that have not been explored?\n"
    "- Has the engagement scope changed in a way that implies new work?\n"
    "- Are there lessons from the verbal memory that suggest a retry "
    "with a different strategy would succeed?\n\n"
    "Respond with a single JSON object:\n"
    '{"should_regress": true, "target_stage": "<stage_name>", '
    '"delta": {<scoped_contract>}, "rationale": "<one sentence>"}\n'
    "or\n"
    '{"should_regress": false}\n\n'
    "Valid target_stage values: recon, enumeration, vuln_triage, exploit, post_exploit, privesc, lateral.\n"
    "Return only the JSON object, no prose."
)

def _format_reflector_input(
    *,
    current_stage: str,
    world_state_slice: dict[str, Any],
    engagement_scope: dict[str, Any],
    verbal_memory: list[dict[str, Any]],
) -> str:
    """Build the user-turn content for the reflector prompt."""
    ws_str = json.dumps(world_state_slice, indent=2, default=str)[:3000]
    scope_str = json.dumps(engagement_scope, indent=2, default=str)[:1000]
    memory_lines: list[str] = []
    for m in (verbal_memory or [])[:10]:
        summary = m.get("summary", "")
        outcome = m.get("outcome", "")
        stage = m.get("stage", "")
        memory_lines.append(f"- [{stage}/{outcome}] {summary}")
    memory_str = "\n".join(memory_lines) if memory_lines else "(no prior reflections)"

    return (
        f"## Current stage (just completed)\n{current_stage}\n\n"
        f"## Engagement scope\n```json\n{scope_str}\n```\n\n"
        f"## World-state delta (current state summary)\n```json\n{ws_str}\n```\n\n"
        f"## Verbal memory (prior reflections)\n{memory_str}\n\n"
        "Given the scope and the current world-state, is there uncovered "
        "in-scope surface that the deterministic triggers missed? "
        "Respond with JSON."
    )


def _parse_suggestion(text: str) -> RegressionSuggestion:
    """Tolerant parser for the reflector's JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("regression_reflector.parse_fail", text_excerpt=text[:300])
        return RegressionSuggestion(should_regress=False)

    should_regress = bool(data.get("should_regress", False))
    if not should_regress:
        return RegressionSuggestion(should_regress=False)

    target_stage = data.get("target_stage")
    delta = data.get("delta", {})
    rationale = str(data.get("rationale", ""))[:500]

    valid_stages = {"recon", "enumeration", "vuln_triage", "exploit", "post_exploit", "privesc", "lateral"}
    if target_stage not in valid_stages:
        log.warning(
            "regression_reflector.invalid_target_stage",
            target_stage=target_stage,
        )
        return RegressionSuggestion(should_regress=False)

    if not isinstance(delta, dict):
        delta = {}

    return RegressionSuggestion(
        should_regress=True,
        target_stage=target_stage,
        delta=delta,
        rationale=rationale,
    )


class RegressionReflector:
    """LLM reflection at stage ADVANCE boundaries.

    Constructs a ``RegressionRequest`` when the LLM identifies uncovered
    surface; the request still goes through the full council.
    """

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    async def reflect(
        self,
        *,
        engagement_id: UUID,
        current_stage: str,
        world_state_slice: dict[str, Any],
        engagement_scope: dict[str, Any],
        verbal_memory: list[dict[str, Any]] | None = None,
    ) -> RegressionRequest | None:
        """Reflect on the stage boundary and optionally suggest a regression.

        Returns a :class:`RegressionRequest` if the LLM identifies
        uncovered surface, or ``None`` if it does not.
        """
        user_content = _format_reflector_input(
            current_stage=current_stage,
            world_state_slice=world_state_slice,
            engagement_scope=engagement_scope,
            verbal_memory=verbal_memory or [],
        )

        try:
            resp = await self.router.complete(
                LLMRequest(
                    system=_REFLECTOR_SYSTEM,
                    user=user_content,
                    site=CallSite.CRITIC,
                    max_tokens=500,
                    temperature=0.0,
                    engagement_id=str(engagement_id),
                )
            )
            suggestion = _parse_suggestion(resp.text)
        except Exception as e:
            log.warning("regression_reflector.reflect_failed", err=str(e))
            return None

        if not suggestion.should_regress or suggestion.target_stage is None:
            return None

        try:
            target = Stage(suggestion.target_stage)
        except ValueError:
            log.warning(
                "regression_reflector.invalid_stage_enum",
                target_stage=suggestion.target_stage,
            )
            return None

        return RegressionRequest(
            target_stage=target,
            trigger_kind=RegressionTriggerKind.LLM_SUGGESTED_REGRESSION,
            delta=suggestion.delta,
            rationale=f"llm_reflection: {suggestion.rationale}"[:1000],
        )
