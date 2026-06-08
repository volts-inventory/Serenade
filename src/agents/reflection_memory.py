"""Reflexion verbal memory (episodic strategy-level lessons).

After each agent step, if step scoring is enabled and the step scored
low or failed, the ``ReflectionMemory`` generates a one-sentence
strategy-level summary of what was tried and what happened. These
summaries are persisted in Postgres and injected into subsequent prompts
as a "Lessons learned" section so the LLM avoids repeating failed
strategies.

Stores *strategy-level* lessons, not target-level facts. Target-level
dedup is handled by the fingerprint replay guard.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog

from llm.router import CallSite, LLMRequest, LLMRouter

log = structlog.get_logger(__name__)


MAX_ENTRIES_PER_STAGE = 10
MAX_SUMMARY_CHARS = 500

_REFLECTION_SYSTEM = (
    "You are a strategy-level reflection agent for an autonomous "
    "penetration testing system. Given the tool results and world-state "
    "context from the latest step, generate a ONE-SENTENCE summary of:\n"
    "1. What strategy was attempted\n"
    "2. What happened (success / failure / partial)\n"
    "3. What lesson should inform the next attempt\n\n"
    "Be specific and actionable. Example: "
    "'nmap SYN scan of 10.0.5.0/24 returned 0 hosts — likely firewall; "
    "try ACK scan or pivot through the jump host at 10.0.1.5.'\n\n"
    "Respond with only the summary sentence, no JSON, no formatting."
)


def _determine_outcome(
    *,
    promise: float | None,
    progress: float | None,
) -> str:
    """Classify the step outcome for the reflection row."""
    if promise is None or progress is None:
        return "partial"
    if promise >= 0.5 and progress >= 0.5:
        return "success"
    if promise < 0.3 and progress < 0.3:
        return "failure"
    return "partial"


class ReflectionMemory:
    """Episodic verbal memory for strategy-level lessons.

    Injected into ``BaseAgent`` via an optional kwarg. When ``None``
    (the default), reflection is skipped.
    """

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    async def write(
        self,
        *,
        engagement_id: UUID,
        stage: str,
        step: int,
        world_state_context: str,
        tool_results: str,
        promise_score: float | None = None,
        progress_score: float | None = None,
    ) -> str | None:
        """Generate and persist a strategy-level reflection summary.

        Returns the summary text, or ``None`` if the step doesn't
        warrant reflection (both scores >= 0.5).
        """
        # When step scoring is not active (both scores are None), skip
        # reflection entirely. Reflection is coupled to scoring -- without
        # scores there is no signal to reflect on, and reflecting on every
        # step would flood the memory with noise.
        if promise_score is None and progress_score is None:
            return None

        if (
            promise_score is not None
            and progress_score is not None
            and promise_score >= 0.5
            and progress_score >= 0.5
        ):
            return None

        user_content = (
            f"## World-state context\n{world_state_context[:1000]}\n\n"
            f"## Latest tool results\n{tool_results[:1000] if tool_results else '(no tool results)'}\n\n"
            "Write a one-sentence strategy-level lesson."
        )

        try:
            resp = await self.router.complete(
                LLMRequest(
                    system=_REFLECTION_SYSTEM,
                    user=user_content,
                    site=CallSite.CRITIC,
                    max_tokens=300,
                    temperature=0.0,
                    engagement_id=str(engagement_id),
                )
            )
            summary = (resp.text or "").strip()[:MAX_SUMMARY_CHARS]
            if not summary:
                return None
        except Exception as e:
            log.warning("reflection_memory.write_failed", err=str(e))
            return None

        outcome = _determine_outcome(promise=promise_score, progress=progress_score)
        await record_reflection(
            engagement_id=engagement_id,
            stage=stage,
            step=step,
            summary=summary,
            outcome=outcome,
        )
        return summary


async def record_reflection(
    engagement_id: UUID,
    stage: str,
    step: int,
    summary: str,
    outcome: str,
) -> None:
    """Persist a reflection row (best-effort)."""
    try:
        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import ReflectionMemoryRow  # noqa: PLC0415

        async with session_scope() as s:
            s.add(ReflectionMemoryRow(
                engagement_id=engagement_id,
                stage=stage,
                step=step,
                summary=summary,
                outcome=outcome,
            ))
    except Exception as e:
        log.warning("reflection_memory.persist_failed", err=str(e))


async def list_reflections(
    engagement_id: UUID,
    stage: str | None = None,
    limit: int = MAX_ENTRIES_PER_STAGE,
) -> list[dict[str, Any]]:
    """Fetch recent reflections for prompt injection.

    Returns plain dicts. Empty list on DB failure so the prompt renders
    without a lessons-learned section.
    """
    try:
        from sqlalchemy import select  # noqa: PLC0415

        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import ReflectionMemoryRow  # noqa: PLC0415

        async with session_scope() as s:
            stmt = select(ReflectionMemoryRow).where(
                ReflectionMemoryRow.engagement_id == engagement_id
            )
            if stage is not None:
                stmt = stmt.where(ReflectionMemoryRow.stage == stage)
            stmt = stmt.order_by(ReflectionMemoryRow.created_at.desc()).limit(limit)
            rows = (await s.execute(stmt)).scalars().all()
    except Exception as e:
        log.warning("reflection_memory.list_failed", err=str(e))
        return []

    return [
        {
            "stage": r.stage,
            "step": r.step,
            "summary": r.summary,
            "outcome": r.outcome,
            "created_at": r.created_at,
        }
        for r in rows
    ]
