"""Engagement cost tracking — plain async functions.

Extracted from ``orchestrator.api.cost``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from orchestrator.cost import estimate_engagement_cost
from orchestrator.db import session_scope
from orchestrator.models import EngagementRow, LLMCallRow
from orchestrator.services.errors import NotFoundError
from orchestrator.settings import get_settings


async def get_cost_estimate(engagement_id: UUID) -> dict[str, Any]:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if row is None:
            raise NotFoundError("engagement not found")
        settings = get_settings()
        est = estimate_engagement_cost(row.scope or {}, row.mode, backend=settings.llm_backend)
        d = est.to_dict()
        return {
            "token_low": d["token_low"],
            "token_high": d["token_high"],
            "usd_low": d["usd_low"],
            "usd_high": d["usd_high"],
            "notes": d["notes"],
        }


async def get_cost_actual(engagement_id: UUID) -> dict[str, Any]:
    async with session_scope() as session:
        stmt = select(
            func.coalesce(func.sum(LLMCallRow.input_tokens), 0).label("in_tok"),
            func.coalesce(func.sum(LLMCallRow.cached_input_tokens), 0).label("cached"),
            func.coalesce(func.sum(LLMCallRow.output_tokens), 0).label("out_tok"),
            func.coalesce(func.sum(LLMCallRow.cost_usd), 0.0).label("cost"),
            func.count(LLMCallRow.id).label("n"),
        ).where(LLMCallRow.engagement_id == engagement_id)
        res = (await session.execute(stmt)).one()
        n = int(res.n or 0)
        cost = float(res.cost or 0.0)
        return {
            "tokens_used": int((res.in_tok or 0) + (res.out_tok or 0)),
            "cached_tokens": int(res.cached or 0),
            "output_tokens": int(res.out_tok or 0),
            "cost_usd": round(cost, 4),
            "calls": n,
            "per_call_avg_usd": round(cost / n, 6) if n else 0.0,
        }
