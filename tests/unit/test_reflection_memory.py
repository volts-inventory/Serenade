"""Tests for reflexion verbal memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agents.reflection_memory import (
    ReflectionMemory,
    _determine_outcome,
    list_reflections,
    record_reflection,
)
from llm.prompts.base import render_reflections, render_user_prompt
from worldstate.schema import Stage


# ---- _determine_outcome ----


class TestDetermineOutcome:
    def test_success(self):
        assert _determine_outcome(promise=0.8, progress=0.7) == "success"

    def test_failure(self):
        assert _determine_outcome(promise=0.1, progress=0.2) == "failure"

    def test_partial(self):
        assert _determine_outcome(promise=0.6, progress=0.2) == "partial"

    def test_none_scores(self):
        assert _determine_outcome(promise=None, progress=None) == "partial"

    def test_one_none(self):
        assert _determine_outcome(promise=0.8, progress=None) == "partial"


# ---- ReflectionMemory.write() ----


class TestReflectionMemoryWrite:
    @pytest.mark.asyncio
    async def test_low_scores_produce_summary(self):
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = "nmap SYN scan failed; try ACK scan or pivot."
        mock_router.complete = AsyncMock(return_value=mock_resp)

        mem = ReflectionMemory(router=mock_router)
        with patch(
            "agents.reflection_memory.record_reflection",
            new_callable=AsyncMock,
        ) as mock_persist:
            summary = await mem.write(
                engagement_id=uuid4(),
                stage="recon",
                step=1,
                world_state_context="hosts: []",
                tool_results="nmap returned 0 hosts",
                promise_score=0.2,
                progress_score=0.1,
            )
        assert summary is not None
        assert "nmap" in summary.lower() or "scan" in summary.lower() or len(summary) > 0
        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        assert call_kwargs[1]["outcome"] == "failure"

    @pytest.mark.asyncio
    async def test_high_scores_return_none(self):
        """No reflection needed when the step went well."""
        mock_router = AsyncMock()
        mem = ReflectionMemory(router=mock_router)
        summary = await mem.write(
            engagement_id=uuid4(),
            stage="recon",
            step=1,
            world_state_context="hosts: [h1, h2]",
            tool_results="found 2 hosts",
            promise_score=0.8,
            progress_score=0.7,
        )
        assert summary is None
        # Router should NOT have been called
        mock_router.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_scores_skip_reflection(self):
        """When both scores are None (scorer not active), skip reflection.

        Reflection is coupled to scoring -- without scores there is no
        signal to reflect on.
        """
        mock_router = AsyncMock()
        mem = ReflectionMemory(router=mock_router)
        summary = await mem.write(
            engagement_id=uuid4(),
            stage="recon",
            step=1,
            world_state_context="",
            tool_results="",
            promise_score=None,
            progress_score=None,
        )
        assert summary is None
        # Router should NOT have been called -- early return
        mock_router.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_none_score_reflects(self):
        """When only one score is None (partial scoring), still reflect."""
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = "step completed with partial scoring"
        mock_router.complete = AsyncMock(return_value=mock_resp)

        mem = ReflectionMemory(router=mock_router)
        with patch(
            "agents.reflection_memory.record_reflection",
            new_callable=AsyncMock,
        ):
            summary = await mem.write(
                engagement_id=uuid4(),
                stage="recon",
                step=1,
                world_state_context="",
                tool_results="",
                promise_score=0.2,
                progress_score=None,
            )
        assert summary is not None

    @pytest.mark.asyncio
    async def test_router_failure_returns_none(self):
        mock_router = AsyncMock()
        mock_router.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        mem = ReflectionMemory(router=mock_router)
        summary = await mem.write(
            engagement_id=uuid4(),
            stage="recon",
            step=1,
            world_state_context="",
            tool_results="",
            promise_score=0.1,
            progress_score=0.1,
        )
        assert summary is None


# ---- record_reflection / list_reflections (mock DB) ----


class TestReflectionPersistence:
    @pytest.mark.asyncio
    async def test_record_reflection_best_effort(self):
        """record_reflection should not raise even on DB failure."""
        with patch(
            "orchestrator.db.session_scope",
            side_effect=RuntimeError("DB down"),
        ):
            # Should not raise
            await record_reflection(
                engagement_id=uuid4(),
                stage="recon",
                step=1,
                summary="test lesson",
                outcome="failure",
            )

    @pytest.mark.asyncio
    async def test_list_reflections_empty_on_failure(self):
        """list_reflections should return [] on DB failure."""
        with patch(
            "orchestrator.db.session_scope",
            side_effect=RuntimeError("DB down"),
        ):
            result = await list_reflections(engagement_id=uuid4())
            assert result == []


# ---- Prompt rendering with reflections ----


class TestRenderReflections:
    def test_empty_reflections(self):
        assert render_reflections([]) == ""

    def test_single_reflection(self):
        reflections = [
            {
                "stage": "recon",
                "step": 1,
                "summary": "SYN scan blocked by firewall",
                "outcome": "failure",
            }
        ]
        result = render_reflections(reflections)
        assert "recon" in result
        assert "SYN scan blocked" in result
        assert "failure" in result

    def test_multiple_reflections(self):
        reflections = [
            {"stage": "recon", "step": 1, "summary": "lesson 1", "outcome": "failure"},
            {"stage": "recon", "step": 2, "summary": "lesson 2", "outcome": "partial"},
        ]
        result = render_reflections(reflections)
        assert "lesson 1" in result
        assert "lesson 2" in result

    def test_truncation_at_600_chars(self):
        reflections = [
            {"stage": "recon", "step": i, "summary": f"lesson {'x' * 100}", "outcome": "failure"}
            for i in range(20)
        ]
        result = render_reflections(reflections)
        assert len(result) <= 600


class TestRenderUserPromptWithReflections:
    def test_no_reflections(self):
        """Existing render_user_prompt call without reflections works."""
        prompt = render_user_prompt(
            stage=Stage.RECON,
            objective="find hosts",
            world_state_slice={"hosts": []},
        )
        assert "Lessons learned" not in prompt

    def test_with_reflections(self):
        prompt = render_user_prompt(
            stage=Stage.RECON,
            objective="find hosts",
            world_state_slice={"hosts": []},
            reflections=[
                {"stage": "recon", "step": 1, "summary": "SYN blocked", "outcome": "failure"},
            ],
        )
        assert "Lessons learned from prior steps" in prompt
        assert "SYN blocked" in prompt

    def test_empty_reflections_list(self):
        prompt = render_user_prompt(
            stage=Stage.RECON,
            objective="find hosts",
            world_state_slice={"hosts": []},
            reflections=[],
        )
        assert "Lessons learned" not in prompt
