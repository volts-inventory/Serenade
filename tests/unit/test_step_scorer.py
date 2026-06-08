"""Tests for per-step scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.step_scorer import StepScore, StepScorer, _parse_score


# ---- StepScore model validation ----


class TestStepScoreModel:
    def test_valid_scores(self):
        s = StepScore(promise=0.8, progress=0.6, reasoning="good step")
        assert s.promise == 0.8
        assert s.progress == 0.6
        assert s.reasoning == "good step"
        assert s.should_backtrack is False

    def test_boundary_values(self):
        s = StepScore(promise=0.0, progress=0.0, reasoning="nothing")
        assert s.promise == 0.0
        assert s.progress == 0.0

        s2 = StepScore(promise=1.0, progress=1.0, reasoning="perfect")
        assert s2.promise == 1.0
        assert s2.progress == 1.0

    def test_invalid_scores_rejected(self):
        with pytest.raises(Exception):
            StepScore(promise=1.5, progress=0.5, reasoning="too high")
        with pytest.raises(Exception):
            StepScore(promise=0.5, progress=-0.1, reasoning="negative")

    def test_should_backtrack_flag(self):
        s = StepScore(
            promise=0.2, progress=0.1, reasoning="bad", should_backtrack=True
        )
        assert s.should_backtrack is True

    def test_defaults(self):
        s = StepScore(promise=0.5, progress=0.5)
        assert s.reasoning == ""
        assert s.should_backtrack is False


# ---- should_backtrack logic ----


class TestShouldBacktrack:
    def test_both_below_threshold(self):
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.2, progress=0.1, reasoning="stuck")
        assert scorer.should_backtrack(score) is True

    def test_promise_above_threshold(self):
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.5, progress=0.1, reasoning="some promise")
        assert scorer.should_backtrack(score) is False

    def test_progress_above_threshold(self):
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.1, progress=0.5, reasoning="some progress")
        assert scorer.should_backtrack(score) is False

    def test_both_above_threshold(self):
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.8, progress=0.7, reasoning="good")
        assert scorer.should_backtrack(score) is False

    def test_exactly_at_threshold(self):
        """Scores exactly at 0.3 should NOT trigger backtrack (< not <=)."""
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.3, progress=0.3, reasoning="border")
        assert scorer.should_backtrack(score) is False

    def test_just_below_threshold(self):
        scorer = StepScorer.__new__(StepScorer)
        scorer.router = MagicMock()
        score = StepScore(promise=0.29, progress=0.29, reasoning="just under")
        assert scorer.should_backtrack(score) is True


# ---- _parse_score ----


class TestParseScore:
    def test_valid_json(self):
        text = '{"promise": 0.7, "progress": 0.4, "reasoning": "ok"}'
        s = _parse_score(text)
        assert s.promise == 0.7
        assert s.progress == 0.4
        assert s.reasoning == "ok"

    def test_code_fenced_json(self):
        text = '```json\n{"promise": 0.6, "progress": 0.3, "reasoning": "fenced"}\n```'
        s = _parse_score(text)
        assert s.promise == 0.6
        assert s.progress == 0.3

    def test_unparseable_returns_neutral(self):
        s = _parse_score("this is not json at all")
        assert s.promise == 0.5
        assert s.progress == 0.5
        assert "unparseable" in s.reasoning

    def test_out_of_range_clamped(self):
        text = '{"promise": 1.5, "progress": -0.3, "reasoning": "oob"}'
        s = _parse_score(text)
        assert s.promise == 1.0
        assert s.progress == 0.0

    def test_missing_fields_default(self):
        text = '{"promise": 0.8}'
        s = _parse_score(text)
        assert s.promise == 0.8
        assert s.progress == 0.5  # default

    def test_backtrack_flag_set_when_low(self):
        text = '{"promise": 0.1, "progress": 0.1, "reasoning": "bad"}'
        s = _parse_score(text)
        assert s.should_backtrack is True

    def test_backtrack_flag_not_set_when_high(self):
        text = '{"promise": 0.8, "progress": 0.8, "reasoning": "good"}'
        s = _parse_score(text)
        assert s.should_backtrack is False

    def test_parse_score_non_numeric_values(self):
        """LLM returns string labels instead of floats — should fall back to 0.5."""
        text = '{"promise": "high", "progress": "low", "reasoning": "qualitative"}'
        s = _parse_score(text)
        assert s.promise == 0.5
        assert s.progress == 0.5
        assert s.reasoning == "qualitative"
        assert s.should_backtrack is False


# ---- StepScorer.score() with mocked router ----


class TestStepScorerScore:
    @pytest.mark.asyncio
    async def test_score_returns_valid_step_score(self):
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"promise": 0.7, "progress": 0.4, "reasoning": "advancing"}'
        mock_router.complete = AsyncMock(return_value=mock_resp)

        scorer = StepScorer(router=mock_router)
        result = await scorer.score(
            engagement_id=uuid4(),
            stage="recon",
            step=1,
            world_state_slice={"hosts": []},
            last_tool_results="nmap found 3 hosts",
            engagement_goal="full pentest",
        )
        assert isinstance(result, StepScore)
        assert result.promise == 0.7
        assert result.progress == 0.4

    @pytest.mark.asyncio
    async def test_score_handles_router_failure(self):
        mock_router = AsyncMock()
        mock_router.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        scorer = StepScorer(router=mock_router)
        result = await scorer.score(
            engagement_id=uuid4(),
            stage="recon",
            step=1,
            world_state_slice={},
            last_tool_results="",
            engagement_goal="",
        )
        # Should return neutral scores, not raise
        assert isinstance(result, StepScore)
        assert result.promise == 0.5
        assert result.progress == 0.5
