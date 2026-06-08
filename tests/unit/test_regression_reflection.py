"""Tests for LLM regression reflection."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.regression import RegressionRequest, RegressionTriggerKind
from agents.regression_reflection import (
    RegressionReflector,
    RegressionSuggestion,
    _parse_suggestion,
)


# ---- RegressionSuggestion model ----


class TestRegressionSuggestion:
    def test_default_no_regress(self):
        s = RegressionSuggestion()
        assert s.should_regress is False
        assert s.target_stage is None
        assert s.delta == {}
        assert s.rationale == ""

    def test_with_regress(self):
        s = RegressionSuggestion(
            should_regress=True,
            target_stage="recon",
            delta={"cidrs": ["10.0.5.0/24"]},
            rationale="missed subnet",
        )
        assert s.should_regress is True
        assert s.target_stage == "recon"


# ---- _parse_suggestion ----


class TestParseSuggestion:
    def test_valid_no_regress(self):
        text = '{"should_regress": false}'
        s = _parse_suggestion(text)
        assert s.should_regress is False

    def test_valid_regress(self):
        text = json.dumps({
            "should_regress": True,
            "target_stage": "recon",
            "delta": {"cidrs": ["10.0.5.0/24"]},
            "rationale": "uncovered subnet in scope",
        })
        s = _parse_suggestion(text)
        assert s.should_regress is True
        assert s.target_stage == "recon"
        assert s.delta == {"cidrs": ["10.0.5.0/24"]}
        assert "uncovered" in s.rationale

    def test_code_fenced_json(self):
        inner = json.dumps({
            "should_regress": True,
            "target_stage": "enumeration",
            "delta": {"host_ids": ["abc"]},
            "rationale": "missed services",
        })
        text = f"```json\n{inner}\n```"
        s = _parse_suggestion(text)
        assert s.should_regress is True
        assert s.target_stage == "enumeration"

    def test_unparseable_returns_no_regress(self):
        s = _parse_suggestion("this is not json")
        assert s.should_regress is False

    def test_invalid_target_stage_returns_no_regress(self):
        text = json.dumps({
            "should_regress": True,
            "target_stage": "report",  # not a valid regression target
            "delta": {},
            "rationale": "bad target",
        })
        s = _parse_suggestion(text)
        assert s.should_regress is False

    def test_missing_target_stage_with_regress(self):
        text = json.dumps({
            "should_regress": True,
            "delta": {},
            "rationale": "no target",
        })
        s = _parse_suggestion(text)
        # target_stage is None but should_regress is True
        # The reflector will catch this (target_stage not in valid_stages)
        assert s.should_regress is False

    def test_valid_stages(self):
        for stage in ("recon", "enumeration", "vuln_triage", "exploit", "lateral"):
            text = json.dumps({
                "should_regress": True,
                "target_stage": stage,
                "delta": {},
                "rationale": "test",
            })
            s = _parse_suggestion(text)
            assert s.should_regress is True
            assert s.target_stage == stage

    def test_rationale_truncated(self):
        text = json.dumps({
            "should_regress": True,
            "target_stage": "recon",
            "delta": {},
            "rationale": "x" * 1000,
        })
        s = _parse_suggestion(text)
        assert len(s.rationale) <= 500


# ---- RegressionTriggerKind ----


def test_llm_suggested_regression_exists():
    assert RegressionTriggerKind.LLM_SUGGESTED_REGRESSION == "llm_suggested_regression"


def test_regression_request_accepts_llm_trigger():
    """A RegressionRequest with LLM_SUGGESTED_REGRESSION trigger validates."""
    from worldstate.schema import Stage

    req = RegressionRequest(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.LLM_SUGGESTED_REGRESSION,
        delta={"cidrs": ["10.0.5.0/24"]},
        rationale="llm_reflection: uncovered subnet",
    )
    assert req.trigger_kind == RegressionTriggerKind.LLM_SUGGESTED_REGRESSION


# ---- RegressionReflector.reflect() ----


class TestRegressionReflector:
    @pytest.mark.asyncio
    async def test_reflect_returns_none_when_no_gap(self):
        """When the LLM says should_regress=false, reflect returns None."""
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"should_regress": false}'
        mock_router.complete = AsyncMock(return_value=mock_resp)

        reflector = RegressionReflector(router=mock_router)
        result = await reflector.reflect(
            engagement_id=uuid4(),
            current_stage="recon",
            world_state_slice={"hosts": []},
            engagement_scope={"cidrs": ["10.0.0.0/24"]},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_reflect_returns_request_when_gap(self):
        """When the LLM says should_regress=true, reflect returns a RegressionRequest."""
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "should_regress": True,
            "target_stage": "recon",
            "delta": {"cidrs": ["10.0.5.0/24"]},
            "rationale": "subnet 10.0.5.0/24 in scope but no hosts enumerated",
        })
        mock_router.complete = AsyncMock(return_value=mock_resp)

        reflector = RegressionReflector(router=mock_router)
        result = await reflector.reflect(
            engagement_id=uuid4(),
            current_stage="enumeration",
            world_state_slice={"hosts": [{"ip": "10.0.1.1"}]},
            engagement_scope={"cidrs": ["10.0.0.0/16"]},
        )
        assert result is not None
        assert isinstance(result, RegressionRequest)
        assert result.trigger_kind == RegressionTriggerKind.LLM_SUGGESTED_REGRESSION
        assert result.delta == {"cidrs": ["10.0.5.0/24"]}
        assert "llm_reflection" in result.rationale

    @pytest.mark.asyncio
    async def test_reflect_handles_router_failure(self):
        """When the router raises, reflect returns None (not crash)."""
        mock_router = AsyncMock()
        mock_router.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        reflector = RegressionReflector(router=mock_router)
        result = await reflector.reflect(
            engagement_id=uuid4(),
            current_stage="recon",
            world_state_slice={},
            engagement_scope={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_reflect_invalid_target_stage_returns_none(self):
        """When the LLM returns an invalid target stage, reflect returns None."""
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "should_regress": True,
            "target_stage": "invalid_stage",
            "delta": {},
            "rationale": "test",
        })
        mock_router.complete = AsyncMock(return_value=mock_resp)

        reflector = RegressionReflector(router=mock_router)
        result = await reflector.reflect(
            engagement_id=uuid4(),
            current_stage="recon",
            world_state_slice={},
            engagement_scope={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_reflect_includes_verbal_memory(self):
        """The verbal memory is included in the prompt."""
        mock_router = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"should_regress": false}'
        mock_router.complete = AsyncMock(return_value=mock_resp)

        reflector = RegressionReflector(router=mock_router)
        await reflector.reflect(
            engagement_id=uuid4(),
            current_stage="recon",
            world_state_slice={},
            engagement_scope={},
            verbal_memory=[
                {"stage": "recon", "outcome": "failure", "summary": "nmap found 0 hosts"},
            ],
        )
        # Verify the router was called (prompt was built)
        assert mock_router.complete.call_count == 1
        call_args = mock_router.complete.call_args
        req = call_args[0][0]
        assert "nmap found 0 hosts" in req.user
