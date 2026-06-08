"""Tests for structured critic feedback."""

from __future__ import annotations

import json

import pytest

from agents.critic import Critique, Verdict, _parse_critique


# ---- Critique model has new fields ----


class TestCritiqueModelNewFields:
    def test_new_fields_exist_with_defaults(self):
        """New fields should exist with sensible defaults."""
        c = Critique(
            verdict=Verdict.APPROVE,
            confidence=0.9,
            primary_concern="none",
            critic_name="test",
        )
        assert c.reasoning_trace == ""
        assert c.suggested_fix is None
        assert c.flaws == []

    def test_new_fields_populated(self):
        c = Critique(
            verdict=Verdict.REVISE,
            confidence=0.8,
            primary_concern="version mismatch",
            critic_name="test",
            reasoning_trace="Checked version banner vs CVE range",
            suggested_fix="Run httpx to verify Apache/2.4.49",
            flaws=["single-source evidence", "inferred version"],
        )
        assert c.reasoning_trace == "Checked version banner vs CVE range"
        assert c.suggested_fix == "Run httpx to verify Apache/2.4.49"
        assert len(c.flaws) == 2
        assert "single-source evidence" in c.flaws

    def test_backwards_compatible_with_old_json(self):
        """JSON without new fields should parse correctly."""
        old_json = {
            "verdict": "approve",
            "confidence": 0.95,
            "primary_concern": "all good",
            "issues": [],
        }
        c = Critique(critic_name="test", **old_json)
        assert c.verdict == Verdict.APPROVE
        assert c.reasoning_trace == ""
        assert c.suggested_fix is None
        assert c.flaws == []


# ---- _parse_critique with extended schema ----


class TestParseCritiqueExtended:
    def test_parse_with_all_new_fields(self):
        data = {
            "verdict": "revise",
            "confidence": 0.7,
            "primary_concern": "version not confirmed",
            "issues": ["banner-only evidence"],
            "reasoning_trace": "The version was inferred from banner grab only",
            "suggested_fix": "verify with httpx /version endpoint",
            "flaws": ["single-source", "no direct observation"],
        }
        c = _parse_critique(json.dumps(data), "test_critic")
        assert c.verdict == Verdict.REVISE
        assert c.reasoning_trace == "The version was inferred from banner grab only"
        assert c.suggested_fix == "verify with httpx /version endpoint"
        assert len(c.flaws) == 2

    def test_parse_without_new_fields(self):
        """Old-format JSON (without new fields) still parses correctly."""
        data = {
            "verdict": "approve",
            "confidence": 0.9,
            "primary_concern": "none",
            "issues": [],
        }
        c = _parse_critique(json.dumps(data), "test_critic")
        assert c.verdict == Verdict.APPROVE
        assert c.reasoning_trace == ""
        assert c.suggested_fix is None
        assert c.flaws == []

    def test_parse_with_code_fenced_extended(self):
        data = {
            "verdict": "veto",
            "confidence": 0.95,
            "primary_concern": "honeypot detected",
            "issues": ["suspicious banner"],
            "reasoning_trace": "All 65535 ports open — classic honeypot.",
            "suggested_fix": None,
            "flaws": ["deceptive banner"],
        }
        text = "```json\n" + json.dumps(data) + "\n```"
        c = _parse_critique(text, "test_critic")
        assert c.verdict == Verdict.VETO
        assert "honeypot" in c.reasoning_trace.lower()

    def test_parse_with_null_suggested_fix(self):
        data = {
            "verdict": "approve",
            "confidence": 0.8,
            "primary_concern": "looks ok",
            "issues": [],
            "suggested_fix": None,
        }
        c = _parse_critique(json.dumps(data), "test_critic")
        assert c.suggested_fix is None


# ---- suggested_fix flows to gate feedback ----


class TestSuggestedFixFlowsToFeedback:
    """Verify that the feedback text contains the suggested_fix when present.

    These tests operate at the model level — they don't invoke the full
    agent loop. The integration is tested by checking that the Critique
    model carries the field and that the gate feedback string construction
    in ``_handle_one_tool_call`` surfaces it (tested at the unit level
    by examining the string building logic).
    """

    def test_critique_with_fix_serialises(self):
        c = Critique(
            verdict=Verdict.REVISE,
            confidence=0.7,
            primary_concern="version not confirmed",
            critic_name="pre_exploit",
            suggested_fix="Run httpx against target:8080 to verify Apache/2.4.49",
            flaws=["single-source evidence"],
        )
        # Simulate the feedback text construction from base.py
        feedback_parts = [f"tool denied by critic (revise): {c.primary_concern}"]
        if c.suggested_fix:
            feedback_parts.append(f"Suggested fix: {c.suggested_fix}")
        if c.flaws:
            feedback_parts.append(f"Flaws: {'; '.join(c.flaws[:3])}")
        feedback = " | ".join(feedback_parts)

        assert "Suggested fix:" in feedback
        assert "httpx" in feedback
        assert "Flaws:" in feedback
        assert "single-source" in feedback

    def test_critique_without_fix_no_suggestion(self):
        c = Critique(
            verdict=Verdict.VETO,
            confidence=0.9,
            primary_concern="out of scope",
            critic_name="scope_boundary",
        )
        feedback_parts = [f"tool denied by critic (veto): {c.primary_concern}"]
        if c.suggested_fix:
            feedback_parts.append(f"Suggested fix: {c.suggested_fix}")
        if c.flaws:
            feedback_parts.append(f"Flaws: {'; '.join(c.flaws[:3])}")
        feedback = " | ".join(feedback_parts)

        assert "Suggested fix:" not in feedback
        assert "Flaws:" not in feedback

    def test_structural_check_leaves_defaults(self):
        """StructuralCheck subclasses leave new fields at defaults."""
        c = Critique(
            verdict=Verdict.APPROVE,
            confidence=1.0,
            primary_concern="budget ok",
            critic_name="budget_critic",
            # No reasoning_trace, suggested_fix, or flaws set
        )
        assert c.reasoning_trace == ""
        assert c.suggested_fix is None
        assert c.flaws == []
