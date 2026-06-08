"""Regression-history digest renderer and prompt integration tests.

Validates:
1. ``render_regression_digest`` produces compact one-line-per-group output.
2. ``render_user_prompt`` includes / omits the digest section based on input.
"""

from __future__ import annotations

import os

os.environ.setdefault("SERENADE_AUTH_ALLOW_DEV_FALLBACK", "1")
os.environ.setdefault("SERENADE_SESSION_SECRET", "x" * 32)

import pytest

from llm.prompts.base import render_regression_digest, render_user_prompt
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# render_regression_digest                                                      #
# --------------------------------------------------------------------------- #


def test_digest_empty_when_no_rows() -> None:
    assert render_regression_digest([]) == ""


def test_digest_groups_by_stage_and_trigger() -> None:
    rows = [
        {
            "target_stage": "RECON",
            "trigger_kind": "new_subnet_discovered",
            "delta_json": {"cidrs": ["10.0.5.0/24"]},
        },
        {
            "target_stage": "RECON",
            "trigger_kind": "new_subnet_discovered",
            "delta_json": {"cidrs": ["10.0.6.0/24"]},
        },
        {
            "target_stage": "VULN_TRIAGE",
            "trigger_kind": "misconfiguration_opens_path",
            "delta_json": {"finding_ids": ["f-381"]},
        },
    ]
    result = render_regression_digest(rows)
    lines = result.strip().splitlines()
    # Header + 2 group lines
    assert lines[0].startswith("Prior re-sweeps")
    group_lines = [l for l in lines if l.strip().startswith("-")]
    assert len(group_lines) == 2
    # Check counts
    assert "2× RECON" in result  # 2x RECON
    assert "1× VULN_TRIAGE" in result  # 1x VULN_TRIAGE


def test_digest_caps_identifiers_at_five() -> None:
    cidrs = [f"10.0.{i}.0/24" for i in range(8)]
    rows = [
        {
            "target_stage": "RECON",
            "trigger_kind": "new_subnet_discovered",
            "delta_json": {"cidrs": cidrs},
        },
    ]
    result = render_regression_digest(rows)
    # Should show 5 CIDRs + the overflow suffix
    assert "...(+3 more)" in result
    # Only 5 CIDRs should appear before the suffix
    shown_cidrs = [c for c in cidrs if c in result]
    assert len(shown_cidrs) == 5


def test_digest_total_hard_cap_600_chars() -> None:
    # Create a massive input that would exceed 600 chars
    rows = [
        {
            "target_stage": f"STAGE_{i}",
            "trigger_kind": f"trigger_kind_with_a_very_long_name_{i}",
            "delta_json": {
                "domains": [f"subdomain-{j}.example-organization-{i}.com" for j in range(5)]
            },
        }
        for i in range(20)
    ]
    result = render_regression_digest(rows)
    assert len(result) <= 600


# --------------------------------------------------------------------------- #
# render_user_prompt integration                                                #
# --------------------------------------------------------------------------- #


def test_render_user_prompt_includes_digest() -> None:
    rows = [
        {
            "target_stage": "RECON",
            "trigger_kind": "new_subnet_discovered",
            "delta_json": {"cidrs": ["10.0.5.0/24"]},
        },
    ]
    prompt = render_user_prompt(
        stage=Stage.RECON,
        objective="Discover hosts",
        world_state_slice={"hosts": []},
        regression_rows=rows,
    )
    assert "## Prior regression activity" in prompt
    assert "Prior re-sweeps this engagement:" in prompt
    assert "10.0.5.0/24" in prompt


def test_render_user_prompt_omits_digest_when_empty() -> None:
    prompt = render_user_prompt(
        stage=Stage.RECON,
        objective="Discover hosts",
        world_state_slice={"hosts": []},
        regression_rows=[],
    )
    assert "## Prior regression activity" not in prompt

    # Also when omitted entirely (None / default)
    prompt2 = render_user_prompt(
        stage=Stage.RECON,
        objective="Discover hosts",
        world_state_slice={"hosts": []},
    )
    assert "## Prior regression activity" not in prompt2
