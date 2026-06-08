"""Tests for Wave 1 of the pentester fix plan."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.cost import estimate_engagement_cost
from tools.operator_exec import OperatorExecInput, _redact_argv


# --------------------------------------------------------------------------- #
# F8 — mode → risk-class derivation                                            #
# --------------------------------------------------------------------------- #


def test_legacy_aggressive_allows_exploit_but_not_destructive():
    """Legacy ``aggressive`` normalizes to ``autonomous``: exploitation is
    unlocked but destructive is never allowed — destructive is finding-only."""
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from orchestrator.models import EngagementRow
    from policy.engagement_dict import build_engagement_context

    now = datetime.now(UTC)
    row = EngagementRow(
        id=uuid4(), name="x", scope={}, objectives=[], mode="aggressive",
        time_window_start=now, time_window_end=now + timedelta(hours=1),
        token_budget=1000, wallclock_hours=1,
        authorization_artifact_sha256="x" * 64, status="draft",
        current_stage="recon", created_at=now, updated_at=now,
    )
    d = build_engagement_context(row)
    assert d["mode"] == "autonomous"
    assert "exploit" in d["allowed_risk_classes"]
    assert "destructive" not in d["allowed_risk_classes"]


# --------------------------------------------------------------------------- #
# F11 — relaxed PoC lint                                                       #
# --------------------------------------------------------------------------- #


def test_poc_lint_allows_uppercase_constants(tmp_path):
    from agents.poc_lint import lint_module

    src = (
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "TIMEOUT_S = 5\n"
        'URL_PATH = "/admin"\n'
        "PORTS = [80, 443, 8080]\n"
        "OPTIONS = {'verify': False, 'allow_redirects': True}\n"
        "def build_invocation(p, c): return None\n"
    )
    p = tmp_path / "exploit.py"
    p.write_text(src)
    ok, reason = lint_module(p)
    assert ok, reason


def test_poc_lint_still_rejects_function_call(tmp_path):
    from agents.poc_lint import lint_module

    src = "from urllib.parse import urlparse\nURL = urlparse('http://x')\n"
    p = tmp_path / "exploit.py"
    p.write_text(src)
    ok, reason = lint_module(p)
    assert not ok


def test_poc_lint_rejects_lowercase_assign(tmp_path):
    from agents.poc_lint import lint_module

    src = "timeout = 5\n"  # lowercase = not allowed
    p = tmp_path / "exploit.py"
    p.write_text(src)
    ok, _ = lint_module(p)
    assert not ok


# --------------------------------------------------------------------------- #
# F3 — cost forecaster                                                         #
# --------------------------------------------------------------------------- #


def test_estimate_scales_with_cidr_size():
    small = estimate_engagement_cost({"cidrs": ["10.0.0.0/29"]}, "normal")  # 8 hosts
    large = estimate_engagement_cost({"cidrs": ["10.0.0.0/24"]}, "normal")  # 256 hosts
    assert large.token_low > small.token_low
    assert large.usd_high > small.usd_high


def test_estimate_mlx_is_zero():
    est = estimate_engagement_cost({"cidrs": ["10.0.0.0/24"]}, "normal", backend="mlx")
    assert est.usd_low == 0.0
    assert est.usd_high == 0.0


def test_estimate_passive_cheaper_than_aggressive():
    scope = {"cidrs": ["10.0.0.0/24"]}
    passive = estimate_engagement_cost(scope, "passive")
    aggressive = estimate_engagement_cost(scope, "aggressive")
    assert aggressive.usd_high > passive.usd_high


def test_estimate_handles_invalid_cidr():
    est = estimate_engagement_cost({"cidrs": ["bogus"]}, "normal")
    # Should fall back to floor of 1 host without crashing
    assert est.token_low > 0


# --------------------------------------------------------------------------- #
# F1 — OperatorExec input validation + argv redaction                          #
# --------------------------------------------------------------------------- #


def test_operator_exec_requires_min_justification():
    with pytest.raises(ValidationError):
        OperatorExecInput(argv=["echo", "hi"], target="x", justification="too short")


def test_operator_exec_bypass_requires_second_justification():
    with pytest.raises(ValidationError):
        OperatorExecInput(
            argv=["echo", "hi"],
            target="x",
            justification="this is a sufficiently long justification ok",
            bypass_critic=True,
            # bypass_justification missing
        )


def test_operator_exec_bypass_with_justification_ok():
    inp = OperatorExecInput(
        argv=["echo", "hi"],
        target="x",
        justification="this is a sufficiently long justification ok",
        bypass_critic=True,
        bypass_justification="verified safe on lab clone of this host already",
    )
    assert inp.bypass_critic is True


def test_redact_argv_strips_password_flag():
    out = _redact_argv(["nc", "-u", "admin", "-p", "Sup3rSecret", "host"])
    assert "Sup3rSecret" not in " ".join(out)
    assert "***REDACTED***" in out


def test_redact_argv_strips_authorization_header_value():
    out = _redact_argv(["curl", "-H", "Authorization: Bearer abc123", "http://x"])
    joined = " ".join(out)
    assert "abc123" not in joined
    assert "REDACTED" in joined


def test_redact_argv_preserves_innocent_args():
    out = _redact_argv(["nmap", "-sV", "-p", "1-1000", "10.0.0.5"])
    # `-p` here is the port flag (different binary) — our heuristic still
    # redacts the value after `-p`, which is intentionally over-eager.
    # The acceptance criterion is "no secret slips through," not "no false
    # redaction." Verify the rest is intact.
    assert "nmap" in out
    assert "10.0.0.5" in out


# --------------------------------------------------------------------------- #
# F7 — DraftFinding has override fields                                        #
# --------------------------------------------------------------------------- #


def test_draft_finding_override_fields_default_none():
    from agents.report import DraftFinding

    f = DraftFinding(title="t", description="d", severity="medium")
    assert f.override_justification is None
    assert f.override_by is None
    assert f.override_at is None


def test_draft_finding_override_set():
    from agents.report import DraftFinding

    f = DraftFinding(
        title="t",
        description="d",
        severity="medium",
        critic_verdict="veto",
        override_justification="confirmed via burp; reinstated by reviewer",
        override_by="operator",
        override_at="2026-05-20T12:00:00Z",
    )
    assert f.override_justification == "confirmed via burp; reinstated by reviewer"
