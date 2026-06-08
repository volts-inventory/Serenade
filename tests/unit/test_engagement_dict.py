from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from orchestrator.models import EngagementRow
from policy.engagement_dict import build_engagement_context


def _row(mode: str) -> EngagementRow:
    now = datetime.now(UTC)
    row = EngagementRow(
        id=uuid4(),
        name="test",
        scope={"cidrs": ["10.0.0.0/24"], "domains": ["lab.local"]},
        objectives=[],
        mode=mode,
        time_window_start=now,
        time_window_end=now + timedelta(hours=4),
        token_budget=500_000,
        wallclock_hours=4,
        authorization_artifact_sha256="x" * 64,
        status="draft",
        current_stage="scope_intake",
        created_at=now,
        updated_at=now,
    )
    return row


def test_recon_only_recon_enum():
    d = build_engagement_context(_row("recon_only"))
    assert d["allowed_risk_classes"] == ["recon", "enum"]
    assert d["mode"] == "recon_only"
    assert d["approval_policy"] == "auto_approve"
    assert "exploit" not in d["allowed_risk_classes"]
    assert "destructive" not in d["allowed_risk_classes"]
    # destructive_opt_in is now surfaced (PoC-path gate), fail-closed False by
    # default. It is NOT a risk class — it never relaxes the policy gate.
    assert d["destructive_opt_in"] is False


def test_assess_recon_enum_only():
    d = build_engagement_context(_row("assess"))
    assert d["allowed_risk_classes"] == ["recon", "enum"]
    assert d["approval_policy"] == "auto_approve"
    assert "exploit" not in d["allowed_risk_classes"]


def test_guided_adds_exploit_phase_gate():
    d = build_engagement_context(_row("guided"))
    assert "exploit" in d["allowed_risk_classes"]
    assert "post-exploit" in d["allowed_risk_classes"]
    assert "destructive" not in d["allowed_risk_classes"]
    assert d["approval_policy"] == "phase_gate"


def test_autonomous_adds_exploit_auto_approve():
    d = build_engagement_context(_row("autonomous"))
    assert "exploit" in d["allowed_risk_classes"]
    assert "post-exploit" in d["allowed_risk_classes"]
    assert "destructive" not in d["allowed_risk_classes"]
    assert d["approval_policy"] == "auto_approve"


def test_legacy_aggressive_normalizes_to_autonomous():
    d = build_engagement_context(_row("aggressive"))
    assert d["mode"] == "autonomous"
    assert "exploit" in d["allowed_risk_classes"]
    assert d["approval_policy"] == "auto_approve"
    assert "destructive" not in d["allowed_risk_classes"]


def test_legacy_normal_normalizes_to_guided():
    d = build_engagement_context(_row("normal"))
    assert d["mode"] == "guided"
    assert "exploit" in d["allowed_risk_classes"]
    assert d["approval_policy"] == "phase_gate"


def test_legacy_passive_normalizes_to_recon_only():
    d = build_engagement_context(_row("passive"))
    assert d["mode"] == "recon_only"
    assert d["allowed_risk_classes"] == ["recon", "enum"]


def test_legacy_lab_normalizes_to_autonomous():
    d = build_engagement_context(_row("lab"))
    assert d["mode"] == "autonomous"
    assert "exploit" in d["allowed_risk_classes"]
    assert d["approval_policy"] == "auto_approve"
    # lab no longer unlocks destructive — destructive is finding-only.
    assert "destructive" not in d["allowed_risk_classes"]


def test_window_nanoseconds_populated():
    d = build_engagement_context(_row("guided"))
    assert d["window"]["start_ns"] > 0
    assert d["window"]["end_ns"] > d["window"]["start_ns"]


def test_scope_fields_present():
    d = build_engagement_context(_row("guided"))
    assert d["scope"]["cidrs"] == ["10.0.0.0/24"]
    assert d["scope"]["domains"] == ["lab.local"]
    assert d["scope"]["excluded_cidrs"] == []


def test_no_exploit_list_default_empty():
    d = build_engagement_context(_row("guided"))
    assert d["no_exploit_list"] == []
