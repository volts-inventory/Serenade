"""Tests for the pure-Python policy engine (replaces OPA HTTP tests)."""

from __future__ import annotations

import pytest

from policy.engine import PolicyDecision, PolicyEngine


@pytest.mark.asyncio
async def test_policy_allow():
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "recon_only",
            "approval_policy": "auto_approve",
        },
        tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.ALLOW
    assert r.reason == "ok"
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_deny_out_of_scope():
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "recon_only",
            "approval_policy": "auto_approve",
        },
        tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        targets=["192.168.1.1"],
    )
    assert r.decision == PolicyDecision.DENY
    assert r.reason == "out_of_scope"
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_deny_risk_class():
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "recon_only",
            "approval_policy": "auto_approve",
        },
        tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.DENY
    assert r.reason == "risk_class_not_allowed"
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_require_approval():
    """guided mode → phase_gate → exploit pauses for the operator."""
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum", "exploit"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "guided",
            "approval_policy": "phase_gate",
        },
        tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.REQUIRE_APPROVAL
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_exploit_no_approval_in_autonomous():
    """autonomous mode → auto_approve → exploit proceeds without a gate."""
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum", "exploit"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "autonomous",
            "approval_policy": "auto_approve",
        },
        tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.ALLOW
    await eng.aclose()


def _exploit_engagement(mode: str = "autonomous", *, destructive_opt_in: bool = False) -> dict:
    return {
        "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
        "allowed_risk_classes": ["recon", "enum", "exploit", "post-exploit"],
        "window": {"start_ns": 0, "end_ns": 10**19},
        "mode": mode,
        "approval_policy": "auto_approve" if mode != "guided" else "phase_gate",
        "destructive_opt_in": destructive_opt_in,
    }


@pytest.mark.asyncio
async def test_policy_destructive_denied_without_optin():
    """A destructive tool is DENIED by default (no destructive_opt_in)."""
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement=_exploit_engagement(destructive_opt_in=False),
        # risk_class is allowed so the destructive flag — not the risk-class
        # gate — is what blocks the call.
        tool={"name": "wiper", "risk_class": "exploit", "destructive": True},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.DENY
    assert r.reason == "destructive_not_authorized"
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_destructive_with_optin_requires_approval():
    """With destructive_opt_in, a destructive tool is ALLOWED but always
    routed through approval (never silent) — in any mode."""
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement=_exploit_engagement(mode="autonomous", destructive_opt_in=True),
        tool={"name": "wiper", "risk_class": "exploit", "destructive": True},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.REQUIRE_APPROVAL
    await eng.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["guided", "autonomous"])
async def test_policy_allows_real_edb_exec(mode: str):
    """The REAL shipped edb_exploit_exec tool is NOT destructive, so the
    policy gate allows it for an in-scope exploit-stage call (auto in
    autonomous, approval-gated in guided). Destructive EDB *modules* (DoS)
    are gated per-module inside the tool, not by a blanket policy deny —
    regression for the blanket-destructive bug that killed the feature.
    """
    from tools.edb_exec import EdbExec

    tool = EdbExec()
    assert tool.destructive is False  # the whole tool must not be destructive
    tool_dict = {
        "name": tool.name,
        "risk_class": tool.risk_class.value,
        "destructive": tool.destructive,
    }
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement=_exploit_engagement(mode=mode),
        tool=tool_dict,
        targets=["10.0.0.5"],  # in-scope
    )
    assert r.decision != PolicyDecision.DENY
    expected = PolicyDecision.REQUIRE_APPROVAL if mode == "guided" else PolicyDecision.ALLOW
    assert r.decision == expected
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_deny_outside_time_window():
    eng = PolicyEngine()
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon", "enum"],
            "window": {"start_ns": 0, "end_ns": 1},  # expired long ago
            "mode": "recon_only",
            "approval_policy": "auto_approve",
        },
        tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.DENY
    assert r.reason == "outside_time_window"
    await eng.aclose()


@pytest.mark.asyncio
async def test_policy_accepts_settings_kwarg():
    """PolicyEngine(settings) must not raise even though it no longer uses OPA."""
    eng = PolicyEngine(settings="anything")
    r = await eng.evaluate(
        engagement={
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": []},
            "allowed_risk_classes": ["recon"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "recon_only",
            "approval_policy": "auto_approve",
        },
        tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        targets=["10.0.0.5"],
    )
    assert r.decision == PolicyDecision.ALLOW
    await eng.aclose()
