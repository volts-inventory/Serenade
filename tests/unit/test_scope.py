"""Tests for the pure-Python scope policy (port of scope.rego).

These tests verify the domain/CIDR matching semantics exactly match the
original Rego rules, including the critical edge case where
"evilexample.com" must NOT match scope domain "example.com".
"""

from __future__ import annotations

from policy.scope import (
    PolicyResult,
    _destructive_blocked,
    _extract_hostname,
    _in_time_window,
    _risk_allowed,
    _target_in_scope,
    evaluate,
)


# ---------------------------------------------------------------------------
# _extract_hostname
# ---------------------------------------------------------------------------

class TestExtractHostname:
    def test_plain_hostname(self):
        assert _extract_hostname("example.com") == "example.com"

    def test_hostname_with_port(self):
        assert _extract_hostname("example.com:8080") == "example.com"

    def test_url_with_scheme(self):
        assert _extract_hostname("https://example.com") == "example.com"

    def test_url_with_scheme_and_port(self):
        assert _extract_hostname("https://example.com:8443") == "example.com"

    def test_url_with_scheme_port_path(self):
        assert _extract_hostname("https://sub.example.com:8443/path/to/thing") == "sub.example.com"

    def test_url_with_path_no_port(self):
        assert _extract_hostname("http://example.com/path") == "example.com"

    def test_ip_address(self):
        assert _extract_hostname("10.0.0.5") == "10.0.0.5"

    def test_ip_address_with_port(self):
        assert _extract_hostname("10.0.0.5:8080") == "10.0.0.5"


# ---------------------------------------------------------------------------
# _target_in_scope — CIDR
# ---------------------------------------------------------------------------

class TestCIDRScope:
    def test_ip_in_cidr(self):
        scope = {"cidrs": ["10.0.0.0/24"]}
        assert _target_in_scope("10.0.0.5", scope) is True

    def test_ip_not_in_cidr(self):
        scope = {"cidrs": ["10.0.0.0/24"]}
        assert _target_in_scope("192.168.1.1", scope) is False

    def test_exact_ip_match(self):
        scope = {"cidrs": ["10.0.0.5/32"]}
        assert _target_in_scope("10.0.0.5", scope) is True

    def test_multiple_cidrs(self):
        scope = {"cidrs": ["10.0.0.0/24", "192.168.1.0/24"]}
        assert _target_in_scope("192.168.1.50", scope) is True

    def test_empty_cidrs(self):
        scope = {"cidrs": []}
        assert _target_in_scope("10.0.0.5", scope) is False


# ---------------------------------------------------------------------------
# _target_in_scope — domain matching (critical: Rego semantics)
# ---------------------------------------------------------------------------

class TestDomainScope:
    def test_exact_match(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("example.com", scope) is True

    def test_exact_match_case_insensitive(self):
        scope = {"domains": ["Example.COM"]}
        assert _target_in_scope("example.com", scope) is True

    def test_subdomain_match(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("sub.example.com", scope) is True

    def test_deep_subdomain_match(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("a.b.c.example.com", scope) is True

    def test_evil_prefix_does_not_match(self):
        """evilexample.com must NOT match example.com — the key edge case."""
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("evilexample.com", scope) is False

    def test_url_domain_match(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("https://sub.example.com:8443/path", scope) is True

    def test_url_domain_no_match(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("https://evilexample.com/path", scope) is False

    def test_empty_domains(self):
        scope = {"domains": []}
        assert _target_in_scope("example.com", scope) is False

    def test_hostname_with_port(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("example.com:8080", scope) is True

    def test_subdomain_case_insensitive(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("SUB.Example.COM", scope) is True


# ---------------------------------------------------------------------------
# _risk_allowed
# ---------------------------------------------------------------------------

class TestRiskAllowed:
    def test_allowed(self):
        assert _risk_allowed("recon", ["recon", "enum"]) is True

    def test_not_allowed(self):
        assert _risk_allowed("exploit", ["recon", "enum"]) is False

    def test_set_input(self):
        assert _risk_allowed("recon", {"recon", "enum"}) is True


# ---------------------------------------------------------------------------
# _in_time_window
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_inside_window(self):
        assert _in_time_window({"start_ns": 0, "end_ns": 10**19}) is True

    def test_before_window(self):
        assert _in_time_window({"start_ns": 10**19, "end_ns": 10**20}) is False

    def test_after_window(self):
        assert _in_time_window({"start_ns": 0, "end_ns": 1}) is False

    def test_empty_window(self):
        # Fail-closed: a missing/empty window is rejected, not always-open
        # (policy-04).
        assert _in_time_window({}) is False

    def test_window_missing_bounds_fails_closed(self):
        assert _in_time_window({"start_ns": 0}) is False
        assert _in_time_window({"end_ns": 10**19}) is False


# ---------------------------------------------------------------------------
# _destructive_blocked
# ---------------------------------------------------------------------------

class TestDestructiveBlocked:
    def test_destructive_blocked_without_optin(self):
        # Denied by default: no destructive_opt_in → blocked (fail-closed).
        assert _destructive_blocked({"destructive": True}, {}) is True

    def test_destructive_blocked_even_with_exploit_allowed(self):
        # Risk-class being allowed does not un-block destructive; only the
        # explicit opt-in does.
        assert (
            _destructive_blocked(
                {"destructive": True},
                {"allowed_risk_classes": ["recon", "enum", "exploit", "post-exploit"]},
            )
            is True
        )

    def test_destructive_unblocked_with_optin(self):
        # With destructive_opt_in the tool is NOT denied here; evaluate()
        # instead routes it through approval (see TestEvaluate).
        assert (
            _destructive_blocked({"destructive": True}, {"destructive_opt_in": True})
            is False
        )

    def test_non_destructive(self):
        assert _destructive_blocked({"destructive": False}, {}) is False

    def test_missing_destructive_key(self):
        assert _destructive_blocked({}, {}) is False


# ---------------------------------------------------------------------------
# evaluate (integration of all checks)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _engagement(self, **overrides):
        base = {
            "scope": {"cidrs": ["10.0.0.0/24"], "domains": ["example.com"]},
            "allowed_risk_classes": ["recon", "enum", "exploit", "post-exploit"],
            "window": {"start_ns": 0, "end_ns": 10**19},
            "mode": "guided",
            "approval_policy": "phase_gate",
        }
        base.update(overrides)
        return base

    def test_allow(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is True
        assert r.reason == "ok"

    def test_deny_out_of_scope(self):
        r = evaluate(
            targets=["192.168.1.1"],
            engagement=self._engagement(),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is False
        assert r.reason == "out_of_scope"

    def test_deny_risk_class(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(allowed_risk_classes=["recon"]),
            tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        )
        assert r.allow is False
        assert r.reason == "risk_class_not_allowed"

    def test_deny_outside_window(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(window={"start_ns": 0, "end_ns": 1}),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is False
        assert r.reason == "outside_time_window"

    def test_deny_destructive_without_optin(self):
        # Destructive is blocked by default (no destructive_opt_in).
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(),
            tool={"name": "wiper", "risk_class": "recon", "destructive": True},
        )
        assert r.allow is False
        assert r.reason == "destructive_not_authorized"

    def test_destructive_with_optin_allowed_but_requires_approval(self):
        # With destructive_opt_in the call is allowed but always routed
        # through approval — never silent.
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(destructive_opt_in=True),
            tool={"name": "wiper", "risk_class": "exploit", "destructive": True},
        )
        assert r.allow is True
        assert r.require_approval is True

    def test_require_approval_exploit_phase_gate(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(mode="guided", approval_policy="phase_gate"),
            tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        )
        assert r.allow is True
        assert r.require_approval is True

    def test_no_approval_exploit_auto_approve(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(mode="autonomous", approval_policy="auto_approve"),
            tool={"name": "exploit_tool", "risk_class": "exploit", "destructive": False},
        )
        assert r.allow is True
        assert r.require_approval is False

    def test_no_approval_recon_phase_gate(self):
        r = evaluate(
            targets=["10.0.0.5"],
            engagement=self._engagement(mode="guided", approval_policy="phase_gate"),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is True
        assert r.require_approval is False

    def test_multiple_targets_all_in_scope(self):
        r = evaluate(
            targets=["10.0.0.1", "10.0.0.2", "sub.example.com"],
            engagement=self._engagement(),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is True

    def test_multiple_targets_one_out_of_scope(self):
        r = evaluate(
            targets=["10.0.0.1", "192.168.1.1"],
            engagement=self._engagement(),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is False
        assert r.reason == "out_of_scope"

    def test_empty_targets(self):
        """No targets = no scope enforcement ran, so fail CLOSED (policy-03)."""
        r = evaluate(
            targets=[],
            engagement=self._engagement(),
            tool={"name": "nmap", "risk_class": "recon", "destructive": False},
        )
        assert r.allow is False
        assert r.reason == "no_targets"
