"""Adversarial regression tests for the policy/authorization gate.

Covers the round-1 critic findings policy-01..policy-08 (excluding policy-06,
owned by another fixer). Each test class maps to a finding ID and exercises the
adversarial case (excluded-inside-included, empty targets, all-internet CIDR,
URL userinfo smuggling, unknown mode, etc.) rather than the happy path.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from orchestrator.models import EngagementRow
from policy.engagement_dict import build_engagement_context
from policy.scope import (
    _extract_hostname,
    _extract_path,
    _in_time_window,
    _target_excluded,
    _target_in_scope,
    evaluate,
)


def _tool(risk_class: str = "recon", destructive: bool = False) -> dict:
    return {"name": "t", "risk_class": risk_class, "destructive": destructive}


def _engagement(scope: dict, **overrides) -> dict:
    base = {
        "scope": scope,
        "allowed_risk_classes": ["recon", "enum", "exploit", "post-exploit"],
        "window": {"start_ns": 0, "end_ns": 10**19},
        "mode": "guided",
        "approval_policy": "phase_gate",
    }
    base.update(overrides)
    return base


def _row(mode: str, scope: dict | None = None) -> EngagementRow:
    now = datetime.now(UTC)
    return EngagementRow(
        id=uuid4(),
        name="test",
        scope=scope if scope is not None else {"cidrs": ["10.0.0.0/24"]},
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


# ---------------------------------------------------------------------------
# policy-01: exclusions enforced, exclusion wins over inclusion
# ---------------------------------------------------------------------------

class TestPolicy01ExclusionsEnforced:
    SCOPE = {
        "cidrs": ["10.0.0.0/24"],
        "excluded_cidrs": ["10.0.0.5/32"],
        "domains": ["example.com"],
        "excluded_domains": ["secret.example.com"],
    }

    def test_excluded_cidr_host_denied_in_scope_check(self):
        assert _target_in_scope("10.0.0.5", self.SCOPE) is False

    def test_excluded_cidr_host_denied_via_evaluate(self):
        r = evaluate(["10.0.0.5"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_excluded_domain_denied_via_evaluate(self):
        r = evaluate(["secret.example.com"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_excluded_subdomain_denied(self):
        # a.secret.example.com is a subdomain of an excluded domain.
        r = evaluate(["a.secret.example.com"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_non_excluded_host_still_allowed(self):
        r = evaluate(["10.0.0.9"], _engagement(self.SCOPE), _tool())
        assert r.allow is True

    def test_non_excluded_subdomain_still_allowed(self):
        r = evaluate(["public.example.com"], _engagement(self.SCOPE), _tool())
        assert r.allow is True

    def test_exclusion_wins_even_when_one_of_many_targets(self):
        r = evaluate(["10.0.0.9", "10.0.0.5"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"


# ---------------------------------------------------------------------------
# policy-02: excluded_paths threaded + enforced for URL targets
# ---------------------------------------------------------------------------

class TestPolicy02ExcludedPaths:
    def test_excluded_paths_threaded_by_builder(self):
        d = build_engagement_context(
            _row("guided", {"domains": ["example.com"], "excluded_paths": ["/admin"]})
        )
        assert d["scope"]["excluded_paths"] == ["/admin"]

    def test_excluded_path_denied(self):
        scope = {"domains": ["example.com"], "excluded_paths": ["/admin"]}
        r = evaluate(["https://example.com/admin/users"], _engagement(scope), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_non_excluded_path_allowed(self):
        scope = {"domains": ["example.com"], "excluded_paths": ["/admin"]}
        r = evaluate(["https://example.com/public"], _engagement(scope), _tool())
        assert r.allow is True

    def test_path_boundary_aware(self):
        # /admin must not exclude /administrator.
        scope = {"domains": ["example.com"], "excluded_paths": ["/admin"]}
        assert _target_excluded("https://example.com/administrator", scope) is False


# ---------------------------------------------------------------------------
# policy2-01: scheme-less targets with query/fragment/params/trailing data
# must not bypass excluded_paths (asymmetry with the urlparse branch).
# ---------------------------------------------------------------------------

class TestPolicy2_01SchemelessPathBypass:
    SCOPE = {"domains": ["example.com"], "excluded_paths": ["/admin"]}

    # --- scheme-less variants that previously ALLOWED but must now DENY ---

    def test_schemeless_query_denied(self):
        r = evaluate(["example.com/admin?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_fragment_denied(self):
        r = evaluate(["example.com/admin#x"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_params_denied(self):
        r = evaluate(["example.com/admin;p"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_trailing_space_denied(self):
        r = evaluate(["example.com/admin "], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_trailing_slash_denied(self):
        r = evaluate(["example.com/admin/"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_subpath_denied(self):
        r = evaluate(["example.com/admin/users?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeless_uppercase_denied(self):
        r = evaluate(["example.com/ADMIN?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    # --- scheme-ful equivalents (control: already correct, must stay DENY) ---

    def test_schemeful_query_denied(self):
        r = evaluate(["http://example.com/admin?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeful_fragment_denied(self):
        r = evaluate(["https://example.com/admin#x"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    def test_schemeful_trailing_slash_denied(self):
        r = evaluate(["https://example.com/admin/"], _engagement(self.SCOPE), _tool())
        assert r.allow is False
        assert r.reason == "excluded"

    # --- non-excluded sibling must STILL be ALLOWED (no over-block) ---

    def test_schemeless_sibling_administrator_allowed(self):
        r = evaluate(["example.com/administrator?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is True

    def test_schemeful_sibling_administrator_allowed(self):
        r = evaluate(["https://example.com/administrator?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is True

    def test_schemeless_unrelated_path_allowed(self):
        r = evaluate(["example.com/public?id=1"], _engagement(self.SCOPE), _tool())
        assert r.allow is True

    # --- direct helper assertions on _target_excluded ---

    def test_target_excluded_schemeless_query(self):
        assert _target_excluded("example.com/admin?id=1", self.SCOPE) is True

    def test_target_excluded_schemeless_sibling(self):
        assert _target_excluded("example.com/administrator?id=1", self.SCOPE) is False


# ---------------------------------------------------------------------------
# policy3-01: dot-segment / double-slash / percent-encoded paths must not
# bypass excluded_paths. A path the target server canonicalizes to /admin
# must DENY; the /administrator sibling must STILL be ALLOWED.
# ---------------------------------------------------------------------------

class TestPolicy3_01DotSegmentPathBypass:
    SCOPE = {"domains": ["example.com"], "excluded_paths": ["/admin"]}

    # --- obfuscated paths that normalize to /admin -> must DENY ---

    @pytest.mark.parametrize(
        "target",
        [
            "example.com//admin",
            "example.com/./admin",
            "example.com/foo/../admin",
            "example.com/admin/../admin",
            "example.com/../admin",
            "example.com///admin",
            "example.com/./foo/../admin",
            "example.com//admin?id=1",
            "example.com/foo/../admin#frag",
            "example.com/%2e%2e/admin",
            "example.com/%2E%2E/admin",
            "example.com/foo/%2e%2e/admin",
            "example.com/%2e/admin",
            "http://example.com//admin",
            "http://example.com/./admin",
            "http://example.com/foo/../admin",
            "http://example.com/admin/../admin",
            "https://example.com/%2e%2e/admin",
        ],
    )
    def test_obfuscated_admin_denied(self, target):
        r = evaluate([target], _engagement(self.SCOPE), _tool())
        assert r.allow is False, target
        assert r.reason == "excluded", target

    def test_obfuscated_admin_excluded_helper(self):
        for target in (
            "example.com//admin",
            "example.com/foo/../admin",
            "example.com/%2e%2e/admin",
        ):
            assert _target_excluded(target, self.SCOPE) is True, target

    # --- the /administrator sibling must NOT be newly matched ---

    @pytest.mark.parametrize(
        "target",
        [
            "example.com//administrator",
            "example.com/./administrator",
            "example.com/foo/../administrator",
            "example.com/%2e%2e/administrator",
            "http://example.com//administrator",
        ],
    )
    def test_obfuscated_administrator_allowed(self, target):
        r = evaluate([target], _engagement(self.SCOPE), _tool())
        assert r.allow is True, target

    def test_administrator_not_excluded_helper(self):
        assert _target_excluded("example.com//administrator", self.SCOPE) is False

    # --- direct _extract_path normalization assertions ---

    @pytest.mark.parametrize(
        "target,expected",
        [
            ("example.com//admin", "/admin"),
            ("example.com/./admin", "/admin"),
            ("example.com/foo/../admin", "/admin"),
            ("example.com/admin/../admin", "/admin"),
            ("example.com/../admin", "/admin"),
            ("example.com/%2e%2e/admin", "/admin"),
            ("example.com/administrator", "/administrator"),
            ("http://example.com//admin", "/admin"),
            ("http://example.com/foo/../admin", "/admin"),
        ],
    )
    def test_extract_path_normalized(self, target, expected):
        assert _extract_path(target) == expected


# ---------------------------------------------------------------------------
# policy3-02: a scheme-less host carrying ?/# (with no path) must resolve its
# host correctly (over-block fix — no authz bypass, but wrong/over-strict).
# ---------------------------------------------------------------------------

class TestPolicy3_02SchemelessHostQuery:
    def test_extract_hostname_query(self):
        assert _extract_hostname("example.com?x=1") == "example.com"

    def test_extract_hostname_fragment(self):
        assert _extract_hostname("example.com#frag") == "example.com"

    def test_extract_hostname_query_with_port(self):
        assert _extract_hostname("example.com:8080?x=1") == "example.com"

    def test_extract_hostname_query_with_userinfo(self):
        # userinfo smuggling still resolves to the real host even with a query.
        assert _extract_hostname("example.com@evil.example.org?x=1") == "evil.example.org"

    def test_inscope_host_with_query_allowed(self):
        scope = {"domains": ["example.com"]}
        r = evaluate(["example.com?x=1"], _engagement(scope), _tool())
        assert r.allow is True

    def test_excluded_host_with_query_denied(self):
        scope = {"domains": ["example.com"], "excluded_domains": ["secret.example.com"]}
        r = evaluate(["secret.example.com?x=1"], _engagement(scope), _tool())
        assert r.allow is False
        assert r.reason == "excluded"


# ---------------------------------------------------------------------------
# policy-03: empty/missing targets fail closed
# ---------------------------------------------------------------------------

class TestPolicy03EmptyTargets:
    def test_empty_list_denied(self):
        r = evaluate([], _engagement({"cidrs": ["10.0.0.0/24"]}), _tool())
        assert r.allow is False
        assert r.reason == "no_targets"

    def test_none_targets_denied(self):
        r = evaluate(None, _engagement({"cidrs": ["10.0.0.0/24"]}), _tool())  # type: ignore[arg-type]
        assert r.allow is False
        assert r.reason == "no_targets"


# ---------------------------------------------------------------------------
# policy-04: time window fail-closed + allowed_hours_utc enforced
# ---------------------------------------------------------------------------

class TestPolicy04TimeWindow:
    def test_empty_window_fails_closed(self):
        assert _in_time_window({}) is False

    def test_missing_bounds_fail_closed(self):
        assert _in_time_window({"start_ns": 0}) is False
        assert _in_time_window({"end_ns": 10**19}) is False

    def test_evaluate_missing_window_denied(self):
        r = evaluate(["10.0.0.5"], _engagement({"cidrs": ["10.0.0.0/24"]}, window={}), _tool())
        assert r.allow is False
        assert r.reason == "outside_time_window"

    def test_allowed_hours_threaded_by_builder(self):
        d = build_engagement_context(
            _row("guided", {"cidrs": ["10.0.0.0/24"], "allowed_hours_utc": (9, 17)})
        )
        assert d["window"]["allowed_hours_utc"] == (9, 17)

    def test_allowed_hours_outside_band_denied(self):
        now = datetime.now(UTC)
        cur = now.hour
        # Pick a one-hour band that definitely excludes the current hour.
        excl = (cur + 2) % 24
        window = {"start_ns": 0, "end_ns": 10**19, "allowed_hours_utc": (excl, excl)}
        assert _in_time_window(window) is False

    def test_allowed_hours_inside_band_allowed(self):
        now = datetime.now(UTC)
        cur = now.hour
        window = {"start_ns": 0, "end_ns": 10**19, "allowed_hours_utc": (cur, cur)}
        assert _in_time_window(window) is True

    def test_allowed_hours_wraparound(self):
        now = datetime.now(UTC)
        cur = now.hour
        # Band that wraps midnight and includes the current hour.
        window = {"start_ns": 0, "end_ns": 10**19, "allowed_hours_utc": (cur, (cur - 1) % 24)}
        assert _in_time_window(window) is True

    def test_malformed_allowed_hours_fails_closed(self):
        window = {"start_ns": 0, "end_ns": 10**19, "allowed_hours_utc": "bogus"}
        assert _in_time_window(window) is False


# ---------------------------------------------------------------------------
# policy-05: scope.urls used for host matching
# ---------------------------------------------------------------------------

class TestPolicy05UrlScope:
    def test_url_only_scope_matches_host(self):
        scope = {"urls": ["https://app.example.com/login"]}
        r = evaluate(["app.example.com"], _engagement(scope), _tool())
        assert r.allow is True

    def test_url_only_scope_matches_subdomain(self):
        scope = {"urls": ["https://example.com/login"]}
        assert _target_in_scope("api.example.com", scope) is True

    def test_url_only_scope_rejects_out_of_scope_host(self):
        scope = {"urls": ["https://app.example.com/login"]}
        r = evaluate(["evil.org"], _engagement(scope), _tool())
        assert r.allow is False
        assert r.reason == "out_of_scope"


# ---------------------------------------------------------------------------
# policy-07: all-internet CIDR guard
# ---------------------------------------------------------------------------

class TestPolicy07UnboundedCidr:
    def test_v4_all_internet_ignored_by_default(self):
        assert _target_in_scope("8.8.8.8", {"cidrs": ["0.0.0.0/0"]}) is False

    def test_v6_all_internet_ignored_by_default(self):
        assert _target_in_scope("2001:4860:4860::8888", {"cidrs": ["::/0"]}) is False

    def test_evaluate_all_internet_denied_by_default(self):
        r = evaluate(["8.8.8.8"], _engagement({"cidrs": ["0.0.0.0/0"]}), _tool())
        assert r.allow is False
        assert r.reason == "out_of_scope"

    def test_all_internet_honored_only_with_explicit_optin(self):
        scope = {"cidrs": ["0.0.0.0/0"], "allow_unbounded_cidrs": True}
        assert _target_in_scope("8.8.8.8", scope) is True

    def test_builder_defaults_optin_false(self):
        d = build_engagement_context(_row("guided", {"cidrs": ["0.0.0.0/0"]}))
        assert d["scope"]["allow_unbounded_cidrs"] is False

    def test_builder_threads_optin_true(self):
        d = build_engagement_context(
            _row("guided", {"cidrs": ["0.0.0.0/0"], "allow_unbounded_cidrs": True})
        )
        assert d["scope"]["allow_unbounded_cidrs"] is True

    def test_excluding_all_internet_is_honored(self):
        # An operator excluding 0.0.0.0/0 must carve everything out even though
        # the *inclusion* path ignores unbounded prefixes.
        scope = {
            "cidrs": ["10.0.0.0/24"],
            "excluded_cidrs": ["0.0.0.0/0"],
            "allow_unbounded_cidrs": True,
        }
        assert _target_in_scope("10.0.0.5", scope) is False


# ---------------------------------------------------------------------------
# policy-01/spoofing: URL userinfo (`user@host`) must resolve to real host
# ---------------------------------------------------------------------------

class TestUrlUserinfoSpoofing:
    def test_url_userinfo_resolves_to_real_host(self):
        # https://example.com@evil.org/ is really evil.org, not example.com.
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("https://example.com@evil.org/", scope) is False

    def test_schemeless_userinfo_resolves_to_real_host(self):
        scope = {"domains": ["example.com"]}
        assert _target_in_scope("example.com@evil.org/path", scope) is False

    def test_userinfo_real_host_in_scope_allowed(self):
        scope = {"domains": ["evil.org"]}
        assert _target_in_scope("https://example.com@evil.org/", scope) is True


# ---------------------------------------------------------------------------
# policy-08: unknown mode falls back to least privilege
# ---------------------------------------------------------------------------

class TestPolicy08UnknownModeLeastPrivilege:
    def test_unknown_mode_falls_back_to_recon_only(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = build_engagement_context(_row("totally_bogus_mode"))
        assert d["mode"] == "recon_only"
        assert d["allowed_risk_classes"] == ["recon", "enum"]
        assert d["approval_policy"] == "auto_approve"
        assert "exploit" not in d["allowed_risk_classes"]
        assert "post-exploit" not in d["allowed_risk_classes"]

    def test_unknown_mode_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_engagement_context(_row("bogus"))
        assert any("least-privilege" in str(x.message) for x in w)

    def test_unknown_mode_not_edb_auto_exploit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = build_engagement_context(_row("bogus"))
        assert d["edb_auto_exploit"] is False

    def test_assess_mode_recon_enum_only(self):
        d = build_engagement_context(_row("assess"))
        assert d["allowed_risk_classes"] == ["recon", "enum"]
        assert "exploit" not in d["allowed_risk_classes"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
