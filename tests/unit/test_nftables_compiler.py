"""Tests for the per-engagement nftables ruleset compiler (Phase 17, Tier A #1)."""

from __future__ import annotations

from sandbox.nftables import compile_ruleset


def test_allows_scope_cidrs():
    rs = compile_ruleset(
        table_name="t1",
        scope={"cidrs": ["10.0.0.0/8", "192.168.1.0/24"], "excluded_cidrs": []},
    )
    assert "ip daddr 10.0.0.0/8 accept" in rs
    assert "ip daddr 192.168.1.0/24 accept" in rs


def test_drops_excluded_cidrs_before_accepting():
    """Order matters in nftables — excluded must appear before allowed."""
    rs = compile_ruleset(
        table_name="t1",
        scope={
            "cidrs": ["10.0.0.0/8"],
            "excluded_cidrs": ["10.0.0.5/32"],
        },
    )
    drop_idx = rs.index("10.0.0.5/32 drop")
    accept_idx = rs.index("10.0.0.0/8 accept")
    assert drop_idx < accept_idx


def test_default_policy_is_drop():
    rs = compile_ruleset(table_name="t1", scope={"cidrs": []})
    assert "policy drop" in rs
    # Default drop chain on both output and input
    assert rs.count("policy drop") >= 2


def test_loopback_always_accepted():
    rs = compile_ruleset(table_name="t1", scope={"cidrs": []})
    assert 'oifname "lo" accept' in rs
    assert 'iifname "lo" accept' in rs


def test_established_related_accepted():
    rs = compile_ruleset(table_name="t1", scope={"cidrs": []})
    assert rs.count("ct state established,related accept") >= 2


def test_proxy_egress_accepted_when_set():
    rs = compile_ruleset(table_name="t1", scope={"cidrs": []}, proxy_ip="172.30.0.50")
    assert "ip daddr 172.30.0.50 tcp dport 8080 accept" in rs


def test_dns_servers_added():
    rs = compile_ruleset(table_name="t1", scope={"cidrs": []}, dns_servers=["1.1.1.1", "8.8.8.8"])
    assert "ip daddr 1.1.1.1 udp dport 53 accept" in rs
    assert "ip daddr 8.8.8.8 udp dport 53 accept" in rs


def test_invalid_cidr_silently_skipped():
    rs = compile_ruleset(
        table_name="t1",
        scope={
            "cidrs": ["10.0.0.0/8", "not-a-cidr", "999.999.999.999/0"],
            "excluded_cidrs": ["bogus/zz"],
        },
    )
    assert "ip daddr 10.0.0.0/8 accept" in rs
    assert "not-a-cidr" not in rs
    assert "999.999.999.999" not in rs
    assert "bogus" not in rs


def test_table_name_appears_in_output():
    rs = compile_ruleset(table_name="serenade_eng_abc123", scope={"cidrs": []})
    assert "table inet serenade_eng_abc123" in rs


def test_counter_on_default_drop():
    """The counter lets the operator see how many out-of-scope blocks fired."""
    rs = compile_ruleset(table_name="t1", scope={"cidrs": ["10.0.0.0/8"]})
    assert "counter drop" in rs
