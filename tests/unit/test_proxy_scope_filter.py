"""Tests for the proxy scope-filter addon's pure matching logic (sandbox-12b).

Tests ``_scope.py`` directly so we don't need mitmproxy installed in the
unit-test environment (it only runs inside the proxy container).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import from sandbox/proxy/_scope.py without going through the mitmproxy addon.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "src" / "sandbox" / "proxy")
)

from _scope import host_in_scope, parse_scope  # type: ignore[import-not-found]


def test_parse_scope_splits_hosts_and_cidrs():
    hosts, nets = parse_scope("10.0.0.0/24\nin-scope.example.com, 192.168.1.5")
    assert "in-scope.example.com" in hosts
    # 192.168.1.5 parses as a /32 network, not a hostname.
    assert any(str(n) == "192.168.1.5/32" for n in nets)
    assert any(str(n) == "10.0.0.0/24" for n in nets)


def test_in_scope_ip_inside_cidr():
    hosts, nets = parse_scope("10.0.0.0/24")
    assert host_in_scope("10.0.0.7", hosts, nets)
    assert not host_in_scope("10.0.1.7", hosts, nets)


def test_in_scope_exact_hostname():
    hosts, nets = parse_scope("in-scope.example.com")
    assert host_in_scope("in-scope.example.com", hosts, nets)
    assert host_in_scope("IN-SCOPE.EXAMPLE.COM", hosts, nets)  # case-insensitive


def test_in_scope_dot_suffix_subdomain():
    hosts, nets = parse_scope("example.com")
    assert host_in_scope("api.example.com", hosts, nets)
    # A look-alike suffix must NOT match (notexample.com endswith example.com).
    assert not host_in_scope("notexample.com", hosts, nets)


def test_trailing_dot_fqdn_in_scope():
    # sandbox3-03: the absolute-FQDN form ``example.com.`` is DNS/HTTP-equivalent
    # to ``example.com`` and must match an in-scope ``example.com``.
    hosts, nets = parse_scope("example.com")
    assert host_in_scope("example.com.", hosts, nets)
    assert host_in_scope("api.example.com.", hosts, nets)
    # A look-alike out-of-scope host with a trailing dot still does NOT match.
    assert not host_in_scope("notexample.com.", hosts, nets)


def test_out_of_scope_host_rejected():
    hosts, nets = parse_scope("10.0.0.0/24\nin-scope.example.com")
    assert not host_in_scope("evil.example.org", hosts, nets)
    assert not host_in_scope("203.0.113.9", hosts, nets)


def test_empty_scope_matches_nothing():
    hosts, nets = parse_scope("")
    assert not hosts and not nets
    assert not host_in_scope("anything.com", hosts, nets)


def test_blank_host_never_in_scope():
    hosts, nets = parse_scope("10.0.0.0/24")
    assert not host_in_scope("", hosts, nets)
