"""Tests for per-engagement mitmproxy lifecycle naming (Phase 18, Tier B #9)."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

import sandbox.proxy_lifecycle as proxy_mod
from sandbox.proxy_lifecycle import (
    _build_proxy_run_argv,
    engagement_proxy_logs_volume,
    engagement_proxy_name,
    teardown_engagement_proxy,
)


def test_proxy_name_uses_full_uuid():
    eid = UUID("12345678-1234-5678-1234-567812345678")
    assert engagement_proxy_name(eid) == "serenade-proxy-12345678-1234-5678-1234-567812345678"


def test_logs_volume_per_engagement():
    eid = UUID("12345678-1234-5678-1234-567812345678")
    assert (
        engagement_proxy_logs_volume(eid)
        == "serenade-proxy-logs-12345678-1234-5678-1234-567812345678"
    )


def test_proxy_name_distinguishes_engagements():
    a = engagement_proxy_name(UUID("12345678-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))
    b = engagement_proxy_name(UUID("12345678-bbbb-bbbb-bbbb-bbbbbbbbbbbb"))
    assert a != b


# --- async-03/10: proxy per-name lock dict must be bounded ---


def test_proxy_lock_dict_is_plain_dict():
    assert not hasattr(proxy_mod._locks, "default_factory")


@pytest.mark.asyncio
async def test_proxy_teardown_evicts_lock_entry(monkeypatch):
    async def _fake_exec(*args, **kwargs):
        class _P:
            returncode = 0

            async def wait(self):
                return 0

            async def communicate(self):
                return b"", b""

        return _P()

    monkeypatch.setattr(proxy_mod.asyncio, "create_subprocess_exec", _fake_exec)

    eid = uuid4()
    name = engagement_proxy_name(eid)
    proxy_mod._lock_for(name)
    assert name in proxy_mod._locks

    await teardown_engagement_proxy(eid)
    assert name not in proxy_mod._locks


# --- sandbox-12b: proxy scope-filter addon plumbing ---


def _argv_for(scope_hosts):
    eid = UUID("12345678-1234-5678-1234-567812345678")
    return _build_proxy_run_argv(
        name=engagement_proxy_name(eid),
        network="serenade-sb-x",
        volume="vol",
        engagement_id=eid,
        scope_hosts=scope_hosts,
    )


def test_proxy_argv_logging_addon_always_present():
    argv = _argv_for(None)
    assert "/scripts/log_evidence.py" in argv
    # No scope filter / scope env when scope is not provided (interface preserved).
    assert "/scripts/scope_filter.py" not in argv
    assert not any(a.startswith("SERENADE_SCOPE_HOSTS=") for a in argv)


def test_proxy_argv_adds_scope_filter_when_scope_given():
    argv = _argv_for(["10.0.0.0/24", "in-scope.example.com"])
    assert "/scripts/scope_filter.py" in argv
    assert "/scripts/log_evidence.py" in argv  # logging still on
    env = [a for a in argv if a.startswith("SERENADE_SCOPE_HOSTS=")]
    assert env, "scope hosts must be injected via trusted env var"
    assert "10.0.0.0/24" in env[0]
    assert "in-scope.example.com" in env[0]


def test_proxy_argv_keeps_image_and_mode():
    argv = _argv_for(["10.0.0.0/24"])
    assert "mitmdump" in argv
    assert "regular@8080" in argv
