"""Tests for the per-engagement sandbox network naming (audit-fix #18)."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

import sandbox.network as network_mod
from sandbox.network import engagement_network_name, teardown_engagement_network


def test_name_format():
    # Verification audit #5 fix: full UUID, no truncation, no collision risk.
    name = engagement_network_name(UUID("12345678-1234-5678-1234-567812345678"))
    assert name == "serenade-sb-12345678-1234-5678-1234-567812345678"


def test_name_collision_resistance():
    """Two engagements sharing an 8-char prefix must produce distinct names."""
    a = engagement_network_name(UUID("12345678-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))
    b = engagement_network_name(UUID("12345678-bbbb-bbbb-bbbb-bbbbbbbbbbbb"))
    assert a != b


def test_name_is_deterministic():
    eid = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert engagement_network_name(eid) == engagement_network_name(eid)


# --- async-03/10: per-name lock dict must be bounded (evicted on teardown) ---


def test_lock_dict_is_plain_dict_not_unbounded_defaultdict():
    # A defaultdict would silently create-and-retain a Lock on every read.
    assert not hasattr(network_mod._name_locks, "default_factory")


def test_lock_created_lazily_and_loop_safe():
    name = "serenade-sb-lazy"
    network_mod._name_locks.pop(name, None)
    assert name not in network_mod._name_locks

    async def _go() -> None:
        lock = network_mod._lock_for(name)
        assert isinstance(lock, asyncio.Lock)
        # Same object on a second call (no duplicate creation).
        assert network_mod._lock_for(name) is lock

    asyncio.run(_go())
    network_mod._name_locks.pop(name, None)


@pytest.mark.asyncio
async def test_teardown_evicts_lock_entry(monkeypatch):
    """teardown_engagement_network must pop its per-name lock so the dict can't
    grow without bound across the worker's lifetime."""

    async def _fake_exec(*args, **kwargs):
        class _P:
            returncode = 0

            async def wait(self):
                return 0

            async def communicate(self):
                return b"", b""

        return _P()

    monkeypatch.setattr(network_mod.asyncio, "create_subprocess_exec", _fake_exec)

    eid = uuid4()
    name = engagement_network_name(eid)
    # Force-create the lock entry as ensure_* would.
    network_mod._lock_for(name)
    assert name in network_mod._name_locks

    await teardown_engagement_network(eid)
    assert name not in network_mod._name_locks  # evicted
