"""R3 Windows-worker hardening tests (Phase 32).

Covers:
- assembly HMAC signing + the orchestrator-side enforcement
- heartbeat pre-flight (stale -> refuses dispatch)
- isolation policy gate (none + high-risk -> refused)
- mimikatz output parser
- mimikatz tool surfaces the new not-ready / unsafe-isolation paths cleanly
"""

from __future__ import annotations

import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

os.environ.setdefault("SERENADE_AUTH_ALLOW_DEV_FALLBACK", "1")


def _enable_worker(monkeypatch):
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 24)


# --------------------------------------------------------------------------- #
# Assembly HMAC                                                                #
# --------------------------------------------------------------------------- #


def test_sign_assembly_refuses_without_key(monkeypatch):
    from orchestrator.windows_worker import (
        WindowsWorkerSigningKeyMissing,
        sign_assembly,
    )

    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_SIGNING_KEY", raising=False)
    with pytest.raises(WindowsWorkerSigningKeyMissing):
        sign_assembly(b"\x4d\x5a\x90")


def test_sign_assembly_refuses_short_key(monkeypatch):
    from orchestrator.windows_worker import (
        WindowsWorkerSigningKeyMissing,
        sign_assembly,
    )

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_SIGNING_KEY", "tooshort")
    with pytest.raises(WindowsWorkerSigningKeyMissing):
        sign_assembly(b"\x4d\x5a\x90")


def test_sign_assembly_matches_hmac(monkeypatch):
    from orchestrator.windows_worker import sign_assembly

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_SIGNING_KEY", "x" * 32)
    sig = sign_assembly(b"hello world")
    expected = hmac.new(b"x" * 32, b"hello world", sha256).hexdigest()
    assert sig == expected


# --------------------------------------------------------------------------- #
# Heartbeat pre-flight                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_assert_worker_ready_passes_with_fresh_heartbeat(monkeypatch):
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue

    q = WindowsJobQueue()
    fresh = json.dumps(
        {"ts": datetime.now(UTC).isoformat(), "isolation": "hyperv-snapshot"}
    )
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=fresh)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    # Should not raise.
    await q.assert_worker_ready("mimikatz")


@pytest.mark.asyncio
async def test_assert_worker_ready_refuses_stale(monkeypatch):
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue, WindowsWorkerNotReady

    q = WindowsJobQueue()
    stale_iso = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=json.dumps({"ts": stale_iso, "isolation": "hyperv-snapshot"}))
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    with pytest.raises(WindowsWorkerNotReady):
        await q.assert_worker_ready("mimikatz")


@pytest.mark.asyncio
async def test_assert_worker_ready_refuses_missing_heartbeat(monkeypatch):
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue, WindowsWorkerNotReady

    q = WindowsJobQueue()
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=None)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    with pytest.raises(WindowsWorkerNotReady):
        await q.assert_worker_ready("shell")


@pytest.mark.asyncio
async def test_heartbeat_accepts_legacy_iso_string(monkeypatch):
    """Phase 29 wrote bare ISO strings; payload-aware reader keeps working."""
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue

    q = WindowsJobQueue()
    iso = datetime.now(UTC).isoformat()
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=iso)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    payload = await q.heartbeat_payload()
    assert payload is not None
    assert payload["ts"] == iso
    assert payload["isolation"] == "unknown"


# --------------------------------------------------------------------------- #
# Isolation policy gate                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_isolation_none_refuses_high_risk(monkeypatch):
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import (
        WindowsJobQueue,
        WindowsWorkerUnsafeIsolation,
    )

    q = WindowsJobQueue()
    payload = json.dumps({"ts": datetime.now(UTC).isoformat(), "isolation": "none"})
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=payload)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION", raising=False)
    with pytest.raises(WindowsWorkerUnsafeIsolation):
        await q.assert_worker_ready("mimikatz")
    with pytest.raises(WindowsWorkerUnsafeIsolation):
        await q.assert_worker_ready("assembly")


@pytest.mark.asyncio
async def test_isolation_none_allows_low_risk(monkeypatch):
    """A `shell` job is not on the high-risk list; isolation=none is permitted."""
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue

    q = WindowsJobQueue()
    payload = json.dumps({"ts": datetime.now(UTC).isoformat(), "isolation": "none"})
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=payload)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION", raising=False)
    # Should not raise — `shell` isn't in _HIGH_RISK_KINDS.
    await q.assert_worker_ready("shell")


@pytest.mark.asyncio
async def test_isolation_none_allowed_with_opt_in(monkeypatch):
    _enable_worker(monkeypatch)
    from orchestrator.windows_worker import WindowsJobQueue

    q = WindowsJobQueue()
    payload = json.dumps({"ts": datetime.now(UTC).isoformat(), "isolation": "none"})
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=payload)
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION", "1")
    # Should not raise once opted in.
    await q.assert_worker_ready("mimikatz")


# --------------------------------------------------------------------------- #
# submit_assembly carries the signature                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_submit_assembly_includes_hmac(monkeypatch):
    _enable_worker(monkeypatch)
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_SIGNING_KEY", "k" * 32)
    from orchestrator.windows_worker import WindowsJobQueue

    q = WindowsJobQueue()
    pushed: dict = {}

    async def fake_rpush(key, value):
        pushed["value"] = value

    async def fake_brpop(keys, timeout):
        return (
            keys[0],
            json.dumps(
                {
                    "job_id": "x",
                    "exit_code": 0,
                    "stdout_b64": "",
                    "stderr_b64": "",
                    "started_at": "2026-05-20T18:00:00+00:00",
                    "ended_at": "2026-05-20T18:00:01+00:00",
                }
            ),
        )

    payload = json.dumps({"ts": datetime.now(UTC).isoformat(), "isolation": "hyperv-snapshot"})
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=payload)
    fake.rpush = fake_rpush
    fake.brpop = fake_brpop
    fake.aclose = AsyncMock()

    async def factory():
        return fake

    monkeypatch.setattr(q, "_redis", factory)
    await q.submit_assembly(uuid4(), assembly_bytes=b"\x4d\x5a\x90", args=["a"])
    job = json.loads(pushed["value"])
    assert job["kind"] == "assembly"
    assert job["assembly_sig"] == hmac.new(b"k" * 32, b"\x4d\x5a\x90", sha256).hexdigest()


# --------------------------------------------------------------------------- #
# mimikatz output parser                                                       #
# --------------------------------------------------------------------------- #


def test_parse_mimikatz_extracts_logonpasswords():
    from tools.mimikatz_via_worker import parse_mimikatz_output

    sample = """
Authentication Id : 0 ; 12345 (00000000:00003039)
Session           : Interactive from 1
User Name         : Administrator

	msv :
	 [00000003] Primary
	 * Username : Administrator
	 * Domain   : CORP
	 * NTLM     : aad3b435b51404eeaad3b435b51404ee
	 * SHA1     : da39a3ee5e6b4b0d3255bfef95601890afd80709
	tspkg :
	 * Username : Administrator
	 * Domain   : CORP
	 * Password : Hunter2!

	wdigest :
	 * Username : svc_sql
	 * Domain   : CORP
	 * Password : verystrongP@ss
"""
    rows = parse_mimikatz_output(sample)
    usernames = sorted([r.username for r in rows])
    assert "Administrator" in usernames
    assert "svc_sql" in usernames
    admin = next(r for r in rows if r.username == "Administrator")
    assert admin.domain == "CORP"
    # Either the NTLM or the password should be captured (or both, deduped).
    # Our dedupe collapses to one row per (username, domain, ntlm).
    sql = next(r for r in rows if r.username == "svc_sql")
    assert sql.password == "verystrongP@ss"


def test_parse_mimikatz_skips_null_users():
    from tools.mimikatz_via_worker import parse_mimikatz_output

    sample = """
	msv :
	 * Username : (null)
	 * Domain   : (null)
	 * NTLM     : 00000000000000000000000000000000
"""
    assert parse_mimikatz_output(sample) == []


def test_parse_mimikatz_empty_input():
    from tools.mimikatz_via_worker import parse_mimikatz_output

    assert parse_mimikatz_output("") == []
    assert parse_mimikatz_output("no creds here") == []


# --------------------------------------------------------------------------- #
# mimikatz tool's not-ready / unsafe-isolation handling                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mimikatz_tool_surfaces_worker_not_ready(monkeypatch):
    _enable_worker(monkeypatch)
    from tools.base import ToolRunContext
    from tools.mimikatz_via_worker import MimikatzViaWorker, MimikatzViaWorkerInput

    # No heartbeat -> assert_worker_ready raises -> propagates through
    # submit_mimikatz -> tool catches and returns -102.
    from orchestrator import windows_worker as ww

    async def fake_assert(self, kind):
        raise ww.WindowsWorkerNotReady("stale")

    monkeypatch.setattr(ww.WindowsJobQueue, "assert_worker_ready", fake_assert)

    tool = MimikatzViaWorker()
    ctx = ToolRunContext(
        tool_run_id=uuid4(), engagement_id=uuid4(), sandbox_id="sb", timeout_seconds=60
    )
    result = await tool.run(MimikatzViaWorkerInput(target="10.0.0.5", args=["x"]), ctx)
    assert result.exit_code == -102
    assert "heartbeat" in result.output.summary.lower()


@pytest.mark.asyncio
async def test_mimikatz_tool_surfaces_unsafe_isolation(monkeypatch):
    _enable_worker(monkeypatch)
    from tools.base import ToolRunContext
    from tools.mimikatz_via_worker import MimikatzViaWorker, MimikatzViaWorkerInput

    from orchestrator import windows_worker as ww

    async def fake_assert(self, kind):
        raise ww.WindowsWorkerUnsafeIsolation("none")

    monkeypatch.setattr(ww.WindowsJobQueue, "assert_worker_ready", fake_assert)

    tool = MimikatzViaWorker()
    ctx = ToolRunContext(
        tool_run_id=uuid4(), engagement_id=uuid4(), sandbox_id="sb", timeout_seconds=60
    )
    result = await tool.run(MimikatzViaWorkerInput(target="10.0.0.5", args=["x"]), ctx)
    assert result.exit_code == -103
    assert "isolation" in result.output.summary.lower()
