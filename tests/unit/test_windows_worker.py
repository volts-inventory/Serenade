"""Tests for F10 Path B — orchestrator-side Windows worker queue."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

os.environ.setdefault("SERENADE_AUTH_ALLOW_DEV_FALLBACK", "1")


def test_is_enabled_false_by_default(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_ENABLED", raising=False)
    assert windows_worker.is_enabled() is False


def test_is_enabled_true_when_set(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    assert windows_worker.is_enabled() is True


def test_queue_refuses_without_feature(monkeypatch):
    from orchestrator.windows_worker import WindowsJobQueue, WindowsJobQueueDisabled

    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_ENABLED", raising=False)
    with pytest.raises(WindowsJobQueueDisabled):
        WindowsJobQueue()


def test_queue_refuses_without_auth_token(monkeypatch):
    from orchestrator.windows_worker import WindowsJobAuthMissing, WindowsJobQueue

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", raising=False)
    with pytest.raises(WindowsJobAuthMissing):
        WindowsJobQueue()


def test_queue_refuses_short_auth_token(monkeypatch):
    from orchestrator.windows_worker import WindowsJobAuthMissing, WindowsJobQueue

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "too-short")
    with pytest.raises(WindowsJobAuthMissing):
        WindowsJobQueue()


def test_queue_accepts_well_configured(monkeypatch):
    from orchestrator.windows_worker import WindowsJobQueue

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    q = WindowsJobQueue()
    assert q.auth_token == "x" * 20
    assert q.job_queue == "serenade:windows_worker:jobs"


@pytest.mark.asyncio
async def test_submit_mimikatz_pushes_and_pops(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    q = windows_worker.WindowsJobQueue()

    fake_redis = AsyncMock()
    pushed_payload: dict[str, str] = {}

    async def fake_rpush(key, value):
        pushed_payload["key"] = key
        pushed_payload["value"] = value

    async def fake_brpop(keys, timeout):
        result = {
            "job_id": "fake",
            "exit_code": 0,
            "stdout_b64": base64.b64encode(b"hello\n").decode(),
            "stderr_b64": base64.b64encode(b"").decode(),
            "started_at": "2026-05-20T18:00:00+00:00",
            "ended_at": "2026-05-20T18:00:01+00:00",
        }
        return (keys[0], json.dumps(result))

    # R3: pre-flight heartbeat check requires a fresh heartbeat payload.
    from datetime import UTC, datetime

    fresh_hb = json.dumps(
        {"ts": datetime.now(UTC).isoformat(), "isolation": "hyperv-snapshot"}
    )
    fake_redis.get = AsyncMock(return_value=fresh_hb)
    fake_redis.rpush = fake_rpush
    fake_redis.brpop = fake_brpop
    fake_redis.aclose = AsyncMock()

    async def fake_redis_factory():
        return fake_redis

    monkeypatch.setattr(q, "_redis", fake_redis_factory)

    result = await q.submit_mimikatz(
        engagement_id=uuid4(), args=["sekurlsa::logonpasswords"], timeout_seconds=30
    )
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.timed_out is False

    # Verify the pushed job carries the auth token and matches the schema.
    job = json.loads(pushed_payload["value"])
    assert job["auth_token"] == "x" * 20
    assert job["kind"] == "mimikatz"
    assert job["args"] == ["sekurlsa::logonpasswords"]
    assert job["timeout_seconds"] == 30


@pytest.mark.asyncio
async def test_submit_timeout_returns_synthesized_result(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    q = windows_worker.WindowsJobQueue()

    from datetime import UTC, datetime

    fake_redis = AsyncMock()
    fresh_hb = json.dumps(
        {"ts": datetime.now(UTC).isoformat(), "isolation": "hyperv-snapshot"}
    )
    fake_redis.get = AsyncMock(return_value=fresh_hb)
    fake_redis.rpush = AsyncMock(return_value=None)
    fake_redis.brpop = AsyncMock(return_value=None)
    fake_redis.aclose = AsyncMock()

    async def fake_redis_factory():
        return fake_redis

    monkeypatch.setattr(q, "_redis", fake_redis_factory)

    result = await q.submit_mimikatz(engagement_id=uuid4(), args=["x"], timeout_seconds=1)
    assert result.timed_out is True
    assert result.exit_code == -99
    assert "timeout" in result.stderr


@pytest.mark.asyncio
async def test_submit_assembly_carries_b64(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    # R3: assembly submission requires the signing key now.
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_SIGNING_KEY", "k" * 32)
    q = windows_worker.WindowsJobQueue()

    from datetime import UTC, datetime

    fake_redis = AsyncMock()
    fresh_hb = json.dumps(
        {"ts": datetime.now(UTC).isoformat(), "isolation": "hyperv-snapshot"}
    )
    fake_redis.get = AsyncMock(return_value=fresh_hb)
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

    fake_redis.rpush = fake_rpush
    fake_redis.brpop = fake_brpop
    fake_redis.aclose = AsyncMock()

    async def fake_redis_factory():
        return fake_redis

    monkeypatch.setattr(q, "_redis", fake_redis_factory)
    await q.submit_assembly(engagement_id=uuid4(), assembly_bytes=b"\x4d\x5a\x90", args=["a", "b"])
    job = json.loads(pushed["value"])
    assert job["kind"] == "assembly"
    assert base64.b64decode(job["assembly_b64"]) == b"\x4d\x5a\x90"
    assert job["args"] == ["a", "b"]


@pytest.mark.asyncio
async def test_heartbeat_age(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    q = windows_worker.WindowsJobQueue()
    fake_redis = AsyncMock()
    iso = datetime.now(UTC).isoformat()
    fake_redis.get = AsyncMock(return_value=iso)
    fake_redis.aclose = AsyncMock()

    async def fake_redis_factory():
        return fake_redis

    monkeypatch.setattr(q, "_redis", fake_redis_factory)
    age = await q.heartbeat_age_seconds()
    assert age is not None
    assert 0 <= age < 5  # set just now


@pytest.mark.asyncio
async def test_heartbeat_none_when_missing(monkeypatch):
    from orchestrator import windows_worker

    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_ENABLED", "1")
    monkeypatch.setenv("SERENADE_WINDOWS_WORKER_AUTH_TOKEN", "x" * 20)
    q = windows_worker.WindowsJobQueue()
    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.aclose = AsyncMock()

    async def fake_redis_factory():
        return fake_redis

    monkeypatch.setattr(q, "_redis", fake_redis_factory)
    age = await q.heartbeat_age_seconds()
    assert age is None


@pytest.mark.asyncio
async def test_mimikatz_tool_disabled_path(monkeypatch):
    """The mimikatz tool returns a disabled-style result when feature is off."""
    monkeypatch.delenv("SERENADE_WINDOWS_WORKER_ENABLED", raising=False)
    from tools.mimikatz_via_worker import (
        MimikatzViaWorker,
        MimikatzViaWorkerInput,
    )
    from tools.base import ToolRunContext

    tool = MimikatzViaWorker()
    ctx = ToolRunContext(
        tool_run_id=uuid4(), engagement_id=uuid4(), sandbox_id="sb", timeout_seconds=60
    )
    result = await tool.run(MimikatzViaWorkerInput(target="10.0.0.5", args=["x"]), ctx)
    assert result.output.worker_disabled is True
    assert result.exit_code == -100


def test_mimikatz_extract_targets():
    from tools.mimikatz_via_worker import MimikatzViaWorker, MimikatzViaWorkerInput

    t = MimikatzViaWorker()
    assert t.extract_targets(MimikatzViaWorkerInput(target="10.0.0.5", args=["x"])) == ["10.0.0.5"]


def test_schema_file_present():
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "clients" / "windows_worker" / "serenade_windows_job.schema.json"
    schema = json.loads(p.read_text())
    assert schema["title"].startswith("Serenade")
    assert "mimikatz" in schema["properties"]["kind"]["enum"]


def test_powershell_agent_present():
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "clients" / "windows_worker" / "agent" / "agent.ps1"
    body = p.read_text()
    assert "BLPOP" in body
    assert "auth_token" in body
    assert "heartbeat" in body.lower()
