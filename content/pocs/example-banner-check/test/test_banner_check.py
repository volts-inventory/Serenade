"""CI test for example-banner-check.

Validates the pure functions (`build_invocation` / `parse_result`) without
spawning a real sandbox. A future integration test brings the sandbox up
against a local netcat-served port.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Load the exploit module directly.
_MOD_PATH = Path(__file__).resolve().parent.parent / "exploit.py"
_spec = importlib.util.spec_from_file_location("example_banner_check_exploit", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
exploit = importlib.util.module_from_spec(_spec)
sys.modules["example_banner_check_exploit"] = exploit
_spec.loader.exec_module(exploit)

from agents.poc_runtime import PoCExecutionContext  # noqa: E402
from sandbox.runner import SandboxResult  # noqa: E402


def _ctx() -> PoCExecutionContext:
    return PoCExecutionContext(
        engagement_id=uuid4(),
        poc_id="example-banner-check",
        manifest_sha256="x" * 64,
        target_host="10.0.0.5",
    )


def _sr(exit_code: int, stderr: bytes = b"") -> SandboxResult:
    now = datetime.now(timezone.utc)
    return SandboxResult(
        exit_code=exit_code,
        stdout=b"",
        stderr=stderr,
        duration_seconds=0.01,
        started_at=now,
        ended_at=now,
    )


def test_build_invocation_emits_nc_argv():
    inv = exploit.build_invocation({"host": "10.0.0.5", "port": 22, "timeout_seconds": 3}, _ctx())
    assert inv.argv == ["nc", "-nvz", "-w", "3", "10.0.0.5", "22"]


def test_build_invocation_defaults_timeout():
    inv = exploit.build_invocation({"host": "x", "port": 80}, _ctx())
    assert "-w" in inv.argv
    assert "5" in inv.argv


def test_parse_success():
    r = exploit.parse_result(_sr(0, b"Connection to 10.0.0.5 22 port [tcp/ssh] succeeded!"), {}, _ctx())
    assert r.outcome == "success"
    assert "succeeded" in r.summary


def test_parse_failure_connection_refused():
    r = exploit.parse_result(_sr(1, b"nc: connect to 10.0.0.5 port 22 (tcp) failed: Connection refused"), {}, _ctx())
    assert r.outcome == "failure"
    assert r.error_message == "nc_exit_1"


def test_parse_timeout():
    r = exploit.parse_result(_sr(124, b"timeout"), {}, _ctx())
    assert r.outcome == "failure"
    assert r.error_message == "timeout"
