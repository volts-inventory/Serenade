"""Fixture self-test: the destructive-gate fixture is classified destructive
and is gated by the destructive opt-in, never auto-run by default."""

from __future__ import annotations

from pathlib import Path

from agents.poc_runtime import PoCDispatcher, PoCLibrary
from sandbox.runner import SandboxRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
POCS_DIR = REPO_ROOT / "content" / "pocs"


def test_fixture_is_destructive_and_gated() -> None:
    lib = PoCLibrary(POCS_DIR)
    poc = lib.get("destructive-gate-fixture")
    assert poc is not None
    assert poc.manifest.risk.destructive is True
    assert poc.manifest.risk.class_.value == "destructive"

    disp = PoCDispatcher(lib, SandboxRunner())
    ok, reason = disp.can_auto_run(poc, {"mode": "autonomous"})
    assert ok is False
    assert reason == "destructive_without_optin"

    ok, reason = disp.can_auto_run(poc, {"destructive_opt_in": True})
    assert ok is True
    assert reason == "ok"
