"""EDB execution gating (issue #112 follow-up).

The whole tool is NOT destructive — non-destructive ExploitDB scripts run;
only DoS modules are gated per-module behind ``destructive_opt_in``. These
tests exercise the early-return guards in ``EdbExec.run()`` without standing
up a real sandbox (the executor is mocked).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tools import edb_exec as edb_mod
from tools.base import ToolRunContext
from tools.edb_exec import EdbExec


def _ctx(engagement: dict) -> ToolRunContext:
    return ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb-test",
        extra_env={"__engagement_json__": json.dumps(engagement)},
    )


def test_edb_is_not_destructive() -> None:
    # The whole tool must not be blanket-destructive (that would make the
    # policy gate deny every EDB call).
    assert EdbExec().destructive is False


@pytest.mark.asyncio
async def test_edb_skips_when_auto_exploit_disabled() -> None:
    inp = EdbExec.Input(edb_id=49112, target_host="10.0.0.5")
    res = await EdbExec().run(inp, _ctx({"edb_auto_exploit": False}))
    assert res.output.outcome == "skipped"
    assert res.output.error_message == "edb_auto_exploit_disabled"


@pytest.mark.asyncio
async def test_edb_dos_module_gated_without_optin(monkeypatch) -> None:
    # edb_auto_exploit on, but the module is DoS and the engagement did NOT
    # opt in to destructive → reported, not executed.
    monkeypatch.setattr(
        edb_mod.ToolExecutor,
        "run_and_capture",
        AsyncMock(return_value=(SimpleNamespace(stdout=b"{}", stderr=b"", exit_code=0), None, None)),
    )
    monkeypatch.setattr(
        edb_mod,
        "_parse_edb_meta",
        lambda *_a, **_k: {"type": "dos", "title": "t", "path": "/x.py", "edb_id": "1"},
    )
    inp = EdbExec.Input(edb_id=1, target_host="10.0.0.5")
    res = await EdbExec().run(inp, _ctx({"edb_auto_exploit": True, "destructive_opt_in": False}))
    assert res.output.outcome == "skipped"
    assert res.output.error_message == "destructive_without_optin"


@pytest.mark.asyncio
async def test_edb_destructive_script_gated_without_optin(monkeypatch) -> None:
    # A non-DoS script whose body removes files is destructive: gated unless
    # the engagement opted in. The meta fetch returns the type, then the
    # script-body read returns destructive content.
    calls = {"n": 0}

    async def fake_run_and_capture(self, argv, **_kw):
        calls["n"] += 1
        # 1st call: searchsploit meta; 2nd: `cat <script>` body scan.
        body = b"{}" if calls["n"] == 1 else b"import os\nos.system('rm -rf /var/www')\n"
        return (SimpleNamespace(stdout=body, stderr=b"", exit_code=0), None, None)

    monkeypatch.setattr(edb_mod.ToolExecutor, "run_and_capture", fake_run_and_capture)
    monkeypatch.setattr(
        edb_mod,
        "_parse_edb_meta",
        lambda *_a, **_k: {"type": "remote", "title": "t", "path": "/x.py", "edb_id": "1"},
    )
    inp = EdbExec.Input(edb_id=1, target_host="10.0.0.5")
    res = await EdbExec().run(inp, _ctx({"edb_auto_exploit": True, "destructive_opt_in": False}))
    assert res.output.outcome == "skipped"
    assert res.output.error_message == "destructive_without_optin"


@pytest.mark.asyncio
async def test_edb_unreadable_body_fails_closed(monkeypatch) -> None:
    # If the script body can't be read (cat nonzero/empty), the scan can't
    # prove it non-destructive → fail CLOSED (report, don't run) without opt-in.
    calls = {"n": 0}

    async def fake_run_and_capture(self, argv, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return (SimpleNamespace(stdout=b"{}", stderr=b"", exit_code=0), None, None)
        return (SimpleNamespace(stdout=b"", stderr=b"no such file", exit_code=1), None, None)

    monkeypatch.setattr(edb_mod.ToolExecutor, "run_and_capture", fake_run_and_capture)
    monkeypatch.setattr(
        edb_mod,
        "_parse_edb_meta",
        lambda *_a, **_k: {"type": "remote", "title": "t", "path": "/x.py", "edb_id": "1"},
    )
    inp = EdbExec.Input(edb_id=1, target_host="10.0.0.5")
    res = await EdbExec().run(inp, _ctx({"edb_auto_exploit": True, "destructive_opt_in": False}))
    assert res.output.outcome == "skipped"
    assert res.output.error_message == "destructive_scan_unreadable"


@pytest.mark.asyncio
async def test_edb_dos_module_runs_with_optin(monkeypatch) -> None:
    # Same DoS module, but the engagement opted in → it is NOT gated as
    # destructive; it proceeds to execution (mocked sandbox).
    monkeypatch.setattr(
        edb_mod.ToolExecutor,
        "run_and_capture",
        AsyncMock(return_value=(SimpleNamespace(stdout=b"done", stderr=b"", exit_code=0), None, None)),
    )
    monkeypatch.setattr(
        edb_mod,
        "_parse_edb_meta",
        lambda *_a, **_k: {"type": "dos", "title": "t", "path": "/x.py", "edb_id": "1"},
    )
    inp = EdbExec.Input(edb_id=1, target_host="10.0.0.5")
    res = await EdbExec().run(inp, _ctx({"edb_auto_exploit": True, "destructive_opt_in": True}))
    assert res.output.error_message != "destructive_without_optin"
