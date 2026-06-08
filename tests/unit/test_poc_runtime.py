from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from agents.poc_runtime import (
    PoCDispatcher,
    PoCExecutionContext,
    PoCLibrary,
    PoCPrecondition,
    PreconditionEvaluator,
)
from sandbox.runner import SandboxResult, SandboxRunner

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _exploit_body(success: bool = True) -> str:
    """Returns a PoC exploit.py that uses the new sandbox-based contract."""
    return f"""\
from agents.poc_runtime import PoCExecutionContext, PoCResult, SandboxInvocation
from sandbox.runner import SandboxResult

def build_invocation(params, ctx):
    return SandboxInvocation(argv=["echo", params["target_url"]], timeout_seconds=10)

def parse_result(sandbox_result, params, ctx):
    text = sandbox_result.stdout.decode("utf-8", errors="replace").strip()
    return PoCResult(
        outcome={"'success'" if success else "'failure'"},
        summary=f"echo returned: {{text}}",
    )
"""


def _not_implemented_body() -> str:
    return """\
def build_invocation(params, ctx):
    raise NotImplementedError
def parse_result(sandbox_result, params, ctx):
    raise NotImplementedError
"""


def _write_poc(
    root: Path,
    *,
    id: str,
    spdx: str,
    auto_allowed: bool = True,
    requires_approval: bool = False,
    destructive: bool = False,
    preconditions: list[dict] | None = None,
    body: str | None = None,
) -> None:
    d = root / id
    d.mkdir(parents=True)
    (d / "test").mkdir()
    manifest = {
        "id": id,
        "provenance": [{"source": "cve_writeup", "ref": "https://example.com"}],
        "target": {"service": "demo", "version_range": ">=1.0,<2.0", "platforms": ["linux"]},
        "parameters": {"schema_ref": "./parameters.schema.json"},
        "risk": {
            "class": "exploit",
            "destructive": destructive,
            "modifies_target": False,
            "modifies_filesystem": False,
            "network_egress_from_target": False,
            "detection_likelihood": "low",
        },
        "auto_run": {
            "allowed": auto_allowed,
            "requires_approval": requires_approval,
            "preconditions": preconditions or [],
        },
        "rollback": {"available": False, "notes": ""},
        "license": {"spdx": spdx, "attribution": "test"},
        "ci": {"test_command": "pytest", "lab": ""},
    }
    (d / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["target_url"],
        "properties": {"target_url": {"type": "string"}},
    }
    (d / "parameters.schema.json").write_text(json.dumps(schema))
    (d / "exploit.py").write_text(body or _exploit_body(success=True))


class _MockSandbox:
    """Stand-in for SandboxRunner that returns canned SandboxResults."""

    def __init__(self, exit_code: int = 0, stdout: bytes = b"ok") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.calls: list[list[str]] = []

    async def run(self, argv, **kwargs):
        from datetime import datetime

        self.calls.append(list(argv))
        return SandboxResult(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=b"",
            duration_seconds=0.01,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )


def _ctx(poc_id: str = "DEMO") -> PoCExecutionContext:
    return PoCExecutionContext(
        engagement_id=uuid4(),
        poc_id=poc_id,
        manifest_sha256="abc",
        target_host="10.0.0.5",
    )


# --------------------------------------------------------------------------- #
# Library tests                                                               #
# --------------------------------------------------------------------------- #


def test_library_loads_valid_poc(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-001", spdx="MIT")
    lib = PoCLibrary(tmp_path)
    assert lib.get("DEMO-001") is not None
    assert len(lib.all()) == 1


def test_library_skips_unknown_license(tmp_path: Path):
    _write_poc(tmp_path, id="BAD-001", spdx="UNKNOWN")
    lib = PoCLibrary(tmp_path)
    assert lib.get("BAD-001") is None
    assert lib.all() == []


def test_library_skips_underscore_prefixed_dirs(tmp_path: Path):
    _write_poc(tmp_path, id="_template", spdx="UNKNOWN")
    lib = PoCLibrary(tmp_path)
    assert lib.all() == []


# --------------------------------------------------------------------------- #
# Dispatcher gate tests                                                       #
# --------------------------------------------------------------------------- #


def test_can_auto_run_blocks_destructive_without_optin(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-004", spdx="MIT", destructive=True)
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, SandboxRunner())
    poc = lib.get("DEMO-004")
    assert poc is not None
    ok, reason = disp.can_auto_run(poc, {"destructive_opt_in": False})
    assert ok is False
    assert "destructive" in reason
    ok, _ = disp.can_auto_run(poc, {"destructive_opt_in": True})
    assert ok is True


def test_real_shipped_destructive_fixture_is_gated():
    """Load the REAL shipped destructive PoC via the production PoCLibrary and
    assert the gate fires against it — not a synthetic tmp_path manifest.

    tracking issue #112, P2 #3: the `destructive_without_optin` gate was dead
    code (nothing shipped set risk.destructive=true). This exercises it against
    the real ``content/pocs/destructive-gate-fixture`` entry.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    lib = PoCLibrary(repo_root / "content" / "pocs")
    poc = lib.get("destructive-gate-fixture")
    assert poc is not None, "shipped destructive-gate-fixture did not load"
    assert poc.manifest.risk.destructive is True
    assert poc.manifest.risk.class_.value == "destructive"

    disp = PoCDispatcher(lib, SandboxRunner())

    # Normal engagement (no opt-in) → blocked with the live reason string.
    for engagement in ({"mode": "autonomous"}, {"destructive_opt_in": False}):
        ok, reason = disp.can_auto_run(poc, engagement)
        assert ok is False
        assert reason == "destructive_without_optin"

    # Operator opt-in (PoC path only) → the destructive gate clears.
    ok, reason = disp.can_auto_run(poc, {"destructive_opt_in": True})
    assert ok is True
    assert reason == "ok"


def test_engagement_context_built_optin_clears_shipped_fixture():
    """End-to-end: an EngagementRow with destructive_opt_in=True flows through
    build_engagement_context into the dispatcher gate for the REAL fixture."""
    from datetime import UTC, datetime, timedelta

    from orchestrator.models import EngagementRow
    from policy.engagement_dict import build_engagement_context

    repo_root = Path(__file__).resolve().parent.parent.parent
    lib = PoCLibrary(repo_root / "content" / "pocs")
    poc = lib.get("destructive-gate-fixture")
    assert poc is not None
    disp = PoCDispatcher(lib, SandboxRunner())

    now = datetime.now(UTC)

    def _row(opt_in: bool) -> EngagementRow:
        return EngagementRow(
            id=uuid4(),
            name="t",
            scope={"cidrs": [], "domains": [], "urls": []},
            objectives=[],
            mode="autonomous",
            time_window_start=now,
            time_window_end=now + timedelta(hours=1),
            token_budget=1,
            wallclock_hours=1,
            authorization_artifact_sha256="",
            per_tool_token_budget={},
            status="draft",
            current_stage="scope_intake",
            destructive_opt_in=opt_in,
        )

    blocked = build_engagement_context(_row(opt_in=False))
    ok, reason = disp.can_auto_run(poc, blocked)
    assert ok is False
    assert reason == "destructive_without_optin"

    allowed = build_engagement_context(_row(opt_in=True))
    ok, reason = disp.can_auto_run(poc, allowed)
    assert ok is True


@pytest.mark.asyncio
async def test_real_shipped_destructive_fixture_never_auto_runs_by_default():
    """dispatch() of the REAL shipped destructive fixture must NOT execute it
    for a default (no opt-in) engagement: can_auto_run blocks before any
    sandbox call, and a destructive PoC is reported, not run."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    lib = PoCLibrary(repo_root / "content" / "pocs")
    mock = _MockSandbox()
    disp = PoCDispatcher(lib, mock)  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="destructive-gate-fixture",
        params={"target_url": "http://10.0.0.5"},
        ctx=_ctx("destructive-gate-fixture"),
        engagement={"mode": "autonomous"},  # no destructive_opt_in
    )
    assert res.outcome == "skipped"
    assert res.error_message == "destructive_without_optin"
    # The sandbox was never invoked — the destructive PoC was not executed.
    assert mock.calls == []


def test_can_auto_run_blocks_when_requires_approval(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-005", spdx="MIT", requires_approval=True)
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, SandboxRunner())
    poc = lib.get("DEMO-005")
    assert poc is not None
    ok, reason = disp.can_auto_run(poc, {"destructive_opt_in": True})
    assert ok is False
    assert "requires_approval" in reason


def test_can_auto_run_blocks_when_disallowed(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-006", spdx="MIT", auto_allowed=False)
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, SandboxRunner())
    poc = lib.get("DEMO-006")
    assert poc is not None
    ok, reason = disp.can_auto_run(poc, {})
    assert ok is False
    assert "allowed=false" in reason


# --------------------------------------------------------------------------- #
# Dispatcher.dispatch tests — audit-fix #2 / #3 / #4                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dispatch_runs_via_sandbox(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-002", spdx="MIT")
    lib = PoCLibrary(tmp_path)
    mock = _MockSandbox(stdout=b"hello")
    disp = PoCDispatcher(lib, mock)  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="DEMO-002",
        params={"target_url": "http://x"},
        ctx=_ctx("DEMO-002"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "success"
    assert "hello" in res.summary
    # Sandbox was invoked with the argv from build_invocation
    assert mock.calls == [["echo", "http://x"]]


@pytest.mark.asyncio
async def test_dispatch_enforces_can_auto_run(tmp_path: Path):
    """Audit-fix #2: dispatch must enforce can_auto_run, not just expose it."""
    _write_poc(tmp_path, id="DEMO-008", spdx="MIT", destructive=True)
    lib = PoCLibrary(tmp_path)
    mock = _MockSandbox()
    disp = PoCDispatcher(lib, mock)  # type: ignore[arg-type]
    # Destructive PoC + no destructive opt-in (the engagement dict never
    # carries one now) — can_auto_run blocks it, so the sandbox is not called.
    res = await disp.dispatch(
        poc_id="DEMO-008",
        params={"target_url": "http://x"},
        ctx=_ctx("DEMO-008"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "skipped"
    assert "destructive" in (res.error_message or "")
    assert mock.calls == []


@pytest.mark.asyncio
async def test_dispatch_enforces_preconditions(tmp_path: Path):
    """Audit-fix #3: dispatcher evaluates structured preconditions."""
    # Precondition requires legacy "aggressive" (normalizes to autonomous);
    # the engagement is guided, so the predicate fails and dispatch skips.
    _write_poc(
        tmp_path,
        id="DEMO-009",
        spdx="MIT",
        preconditions=[{"engagement_mode_in": ["aggressive"]}],
    )
    lib = PoCLibrary(tmp_path)
    mock = _MockSandbox()
    disp = PoCDispatcher(lib, mock)  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="DEMO-009",
        params={"target_url": "http://x"},
        ctx=_ctx("DEMO-009"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "skipped"
    assert "precondition" in (res.error_message or "")
    assert mock.calls == []


@pytest.mark.asyncio
async def test_dispatch_rejects_invalid_params(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-003", spdx="MIT")
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, _MockSandbox())  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="DEMO-003",
        params={"WRONG": "x"},
        ctx=_ctx("DEMO-003"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "error"
    assert "invalid" in res.summary.lower()


@pytest.mark.asyncio
async def test_dispatch_handles_unknown_poc(tmp_path: Path):
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, _MockSandbox())  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="MISSING",
        params={},
        ctx=_ctx("MISSING"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "error"
    assert res.error_message == "manifest_not_loaded"


@pytest.mark.asyncio
async def test_dispatch_handles_not_implemented(tmp_path: Path):
    _write_poc(tmp_path, id="DEMO-007", spdx="MIT", body=_not_implemented_body())
    lib = PoCLibrary(tmp_path)
    disp = PoCDispatcher(lib, _MockSandbox())  # type: ignore[arg-type]
    res = await disp.dispatch(
        poc_id="DEMO-007",
        params={"target_url": "http://x"},
        ctx=_ctx("DEMO-007"),
        engagement={"mode": "guided"},
    )
    assert res.outcome == "skipped"
    assert res.error_message == "not_implemented"


# --------------------------------------------------------------------------- #
# Precondition evaluator                                                       #
# --------------------------------------------------------------------------- #


def test_precondition_empty_fails_closed():
    ev = PreconditionEvaluator()
    ok, reason = ev.evaluate([PoCPrecondition()], engagement={"mode": "guided"}, target_host="x")
    assert ok is False
    assert "empty" in reason


def test_precondition_target_no_exploit_list():
    ev = PreconditionEvaluator()
    pcs = [PoCPrecondition(target_not_in_no_exploit_list=True)]
    eng = {"mode": "guided", "no_exploit_list": ["prod-db-1", "prod-db-2"]}
    ok, _ = ev.evaluate(pcs, engagement=eng, target_host="prod-app-1")
    assert ok is True
    ok, reason = ev.evaluate(pcs, engagement=eng, target_host="prod-db-1")
    assert ok is False
    assert "no_exploit_list" in reason


def test_precondition_finding_cve():
    ev = PreconditionEvaluator()
    pcs = [PoCPrecondition(world_state_has_finding_with_cve="CVE-2021-44228")]
    ok, _ = ev.evaluate(
        pcs,
        engagement={},
        target_host="x",
        world_state_findings_cves=["CVE-2021-44228"],
    )
    assert ok is True
    ok, _ = ev.evaluate(
        pcs,
        engagement={},
        target_host="x",
        world_state_findings_cves=["CVE-2020-1472"],
    )
    assert ok is False


def test_precondition_destructive_opt_in_never_matches():
    """destructive is finding-only now: the engagement-context dict no longer
    carries a `destructive_opt_in` key, so a manifest precondition that
    requires `engagement_destructive_opt_in: true` can never be satisfied."""
    ev = PreconditionEvaluator()
    pcs = [PoCPrecondition(engagement_destructive_opt_in=True)]
    # A realistic engagement dict (no destructive_opt_in key) → fails closed.
    ok, _reason = ev.evaluate(pcs, engagement={"mode": "autonomous"}, target_host="x")
    assert ok is False
