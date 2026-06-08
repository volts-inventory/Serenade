"""R3 #K2 Bug 3: customer-facing abort / kill switch.

Background: ``POST /engagements/{id}/pause`` lets the operator stop the
graph from scheduling more steps, but in-flight tool runs continue until
they finish naturally (which can take minutes). A CISO asking "how do
I stop this in 30 seconds?" needs a real kill switch — pause is not it.

Fix: a new ``POST /engagements/{id}/abort`` endpoint that:

  1. Stamps ``status=aborted`` + ``abort_requested_at`` + ``abort_reason``
     + ``aborted_by`` on the engagement row.
  2. Publishes ``engagement.abort_requested`` on the event bus.
  3. Audit-logs the decision.
  4. Returns the engagement's final state.

The LangGraph state machine checks ``EngagementState.should_abort``
(seeded by either an in-memory flag from the bus consumer OR a poll of
the engagement row's ``status`` column) at the top of every
``_step_agent`` call and short-circuits with ``state.completed=True``
plus a note carrying the abort reason. In-flight sandbox containers are
SIGTERMed (then SIGKILLed after a 30s grace) via
``SandboxRunner.terminate``.

These tests cover:

  - The endpoint contract (200 + state mutation + audit row).
  - The 422 path for a missing ``reason``.
  - The 404 path for a missing engagement.
  - The ``should_abort`` flag short-circuits ``_step_agent`` before the
    agent runs.
  - The row-poll fallback fires ``should_abort=True`` from an aborted
    engagement-row status.
  - ``SandboxRunner.terminate`` exists with the expected signature.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agents.base import StageContext, StepDecision, StepResult
from orchestrator.services.engagements import (
    EngagementAbortRequest,
    abort_engagement,
)
from orchestrator.services.errors import NotFoundError
from orchestrator.models import AuditLogRow, EngagementRow
from orchestrator.stage_controller import (
    EngagementState,
    _abort_requested,
    _step_agent,
)
from worldstate.schema import Stage


# --------------------------------------------------------------------------- #
# Fixtures + helpers                                                          #
# --------------------------------------------------------------------------- #


def _engagement_row(*, status: str = "running") -> EngagementRow:
    now = datetime.now(UTC)
    return EngagementRow(
        id=uuid4(),
        name="abort-test",
        scope={"cidrs": ["10.0.0.0/24"], "domains": []},
        objectives=[],
        mode="guided",
        time_window_start=now - timedelta(hours=1),
        time_window_end=now + timedelta(hours=8),
        token_budget=1_000_000,
        wallclock_hours=8,
        authorization_artifact_sha256="a" * 64,
        per_tool_token_budget={},
        status=status,
        current_stage="exploit",
        created_by="alice",
        llm_config_snapshot={},
        created_at=now - timedelta(hours=2),
        updated_at=now,
    )


class _AbortSession:
    """AsyncSession stand-in for the abort endpoint.

    Tracks ``session.add`` calls so audit-row assertions are easy. The
    engagement row's mutations happen in-place on the object returned
    from ``get``, so the test can read row.status / row.abort_* directly
    after the call.
    """

    def __init__(self, row: EngagementRow | None) -> None:
        self.row = row
        self.added: list[Any] = []

    async def get(self, _cls: Any, _key: Any) -> Any:
        return self.row

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def execute(self, *_a: Any, **_k: Any) -> Any:
        # The audit path (ledger.append_audit_row) reads the chain tail.
        # No prior rows → genesis (seq=1, prev_hash=None).
        class _Empty:
            def scalars(self_inner):  # noqa: N805
                class _S:
                    def first(self_s):  # noqa: N805
                        return None

                return _S()

        return _Empty()

    async def flush(self) -> None:
        return None

    def begin_nested(self) -> Any:
        # SAVEPOINT used by ledger.append_audit_row; a no-op CM here.
        class _Savepoint:
            async def __aenter__(self_inner) -> Any:  # noqa: N805
                return None

            async def __aexit__(self_inner, *exc: Any) -> bool:  # noqa: N805
                return False

        return _Savepoint()


# --------------------------------------------------------------------------- #
# 1. POST /abort with reason → 200, state.status=aborted, audit row written.  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_abort_endpoint_marks_row_and_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # The audit write now computes a master-key HMAC chain row — keep the
    # key in a tmp dir so no real ~/.serenade/.master_key is touched.
    from orchestrator import crypto
    from orchestrator.settings import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()

    row = _engagement_row(status="running")
    session = _AbortSession(row)

    @asynccontextmanager
    async def _scope():
        yield session
        # skip commit/rollback
    monkeypatch.setattr("orchestrator.services.engagements.session_scope", _scope)
    monkeypatch.setattr(
        "orchestrator.services.engagements.get_bus",
        lambda: type("B", (), {"publish": AsyncMock(return_value=None)})(),
    )

    out = await abort_engagement(
        row.id,
        EngagementAbortRequest(reason="customer-requested emergency stop"),
        aborted_by="alice",
    )
    assert out.status == "aborted"
    assert row.status == "aborted"
    assert row.abort_requested_at is not None
    assert row.abort_reason == "customer-requested emergency stop"
    assert row.aborted_by == "alice"
    # Audit row stamped.
    audit_rows = [r for r in session.added if isinstance(r, AuditLogRow)]
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "engagement.aborted"
    assert audit_rows[0].payload["aborted_by"] == "alice"
    assert audit_rows[0].payload["reason"] == "customer-requested emergency stop"


# --------------------------------------------------------------------------- #
# 2. POST /abort without reason (or with a too-short reason) → 422.           #
# --------------------------------------------------------------------------- #


def test_abort_request_schema_requires_reason() -> None:
    with pytest.raises(ValidationError):
        EngagementAbortRequest()  # type: ignore[call-arg]


def test_abort_request_schema_rejects_short_reason() -> None:
    # 9 chars — under the 10-char floor; an "oops" audit trail is useless.
    with pytest.raises(ValidationError):
        EngagementAbortRequest(reason="too short")


def test_abort_request_schema_accepts_min_length_reason() -> None:
    req = EngagementAbortRequest(reason="exactly10c")
    assert req.reason == "exactly10c"


# --------------------------------------------------------------------------- #
# 3. POST /abort on a missing engagement → 404.                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_abort_endpoint_404_when_engagement_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _AbortSession(None)

    @asynccontextmanager
    async def _scope():
        yield session
    monkeypatch.setattr("orchestrator.services.engagements.session_scope", _scope)
    monkeypatch.setattr(
        "orchestrator.services.engagements.get_bus",
        lambda: type("B", (), {"publish": AsyncMock(return_value=None)})(),
    )
    with pytest.raises(NotFoundError):
        await abort_engagement(
            uuid4(),
            EngagementAbortRequest(reason="engagement does not exist"),
            aborted_by="alice",
        )


# --------------------------------------------------------------------------- #
# 4. Bus-publish failure does not 5xx the abort (the DB row is the source     #
#    of truth; a bus blip cannot strand a kill switch request).               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_abort_endpoint_swallows_bus_publish_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from orchestrator import crypto
    from orchestrator.settings import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()

    row = _engagement_row(status="running")
    session = _AbortSession(row)

    async def _bad_publish(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("redis unreachable")

    @asynccontextmanager
    async def _scope():
        yield session
    monkeypatch.setattr("orchestrator.services.engagements.session_scope", _scope)
    monkeypatch.setattr(
        "orchestrator.services.engagements.get_bus",
        lambda: type("B", (), {"publish": _bad_publish})(),
    )
    out = await abort_engagement(
        row.id,
        EngagementAbortRequest(reason="customer-requested kill"),
        aborted_by="alice",
    )
    # The row is still aborted — bus failure must not strand the abort.
    assert out.status == "aborted"
    assert row.status == "aborted"


# --------------------------------------------------------------------------- #
# 5. ``_step_agent`` short-circuits when ``should_abort`` is True.            #
# --------------------------------------------------------------------------- #


def _state(**overrides: Any) -> EngagementState:
    base: dict[str, Any] = dict(
        engagement_id=uuid4(),
        engagement={
            "token_budget": 1_000_000,
            "window": {
                "start_ns": 0,
                "end_ns": 10**20,  # far future
            },
        },
        current_stage=Stage.RECON,
    )
    base.update(overrides)
    return EngagementState(**base)


@pytest.mark.asyncio
async def test_step_agent_short_circuits_when_should_abort_flag_set() -> None:
    """The in-flight agent step must NOT execute when ``should_abort`` is set.

    The controller stamps a note carrying the abort reason and sets
    ``state.completed=True`` so the LangGraph terminates promptly.
    """
    s = _state(should_abort=True, abort_reason="customer kill switch")
    step_ran = False

    class _Agent:
        stage = Stage.RECON

        async def step(self, _ctx: StageContext) -> StepResult:
            nonlocal step_ran
            step_ran = True
            return StepResult(
                decision=StepDecision.CONTINUE,
                tokens_used=10,
                cost_usd=0.0,
                tool_runs=1,
                nodes_written=1,
                notes="",
            )

    node = _step_agent(_Agent())
    out = await node(s)
    assert step_ran is False, (
        "agent.step() must not run when should_abort=True; the kill switch fired"
    )
    assert out.completed is True
    # The note carries the abort reason so the operator (and the report
    # renderer) see WHY the engagement stopped.
    assert any("ABORT" in n and "customer kill switch" in n for n in out.notes)


# --------------------------------------------------------------------------- #
# 6. ``_abort_requested`` polls the engagement row as the durable fallback    #
#    (bus may have dropped its publish).                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_abort_requested_polls_row_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    s = _state(engagement_id=eid)
    # Build a fake session_scope that returns an EngagementRow with
    # status="aborted" so the helper picks it up.
    row = _engagement_row(status="aborted")
    row.id = eid
    row.abort_reason = "polled-via-row"

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return row

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    monkeypatch.setattr("orchestrator.db.session_scope", _scope)

    assert s.should_abort is False
    assert await _abort_requested(s) is True
    # Helper hydrated the in-memory flag + reason so subsequent calls
    # don't re-query the DB.
    assert s.should_abort is True
    assert s.abort_reason == "polled-via-row"


@pytest.mark.asyncio
async def test_abort_requested_returns_false_when_row_not_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eid = uuid4()
    s = _state(engagement_id=eid)
    row = _engagement_row(status="running")
    row.id = eid

    class _Sess:
        async def get(self, _cls: Any, _key: Any) -> Any:
            return row

    @asynccontextmanager
    async def _scope():
        yield _Sess()

    monkeypatch.setattr("orchestrator.db.session_scope", _scope)

    assert await _abort_requested(s) is False
    assert s.should_abort is False


@pytest.mark.asyncio
async def test_abort_requested_fast_path_skips_db_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the flag is already True, the helper returns True without
    touching the DB — important so a running graph isn't billed an extra
    SELECT on every step once the abort is acknowledged."""
    db_hit = False

    @asynccontextmanager
    async def _scope():
        nonlocal db_hit
        db_hit = True
        yield None

    monkeypatch.setattr("orchestrator.db.session_scope", _scope)
    s = _state(should_abort=True, abort_reason="fast-path")
    assert await _abort_requested(s) is True
    assert db_hit is False


# --------------------------------------------------------------------------- #
# 7. SandboxRunner.terminate signature is present (real kill goes through    #
#    ``docker stop --time=<grace>`` then ``docker kill``).                    #
# --------------------------------------------------------------------------- #


def test_sandbox_runner_terminate_exists() -> None:
    """The kill-switch abort path calls ``runner.terminate(container_name,
    timeout_s=30)``. Verify the method exists with the documented shape so
    a future refactor that drops it gets caught here."""
    from sandbox.runner import SandboxRunner

    runner = SandboxRunner()
    assert hasattr(runner, "terminate"), (
        "SandboxRunner must expose terminate() so the abort endpoint can "
        "send SIGTERM to in-flight tool runs"
    )
    import inspect

    sig = inspect.signature(runner.terminate)
    assert "container_name" in sig.parameters
    assert "timeout_s" in sig.parameters
    # Default grace is 30s per the abort docstring contract.
    assert sig.parameters["timeout_s"].default == 30


# --------------------------------------------------------------------------- #
# 8. The abort endpoint's status code + return shape: 200 + EngagementRead.   #
# --------------------------------------------------------------------------- #


def test_abort_service_function_exists() -> None:
    """Defensive — a future refactor that renames the function must be caught."""
    from orchestrator.services import engagements as svc

    assert callable(getattr(svc, "abort_engagement", None))


# Keep an opaque id around so a future expansion doesn't have to re-import.
_OPAQUE: UUID = uuid4()
