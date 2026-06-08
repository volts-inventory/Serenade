"""Phase 43: per-engagement approval_policy + auto_start launch flow.

Covers the two surfaces:

1. `_resolve_approval_gate()` — the agent helper that consults the
   engagement's approval_policy and writes an ApprovalRow with the
   right `auto_decision` set, returning "approve" | "reject" | "queue"
   so the caller knows what to do next.

2. The engagement-dict builder includes the policy so the agent sees it.

We mock `_persist_approval` to avoid the DB; the contract under test is
the policy → decision mapping, not the persistence path itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agents import base as agent_base
from orchestrator.models import EngagementRow
from policy.engagement_dict import build_engagement_context


def _row(mode: str = "guided") -> EngagementRow:
    """Build a minimal EngagementRow for the engagement-dict builder."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    return EngagementRow(
        id=uuid4(),
        name="t",
        scope={"cidrs": [], "domains": [], "urls": [], "excluded_cidrs": [], "excluded_domains": []},
        objectives=[],
        mode=mode,
        time_window_start=now,
        time_window_end=now + timedelta(hours=1),
        token_budget=1,
        wallclock_hours=1,
        authorization_artifact_sha256="",
        per_tool_token_budget={},
        status="draft",
        current_stage="scope_intake",
    )


# --------------------------------------------------------------------------- #
# build_engagement_context derives approval_policy from mode                   #
# --------------------------------------------------------------------------- #


def test_engagement_dict_guided_derives_phase_gate() -> None:
    # guided mode gates exploit/post-exploit behind a phase gate.
    d = build_engagement_context(_row(mode="guided"))
    assert d["approval_policy"] == "phase_gate"


def test_engagement_dict_autonomous_derives_auto_approve() -> None:
    # autonomous mode runs exploitation without pausing.
    d = build_engagement_context(_row(mode="autonomous"))
    assert d["approval_policy"] == "auto_approve"


def test_engagement_dict_recon_only_derives_auto_approve() -> None:
    # recon_only/assess never reach exploit, so the policy is auto_approve.
    d = build_engagement_context(_row(mode="recon_only"))
    assert d["approval_policy"] == "auto_approve"


def test_engagement_dict_surfaces_destructive_opt_in() -> None:
    """tracking issue #112, P2 #3: the operator destructive_opt_in flag on the
    EngagementRow is surfaced into the engagement context (PoC path only).
    Fail-closed default is False when the row omits it."""
    # Default row (no destructive_opt_in set) → fail-closed False.
    assert build_engagement_context(_row()).get("destructive_opt_in") is False

    # Operator opt-in → surfaced as True.
    row = _row()
    row.destructive_opt_in = True
    assert build_engagement_context(row)["destructive_opt_in"] is True


# --------------------------------------------------------------------------- #
# _resolve_approval_gate maps policy -> decision + persists right row          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_persist(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace _persist_approval with an async mock so we can inspect calls
    without touching the DB."""
    m = AsyncMock(return_value=None)
    monkeypatch.setattr(agent_base, "_persist_approval", m)
    return m


async def test_resolve_gate_manual_returns_queue(mock_persist: AsyncMock) -> None:
    """Default policy: write the row with no decision; tell caller to queue."""
    eng = {"approval_policy": "manual"}
    eng_id = uuid4()
    result = await agent_base._resolve_approval_gate(
        engagement=eng,
        engagement_id=eng_id,
        kind="policy_require_approval:nmap",
        payload={"tool": "nmap", "input": {}, "reason": "out-of-scope"},
    )
    assert result == "queue"
    mock_persist.assert_awaited_once()
    call_kwargs = mock_persist.call_args.kwargs
    assert call_kwargs["engagement_id"] == eng_id
    assert call_kwargs["kind"] == "policy_require_approval:nmap"
    # No auto_decision passed -> ApprovalRow.decision stays NULL
    assert "auto_decision" not in call_kwargs or call_kwargs.get("auto_decision") is None


async def test_resolve_gate_auto_approve_returns_approve(mock_persist: AsyncMock) -> None:
    eng = {"approval_policy": "auto_approve"}
    result = await agent_base._resolve_approval_gate(
        engagement=eng,
        engagement_id=uuid4(),
        kind="critic_revise:exploit_x",
        payload={"tool": "exploit_x", "input": {}, "rationale": "uncertain"},
    )
    assert result == "approve"
    kwargs = mock_persist.call_args.kwargs
    assert kwargs["auto_decision"] == "approve"
    # Policy stamped into the payload for audit clarity.
    assert kwargs["payload"]["auto_policy"] == "auto_approve"


async def test_resolve_gate_auto_reject_returns_reject(mock_persist: AsyncMock) -> None:
    eng = {"approval_policy": "auto_reject"}
    result = await agent_base._resolve_approval_gate(
        engagement=eng,
        engagement_id=uuid4(),
        kind="policy_require_approval:nuclei",
        payload={"tool": "nuclei", "input": {}, "reason": "destructive"},
    )
    assert result == "reject"
    kwargs = mock_persist.call_args.kwargs
    assert kwargs["auto_decision"] == "reject"
    assert kwargs["payload"]["auto_policy"] == "auto_reject"


async def test_resolve_gate_missing_policy_defaults_to_manual(
    mock_persist: AsyncMock,
) -> None:
    """An engagement dict with no `approval_policy` key — legacy or partial
    construction — must NOT auto-decide. Manual is the safe default."""
    result = await agent_base._resolve_approval_gate(
        engagement={},
        engagement_id=uuid4(),
        kind="policy_require_approval:x",
        payload={},
    )
    assert result == "queue"
    kwargs = mock_persist.call_args.kwargs
    assert kwargs.get("auto_decision") is None


async def test_resolve_gate_unknown_policy_defaults_to_manual(
    mock_persist: AsyncMock,
) -> None:
    """Defence-in-depth: an unknown value (config drift, typo) must NOT
    silently auto-approve. Fall back to manual."""
    result = await agent_base._resolve_approval_gate(
        engagement={"approval_policy": "auto_yolo"},
        engagement_id=uuid4(),
        kind="policy_require_approval:x",
        payload={},
    )
    assert result == "queue"
    kwargs = mock_persist.call_args.kwargs
    assert kwargs.get("auto_decision") is None


async def test_resolve_gate_case_insensitive(mock_persist: AsyncMock) -> None:
    """Policy values normalise via .lower() so AUTO_REJECT etc. still work."""
    result = await agent_base._resolve_approval_gate(
        engagement={"approval_policy": "AUTO_REJECT"},
        engagement_id=uuid4(),
        kind="policy_require_approval:x",
        payload={},
    )
    assert result == "reject"
    assert mock_persist.call_args.kwargs["auto_decision"] == "reject"
