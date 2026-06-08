"""Unit tests for regression types: StepDecision.REGRESS_TO,
RegressionRequest round-trip, RegressionTriggerKind enum, and
StepResult.regression_request defaults.
"""

from __future__ import annotations

import json

import pydantic
import pytest

from agents.base import StepDecision, StepResult
from agents.regression import RegressionRequest, RegressionTriggerKind
from worldstate.schema import Stage


def test_step_decision_regress_to_exists_and_is_distinct() -> None:
    """REGRESS_TO is a distinct StepDecision variant."""
    assert StepDecision.REGRESS_TO.value == "regress_to"
    # Distinct from every other variant — the controller branches on identity.
    assert StepDecision.REGRESS_TO != StepDecision.CONTINUE
    assert StepDecision.REGRESS_TO != StepDecision.ADVANCE
    assert StepDecision.REGRESS_TO != StepDecision.ESCALATE_HUMAN
    # And every other variant survives — we did not accidentally rename one.
    members = {m.value for m in StepDecision}
    assert {"continue", "advance", "escalate_human", "regress_to"} <= members


def test_regression_trigger_kind_has_expected_values() -> None:
    """All expected trigger kinds are present."""
    values = {m.value for m in RegressionTriggerKind}
    assert values == {
        "new_subnet_discovered",
        "new_service_class_discovered",
        "credentials_imply_pivot",
        "misconfiguration_opens_path",
        "llm_suggested_regression",
        "privesc_finding_discovered",
        "authenticated_access_gained",
    }


def test_regression_request_minimal_construction() -> None:
    """Non-required fields have sensible defaults."""
    req = RegressionRequest(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.NEW_SUBNET_DISCOVERED,
    )
    assert req.delta == {}
    assert req.evidence == []
    assert req.rationale == ""


def test_regression_request_round_trips_through_model_dump() -> None:
    """model_dump -> JSON -> model_validate preserves all fields."""
    original = RegressionRequest(
        target_stage=Stage.ENUMERATION,
        trigger_kind=RegressionTriggerKind.NEW_SERVICE_CLASS_DISCOVERED,
        delta={"host_id": "h-1", "ports": [445, 3389]},
        evidence=["service-node-1", "service-node-2"],
        rationale="loot pass revealed SMB on a host whose only enumerated port was 80",
    )

    dumped = original.model_dump()
    # JSON-serialisable shape: every value must survive json encode/decode.
    serialised = json.dumps(dumped, default=str)
    reloaded = json.loads(serialised)

    restored = RegressionRequest.model_validate(reloaded)
    assert restored.target_stage == original.target_stage
    assert restored.trigger_kind == original.trigger_kind
    assert restored.delta == original.delta
    assert restored.evidence == original.evidence
    assert restored.rationale == original.rationale


def test_regression_request_rejects_unknown_trigger_kind() -> None:
    """Unknown trigger kinds fail validation."""
    with pytest.raises(pydantic.ValidationError):
        RegressionRequest(
            target_stage=Stage.RECON,
            trigger_kind="totally_made_up_trigger",  # type: ignore[arg-type]
        )


def test_step_result_regression_request_defaults_none() -> None:
    """Non-REGRESS_TO step results do not carry a regression payload."""
    result = StepResult(
        decision=StepDecision.ADVANCE,
        tokens_used=100,
        cost_usd=0.0,
        tool_runs=1,
        nodes_written=2,
    )
    assert result.regression_request is None


def test_step_result_carries_regression_request_when_set() -> None:
    """REGRESS_TO step results carry the typed request."""
    req = RegressionRequest(
        target_stage=Stage.RECON,
        trigger_kind=RegressionTriggerKind.CREDENTIALS_IMPLY_PIVOT,
        delta={"cidrs": ["10.0.5.0/24"]},
        evidence=["cred-node-1"],
        rationale="LATERAL spray succeeded on host whose /24 was outside the enumerated set",
    )
    result = StepResult(
        decision=StepDecision.REGRESS_TO,
        tokens_used=42,
        cost_usd=0.0,
        tool_runs=0,
        nodes_written=0,
        regression_request=req,
    )
    assert result.decision == StepDecision.REGRESS_TO
    assert result.regression_request is not None
    assert result.regression_request.target_stage == Stage.RECON
    assert result.regression_request.delta == {"cidrs": ["10.0.5.0/24"]}
