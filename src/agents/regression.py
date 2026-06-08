"""Regression request types.

Typed signal carriers for delta-scoped stage regression. Detection
belongs in agents, verification in the council, routing in the
stage controller.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from worldstate.schema import Stage


class RegressionTriggerKind(StrEnum):
    """Enumerated structural triggers for stage regression.

    New kinds require a matching council critic for verification.
    """

    NEW_SUBNET_DISCOVERED = "new_subnet_discovered"
    NEW_SERVICE_CLASS_DISCOVERED = "new_service_class_discovered"
    CREDENTIALS_IMPLY_PIVOT = "credentials_imply_pivot"
    MISCONFIGURATION_OPENS_PATH = "misconfiguration_opens_path"
    LLM_SUGGESTED_REGRESSION = "llm_suggested_regression"
    # Post-compromise → vuln_triage feedback loops.
    PRIVESC_FINDING_DISCOVERED = "privesc_finding_discovered"
    AUTHENTICATED_ACCESS_GAINED = "authenticated_access_gained"


class RegressionRequest(BaseModel):
    """Deterministic regression request from an agent to re-enter an earlier stage.

    JSON round-trip stable: ``RegressionRow`` stores ``delta`` and ``evidence``
    as JSONB columns from ``model_dump()``.
    """

    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    target_stage: Stage = Field(
        description="The earlier stage we want the controller to re-enter."
    )
    trigger_kind: RegressionTriggerKind = Field(
        description="Which enumerated structural condition fired."
    )
    delta: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The scoped re-entry contract. Shape depends on trigger_kind: "
            "e.g. {'cidrs': ['10.0.5.0/24']} for NEW_SUBNET_DISCOVERED, "
            "{'host_id': '...', 'ports': [445, 3389]} for "
            "NEW_SERVICE_CLASS_DISCOVERED. The re-entered agent MUST scope "
            "its work to this delta and clear it on completion."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "World-state graph node IDs that motivated this request. The "
            "Novelty critic queries these to verify the discovered surface "
            "isn't already represented in prior work."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Short human-readable explanation. Surfaced in the audit row "
            "and the report's methodology section."
        ),
    )
