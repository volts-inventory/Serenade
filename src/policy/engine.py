"""Policy engine — pure Python, no OPA container.

Every tool call goes through this gate. Decisions: ALLOW | DENY | REQUIRE_APPROVAL.

This module preserves the public interface (``PolicyEngine``, ``PolicyDecision``,
``PolicyResult``) so callers (``agents/base.py``, ``worker/runtime.py``,
``orchestrator/api/operator_exec.py``) do not need changes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel

from policy import scope as _scope

log = structlog.get_logger(__name__)


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyResult(BaseModel):
    decision: PolicyDecision
    reason: str
    raw: dict[str, Any] = {}


class PolicyEngine:
    """Evaluate scope / risk / time / destructive policy for a tool call.

    Previously an HTTP client to the OPA container.  Now delegates to the
    pure-Python ``policy.scope.evaluate()`` function.
    """

    def __init__(self, settings: Any | None = None) -> None:
        # ``settings`` is accepted (but ignored) for backwards compatibility
        # with call sites that pass ``Settings`` — e.g. ``PolicyEngine(s)``.
        pass

    async def aclose(self) -> None:
        """No-op — kept so callers that manage the lifecycle don't break."""

    async def evaluate(
        self,
        *,
        engagement: dict[str, Any],
        tool: dict[str, Any],
        targets: list[str],
    ) -> PolicyResult:
        try:
            result = _scope.evaluate(targets=targets, engagement=engagement, tool=tool)
        except Exception as e:
            log.error("policy.evaluate_error", err=str(e))
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"policy_error: {e}",
            )

        raw = {
            "allow": result.allow,
            "require_approval": result.require_approval,
            "reason": result.reason,
        }

        if not result.allow:
            decision = PolicyDecision.DENY
        elif result.require_approval:
            decision = PolicyDecision.REQUIRE_APPROVAL
        else:
            decision = PolicyDecision.ALLOW

        return PolicyResult(decision=decision, reason=result.reason, raw=raw)


__all__ = ["PolicyDecision", "PolicyEngine", "PolicyResult"]
