"""Operator-driven one-off command execution — plain async functions.

Extracted from ``orchestrator.api.operator_exec``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from orchestrator.db import session_scope
from orchestrator.ledger import append_audit_row
from orchestrator.models import ApprovalRow, EngagementRow, ToolRunRow
from orchestrator.services.errors import (
    ApprovalRequired,
    CriticVeto,
    NotFoundError,
    PolicyViolation,
)
from orchestrator.settings import get_settings
from policy.engagement_dict import build_engagement_context
from policy.engine import PolicyDecision, PolicyEngine
from tools.base import ToolRunContext
from tools.operator_exec import OperatorExec, OperatorExecInput, _redact_argv


class OperatorExecRequest(BaseModel):
    argv: list[str]
    target: str
    justification: str
    timeout_seconds: int = 120
    bypass_critic: bool = False
    bypass_justification: str | None = None
    actor: str = "operator"


async def run_operator_exec(
    engagement_id: UUID,
    payload: OperatorExecRequest,
    principal: str = "local",
) -> dict[str, Any]:
    async with session_scope() as session:
        row = await session.get(EngagementRow, engagement_id)
        if row is None:
            raise NotFoundError("engagement not found")

        actor = principal

        async def _audit(action: str, payload: dict[str, Any]) -> None:
            # db-03: write the audit row into THIS open session/transaction
            # instead of opening an independent nested ``session_scope()`` (a
            # second connection on the same SQLite file). Keeps the whole
            # operator-exec record — approval/tool-run/audit — in one atomic
            # commit boundary and avoids the self-lock risk of a nested writer.
            await append_audit_row(
                session,
                engagement_id=engagement_id,
                actor=actor,
                action=action,
                payload=payload,
            )
        inp = OperatorExecInput.model_validate(payload.model_dump(exclude={"actor"}))
        tool = OperatorExec()
        engagement_dict = build_engagement_context(row)
        gates: dict[str, Any] = {}

        # Gate 1: Scope/risk/time policy
        settings = get_settings()
        policy_engine = PolicyEngine(settings)
        try:
            policy_result = await policy_engine.evaluate(
                engagement=engagement_dict,
                tool={
                    "name": tool.name,
                    "risk_class": tool.risk_class.value,
                    "destructive": tool.destructive,
                },
                targets=tool.extract_targets(inp),
            )
        finally:
            await policy_engine.aclose()
        gates["policy"] = {"decision": policy_result.decision.value, "reason": policy_result.reason}
        if policy_result.decision == PolicyDecision.DENY:
            await _audit(
                action="operator_exec.denied_by_policy",
                payload={"target": inp.target, "reason": policy_result.reason},
            )
            # Persist the denial audit before unwinding (the raise rolls back
            # the session otherwise).
            await session.commit()
            raise PolicyViolation(f"operator_exec denied by scope policy: {policy_result.reason}")
        if policy_result.decision == PolicyDecision.REQUIRE_APPROVAL:
            session.add(
                ApprovalRow(
                    engagement_id=engagement_id,
                    kind="operator_exec_policy_required_approval",
                    requested_payload={
                        "target": inp.target,
                        "redacted_argv": _redact_argv(inp.argv),
                        "justification": inp.justification,
                        "actor": actor,
                    },
                )
            )
            await _audit(
                action="operator_exec.policy_required_approval_queued",
                payload={"target": inp.target, "reason": policy_result.reason},
            )
            await session.commit()
            raise ApprovalRequired(
                f"Policy returned REQUIRE_APPROVAL: {policy_result.reason}; queued"
            )

        # Gate 2: PreExploitCritic
        if inp.bypass_critic:
            gates["critic"] = {"bypassed": True, "justification": inp.bypass_justification}
            await _audit(
                action="operator_exec.critic_bypassed",
                payload={
                    "target": inp.target,
                    "justification": inp.justification,
                    "bypass_justification": inp.bypass_justification,
                },
            )
        else:
            from agents.critic import PreExploitCritic, Verdict
            from llm.router import build_router_async

            snapshot = getattr(row, "llm_config_snapshot", {}) or {}
            router_ = await build_router_async(settings, snapshot=snapshot)
            critic = PreExploitCritic(router_)
            subject = {
                "tool": tool.name,
                "argv": _redact_argv(inp.argv),
                "target": inp.target,
                "justification": inp.justification,
                "engagement_mode": engagement_dict["mode"],
            }
            critique = await critic.review(subject, engagement_id=engagement_id)
            gates["critic"] = {
                "verdict": critique.verdict.value,
                "concern": critique.primary_concern,
            }
            if critique.verdict == Verdict.VETO:
                await _audit(
                    action="operator_exec.critic_vetoed",
                    payload={"target": inp.target, "concern": critique.primary_concern},
                )
                await session.commit()
                raise CriticVeto(f"PreExploitCritic vetoed: {critique.primary_concern}")
            if critique.verdict in (Verdict.ESCALATE, Verdict.REVISE):
                session.add(
                    ApprovalRow(
                        engagement_id=engagement_id,
                        kind=f"operator_exec_critic_{critique.verdict.value}",
                        requested_payload={
                            **payload.model_dump(),
                            "argv": _redact_argv(payload.argv),
                        },
                    )
                )
                await session.commit()
                raise ApprovalRequired(
                    f"critic requested {critique.verdict.value}: {critique.primary_concern}; queued as approval"
                )

        # Execute
        tool_run_id = tool.new_run_id()
        started = datetime.now(UTC)
        redacted_argv = _redact_argv(inp.argv)
        # db-03: write the tool-run row and the dispatch audit row into the
        # SAME open session as the approval/engagement writes, then commit ONCE
        # so the whole operator-exec record is atomic (no partial-state window
        # if the process dies between independent transactions).
        session.add(
            ToolRunRow(
                id=tool_run_id,
                engagement_id=engagement_id,
                tool=tool.name,
                input_json={"argv": redacted_argv, "target": inp.target},
                sandbox_id="dev",
                started_at=started,
                ended_at=None,
                exit_code=0,
                stdout_blob_id=None,
                stderr_blob_id=None,
            )
        )
        await _audit(
            action="operator_exec.dispatched",
            payload={
                "tool_run_id": str(tool_run_id),
                "target": inp.target,
                "justification": inp.justification,
                "bypass_critic": inp.bypass_critic,
                "redacted_argv": redacted_argv,
            },
        )
        # Commit the provenance record before the (potentially long) sandbox
        # run so it's durable even if the run itself crashes.
        await session.commit()

        ctx = ToolRunContext(
            tool_run_id=tool_run_id,
            engagement_id=engagement_id,
            sandbox_id="dev",
        )
        result = await tool.run(inp, ctx)

        return {
            "tool_run_id": str(tool_run_id),
            "exit_code": result.output.exit_code,
            "stdout": result.output.stdout,
            "stderr": result.output.stderr,
            "duration_seconds": result.output.duration_seconds,
            "truncated": result.output.truncated,
            "gates": gates,
        }
