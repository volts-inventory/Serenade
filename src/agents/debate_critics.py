"""Debate-mode critic personas for exploit/lateral high-stakes gates.

Two adversarial critics with opposing biases used by the multi-agent
debate council. Both inherit from ``LLMCritic`` and route via
``CallSite.CRITIC`` (E4B model for cost efficiency).
"""

from __future__ import annotations

import json
from typing import Any

from agents.critic import LLMCritic

__all__ = ["AggressivePentester", "CautiousDefender"]


def _format_review_subject(subject: dict[str, Any]) -> str:
    """Shared subject formatter for debate critics."""
    return (
        "## Proposed action under review\n```json\n"
        + json.dumps(subject, indent=2, default=str)
        + "\n```"
    )


class AggressivePentester(LLMCritic):
    """Red-side debate persona: favours exploitation, assesses viability."""

    name = "aggressive_pentester"
    perspective = "Favours exploitation; assesses attack viability and reasons to proceed."

    def _system_prompt(self) -> str:
        return (
            "You are a senior offensive security operator reviewing a proposed "
            "action. Your lens is ATTACK VIABILITY — you look for reasons the "
            "action SHOULD proceed. Consider:\n"
            "- Is the version match exact and directly observed?\n"
            "- Is the PoC from the curated library with a known-good track record?\n"
            "- Is the blast radius contained and acceptable for this engagement?\n"
            "- Are the prerequisites for the exploit clearly satisfied?\n"
            "- Is there a clear path to impact (proof of exploitation)?\n\n"
            "You are NOT reckless — you approve only when the technical "
            "evidence supports success. But you bias toward action over "
            "caution when the evidence is solid. Veto only when the "
            "evidence is clearly insufficient or the target is obviously "
            "wrong. Revise when minor pre-checks would strengthen confidence."
        )

    def _format_subject(self, subject: dict[str, Any]) -> str:
        return _format_review_subject(subject)


class CautiousDefender(LLMCritic):
    """Blue-side debate persona: highlights risks, scope, and detection."""

    name = "cautious_defender"
    perspective = "Highlights risks, scope concerns, detection likelihood, and blast radius."

    def _system_prompt(self) -> str:
        return (
            "You are a defensive security analyst reviewing a proposed "
            "offensive action. Your lens is RISK AND DETECTION — you look "
            "for reasons the action could go wrong. Consider:\n"
            "- Will this action trigger high-confidence detection (IDS/IPS, "
            "EDR, SIEM correlation)?\n"
            "- Is there account lockout risk from authentication attempts?\n"
            "- Is the target clearly in scope per the engagement letter?\n"
            "- Could this action affect shared infrastructure or other tenants?\n"
            "- Does the host fingerprint match known honeypot patterns?\n"
            "- Is the evidence fresh or could it be stale (ports closed, "
            "services patched since last scan)?\n"
            "- Is there a rollback plan if the action causes damage?\n\n"
            "You bias toward caution — approve only when risks are clearly "
            "mitigated. Veto when detection likelihood is high or scope is "
            "ambiguous. Revise when additional pre-checks would reduce risk."
        )

    def _format_subject(self, subject: dict[str, Any]) -> str:
        return _format_review_subject(subject)
