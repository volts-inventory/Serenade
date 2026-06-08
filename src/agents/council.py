"""Council pattern: multiple critics with different priors vote, a synthesizer
produces a single verdict.

Used at stage-transition decision points where single-agent bias is the
biggest risk (vuln-triage ranking, attack-path selection, finding consolidation).

Council members are cheap (Haiku). The synthesizer is one tier up (Sonnet) to
make the judgment call. This is still vastly cheaper than the planner LLM.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError

from agents.critic import (
    _PROMPT_INJECTION_GUARD,
    BaseCritic,
    Critique,
    LLMCritic,
    StructuralCheck,
    Verdict,
    _truncate_subject,
)
from llm.router import CallSite, LLMRequest, LLMRouter

# ``Council.members`` is a heterogeneous list of either LLM
# critics or structural checks. Both honour the same
# ``review(subject, *, engagement_id) -> Critique`` shape.
CouncilMember = LLMCritic | StructuralCheck

log = structlog.get_logger(__name__)


class CouncilVerdict(BaseModel):
    decision: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    member_critiques: list[Critique]
    dissent_summary: str | None = None
    total_cost_usd: float


class Council:
    """Runs N critics in parallel, then a synthesizer decides.

    Each critic returns a Critique. The synthesizer:
    - resolves splits (e.g. 3 approve, 2 veto -> revise with dissent surfaced)
    - extracts the strongest concern across members
    - decides the final action
    """

    def __init__(
        self,
        *,
        name: str,
        router: LLMRouter,
        members: list[CouncilMember],
    ) -> None:
        if len(members) < 2:
            raise ValueError("A council needs at least 2 members")
        self.name = name
        self.router = router
        self.members = members

    async def deliberate(
        self,
        subject: dict[str, Any],
        *,
        engagement_id: UUID | None = None,
        mode: str = "full",
        debate_rounds: int = 1,
    ) -> CouncilVerdict:
        """`mode="fast"` runs ONE critic (false-positive skeptic if present,
        else the first member) and skips the council synthesis. Saves 3-4 LLM
        calls per triage batch. Use for large-volume low-stakes batches; the
        full council still runs by default.

        When ``debate_rounds > 1``, runs an iterative debate: after the
        initial round of member critiques, subsequent rounds inject the
        prior round's critiques into the subject so each member can see
        and respond to the others' reasoning. Short-circuit rules still
        apply: any VETO in any round triggers an immediate VETO; unanimous
        high-confidence approve skips remaining rounds.
        """
        if mode == "fast":
            picked = next(
                (m for m in self.members if m.name == "false_positive_skeptic"),
                self.members[0],
            )
            single = await picked.review(subject, engagement_id=engagement_id)
            return CouncilVerdict(
                decision=single.verdict,
                confidence=single.confidence,
                rationale=f"fast-mode: {single.critic_name} — {single.primary_concern}",
                member_critiques=[single],
                dissent_summary=None,
                total_cost_usd=single.cost_usd,
            )

        rounds = max(1, min(debate_rounds, 5))
        all_critiques: list[Critique] = []
        total_cost = 0.0

        for round_num in range(rounds):
            # Build per-round subject: original subject + prior round context
            if round_num == 0:
                round_subject = subject
            else:
                round_subject = dict(subject)
                round_subject["_debate_round"] = round_num + 1
                round_subject["_prior_critiques"] = [
                    c.model_dump(mode="json") for c in all_critiques
                ]

            critiques = await asyncio.gather(
                *(m.review(round_subject, engagement_id=engagement_id) for m in self.members)
            )
            round_critiques = list(critiques)
            all_critiques.extend(round_critiques)
            total_cost += sum(c.cost_usd for c in round_critiques)

            # Short-circuit: any VETO in any round -> immediate VETO
            vetoes = [c for c in round_critiques if c.verdict == Verdict.VETO]
            if vetoes:
                strongest = max(vetoes, key=lambda c: c.confidence)
                return CouncilVerdict(
                    decision=Verdict.VETO,
                    confidence=strongest.confidence,
                    rationale=f"vetoed by {strongest.critic_name} (round {round_num + 1}): {strongest.primary_concern}",
                    member_critiques=all_critiques,
                    dissent_summary=_summarize_dissent(all_critiques),
                    total_cost_usd=total_cost,
                )

            # Short-circuit: unanimous high-confidence approve -> skip remaining rounds
            if all(c.verdict == Verdict.APPROVE and c.confidence >= 0.8 for c in round_critiques):
                return CouncilVerdict(
                    decision=Verdict.APPROVE,
                    confidence=min(c.confidence for c in round_critiques),
                    rationale=f"unanimous approval (round {round_num + 1})",
                    member_critiques=all_critiques,
                    dissent_summary=None,
                    total_cost_usd=total_cost,
                )

        # Use the final round's critiques for the synthesis decision
        final_round = all_critiques[-len(self.members):]

        # Check for unanimous approve in the final round (may not be high-confidence)
        if all(c.verdict == Verdict.APPROVE and c.confidence >= 0.8 for c in final_round):
            return CouncilVerdict(
                decision=Verdict.APPROVE,
                confidence=min(c.confidence for c in final_round),
                rationale="unanimous approval (final round)",
                member_critiques=all_critiques,
                dissent_summary=None,
                total_cost_usd=total_cost,
            )

        # Synthesizer resolves when no short-circuit fired.
        # Feed the final round's critiques to the synthesizer for the verdict.
        critiques = final_round

        # The synthesizer is the deciding vote. Like the member critics
        # (LLMCritic.review), bound attacker-controlled strings in the
        # subject and prepend the prompt-injection guard so embedded
        # directives in tool output can't steer the final verdict (llm-03).
        safe_subject = _truncate_subject(subject)
        synth_resp = await self.router.complete(
            LLMRequest(
                system=_PROMPT_INJECTION_GUARD + _SYNTH_SYSTEM,
                user=_format_synth_user(safe_subject, critiques),
                site=CallSite.CRITIC,
                max_tokens=1200,
                temperature=0.0,
                engagement_id=str(engagement_id) if engagement_id else None,
            )
        )
        verdict, conf, rationale = _parse_synth(synth_resp.text)
        return CouncilVerdict(
            decision=verdict,
            confidence=conf,
            rationale=rationale,
            member_critiques=all_critiques,
            dissent_summary=_summarize_dissent(all_critiques),
            total_cost_usd=total_cost + synth_resp.cost_usd,
        )


_SYNTH_SYSTEM = (
    "You are the chair of a council of offensive-security critics. Each member "
    "has reviewed a proposed action and produced a structured critique. Your job "
    "is to produce the final verdict.\n\n"
    "Rules:\n"
    "- If any member's veto is well-supported, the council vetoes.\n"
    "- If the majority approves but a critic raises a fixable concern, return 'revise' "
    "with the concern as the rationale.\n"
    "- If members disagree substantively and the issue is one of policy or scope, "
    "return 'escalate'.\n"
    "- Confidence should reflect council agreement (unanimous=high, split=low).\n\n"
    "Output a single JSON object on one line:\n"
    '{"verdict": "approve|revise|veto|escalate", "confidence": 0.0-1.0, "rationale": "..."}'
)


def _format_synth_user(subject: dict[str, Any], critiques: list[Critique]) -> str:
    return (
        "## Subject under review\n```json\n"
        + json.dumps(subject, indent=2, default=str)
        + "\n```\n\n## Member critiques\n```json\n"
        + json.dumps([c.model_dump() for c in critiques], indent=2, default=str)
        + "\n```\n\nProduce the final verdict JSON now."
    )


def _parse_synth(text: str) -> tuple[Verdict, float, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
        v = Verdict(data["verdict"])
        c = float(data.get("confidence", 0.5))
        r = str(data.get("rationale", ""))
        return v, max(0.0, min(1.0, c)), r
    except (json.JSONDecodeError, KeyError, ValueError, ValidationError):
        return Verdict.ESCALATE, 0.0, "synthesizer produced unparseable verdict"


def _summarize_dissent(critiques: list[Critique]) -> str | None:
    counts = Counter(c.verdict for c in critiques)
    if len(counts) == 1:
        return None
    return ", ".join(f"{n}x {v.value}" for v, n in counts.most_common())


# --------------------------------------------------------------------------- #
# Council recipes — pre-built compositions used by the orchestrator           #
# --------------------------------------------------------------------------- #


def _format_candidate_subject(subject: dict[str, Any]) -> str:
    """Shared subject formatter for triage council members."""
    return (
        "## Candidate ranking under review\n```json\n"
        + json.dumps(subject, indent=2, default=str)
        + "\n```"
    )


class _ExploitabilityFocused(BaseCritic):
    """Council member: technical viability of the exploit."""

    name = "exploitability_focused"
    perspective = "Focuses on whether the exploit is technically viable on the target."

    def _system_prompt(self) -> str:
        return (
            "You are a council member evaluating exploit candidates. Your lens is "
            "technical viability: does the version match, is the prerequisite "
            "present, has the exploit been observed to work on this exact stack? "
            "Penalize candidates with version inference, missing prereqs, or untested PoCs."
        )

    def _format_subject(self, subject: dict[str, Any]) -> str:
        return _format_candidate_subject(subject)


class _BlastRadiusFocused(BaseCritic):
    """Council member: collateral damage and detection footprint."""

    name = "blast_radius_focused"
    perspective = "Focuses on collateral damage and detection footprint."

    def _system_prompt(self) -> str:
        return (
            "You are a council member evaluating exploit candidates. Your lens is "
            "blast radius: what's the worst case if this exploit misfires? "
            "Considerations: shared infrastructure, persistent state, detection "
            "footprint, account lockouts, irreversible operations. Penalize "
            "candidates with high blast radius unless the engagement explicitly "
            "authorizes it."
        )

    def _format_subject(self, subject: dict[str, Any]) -> str:
        return _format_candidate_subject(subject)


class _FalsePositiveSkeptic(BaseCritic):
    """Council member: evidence quality and finding deduplication."""

    name = "false_positive_skeptic"
    perspective = "Hard skeptic about evidence quality and finding deduplication."

    def _system_prompt(self) -> str:
        return (
            "You are a council member evaluating exploit candidates. Your lens is "
            "false-positive avoidance: is the finding well-evidenced? Is it the "
            "same as another finding with a different name? Is the affected-asset "
            "list inferred or directly observed? Penalize anything with weak or "
            "single-source evidence."
        )

    def _format_subject(self, subject: dict[str, Any]) -> str:
        return _format_candidate_subject(subject)


def triage_council(router: LLMRouter) -> Council:
    """Three-member council used at VULN_TRIAGE consolidation."""
    return Council(
        name="triage_council",
        router=router,
        members=[
            _ExploitabilityFocused(router),
            _BlastRadiusFocused(router),
            _FalsePositiveSkeptic(router),
        ],
    )


def debate_council(router: LLMRouter) -> Council:
    """Two-member debate council for exploit/lateral high-stakes gates."""
    from agents.debate_critics import AggressivePentester, CautiousDefender  # noqa: PLC0415

    return Council(
        name="debate_council",
        router=router,
        members=[
            AggressivePentester(router),
            CautiousDefender(router),
        ],
    )
