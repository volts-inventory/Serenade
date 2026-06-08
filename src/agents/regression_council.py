"""StageRegressionCouncil and its three structural critics.

Verifies a :class:`RegressionRequest` before the stage controller is
allowed to write the backwards transition. Three lenses vote in parallel:

- :class:`NoveltyCritic` — "is the delta actually new world-state evidence?"
- :class:`ScopeCritic` — "is the delta inside the engagement's authorisation?"
- :class:`BudgetCritic` — "can we afford this regression?" (deterministic)

The :class:`StageRegressionCouncil` aggregates the three critiques using
the existing :class:`Council` synthesiser pattern but biases the
synthesiser with structural priors so a deterministic VETO from any
member is final.

Design notes:

- Public entry is :func:`StageRegressionCouncil.evaluate`, which threads
  the per-engagement context (``repo``, ``engagement`` dict,
  ``EngagementState``, ``engagement_id``) at call time rather than at
  council-construction time. This keeps the council a long-lived,
  engagement-agnostic object — the same object can verify regressions
  across many engagements in the same worker.
- All three members are :class:`StructuralCheck` subclasses (the
  LLM-free sibling of :class:`LLMCritic`). None invokes the LLM —
  :class:`BudgetCritic` is pure arithmetic, :class:`ScopeCritic` is
  literal CIDR/domain membership, :class:`NoveltyCritic` queries the
  world-state graph.
- Audit trail: the council writes ONLY the verdict object. The stage
  controller is the only writer of state mutations
  (``RegressionRow.council_decision`` / ``council_rationale`` columns
  and the in-memory event publish).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import structlog
from pydantic import ValidationError

from agents.council import (
    Council,
    CouncilVerdict,
    _summarize_dissent,
)
from agents.critic import Critique, StructuralCheck, Verdict
from agents.regression import RegressionRequest
from llm.router import LLMRouter
from orchestrator.models import (
    REGRESSION_BUDGET_MAX_DEFAULT,
    REGRESSION_TOKEN_RESERVE_PCT,
)
from worldstate.neo4j_repo import InMemoryGraphRepo

log = structlog.get_logger(__name__)


# ----- delta fingerprinting ----------------------------------------------- #

# Wall-clock window for the regression-storm escalation guard. A
# producer agent firing >5 distinct-fingerprint regressions on the same
# ``(target_stage, trigger_kind)`` pair within this window is treated as
# a likely bug, not legitimate enterprise discovery — the NoveltyCritic
# ESCALATEs to a human so the operator can decide. This is the ONLY
# wall-clock-based throttle on the regression mechanism; the per-delta
# replay guard handles ordinary loop-safety.
REGRESSION_STORM_WINDOW_MINUTES: int = 10
REGRESSION_STORM_THRESHOLD: int = 5


def _canonicalize_delta(delta: dict[str, Any]) -> Any:
    """Return a deterministic Python structure equivalent to ``delta``.

    Used as the input to the fingerprint hash. Rules:

    - CIDR-like strings under keys ``cidrs`` / ``excluded_cidrs`` are
      normalised via :class:`ipaddress.ip_network` (``strict=False``) so
      ``10.0.5.0/24`` and ``10.0.5.1/24`` collapse to the same canonical
      ``10.0.5.0/24``.
    - Hostname-like strings under keys ``domains`` / ``hostnames`` /
      ``urls`` are lowercased and stripped of scheme/port via the
      existing :func:`_to_hostname` helper.
    - List values are deduplicated (set conversion) and sorted
      lexicographically so order-irrelevant deltas collide.
    - Dict values are recursed with sorted keys.
    - Scalars pass through.

    The result is JSON-serialisable so the caller can pass it through
    :func:`json.dumps` with ``sort_keys=True`` for a stable byte
    representation. Unrecognised keys are still canonicalised (sorted
    lists, recursed dicts) — the goal is *any* deterministic shape, not
    a curated allow-list.
    """
    _cidr_keys = {"cidrs", "excluded_cidrs"}
    _hostname_keys = {"domains", "hostnames", "urls", "excluded_domains"}

    def _norm_cidr(s: str) -> str:
        try:
            return str(ipaddress.ip_network(str(s), strict=False))
        except (ValueError, TypeError):
            return str(s)

    def _norm_hostname(s: str) -> str:
        h = _to_hostname(str(s))
        return h if h else str(s).lower()

    def _norm_list(key: str, values: list[Any]) -> list[Any]:
        if key in _cidr_keys:
            normed = [_norm_cidr(v) for v in values]
        elif key in _hostname_keys:
            normed = [_norm_hostname(v) for v in values]
        else:
            normed = [_canon_value(v) for v in values]
        # Deduplicate while preserving JSON-serialisable types. Sorting
        # on the JSON encoding gives a stable order for heterogeneous
        # lists without requiring orderable element types.
        seen: dict[str, Any] = {}
        for item in normed:
            seen.setdefault(json.dumps(item, sort_keys=True), item)
        return [seen[k] for k in sorted(seen)]

    def _canon_value(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _canon_value(v[k]) for k in sorted(v)}
        if isinstance(v, list):
            return _norm_list("__inner__", v)
        return v

    out: dict[str, Any] = {}
    for k in sorted(delta):
        v = delta[k]
        if isinstance(v, list):
            out[k] = _norm_list(k, v)
        elif isinstance(v, dict):
            out[k] = _canon_value(v)
        else:
            out[k] = v
    return out


def fingerprint_delta(delta: dict[str, Any]) -> str:
    """sha256 hex of the canonical-JSON delta payload.

    Deterministic by construction: same delta -> same hex string.
    Order-independent for list values (sorted set), case-insensitive
    for hostnames, and CIDR-canonicalised for IP ranges. See
    :func:`_canonicalize_delta` for the canonicalisation rules.

    Returns a 64-char lowercase hex string suitable for the
    ``regressions.delta_fingerprint`` column.
    """
    canon = _canonicalize_delta(delta or {})
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Member critics                                                              #
# --------------------------------------------------------------------------- #


class NoveltyCritic(StructuralCheck):
    """Lens: is the delta actually new world-state evidence?

    The novelty lens prevents the regression mechanism from degenerating into
    the agent looping back to redo work it has already done. The check is
    deterministic: every host / service / finding ID referenced in the delta
    must either be absent from the world-state graph or contradict a node
    already there (e.g. a CIDR with no host coverage). The LLM is NOT
    consulted — this is a graph query against ``InMemoryGraphRepo``.

    Veto criteria:
    - The delta JSON is empty (nothing to act on).
    - Every entity in the delta is already represented in the world-state.
    - The same ``(target_stage, trigger_kind)`` was already
      regressed-to-and-applied this engagement (replay loop guard).

    Confidence is 1.0 for the clean approve/veto cases; we report 0.6 when
    the delta is partially novel (some entities exist, some don't) so the
    synthesiser knows the signal is mixed but lets the request through.

    Inherits from :class:`StructuralCheck` (LLM-free) instead of
    :class:`LLMCritic`. The router argument on ``__init__`` is preserved
    so call-sites that pass it still work, but it's discarded — the
    structural-check contract makes the LLM-free shape explicit.
    """

    name = "novelty_critic"
    perspective = "Graph-side check: does the delta point at world-state we don't already cover?"

    def __init__(self, router: LLMRouter | None = None) -> None:
        # ``router`` is accepted for backwards compatibility with
        # callers that still pass it; ignored because StructuralCheck
        # subclasses never call the LLM. New callers can omit it.
        _ = router  # explicit no-op for the linter
        # Per-call context, populated by StageRegressionCouncil.evaluate
        # before invoking the council. The fields exist so the override of
        # review() has a stable place to read them; the council resets them
        # on every call so cross-engagement leakage is impossible.
        self._request: RegressionRequest | None = None
        self._repo: InMemoryGraphRepo | None = None
        self._engagement_id: UUID | None = None
        # Set of approved-and-applied ``(target_stage, trigger_kind,
        # delta_fingerprint)`` triples for this engagement, read from
        # Postgres at evaluate-time. Triples (not pairs) so genuinely-
        # novel deltas with the same ``(stage, trigger)`` aren't treated
        # as replays — that mattered for real multi-segment engagements
        # where lateral discovers three different uncovered CIDRs.
        self._applied_keys: set[tuple[str, str, str]] = set()
        # Pre-computed fingerprint of the current request, threaded in
        # by the council so the replay check is O(1).
        self._current_fingerprint: str = ""
        # Count of approved-and-applied regressions for the current
        # ``(target_stage, trigger_kind)`` pair within the last
        # ``REGRESSION_STORM_WINDOW_MINUTES`` wall-clock minutes. The
        # council seeds this before review(); >threshold ESCALATEs.
        self._recent_pair_count: int = 0

    async def review(  # noqa: PLR0911 — each verdict shape needs its own early-return path
        self,
        subject: dict[str, Any],
        *,
        engagement_id: UUID | None = None,
    ) -> Critique:
        """Deterministic graph-side novelty check.

        :class:`StructuralCheck` implementation — no LLM call.
        The StageRegressionCouncil installs ``self._request`` /
        ``self._repo`` / ``self._engagement_id`` immediately before
        invoking this method so we have everything we need without
        re-parsing ``subject``.
        """
        req = self._request
        repo = self._repo
        eid = self._engagement_id
        if req is None or repo is None or eid is None:
            # Defensive: a caller invoked review() outside of evaluate(). The
            # behaviour matches the parser-failure shape from BaseCritic.
            return Critique(
                verdict=Verdict.ESCALATE,
                confidence=0.0,
                primary_concern="novelty critic called without engagement context",
                issues=["missing_context"],
                critic_name=self.name,
            )

        delta = dict(req.delta or {})
        # 1. Empty delta → there is nothing to do on re-entry.
        if not delta:
            return Critique(
                verdict=Verdict.VETO,
                confidence=1.0,
                primary_concern=(
                    "regression has empty delta — nothing for the re-entered stage to act on"
                ),
                issues=["empty_delta"],
                critic_name=self.name,
            )

        # 2. Replay-loop guard, keyed on the full ``(target_stage,
        # trigger_kind, delta_fingerprint)`` triple. The coarser
        # (stage, trigger) pair-key silently rejected the second and
        # third in-scope CIDR discovery on real multi-segment
        # engagements. Triple-key means only identical-delta replays
        # are rejected; genuinely-novel CIDRs sail through.
        target_stage_str = str(
            req.target_stage.value if hasattr(req.target_stage, "value") else req.target_stage
        )
        trigger_kind_str = str(
            req.trigger_kind.value if hasattr(req.trigger_kind, "value") else req.trigger_kind
        )
        fp = self._current_fingerprint or fingerprint_delta(delta)
        triple = (target_stage_str, trigger_kind_str, fp)
        if triple in self._applied_keys:
            return Critique(
                verdict=Verdict.VETO,
                confidence=1.0,
                primary_concern=(
                    f"regression to {target_stage_str}/{trigger_kind_str} with "
                    f"this exact delta (fp={fp[:12]}…) was already applied this "
                    "engagement"
                ),
                issues=["replay_loop"],
                critic_name=self.name,
            )

        # 2a. Regression-storm guard. Many genuinely-novel deltas on the
        # same ``(stage, trigger)`` pair in a short wall-clock window
        # imply a producer bug (e.g. an emitter looping on graph-query
        # results). ESCALATE so the operator can decide; never VETO
        # silently, since the underlying deltas might be legitimate.
        if self._recent_pair_count >= REGRESSION_STORM_THRESHOLD:
            return Critique(
                verdict=Verdict.ESCALATE,
                confidence=1.0,
                primary_concern=(
                    f"regression storm: {self._recent_pair_count} approved "
                    f"{target_stage_str}/{trigger_kind_str} regressions within "
                    f"the last {REGRESSION_STORM_WINDOW_MINUTES} minutes — "
                    "operator review required"
                ),
                issues=[f"regression_storm_{target_stage_str}_{trigger_kind_str}"],
                critic_name=self.name,
            )

        # 3. Graph existence check. Each delta key is matched against the
        # corresponding world-state node kind:
        #   - cidrs → host-coverage check (ws-07): a CIDR whose every
        #                   in-scope address is already a Host counts as
        #                   existing, not novel, so a re-swept subnet hits the
        #                   already-known VETO instead of re-triggering recon.
        #   - host_ids → look up by id.
        #   - service_ids → look up by id.
        #   - finding_ids → look up by id.
        #   - hosts → IP-string lookup against Host.ip
        try:
            novel, existing = await _classify_delta_entities(repo, eid, delta)
        except Exception as e:
            log.warning("novelty_critic.graph_query_failed", err=str(e))
            return Critique(
                verdict=Verdict.ESCALATE,
                confidence=0.0,
                primary_concern=f"world-state graph unavailable: {e}",
                issues=["graph_unavailable"],
                critic_name=self.name,
            )

        if novel == 0 and existing > 0:
            return Critique(
                verdict=Verdict.VETO,
                confidence=1.0,
                primary_concern=(
                    "every entity referenced in the delta already exists in the world-state"
                ),
                issues=["delta_already_known"],
                critic_name=self.name,
            )
        if novel > 0 and existing == 0:
            return Critique(
                verdict=Verdict.APPROVE,
                confidence=1.0,
                primary_concern=f"{novel} novel entity/entities in delta",
                issues=[],
                critic_name=self.name,
            )
        if novel == 0 and existing == 0:
            # No identifiable entities in delta (e.g. only opaque blobs). Treat
            # as ambiguous and let the council escalate via the synthesiser.
            return Critique(
                verdict=Verdict.APPROVE,
                confidence=0.6,
                primary_concern="delta has no graph-checkable entities (CIDR-only or opaque)",
                issues=["unverifiable_delta"],
                critic_name=self.name,
            )
        # Mixed: some novel, some already known — partial novelty.
        return Critique(
            verdict=Verdict.APPROVE,
            confidence=0.6,
            primary_concern=f"partial novelty: {novel} novel, {existing} already in graph",
            issues=["partial_novelty"],
            critic_name=self.name,
        )


class ScopeCritic(StructuralCheck):
    """Lens: is the delta inside the engagement's authorisation scope?

    Reuses the literal-scope logic familiar from :class:`ScopeBoundaryCritic`:
    every IP, CIDR, domain, and URL referenced in the delta must match the
    engagement's allow-list and miss the engagement's exclude-list. The check
    is deterministic; we VETO on any out-of-scope hit and ESCALATE when a
    domain wildcard could resolve to CDN-class infrastructure outside the
    customer's perimeter.

    Now a :class:`StructuralCheck` (LLM-free). The ``router``
    argument is accepted for backwards-compat but ignored.
    """

    name = "scope_critic"
    perspective = "Literal scope reader; assume the engagement letter is the only source of truth."

    def __init__(self, router: LLMRouter | None = None) -> None:
        _ = router  # no-LLM check; router accepted for compat
        self._request: RegressionRequest | None = None
        self._scope: dict[str, list[str]] = {}

    async def review(  # noqa: PLR0911 — each scope verdict shape needs its own early-return
        self,
        subject: dict[str, Any],
        *,
        engagement_id: UUID | None = None,
    ) -> Critique:
        """Deterministic scope-membership check.

        Each entity in the delta gets evaluated against four lists:
        ``cidrs`` and ``excluded_cidrs`` (IP / CIDR), ``domains`` and
        ``excluded_domains`` (hostname / URL). One out-of-scope hit vetoes
        the whole regression; one wildcard-ambiguous hit escalates.
        """
        req = self._request
        scope = self._scope
        if req is None:
            return Critique(
                verdict=Verdict.ESCALATE,
                confidence=0.0,
                primary_concern="scope critic called without request context",
                issues=["missing_context"],
                critic_name=self.name,
            )

        ips_cidrs, hostnames = _extract_scope_entities(dict(req.delta or {}))
        if not ips_cidrs and not hostnames:
            # Nothing to check against scope — the delta is, e.g., only opaque
            # blob refs. Treat as approve; the novelty critic owns the empty
            # case, and the audit row preserves the delta for review.
            return Critique(
                verdict=Verdict.APPROVE,
                confidence=0.6,
                primary_concern="delta has no scope-checkable entities",
                issues=[],
                critic_name=self.name,
            )

        cidrs = list(scope.get("cidrs", []))
        excluded_cidrs = list(scope.get("excluded_cidrs", []))
        domains = list(scope.get("domains", []))
        excluded_domains = list(scope.get("excluded_domains", []))

        # IP / CIDR membership.
        for entity in ips_cidrs:
            decision = _ip_or_cidr_in_scope(entity, cidrs, excluded_cidrs)
            if decision == "out":
                return Critique(
                    verdict=Verdict.VETO,
                    confidence=1.0,
                    primary_concern=f"delta entity '{entity}' is outside engagement CIDR scope",
                    issues=["out_of_scope_ip"],
                    critic_name=self.name,
                )
            if decision == "excluded":
                return Critique(
                    verdict=Verdict.VETO,
                    confidence=1.0,
                    primary_concern=f"delta entity '{entity}' is in an excluded CIDR",
                    issues=["excluded_ip"],
                    critic_name=self.name,
                )

        # Hostname / domain membership.
        ambiguous = False
        for entity in hostnames:
            hostname = _to_hostname(entity)
            if not hostname:
                ambiguous = True
                continue
            decision = _domain_in_scope(hostname, domains, excluded_domains)
            if decision == "out":
                return Critique(
                    verdict=Verdict.VETO,
                    confidence=1.0,
                    primary_concern=f"delta entity '{entity}' is outside engagement domain scope",
                    issues=["out_of_scope_domain"],
                    critic_name=self.name,
                )
            if decision == "excluded":
                return Critique(
                    verdict=Verdict.VETO,
                    confidence=1.0,
                    primary_concern=f"delta entity '{entity}' is in an excluded domain",
                    issues=["excluded_domain"],
                    critic_name=self.name,
                )
            if decision == "wildcard":
                ambiguous = True

        if ambiguous:
            return Critique(
                verdict=Verdict.ESCALATE,
                confidence=0.5,
                primary_concern=(
                    "delta contains a wildcard or unresolved hostname; CDN / shared-tenant "
                    "membership cannot be confirmed automatically"
                ),
                issues=["wildcard_ambiguous"],
                critic_name=self.name,
            )

        return Critique(
            verdict=Verdict.APPROVE,
            confidence=1.0,
            primary_concern="every delta entity is unambiguously in scope",
            issues=[],
            critic_name=self.name,
        )


class BudgetCritic(StructuralCheck):
    """Lens: can we afford this regression? (deterministic, LLM-free).

    A true :class:`StructuralCheck` — no LLM call ever fires.
    The budget gate is purely arithmetic against
    :class:`EngagementState.regressions_used`, the engagement's
    ``regression_budget_max`` column, and the engagement's
    ``token_budget``, so the check is free and predictable.

    The per-engagement ``regression_budget_max`` is a runaway-loop
    circuit breaker, not a design constraint. A value of ``0`` means
    unlimited (the default for new engagements). When non-zero and met,
    the rationale string is deliberately ``runaway_loop_circuit_breaker``
    so anyone reading audit logs in a year understands this is NOT a
    normal stop condition — loop safety lives in NoveltyCritic's
    per-fingerprint replay rejection and ScopeCritic's authorisation.

    Veto criteria:
    - ``regression_budget_max > 0`` and
      ``regressions_used >= regression_budget_max``.
    - Token reserve exhausted: ``total_tokens > token_budget * (1 - PCT)``.
    """

    name = "budget_critic"
    perspective = "Deterministic budget gate; no LLM."

    def __init__(self, router: LLMRouter | None = None) -> None:
        _ = router  # no-LLM check; router accepted for compat
        self._regressions_used: int = 0
        self._regression_budget_max: int = 0  # 0 = unlimited
        self._token_budget: int = 0
        self._tokens_used: int = 0

    async def review(
        self,
        subject: dict[str, Any],
        *,
        engagement_id: UUID | None = None,
    ) -> Critique:
        """Deterministic budget check; never calls the LLM."""
        cap = self._regression_budget_max
        if cap > 0 and self._regressions_used >= cap:
            return Critique(
                verdict=Verdict.VETO,
                confidence=1.0,
                primary_concern=(
                    f"runaway-loop circuit breaker tripped: {self._regressions_used} "
                    f"regressions used >= {cap} cap (this is not a normal stop; "
                    "loop safety lives in NoveltyCritic per-fingerprint replay)"
                ),
                issues=["runaway_loop_circuit_breaker"],
                critic_name=self.name,
            )

        # Token reserve check: if the engagement has already burned more than
        # (1 - reserve_pct) of its token budget, refuse to spend more on a
        # regression-driven sub-pipeline. token_budget == 0 means "no cap"
        # — skip the reserve check entirely.
        if self._token_budget > 0:
            spend_ceiling = int(self._token_budget * (1.0 - REGRESSION_TOKEN_RESERVE_PCT))
            if self._tokens_used > spend_ceiling:
                pct_burned = self._tokens_used / self._token_budget
                return Critique(
                    verdict=Verdict.VETO,
                    confidence=1.0,
                    primary_concern=(
                        f"token reserve exhausted: used {self._tokens_used} / "
                        f"{self._token_budget} ({pct_burned:.0%}); "
                        f"reserve floor is {REGRESSION_TOKEN_RESERVE_PCT:.0%}"
                    ),
                    issues=["token_reserve_exhausted"],
                    critic_name=self.name,
                )

        cap_str = "unlimited" if cap == 0 else str(cap)
        return Critique(
            verdict=Verdict.APPROVE,
            confidence=1.0,
            primary_concern=(
                f"budget headroom available: {self._regressions_used}/{cap_str} "
                "regressions used"
            ),
            issues=[],
            critic_name=self.name,
        )


# --------------------------------------------------------------------------- #
# Council                                                                     #
# --------------------------------------------------------------------------- #


class StageRegressionCouncil(Council):
    """Three-member council that votes on a :class:`RegressionRequest`.

    Constructed once per worker; reused across engagements. The per-call
    context (request + repo + engagement dict + state + id) is threaded
    through :meth:`evaluate`, which seeds the members' deterministic-review
    paths and then runs the standard :class:`Council` synthesiser.

    **Not safe for concurrent ``evaluate()`` calls.** Member instance
    attributes are seeded then read without synchronisation; the single-
    writer-per-engagement invariant (one worker slot, one step at a time)
    guarantees sequential access in production. Do not wrap ``evaluate()``
    in ``asyncio.gather()``.

    Synthesis rule (matches :class:`Council` defaults plus structural priors):

    - Any member VETO → council VETO with the vetoing member's rationale.
    - Any member ESCALATE (and no VETO) → council ESCALATE.
    - All members APPROVE → council APPROVE with the lowest member confidence.

    Implementation note: :class:`Council.deliberate` already implements the
    "any veto → council VETO" and "unanimous high-confidence approve →
    APPROVE" shortcuts. The only behavioural gap is "any escalate, no veto
    → council ESCALATE" — :meth:`Council.deliberate` would otherwise punt to
    the synthesiser. We pre-empt that case in :meth:`evaluate` so the
    structural verdict is deterministic and free for the escalate path.
    """

    def __init__(self, router: LLMRouter) -> None:
        super().__init__(
            name="stage_regression_council",
            router=router,
            members=[
                NoveltyCritic(router),
                ScopeCritic(router),
                BudgetCritic(router),
            ],
        )

    async def evaluate(
        self,
        request: RegressionRequest,
        engagement_state: Any,
        engagement: dict[str, Any],
        repo: InMemoryGraphRepo,
        engagement_id: UUID,
    ) -> CouncilVerdict:
        """Seed per-call context onto the members and deliberate.

        ``engagement_state`` is typed ``Any`` to avoid a circular import
        with :mod:`orchestrator.stage_controller`; the only attributes
        read are ``regressions_used`` and ``total_tokens``, both present
        on :class:`EngagementState`.
        """
        # Find each member by name; the council was built with exactly these.
        novelty = _find_member(self, NoveltyCritic, "novelty_critic")
        scope_critic = _find_member(self, ScopeCritic, "scope_critic")
        budget = _find_member(self, BudgetCritic, "budget_critic")

        # Pre-compute the fingerprint once so the audit row, the
        # NoveltyCritic, and any downstream consumer see the same value.
        current_fp = fingerprint_delta(dict(request.delta or {}))
        target_stage_str = str(
            request.target_stage.value
            if hasattr(request.target_stage, "value")
            else request.target_stage
        )
        trigger_kind_str = str(
            request.trigger_kind.value
            if hasattr(request.trigger_kind, "value")
            else request.trigger_kind
        )

        # Seed NoveltyCritic.
        novelty._request = request
        novelty._repo = repo
        novelty._engagement_id = engagement_id
        novelty._current_fingerprint = current_fp
        novelty._applied_keys = await _applied_keys(engagement_id)
        novelty._recent_pair_count = await _recent_pair_count(
            engagement_id, target_stage_str, trigger_kind_str
        )

        # Seed ScopeCritic.
        scope_critic._request = request
        scope_critic._scope = dict(engagement.get("scope") or {})

        # Seed BudgetCritic. ``regression_budget_max`` is read from the
        # engagement dict (set by ``build_engagement_context``). Falls back
        # to ``REGRESSION_BUDGET_MAX_DEFAULT`` (50) only when the dict
        # is missing the key entirely — legacy tests + degraded paths.
        budget._regressions_used = int(getattr(engagement_state, "regressions_used", 0) or 0)
        budget._regression_budget_max = int(
            engagement.get("regression_budget_max", REGRESSION_BUDGET_MAX_DEFAULT) or 0
        )
        budget._token_budget = int(engagement.get("token_budget", 0) or 0)
        budget._tokens_used = int(getattr(engagement_state, "total_tokens", 0) or 0)

        subject = {
            "regression_request": request.model_dump(mode="json"),
            "regressions_used": budget._regressions_used,
            "token_budget": budget._token_budget,
            "tokens_used": budget._tokens_used,
        }

        # Pre-compute member critiques so we can short-circuit the synthesiser
        # on the deterministic-escalate case. Council.deliberate would
        # otherwise pay for a synthesiser LLM call to resolve an unambiguous
        # ESCALATE — wasted spend for a structurally-decided verdict.
        critiques = [await m.review(subject, engagement_id=engagement_id) for m in self.members]
        vetoes = [c for c in critiques if c.verdict == Verdict.VETO]
        if vetoes:
            strongest = max(vetoes, key=lambda c: c.confidence)
            return CouncilVerdict(
                decision=Verdict.VETO,
                confidence=strongest.confidence,
                rationale=strongest.primary_concern,
                member_critiques=critiques,
                dissent_summary=None,
                total_cost_usd=sum(c.cost_usd for c in critiques),
            )
        escalates = [c for c in critiques if c.verdict == Verdict.ESCALATE]
        if escalates:
            strongest = max(escalates, key=lambda c: c.confidence)
            return CouncilVerdict(
                decision=Verdict.ESCALATE,
                confidence=strongest.confidence,
                rationale=strongest.primary_concern,
                member_critiques=critiques,
                dissent_summary=None,
                total_cost_usd=sum(c.cost_usd for c in critiques),
            )
        # All approve. Pick the lowest member confidence so the controller
        # sees the most cautious member's signal.
        approves = [c for c in critiques if c.verdict == Verdict.APPROVE]
        if len(approves) == len(critiques):
            lowest = min(approves, key=lambda c: c.confidence)
            return CouncilVerdict(
                decision=Verdict.APPROVE,
                confidence=lowest.confidence,
                rationale="; ".join(c.primary_concern for c in critiques),
                member_critiques=critiques,
                dissent_summary=None,
                total_cost_usd=sum(c.cost_usd for c in critiques),
            )

        # The "mixed verdicts that don't match
        # structural rules" path was dead code. All three members of
        # this council are deterministic and only emit one of
        # {APPROVE, VETO, ESCALATE} (see ``NoveltyCritic.review``,
        # ``ScopeCritic.review``, ``BudgetCritic.review``). The three
        # branches above (any-veto / any-escalate / all-approve) cover
        # every reachable verdict combination. The previous fallback
        # synthesiser fired an LLM call on a verdict shape no critic
        # could produce — pure cost variance for a structurally-decided
        # council. Replace it with a hard ESCALATE so an operator sees
        # the impossible case if a future critic regression reintroduces
        # REVISE without updating the synthesiser path.
        log.warning(
            "regression_council.unreachable_verdict_combo",
            critiques=[
                {"name": c.critic_name, "verdict": c.verdict.value}
                for c in critiques
            ],
            note=(
                "all three structural critics should only emit "
                "APPROVE/VETO/ESCALATE; reaching this branch implies a "
                "subclass regression"
            ),
        )
        return CouncilVerdict(
            decision=Verdict.ESCALATE,
            confidence=0.0,
            rationale=(
                "unreachable mixed-verdict combination from structural critics — "
                "operator review required"
            ),
            member_critiques=critiques,
            dissent_summary=_summarize_dissent(critiques),
            total_cost_usd=sum(c.cost_usd for c in critiques),
        )


# Removed _REGRESSION_SYNTH_SYSTEM and _synthesise() —
# they were the LLM-call body of the dead-code path above. Kept the
# import of LLM router types out of the module's runtime hot path
# (they're still imported at module level for type clarity).


def build_regression_council(router: LLMRouter) -> StageRegressionCouncil:
    """Public factory. One instance per worker; reused across
    engagements. The per-call context is threaded through
    :meth:`StageRegressionCouncil.evaluate`."""
    return StageRegressionCouncil(router)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _find_member(council: Council, cls: type, name: str) -> Any:
    """Locate a member by name, raise if absent.

    Keeps :meth:`StageRegressionCouncil.evaluate` honest about which members
    are expected to be on the council. ``cls`` is informational; we match by
    ``name`` because the test fixtures construct members directly.
    """
    for m in council.members:
        if m.name == name:
            return m
    raise RuntimeError(f"StageRegressionCouncil missing member: {name}")


async def _classify_delta_entities(
    repo: InMemoryGraphRepo,
    engagement_id: UUID,
    delta: dict[str, Any],
) -> tuple[int, int]:
    """Return (novel_count, existing_count) for graph-checkable IDs
    in the delta.

    Known delta-key shapes (kept loose to match the open-ended
    :class:`RegressionRequest.delta` contract):

    - ``cidrs`` — list[str]. ws-07: a CIDR is novel ONLY if it has at least one
      in-scope address with no Host coverage in the graph. A re-swept subnet
      whose every host already exists is counted as ``existing`` so the
      already-known VETO path can fire (closing the supply side of the
      regression loop). A malformed/huge CIDR we cannot enumerate cheaply is
      treated as novel (fail-safe toward letting recon look).
    - ``host_ids`` / ``service_ids`` / ``finding_ids`` — lists of UUID
      strings. Match against the corresponding node kind.
    - ``hosts`` — list of IP strings. Match against ``Host.ip``.
    """
    novel = 0
    existing = 0

    # CIDR coverage check (ws-07): a CIDR every in-scope address of which is
    # already a Host in the graph is NOT novel — counting it novel let an
    # already-enumerated subnet re-trigger recon.
    cidrs = delta.get("cidrs") or []
    if cidrs:
        existing_ips = await _existing_host_ips(repo, engagement_id)
        for cidr in cidrs:
            if _cidr_fully_covered(str(cidr), existing_ips):
                existing += 1
            else:
                novel += 1

    # ID-keyed lookups.
    for delta_key, node_kind in (
        ("host_ids", "Host"),
        ("service_ids", "Service"),
        ("finding_ids", "Finding"),
    ):
        ids = delta.get(delta_key) or []
        if not ids:
            continue
        existing_ids = await _existing_ids(repo, engagement_id, node_kind)
        for nid in ids:
            nid_str = str(nid)
            if nid_str in existing_ids:
                existing += 1
            else:
                novel += 1

    # Plain-IP lookup.
    ips = delta.get("hosts") or []
    if ips:
        existing_ips = await _existing_host_ips(repo, engagement_id)
        for ip in ips:
            if str(ip) in existing_ips:
                existing += 1
            else:
                novel += 1

    return novel, existing


async def _existing_ids(repo: InMemoryGraphRepo, engagement_id: UUID, kind: str) -> set[str]:
    """Return the set of node IDs of ``kind`` in the engagement's world-state."""
    nodes = await repo.get_nodes_by_kind(engagement_id, kind)
    return {str(n.get("id")) for n in nodes if n.get("id") is not None}


async def _existing_host_ips(repo: InMemoryGraphRepo, engagement_id: UUID) -> set[str]:
    """Return the set of Host.ip strings already in the world-state."""
    nodes = await repo.get_nodes_by_kind(engagement_id, "Host")
    return {str(n.get("ip")) for n in nodes if n.get("ip")}


# Upper bound on the number of usable addresses we will enumerate when
# coverage-checking a CIDR (ws-07). A subnet larger than this is treated as
# novel without enumeration — checking a /8 host-by-host would be wasteful and
# such a sweep almost certainly has un-enumerated space anyway.
_CIDR_COVERAGE_MAX_HOSTS = 4096


def _cidr_fully_covered(cidr: str, existing_ips: set[str]) -> bool:
    """True iff every usable in-scope address of ``cidr`` is already a Host.ip.

    Used by the NoveltyCritic (ws-07) to decide whether a ``cidrs`` delta is a
    re-sweep of already-enumerated space (``existing``) or genuinely new
    (``novel``). Returns ``False`` (treat as novel) when the CIDR is malformed,
    too large to enumerate cheaply, or has zero usable hosts — i.e. we only
    declare "covered" when we can positively prove every host is present.
    """
    try:
        net = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return False
    hosts = net.hosts() if net.num_addresses > 2 else net
    covered_any = False
    count = 0
    for addr in hosts:
        count += 1
        if count > _CIDR_COVERAGE_MAX_HOSTS:
            return False
        if str(addr) not in existing_ips:
            return False
        covered_any = True
    # An empty host set (e.g. /32 reduced to nothing) is not provably covered.
    return covered_any


async def _applied_keys(
    engagement_id: UUID,
) -> set[tuple[str, str, str]]:
    """Return ``(target_stage, trigger_kind, delta_fingerprint)`` triples
    for every approved-and-applied regression this engagement.

    Reads ``RegressionRow`` rows where ``engagement_id`` matches AND
    ``applied_at IS NOT NULL`` (the marker that the controller committed
    the backwards transition). The NoveltyCritic uses this set to VETO
    an identical-delta replay — the only fingerprint-equal regression
    counts as a replay; genuinely-novel deltas with the same
    ``(stage, trigger)`` sail through.

    Rows with NULL ``delta_fingerprint`` (legacy rows written before
    the column existed) are skipped: a NULL row could never match the
    canonical hash of any current request, so excluding them is
    behaviour-preserving and avoids spurious VETOes on legacy data.

    Best-effort, matching the existing audit-trail tolerance pattern
    (``record_tool_run`` / ``record_regression``): a Postgres outage
    logs a warning and returns an empty set. A degraded read here only
    means the replay guard is temporarily soft; the budget critic
    (when an operator has set a cap) and the scope critic still bound
    the engagement.
    """
    try:
        # Local imports keep the critic module's top-level dependencies
        # narrow — only the controller-side resolver pays for the SQL
        # session machinery at import time.
        from sqlalchemy import select  # noqa: PLC0415

        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import RegressionRow  # noqa: PLC0415

        async with session_scope() as s:
            stmt = select(
                RegressionRow.target_stage,
                RegressionRow.trigger_kind,
                RegressionRow.delta_fingerprint,
            ).where(
                RegressionRow.engagement_id == engagement_id,
                RegressionRow.applied_at.is_not(None),
                RegressionRow.delta_fingerprint.is_not(None),
            )
            rows = (await s.execute(stmt)).all()
            return {(str(t), str(k), str(f)) for t, k, f in rows}
    except Exception as e:
        log.warning("regression_council.applied_keys_query_failed", err=str(e))
        return set()


async def _recent_pair_count(
    engagement_id: UUID,
    target_stage: str,
    trigger_kind: str,
) -> int:
    """Count approved-and-applied regressions on the given
    ``(target_stage, trigger_kind)`` pair within the last
    :data:`REGRESSION_STORM_WINDOW_MINUTES` minutes wall-clock.

    NoveltyCritic uses this for the regression-storm ESCALATE guard.
    Best-effort like its sibling; returns ``0`` on persistence failure
    so the storm guard fails open rather than blocking a healthy
    engagement on a transient Postgres hiccup.
    """
    try:
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from sqlalchemy import func, select  # noqa: PLC0415

        from orchestrator.db import session_scope  # noqa: PLC0415
        from orchestrator.models import RegressionRow  # noqa: PLC0415

        cutoff = datetime.now(UTC) - timedelta(minutes=REGRESSION_STORM_WINDOW_MINUTES)
        async with session_scope() as s:
            stmt = select(func.count(RegressionRow.id)).where(
                RegressionRow.engagement_id == engagement_id,
                RegressionRow.target_stage == target_stage,
                RegressionRow.trigger_kind == trigger_kind,
                RegressionRow.applied_at.is_not(None),
                RegressionRow.applied_at >= cutoff,
                # Exclude legacy NULL-fingerprint rows so a freshly
                # migrated deployment doesn't inflate the storm count.
                RegressionRow.delta_fingerprint.is_not(None),
            )
            result = await s.execute(stmt)
            return int(result.scalar_one() or 0)
    except Exception as e:
        log.warning("regression_council.recent_pair_count_query_failed", err=str(e))
        return 0


# --------------------------------------------------------------------------- #
# Scope helpers — literal CIDR / domain membership                            #
# --------------------------------------------------------------------------- #


def _extract_scope_entities(delta: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Pull IPs/CIDRs and hostnames/URLs out of a delta blob.

    Returns ``(ip_or_cidr, hostname_or_url)``. The shape of the delta is
    open-ended (it depends on the trigger kind), so we look at every key we
    recognise and aggregate.
    """
    ips_cidrs: list[str] = []
    hostnames: list[str] = []
    for key in ("cidrs", "hosts", "ips"):
        for v in delta.get(key) or []:
            ips_cidrs.append(str(v))
    for key in ("domains", "urls", "hostnames"):
        for v in delta.get(key) or []:
            hostnames.append(str(v))
    return ips_cidrs, hostnames


def _ip_or_cidr_in_scope(entity: str, cidrs: list[str], excluded_cidrs: list[str]) -> str:
    """Return one of ``"in"`` / ``"out"`` / ``"excluded"`` for an IP or CIDR.

    Treats both single IPs and CIDR ranges uniformly: an entity is "in" iff
    every IP it covers is contained by at least one allow-list CIDR; it is
    "excluded" if it overlaps any excluded CIDR.
    """
    try:
        net = ipaddress.ip_network(entity, strict=False)
    except (ValueError, TypeError):
        return "out"

    for x in excluded_cidrs or []:
        try:
            ex = ipaddress.ip_network(x, strict=False)
        except (ValueError, TypeError):
            continue
        if net.overlaps(ex):
            return "excluded"

    for c in cidrs or []:
        try:
            allow = ipaddress.ip_network(c, strict=False)
        except (ValueError, TypeError):
            continue
        # Version-agnostic containment: explicit int range comparison avoids
        # `subnet_of`'s same-version arg constraint. The two ranges must both
        # be IPv4 or both IPv6 for containment to be meaningful; the version
        # mismatch path falls through to "out" naturally.
        if net.version != allow.version:
            continue
        if int(net.network_address) >= int(allow.network_address) and int(
            net.broadcast_address
        ) <= int(allow.broadcast_address):
            return "in"
    return "out"


def _to_hostname(s: str) -> str | None:
    """Strip scheme / path / port from a URL-ish string. Returns ``None`` for
    obvious junk."""
    if not s:
        return None
    if "://" in s:
        try:
            parsed = urlparse(s)
            if parsed.hostname:
                return parsed.hostname
        except Exception:
            return None
    # Treat bare strings: drop port and trailing slash/path.
    s2 = s.split("/", 1)[0]
    s2 = s2.split(":", 1)[0]
    return s2.lower() if s2 else None


def _domain_in_scope(hostname: str, domains: list[str], excluded_domains: list[str]) -> str:
    """Return ``"in"`` / ``"out"`` / ``"excluded"`` / ``"wildcard"`` for a
    hostname against engagement domain rules.

    ``"wildcard"`` is returned when the match is via a leading ``*``-glob in
    the allow-list (e.g. ``*.example.com``) — the rego scope policy treats
    this as in-scope but the council escalates because the resolved subdomain
    could be CDN-class shared infrastructure.
    """
    hn = hostname.lower()
    for ex in excluded_domains or []:
        if _match_domain(hn, ex.lower()):
            return "excluded"
    for d in domains or []:
        dl = d.lower()
        if dl.startswith("*.") and _match_domain(hn, dl[2:]):
            return "wildcard"
        if _match_domain(hn, dl):
            return "in"
    return "out"


def _match_domain(hostname: str, domain: str) -> bool:
    """Exact or true-subdomain match. ``foo.example.com`` matches
    ``example.com``; ``evilexample.com`` does NOT."""
    if hostname == domain:
        return True
    return hostname.endswith("." + domain)


__all__ = [
    "REGRESSION_STORM_THRESHOLD",
    "REGRESSION_STORM_WINDOW_MINUTES",
    "BudgetCritic",
    "NoveltyCritic",
    "ScopeCritic",
    "StageRegressionCouncil",
    "build_regression_council",
    "fingerprint_delta",
]


# Silence "imported but unused" if a future refactor drops ValidationError.
_ = ValidationError
