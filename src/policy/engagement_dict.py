"""Build the policy-input engagement dict from an EngagementRow.

This module is the single producer of the engagement-context shape — agents
must not hand-roll their own dict. It populates allowed_risk_classes and
window.start_ns/end_ns, both of which the policy needs but agents tended
to omit.

Mode is a single axis controlling autonomy. Each mode maps to
(allowed_risk_classes, approval_policy): recon_only/assess allow recon+enum
only (auto_approve); guided allows exploit+post-exploit with phase gates;
autonomous allows the same with auto_approve. Non-destructive EDB scripts
(remote/local/privesc/webapp/bypass) auto-run under autonomous via
``edb_auto_exploit``; in guided they surface to the operator.

Destructiveness is a SEPARATE axis from mode: the per-engagement
``destructive_opt_in`` flag (fail-closed ``False``). When off, destructive
tools/PoCs and DoS EDB modules are blocked (policy DENY
``destructive_not_authorized`` / PoC ``destructive_without_optin``). When
on, they are allowed but always routed through human/critic approval — a
destructive action is never executed silently. The flag is surfaced here
and honored by ``policy.scope.evaluate``, ``poc_runtime`` and
``tools.edb_exec``.
"""

from __future__ import annotations

import warnings
from typing import Any

from orchestrator.models import EngagementRow
from worldstate.schema import normalize_mode

# Mode → (allowed_risk_classes, approval_policy). Destructive is never
# included; destructive attack paths are reported as findings, not executed.
#
# ``assess`` runs the full triage/chain-planning pipeline but performs *no
# exploitation*; its allow-set is recon+enum, matching ``recon_only``. (If a
# chain-planning tool were ever tagged ``exploit``-class it would be denied
# here by design — assess never exploits.)
_MODE_MAP: dict[str, tuple[list[str], str]] = {
    "recon_only": (["recon", "enum"], "auto_approve"),
    "assess": (["recon", "enum"], "auto_approve"),
    "guided": (["recon", "enum", "exploit", "post-exploit"], "phase_gate"),
    "autonomous": (["recon", "enum", "exploit", "post-exploit"], "auto_approve"),
}

# Least-privilege fallback for an unrecognized / future / typo'd mode value.
# Unknown modes must fail toward the most-restrictive recon-only posture,
# NOT toward exploitation-capable ``guided`` (policy-08).
_FALLBACK_MODE = "recon_only"


def build_engagement_context(row: EngagementRow) -> dict[str, Any]:
    """Build the engagement dict consumed by ``policy.scope.evaluate()``.

    The returned dict is also consumed by agents (``allowed_risk_classes``,
    ``per_tool_token_budget``, ``window``, etc.) so the structure is shared
    between policy and agent code.
    """
    mode = normalize_mode(row.mode or "guided")
    if mode not in _MODE_MAP:
        warnings.warn(
            f"unknown engagement mode {mode!r}; falling back to least-privilege "
            f"{_FALLBACK_MODE!r} (recon-only)",
            RuntimeWarning,
            stacklevel=2,
        )
        mode = _FALLBACK_MODE
    allowed, approval_policy = _MODE_MAP[mode]

    scope_raw = row.scope or {}
    # Optional operator opt-in to permit an all-internet CIDR (0.0.0.0/0,
    # ::/0) in scope. Fail-closed default: absent/false means such prefixes
    # are ignored by the scope check (policy-07).
    allow_unbounded = bool(scope_raw.get("allow_unbounded_cidrs", False))
    # Optional recurring business-hours band, carried into the policy window
    # so the time-window check can enforce it (policy-04).
    allowed_hours_utc = scope_raw.get("allowed_hours_utc")

    window: dict[str, Any] = {
        "start_ns": int(row.time_window_start.timestamp() * 1_000_000_000),
        "end_ns": int(row.time_window_end.timestamp() * 1_000_000_000),
    }
    if allowed_hours_utc is not None:
        window["allowed_hours_utc"] = allowed_hours_utc

    return {
        "id": str(row.id),
        "name": row.name,
        "mode": mode,
        "allowed_risk_classes": list(allowed),
        # Reserved for an operator-managed exclusion list; empty = no exclusions.
        "no_exploit_list": [],
        "scope": {
            "cidrs": scope_raw.get("cidrs", []),
            "domains": scope_raw.get("domains", []),
            "urls": scope_raw.get("urls", []),
            "excluded_cidrs": scope_raw.get("excluded_cidrs", []),
            "excluded_domains": scope_raw.get("excluded_domains", []),
            # Path carve-outs (e.g. /admin, /billing) for URL targets. Was
            # silently dropped before; now threaded so the policy gate can
            # enforce it (policy-02).
            "excluded_paths": scope_raw.get("excluded_paths", []),
            # Operator opt-in to accept an all-internet CIDR. Fail-closed
            # default False (policy-07).
            "allow_unbounded_cidrs": allow_unbounded,
        },
        "window": window,
        "authorization_artifact_sha256": row.authorization_artifact_sha256,
        # Per-tool token ceilings for the BaseAgent.step gate.
        "per_tool_token_budget": dict(row.per_tool_token_budget or {}),
        # Agent consults this when a REQUIRE_APPROVAL policy result or
        # critic ESCALATE/REVISE would otherwise queue for a human. Derived
        # from mode: phase_gate (guided) pauses between stages; auto_approve
        # (recon_only/assess/autonomous) proceeds without pausing.
        "approval_policy": approval_policy,
        # Runaway-loop circuit breaker for the regression mechanism.
        # ``0`` (the default) means unlimited — loop safety lives in
        # NoveltyCritic per-fingerprint replay rejection and
        # ScopeCritic literal authorisation, not in this number. The
        # BudgetCritic reads this directly.
        "regression_budget_max": int(
            getattr(row, "regression_budget_max", 0) or 0
        ),
        # Engagement-wide token budget — the BudgetCritic's token-reserve
        # check reads this to refuse a regression when the engagement
        # has already burned most of its budget.
        "token_budget": int(getattr(row, "token_budget", 0) or 0),
        # Operator opt-in for destructive actions (DoS, wipe, lockout —
        # irreversible harm to the target). Fail-closed default False. When
        # False the policy gate DENYs destructive tools, destructive curated
        # PoCs are gated ``destructive_without_optin``, and DoS EDB modules are
        # reported-not-run. When True they are allowed but always routed
        # through human/critic approval. Independent of mode.
        "destructive_opt_in": bool(getattr(row, "destructive_opt_in", False)),
        # Auto-run non-destructive EDB exploit scripts (remote/local/privesc/
        # webapp/bypass) without pausing. Derived from mode: on under
        # autonomous, off otherwise (guided surfaces EDB candidates to the
        # operator). DoS EDB modules remain gated by destructive_opt_in.
        "edb_auto_exploit": mode == "autonomous",
    }


def build_opa_engagement(row: EngagementRow) -> dict[str, Any]:
    """Deprecated alias — use ``build_engagement_context()`` instead.

    Kept for backwards compatibility during migration. Will be removed
    in a future release.
    """
    warnings.warn(
        "build_opa_engagement() is deprecated; use build_engagement_context()",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_engagement_context(row)
