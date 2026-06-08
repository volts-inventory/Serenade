"""Engagement cost forecasting.

Heuristic estimator based on scope size + mode that gives the operator
a number BEFORE they hit Start. Refines over time by joining against
`llm_calls` historical data; for v1 it's pure heuristic.

The heuristic is intentionally conservative (over-estimates) so the
operator's first surprise is "cheaper than I expected" rather than the
other direction.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any

from worldstate.schema import normalize_mode


@dataclass
class CostEstimate:
    token_low: int
    token_high: int
    usd_low: float
    usd_high: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_low": self.token_low,
            "token_high": self.token_high,
            "usd_low": round(self.usd_low, 2),
            "usd_high": round(self.usd_high, 2),
            "notes": self.notes,
        }


# Per-host token cost ranges, based on observed Sonnet+Haiku mix:
#   - planner (Sonnet): ~3k tokens/turn, ~5 turns/host
#   - parser (Haiku): ~1k tokens/parse, ~7 parses/host
#   - triage (Haiku): ~500 tokens/candidate, ~10 candidates/host
#   - critic (Haiku): ~800 tokens/review, fires for ~30% of decisions
# Total per fully-enumerated host: ~25k low, ~80k high
_TOKENS_PER_HOST_LOW = 25_000
_TOKENS_PER_HOST_HIGH = 80_000

# Blended $/1M tokens — Anthropic mix (Sonnet planner + Haiku parser/triage/critic).
# Prompt caching brings effective input cost down by ~5x for the engagement-stable prefix.
_USD_PER_M_TOKENS_LOW = 1.20  # heavy caching, Haiku-dominant
_USD_PER_M_TOKENS_HIGH = 4.50  # cold start, Sonnet-heavy planning

# Mode multipliers — fuller pipelines run more attempts.
_MODE_MULT = {
    "recon_only": 0.4,  # recon + enum only
    "assess": 0.7,      # full pipeline, planning only (no exploitation)
    "guided": 1.0,
    "autonomous": 1.5,  # auto exploitation across the candidate set
}


def estimate_engagement_cost(
    scope: dict[str, Any],
    mode: str,
    *,
    backend: str = "anthropic",
) -> CostEstimate:
    """Forecast token + USD spend for an engagement.

    `scope.cidrs` and `scope.domains` drive the host-count estimate; mode
    drives the multiplier.
    """
    if backend != "anthropic":
        # MLX local mode = no API cost
        return CostEstimate(
            token_low=0,
            token_high=0,
            usd_low=0.0,
            usd_high=0.0,
            notes=["SERENADE_LLM_BACKEND=mlx — local inference, no API spend"],
        )

    estimated_hosts = _estimate_host_count(scope)
    multiplier = _MODE_MULT.get(normalize_mode(mode), 1.0)

    tok_low = int(estimated_hosts * _TOKENS_PER_HOST_LOW * multiplier)
    tok_high = int(estimated_hosts * _TOKENS_PER_HOST_HIGH * multiplier)
    usd_low = tok_low * _USD_PER_M_TOKENS_LOW / 1_000_000
    usd_high = tok_high * _USD_PER_M_TOKENS_HIGH / 1_000_000

    notes = [
        f"Estimated hosts in scope: {estimated_hosts}",
        f"Mode multiplier ({mode}): {multiplier}x",
        "Heuristic; refines once 50+ historical engagements have completed.",
    ]
    if estimated_hosts > 256:
        notes.append("Large scope — consider running in segments to bound cost.")

    return CostEstimate(
        token_low=tok_low,
        token_high=tok_high,
        usd_low=usd_low,
        usd_high=usd_high,
        notes=notes,
    )


def _estimate_host_count(scope: dict[str, Any]) -> int:
    """Total host-equivalents in scope. CIDRs expand to their address count
    (capped); each domain counts as 5 hosts (rough subdomain estimate)."""
    total = 0
    for cidr in scope.get("cidrs", []) or []:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
            # Cap at /16 to avoid wildly inflated estimates from typo'd /8s
            total += min(net.num_addresses, 65_536)
        except (ValueError, TypeError):
            continue
    domain_count = len(scope.get("domains", []) or [])
    total += domain_count * 5
    # Reasonable floor: 1 host even if scope is just URLs
    if total == 0:
        total = max(1, len(scope.get("urls", []) or []))
    return total
