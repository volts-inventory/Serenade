"""Per-engagement nftables ruleset compiler.

L3 network egress filtering — compile the engagement's
`Scope` spec into an nftables ruleset that allows traffic only to scope
CIDRs (plus the mitmproxy egress for HTTP). Apply on engagement start;
flush on engagement end.

ARCHITECTURE.md §7 describes L3 as the third layer of scope enforcement
on top of L2 (per-engagement Docker network). This module implements
L3.

Architecture:
- One nftables table per engagement: `serenade_eng_{prefix}`.
- Chains: `output_filter` (egress decisions) and `input_filter` (DROP by
  default; per-engagement sandbox containers don't need ingress).
- Allowed: scope.cidrs + scope.excluded_cidrs (explicitly DROPPED) + the
  proxy-logger container IP for HTTP egress.
- Everything else: DROP.

Application model:
- Dev (Docker): run a sidecar container with `--cap-add=NET_ADMIN
  --network=container:<sandbox>` that applies the ruleset to the sandbox
  network namespace. The sidecar's image (`serenade/nftables-sidecar:dev`)
  exposes `nftables-apply <json>` and exits.
- Prod (Firecracker): the host applies the ruleset to the microVM's TAP
  device. Same compiler; different applier.

This module ships the compiler (pure function, fully testable) and the
async applier (calls `docker run` with the ruleset stdin). The applier is
best-effort in dev — if NET_ADMIN isn't available, we log a WARN and let
L2-only operate; this matches ARCHITECTURE.md §7 which acknowledges L3 is
graceful-degraded.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def compile_ruleset(
    *,
    table_name: str,
    scope: dict[str, Any],
    proxy_ip: str | None = None,
    dns_servers: list[str] | None = None,
) -> str:
    """Compile a `Scope` dict into an nftables ruleset string.

    WIRING STATUS (sandbox5-01): this compiler is implemented and unit-tested,
    and `SandboxRunner` accepts the resulting ruleset via its
    `nftables_ruleset=` argument — but no production call site currently
    constructs a `SandboxRunner` WITH a scope-compiled ruleset, so scoped L3
    egress filtering is NOT yet active. The egress control that IS active by
    default is the per-engagement Docker `--internal` network
    (`ensure_engagement_network(internal=True)`), which blocks ALL egress off
    the engagement network (host gateway + internet) — a coarser, scope-blind
    control. The remaining gap is therefore *intra-network* scope distinction
    (in-scope vs out-of-scope hosts reachable on the same engagement network),
    bounded by the operator-driven, audit-gated threat model. Wiring this
    compiler into the runner (threading engagement scope to the construction
    sites) is tracked follow-up and requires Docker+nft to verify end-to-end.

    Inputs:
      scope.cidrs  — allow egress to these CIDRs
      scope.excluded_cidrs — explicit DROP (overrides .cidrs if overlapping)
      scope.domains — informational only at L3; OPA handles domain checks
      proxy_ip     — IP of proxy-logger (allow egress to it on 8080 always)
      dns_servers  — UDP/53 allowed to these servers (default: none — DNS
                     resolution happens via Docker's embedded resolver inside
                     the network namespace)

    Output: a multi-line nftables ruleset suitable for `nft -f -`.
    """
    cidrs = _validate_cidrs(scope.get("cidrs", []))
    excluded = _validate_cidrs(scope.get("excluded_cidrs", []))

    lines = [
        f"table inet {table_name} {{",
        # Drop-everything by default for both directions; specific accepts below.
        "    chain output_filter {",
        "        type filter hook output priority 0; policy drop;",
        # loopback always
        '        oifname "lo" accept',
        # established/related
        "        ct state established,related accept",
    ]

    # Proxy egress — for HTTP requests routed via mitmproxy. The proxy IP may
    # be v4 or v6; emit the matching family keyword so an IPv6 proxy doesn't
    # produce an invalid `ip daddr` rule that voids the whole ruleset.
    if proxy_ip:
        kw = _daddr_keyword(proxy_ip)
        if kw:
            lines.append(f"        {kw} {proxy_ip} tcp dport 8080 accept")
        else:
            log.warning("nftables.skip_invalid_proxy_ip", proxy_ip=proxy_ip)

    # DNS — narrowly allowed to listed servers
    for dns in dns_servers or []:
        kw = _daddr_keyword(dns)
        if kw is None:
            log.warning("nftables.skip_invalid_dns", dns=dns)
            continue
        lines.append(f"        {kw} {dns} udp dport 53 accept")
        lines.append(f"        {kw} {dns} tcp dport 53 accept")

    # Explicit drops first (excluded_cidrs override). Branch on address family
    # so IPv6 CIDRs use `ip6 daddr` (an `ip daddr` with a v6 CIDR is an invalid
    # rule that makes `nft -f -` reject the ENTIRE ruleset — which, per the L3
    # fail path, silently downgrades the container to no-ACL).
    for cidr, version in excluded:
        kw = "ip6 daddr" if version == 6 else "ip daddr"
        lines.append(f"        {kw} {cidr} drop")

    # Allowed CIDRs
    for cidr, version in cidrs:
        kw = "ip6 daddr" if version == 6 else "ip daddr"
        lines.append(f"        {kw} {cidr} accept")

    # Counter on the default-drop so operator can see how often scope blocks fired
    lines.append("        counter drop")
    lines.append("    }")

    lines += [
        "    chain input_filter {",
        "        type filter hook input priority 0; policy drop;",
        '        iifname "lo" accept',
        "        ct state established,related accept",
        "        counter drop",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _validate_cidrs(items: list[str]) -> list[tuple[str, int]]:
    """Return ``(cidr, version)`` for valid IPv4/IPv6 CIDRs; drop the rest.

    Returning the IP version lets the compiler emit the correct nftables
    family keyword (``ip daddr`` for v4, ``ip6 daddr`` for v6). A v6 CIDR
    rendered as ``ip daddr`` is an invalid rule that makes ``nft -f -`` reject
    the ENTIRE ruleset — which (per the L3 fail path) silently downgrades the
    container to no-ACL. Defense-in-depth: a malformed scope spec is skipped
    rather than allowed to void the whole ruleset.
    """
    out: list[tuple[str, int]] = []
    for c in items:
        try:
            net = ipaddress.ip_network(c, strict=False)
        except (ValueError, TypeError):
            log.warning("nftables.skip_invalid_cidr", cidr=c)
            continue
        out.append((c, net.version))
    return out


def _daddr_keyword(addr: str) -> str | None:
    """Return the nftables destination-address keyword for a bare IP address
    (``ip daddr`` for IPv4, ``ip6 daddr`` for IPv6), or ``None`` if *addr* is
    not a valid IP. Used for proxy/DNS single-host rules."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return None
    return "ip6 daddr" if ip.version == 6 else "ip daddr"


async def apply_ruleset_to_sandbox(
    *,
    sandbox_container_name: str,
    ruleset: str,
    sidecar_image: str = "serenade/nftables-sidecar:dev",
) -> bool:
    """Best-effort: apply `ruleset` to the network namespace of the sandbox container.

    Spawns a privileged sidecar that pipes the ruleset into `nft -f -`.
    Returns True on success; False on any error (logged). The caller decides
    whether to refuse the engagement when L3 can't be enforced — for now we
    log and let L2-only proceed (ARCHITECTURE.md §7 graceful-degradation).
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--cap-add",
        "NET_ADMIN",
        "--network",
        f"container:{sandbox_container_name}",
        "-i",
        sidecar_image,
        "nft",
        "-f",
        "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=ruleset.encode("utf-8"))
        if proc.returncode == 0:
            log.info(
                "nftables.applied",
                container=sandbox_container_name,
                rules=ruleset.count("\n"),
            )
            return True
        log.warning(
            "nftables.apply_failed",
            container=sandbox_container_name,
            err=stderr.decode("utf-8", errors="replace"),
            stdout=stdout.decode("utf-8", errors="replace"),
        )
        return False
    except Exception as e:
        log.warning("nftables.apply_error", err=str(e))
        return False
