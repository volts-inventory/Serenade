"""Pure scope-matching helper for the proxy scope-filter addon (sandbox-12b).

Importable without mitmproxy so the scope logic is unit-testable. The addon
``scope_filter.py`` imports these helpers and applies them to live flows.
"""

from __future__ import annotations

import ipaddress


def parse_scope(raw: str) -> tuple[frozenset[str], tuple[object, ...]]:
    """Split a ``SERENADE_SCOPE_HOSTS`` value into (hostnames, networks).

    Entries are newline- or comma-separated. An entry that parses as an
    IPv4/IPv6 network (or bare IP, via ``strict=False``) becomes a CIDR; every
    other non-empty entry is treated as a hostname (lower-cased).
    """
    hosts: set[str] = set()
    nets: list[object] = []
    for chunk in raw.replace(",", "\n").splitlines():
        entry = chunk.strip().lower()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
            continue
        except ValueError:
            pass
        hosts.add(entry)
    return frozenset(hosts), tuple(nets)


def host_in_scope(
    host: str,
    scope_hosts: frozenset[str],
    scope_nets: tuple[object, ...],
) -> bool:
    """True if *host* is an allowed hostname (exact or dot-suffix) or falls
    inside an allowed CIDR. Empty/blank host is never in scope."""
    # Strip a single trailing dot (absolute-FQDN form ``example.com.`` is
    # DNS/HTTP-equivalent to ``example.com``) so an in-scope host in that form
    # still matches (sandbox3-03).
    h = (host or "").strip().lower().strip("[]")
    if h.endswith(".") and not h.endswith(".."):
        h = h[:-1]
    if not h:
        return False
    try:
        ip = ipaddress.ip_address(h)
        return any(ip in net for net in scope_nets)  # type: ignore[operator]
    except ValueError:
        pass
    for allowed in scope_hosts:
        if h == allowed or h.endswith("." + allowed):
            return True
    return False
