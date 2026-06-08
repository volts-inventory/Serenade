"""Pure Python policy engine — replaces the OPA Rego bundle.

Port of ``src/policy/rego/scope.rego`` (102 lines of Rego) into Python.
Every inclusion rule preserves the original semantics exactly, including
the dotted-suffix domain matching that prevents "evilexample.com" from
matching scope domain "example.com".

Beyond the original Rego, this module adds fail-closed enforcement of
scope **exclusions** (excluded_cidrs / excluded_domains / excluded_paths),
URL-host scope, an all-internet (``0.0.0.0/0`` / ``::/0``) tripwire, and a
fail-closed time-window default. Exclusions always win over inclusions.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Mirrors the shape that engine.py and callers expect."""

    allow: bool
    require_approval: bool = False
    reason: str = "ok"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_hostname(target: str) -> str:
    """Strip scheme, path, port, and query from a URL-or-hostname string.

    Examples:
        https://sub.example.com:8443/path  ->  sub.example.com
        sub.example.com:8080              ->  sub.example.com
        sub.example.com                   ->  sub.example.com
    """
    # If the target contains "://", use urlparse for reliable extraction.
    # urlparse also correctly handles userinfo (``user@host``), so a target
    # like ``https://example.com@evil.example.org/`` resolves to the real
    # host ``evil.example.org`` rather than the spoofed ``example.com``.
    if "://" in target:
        parsed = urlparse(target)
        host = parsed.hostname or target
        return host

    # No scheme — strip query/fragment, then path, then port manually
    # (matches Rego logic). A scheme-less host can carry a ``?``/``#`` directly
    # (e.g. ``example.com?x=1``) with no path; split those off before the ``/``
    # split so the host resolves correctly rather than becoming
    # ``example.com?x=1`` (policy3-02).
    no_qf = target.split("?", 1)[0].split("#", 1)[0]
    no_path = no_qf.split("/", 1)[0]
    # Defend against userinfo smuggling in scheme-less targets
    # (e.g. ``example.com@evil.example.org``): the real host is the part
    # after the last ``@``.
    if "@" in no_path:
        no_path = no_path.rsplit("@", 1)[1]
    host = no_path.split(":", 1)[0]
    return host


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4 remove_dot_segments + collapse of repeated slashes.

    Resolves ``.`` and ``..`` segments and collapses runs of ``/`` so that a
    path a target web server would normalize (``//admin``, ``/./admin``,
    ``/foo/../admin``) compares equal to the canonical ``/admin``. ``..`` can
    never escape above the root (extra ``..`` at the top are discarded), so
    ``/admin/../admin`` -> ``/admin`` and ``/../admin`` -> ``/admin``.
    """
    if not path.startswith("/"):
        path = "/" + path
    out: list[str] = []
    # Splitting an absolute path on "/" yields a leading "" we ignore; a
    # trailing "/" yields a trailing "" we use to preserve a trailing slash.
    segments = path.split("/")
    for i, seg in enumerate(segments):
        if seg in ("", "."):
            # Preserve a single trailing slash for a trailing empty/"."
            # segment; intermediate empties (double slashes) are collapsed.
            if i == len(segments) - 1 and out:
                out.append("")
            continue
        if seg == "..":
            if out:
                out.pop()
            # Trailing ".." should leave a trailing slash (directory).
            if i == len(segments) - 1:
                out.append("")
            continue
        out.append(seg)
    return "/" + "/".join(out)


def _normalize_path(raw: str) -> str:
    """Percent-decode then RFC 3986-normalize a path for exclusion matching.

    Percent-encoded dot/slash forms (``%2e``, %2E, ``%2f``) are decoded first
    so ``/%2e%2e/admin`` and ``/foo/%2e%2e/admin`` cannot dodge the carve-out,
    then dot-segments/double-slashes are resolved. Surrounding whitespace is
    trimmed. The result always begins with ``/``.
    """
    raw = raw.strip()
    # Decode percent-encoding so obfuscated separators/dot-segments
    # (``%2f`` -> ``/``, ``%2e`` -> ``.``) are normalized too. unquote also
    # decodes a literal ``%`` payload harmlessly.
    raw = unquote(raw)
    if not raw.startswith("/"):
        raw = "/" + raw
    return _remove_dot_segments(raw)


def _extract_path(target: str) -> str:
    """Return the normalized URL path component of *target* (``/`` if none).

    The query string (``?...``), fragment (``#...``) and ``;params`` are
    stripped from BOTH the scheme-ful and scheme-less branches so that
    excluded-path matching is consistent regardless of input shape — a
    scheme-less target like ``example.com/admin?id=1`` must yield ``/admin``,
    not ``/admin?id=1`` (policy2-01). Surrounding whitespace is also trimmed.

    The path is then percent-decoded and RFC 3986-normalized (collapse ``//``,
    resolve ``.``/``..``) so obfuscated forms a server would canonicalize —
    ``//admin``, ``/./admin``, ``/foo/../admin``, ``/%2e%2e/admin`` — cannot
    bypass an ``/admin`` carve-out (policy3-01).
    """
    if "://" in target:
        parsed = urlparse(target)
        raw = parsed.path or "/"
        return _normalize_path(raw)
    # Scheme-less: take everything after the first slash, if any, then strip
    # query / fragment / params and whitespace so the result matches what
    # urlparse would have produced for the scheme-ful equivalent.
    if "/" in target:
        raw = "/" + target.split("/", 1)[1]
    else:
        raw = "/"
    raw = raw.strip()
    raw = raw.split("?", 1)[0]
    raw = raw.split("#", 1)[0]
    raw = raw.split(";", 1)[0]
    return _normalize_path(raw or "/")


def _is_unbounded_cidr(net: ipaddress.IPv4Network | ipaddress.IPv6Network) -> bool:
    """True for an all-internet prefix (``0.0.0.0/0`` or ``::/0``)."""
    return net.prefixlen == 0


def _ip_in_any_cidr(ip: Any, cidrs: list[str], *, allow_unbounded: bool = False) -> bool:
    """True if *ip* (an ``ip_address``) is inside any of *cidrs*.

    ``0.0.0.0/0`` / ``::/0`` are ignored unless *allow_unbounded* is set —
    a stray all-internet prefix must never silently disable IP confinement
    (policy-07).
    """
    for cidr in cidrs:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if _is_unbounded_cidr(net) and not allow_unbounded:
            continue
        if ip in net:
            return True
    return False


def _domain_matches(hostname: str, domains: list[str]) -> bool:
    """True if *hostname* exactly matches or is a true subdomain of a domain.

    The dotted-suffix check (``.example.com``) prevents ``evilexample.com``
    from matching ``example.com`` (the critical Rego edge case).
    """
    hostname = hostname.lower()
    for domain in domains:
        d = domain.lower().strip(".")
        if not d:
            continue
        if hostname == d or hostname.endswith("." + d):
            return True
    return False


def _path_excluded(target: str, excluded_paths: list[str]) -> bool:
    """True if the URL path of *target* is carved out by *excluded_paths*.

    A path is excluded when it equals an excluded prefix or sits beneath it
    (``/admin`` excludes ``/admin`` and ``/admin/users``). Matching is
    boundary-aware so ``/admin`` does not exclude ``/administrator``.
    """
    if not excluded_paths:
        return False
    path = _extract_path(target)
    # Normalize the candidate path: trim a trailing slash so ``/admin/`` and
    # ``/admin`` compare equal, and lower-case for case-insensitive matching.
    path_norm = (path.rstrip("/") or "/").lower()
    for ex in excluded_paths:
        ex = ex if ex.startswith("/") else "/" + ex
        ex_norm = (ex.rstrip("/") or "/").lower()
        if ex_norm == "/":
            # An excluded "/" carves out the entire host.
            return True
        # Boundary-aware: exact match or a true sub-path beneath the prefix
        # (``/admin`` excludes ``/admin`` and ``/admin/users`` but never
        # ``/administrator``).
        if path_norm == ex_norm or path_norm.startswith(ex_norm + "/"):
            return True
    return False


def _target_excluded(target: str, scope: dict[str, Any]) -> bool:
    """True if *target* matches any exclusion — exclusions win over inclusion.

    Enforces ``excluded_cidrs`` (CIDR membership), ``excluded_domains``
    (exact + dotted-suffix), and ``excluded_paths`` (URL path carve-out).
    """
    # Excluded CIDRs (IP targets). Unbounded excludes are honored here —
    # excluding 0.0.0.0/0 is a valid (if extreme) operator intent.
    try:
        ip = ipaddress.ip_address(target)
    except ValueError:
        ip = None
    if ip is not None and _ip_in_any_cidr(
        ip, scope.get("excluded_cidrs", []), allow_unbounded=True
    ):
        return True

    # Excluded domains (host targets).
    hostname = _extract_hostname(target)
    if _domain_matches(hostname, scope.get("excluded_domains", [])):
        return True

    # Excluded URL paths.
    if _path_excluded(target, scope.get("excluded_paths", [])):
        return True

    return False


def _url_hosts(scope: dict[str, Any]) -> list[str]:
    """Hostnames derived from ``scope["urls"]`` (for host-scope matching)."""
    hosts: list[str] = []
    for url in scope.get("urls", []):
        host = _extract_hostname(url)
        if host:
            hosts.append(host)
    return hosts


def _target_in_scope(target: str, scope: dict[str, Any]) -> bool:
    """Return True if *target* falls within the engagement scope.

    Order of evaluation (fail-closed):
    1. Exclusions win — an excluded host/IP/path is never in scope.
    2. CIDR membership for IP targets (all-internet prefixes ignored).
    3. Domain match — exact or true subdomain (dotted-suffix).
    4. URL-host match — hosts derived from ``scope["urls"]``.
    """
    # --- Exclusion check (takes precedence over every inclusion) ---
    if _target_excluded(target, scope):
        return False

    allow_unbounded = bool(scope.get("allow_unbounded_cidrs", False))

    # --- CIDR check ---
    try:
        ip = ipaddress.ip_address(target)
    except ValueError:
        ip = None
    if ip is not None and _ip_in_any_cidr(
        ip, scope.get("cidrs", []), allow_unbounded=allow_unbounded
    ):
        return True

    # --- Domain check (included domains + hosts derived from scope URLs) ---
    hostname = _extract_hostname(target)
    if _domain_matches(hostname, scope.get("domains", [])):
        return True
    if _domain_matches(hostname, _url_hosts(scope)):
        return True

    return False


def _risk_allowed(risk_class: str, allowed: set[str] | list[str]) -> bool:
    """Return True if *risk_class* is in the allowed set."""
    return risk_class in allowed


def _in_time_window(window: dict[str, Any]) -> bool:
    """Return True if the current UTC time is within the engagement window.

    Fail-closed: a missing/empty window (no ``start_ns``/``end_ns``) is
    rejected rather than treated as always-open (policy-04). If present,
    ``allowed_hours_utc`` — an inclusive ``(start_hour, end_hour)`` band —
    is also enforced, supporting wrap-around windows (e.g. 22..6).
    """
    if not window:
        return False
    start_ns = window.get("start_ns")
    end_ns = window.get("end_ns")
    if start_ns is None or end_ns is None:
        return False

    now = datetime.now(UTC)
    now_ns = int(now.timestamp() * 1_000_000_000)
    if not (start_ns <= now_ns <= end_ns):
        return False

    allowed_hours = window.get("allowed_hours_utc")
    if allowed_hours is not None:
        try:
            start_hour, end_hour = int(allowed_hours[0]), int(allowed_hours[1])
        except (TypeError, ValueError, IndexError):
            # Malformed band → fail closed.
            return False
        hour = now.hour
        if start_hour <= end_hour:
            if not (start_hour <= hour <= end_hour):
                return False
        # Wrap-around band (e.g. 22..6 spans midnight).
        elif not (hour >= start_hour or hour <= end_hour):
            return False

    return True


def _destructive_blocked(tool: dict[str, Any], engagement: dict[str, Any]) -> bool:
    """Return True when a destructive tool must be DENIED at the policy gate.

    Destructive actions cause irreversible harm to the *target* (the sandbox
    isolates our host, not the client's box), so they are blocked by
    default. An operator can authorize them per-engagement via the
    ``destructive_opt_in`` flag (fail-closed ``False``). When opted in the
    call is not denied here — ``evaluate`` instead forces it through human/
    critic approval so a destructive action is never executed silently.
    """
    if not bool(tool.get("destructive")):
        return False
    return not bool(engagement.get("destructive_opt_in", False))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    targets: list[str],
    engagement: dict[str, Any],
    tool: dict[str, Any],
) -> PolicyResult:
    """Evaluate policy for a tool invocation.

    Returns a ``PolicyResult`` with ``allow``, ``require_approval``, and
    ``reason`` fields — matching the shape the old OPA ``decision`` object
    produced.
    """
    scope = engagement.get("scope", {})

    # 0. Fail-closed: a tool call with no targets gets NO scope enforcement,
    # so it must be denied rather than allowed through (policy-03).
    if not targets:
        return PolicyResult(allow=False, reason="no_targets")

    # 1. Scope check — every target must be in scope and not excluded.
    for t in targets:
        if _target_excluded(t, scope):
            return PolicyResult(allow=False, reason="excluded")
        if not _target_in_scope(t, scope):
            return PolicyResult(allow=False, reason="out_of_scope")

    # 2. Risk class check.
    if not _risk_allowed(tool.get("risk_class", ""), engagement.get("allowed_risk_classes", [])):
        return PolicyResult(allow=False, reason="risk_class_not_allowed")

    # 3. Time window check.
    if not _in_time_window(engagement.get("window", {})):
        return PolicyResult(allow=False, reason="outside_time_window")

    # 4. Destructive tool check. Denied unless the engagement opted in.
    if _destructive_blocked(tool, engagement):
        return PolicyResult(allow=False, reason="destructive_not_authorized")

    # 5. Approval requirement. Exploit/post-exploit calls pause for the
    # operator under the phase_gate policy (guided mode); auto_approve
    # modes (recon_only/assess/autonomous) proceed without pausing.
    # A destructive tool that reached here is opted-in — it is ALWAYS routed
    # through approval (human in guided, critic + auto-approve record in
    # autonomous) so a destructive action is never executed silently.
    require_approval = (
        tool.get("risk_class") in {"exploit", "post-exploit"}
        and engagement.get("approval_policy") == "phase_gate"
    ) or bool(tool.get("destructive"))

    return PolicyResult(allow=True, require_approval=require_approval, reason="ok")
