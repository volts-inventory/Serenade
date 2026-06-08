"""mitmproxy addon: drop out-of-scope HTTP(S) egress at the proxy (sandbox-12b).

Defence-in-depth second layer for the HTTP path. The per-engagement proxy is
dual-homed onto ``serenade_default`` so it can reach real targets on behalf of
a sandbox; the ``--internal`` engagement network only constrains the sandbox's
*direct* egress, not what it asks the proxy to fetch. Without this addon a gap
in the extract_targets/policy gate would let an out-of-scope URL through the
proxy unchecked.

This addon enforces the engagement scope a SECOND time, at the proxy, mirroring
the L3 nftables ACL for the HTTP path. The allow-list is injected by
``proxy_lifecycle.ensure_engagement_proxy`` via the trusted, server-side
``SERENADE_SCOPE_HOSTS`` container env var (newline- or comma-separated bare
hostnames and/or IPv4/IPv6 CIDRs). It is NOT client-controlled, so a sandbox
cannot widen its own scope.

Behaviour:
- If ``SERENADE_SCOPE_HOSTS`` is unset/empty, the addon is a no-op (fail-open):
  scope still rests on the (tightened) extract_targets gate. This preserves the
  default proxy interface when scope data is not wired in.
- If set, any request whose host is neither an exact/suffix hostname match nor
  inside an allowed CIDR is answered with ``403`` and never forwarded.

The pure matching logic lives in ``_scope.py`` so it is unit-testable without
mitmproxy installed.
"""

from __future__ import annotations

import os

from _scope import host_in_scope, parse_scope  # see sandbox/proxy/_scope.py
from mitmproxy import http

_ENV = "SERENADE_SCOPE_HOSTS"

_SCOPE_HOSTS, _SCOPE_NETS = parse_scope(os.environ.get(_ENV, ""))
# Empty scope ⇒ disabled (fail-open). A configured-but-empty value would block
# everything, which is never the intent for an unset variable.
_ENABLED = bool(_SCOPE_HOSTS or _SCOPE_NETS)


def request(flow: http.HTTPFlow) -> None:
    if not _ENABLED:
        return
    if host_in_scope(flow.request.host, _SCOPE_HOSTS, _SCOPE_NETS):
        return
    flow.response = http.Response.make(
        403,
        b"serenade: out-of-scope host blocked by proxy scope filter\n",
        {"Content-Type": "text/plain"},
    )
