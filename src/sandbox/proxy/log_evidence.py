"""mitmproxy addon: log every flow as evidence (5-tuple + bytes + SNI).

The proxy is spawned ONE-PER-ENGAGEMENT (see proxy_lifecycle.py) with the
authoritative engagement id in the ``ENGAGEMENT_ID`` container env var. We
bucket evidence by THAT value, NOT by the client-supplied
``X-Serenade-Engagement`` request header.

Why: the header is set by the very sandbox container being audited. A tool (or
an injected payload) could stamp a *different* engagement's UUID and file its
HTTP evidence under another engagement, or set junk to evade attribution
(sandbox-12). The header is therefore recorded only as untrusted metadata and
flagged when it disagrees with the trusted env value — it never decides the
bucket. Requests without a configured ``ENGAGEMENT_ID`` fall to
``unattributed.jsonl`` so the gap is visible rather than silently merged.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from _bucket import safe_bucket  # see sandbox/proxy/_bucket.py
from mitmproxy import http

LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_ENG_HEADER = "X-Serenade-Engagement"

# Authoritative engagement id for THIS proxy, set by proxy_lifecycle when the
# per-engagement proxy container is spawned. Trusted; not client-controlled.
_TRUSTED_ENGAGEMENT_ID = os.environ.get("ENGAGEMENT_ID", "")


def response(flow: http.HTTPFlow) -> None:
    # Bucket strictly by the trusted, server-side engagement id — never the
    # client header. The header is kept only as untrusted metadata.
    bucket = safe_bucket(_TRUSTED_ENGAGEMENT_ID) if _TRUSTED_ENGAGEMENT_ID else "unattributed"
    client_header = flow.request.headers.get(_ENG_HEADER, "") or None

    record = {
        "ts": datetime.now(UTC).isoformat(),
        "client": flow.client_conn.peername[0] if flow.client_conn.peername else None,
        "method": flow.request.method,
        "scheme": flow.request.scheme,
        "host": flow.request.host,
        "port": flow.request.port,
        "path": flow.request.path,
        "status": flow.response.status_code if flow.response else None,
        "req_bytes": len(flow.request.raw_content or b""),
        "resp_bytes": len(flow.response.raw_content or b"") if flow.response else 0,
        "sni": flow.client_conn.sni if hasattr(flow.client_conn, "sni") else None,
        # Authoritative attribution (trusted env), plus the untrusted header
        # for forensics. A mismatch flags a possible spoofing attempt.
        "engagement_id": _TRUSTED_ENGAGEMENT_ID or None,
        "client_engagement_header": client_header,
        "header_mismatch": bool(
            client_header and client_header != _TRUSTED_ENGAGEMENT_ID
        ),
    }
    out = LOG_DIR / f"{bucket}.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
