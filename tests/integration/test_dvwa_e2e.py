"""End-to-end test against the bundled DVWA target.

Prerequisites:
- `make up` to bring up the Serenade stack
- `docker compose -f content/targets/dvwa/docker-compose.yml up -d`
- DVWA initialized (visit http://127.0.0.1:8080 once and create the DB)
- ANTHROPIC_API_KEY set OR LLM_BACKEND=mlx + mlx_lm.server running

Run with: `pytest -m integration tests/integration/test_dvwa_e2e.py`

The test:
1. Creates an engagement scoped to 172.30.0.0/24
2. Starts it
3. Polls /coverage until at least one host is discovered (or timeout)
4. Asserts a non-zero number of findings landed
5. Renders the report HTML; asserts findings appear
6. Cleans up

This test is intentionally lenient about what specifically gets found —
DVWA's exposed surface changes across versions. The contract under test is
that the platform produces *some* evidence-chained findings when pointed at
a known-vulnerable target.
"""

from __future__ import annotations

import hashlib
import os
import time

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_E2E", "").strip(),
        reason="E2E tests need RUN_E2E=1 and the dvwa stack up",
    ),
]


_API = os.environ.get("SERENADE_API", "http://localhost:8000")
_DVWA_IP = "172.30.0.100"


def _auth_sha() -> str:
    return hashlib.sha256(b"e2e test authorization").hexdigest()


def test_dvwa_engagement_full_lifecycle():
    with httpx.Client(base_url=_API, timeout=30.0) as client:
        # 1. Create engagement
        resp = client.post(
            "/engagements",
            json={
                "name": "DVWA E2E",
                "scope": {
                    "cidrs": ["172.30.0.0/24"],
                    "domains": [],
                    "urls": [f"http://{_DVWA_IP}"],
                },
                "objectives": ["xss_demo", "sqli_demo"],
                "mode": "normal",
                "token_budget": 500_000,
                "wallclock_hours": 1,
                "authorization_artifact_sha256": _auth_sha(),
            },
        )
        assert resp.status_code == 201, resp.text
        engagement = resp.json()
        eid = engagement["id"]

        # 2. Start
        resp = client.post(f"/engagements/{eid}/start")
        assert resp.status_code == 200, resp.text

        # 3. Poll for discovery — wait up to 5 minutes for the worker to find DVWA
        deadline = time.monotonic() + 300
        hosts_seen = 0
        while time.monotonic() < deadline:
            resp = client.get(f"/engagements/{eid}/coverage")
            assert resp.status_code == 200
            cov = resp.json()
            hosts_seen = cov.get("hosts", 0)
            if hosts_seen > 0 and cov.get("services", 0) > 0:
                break
            time.sleep(10)
        assert hosts_seen > 0, "no hosts discovered within timeout — is DVWA up on 172.30.0.100?"

        # 4. Continue polling for findings
        deadline = time.monotonic() + 600
        findings_count = 0
        while time.monotonic() < deadline:
            resp = client.get(f"/engagements/{eid}/findings")
            assert resp.status_code == 200
            buckets = resp.json()
            findings_count = sum(len(v) for v in buckets.values())
            if findings_count > 0:
                break
            time.sleep(15)
        assert findings_count > 0, "no findings landed within timeout"

        # 5. Render report HTML
        resp = client.get(f"/engagements/{eid}/report/html")
        assert resp.status_code == 200
        html = resp.text
        assert "DVWA E2E" in html
        # Some severity-tinted finding must appear in the report
        assert any(s in html for s in ("severity-critical", "severity-high", "severity-medium"))

        # 6. Best-effort cleanup
        client.patch(f"/engagements/{eid}", json={"status": "completed"})
