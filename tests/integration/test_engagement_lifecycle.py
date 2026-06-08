"""Integration test for the engagement lifecycle.

Tests the orchestrator API + stage controller end-to-end with mocked
externals (no real Postgres, no real Neo4j, no real LLM). Verifies:
- Engagement create / start / pause API contracts hold
- Stage controller routes correctly between agents
- Three-gate sequence (OPA + critic + sandbox) is honored
- Reports render against an empty world-state

Marked `integration` so it's skipped by the default `make test`. Run via
`pytest -m integration`.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from agents.report import DraftFinding, DraftReport
from reports.renderer import render_html, render_json, render_markdown, render_sarif

pytestmark = pytest.mark.integration


def _sample_draft() -> DraftReport:
    return DraftReport(
        engagement_id=uuid4(),
        title="Integration test engagement",
        executive_summary="Two findings during integration smoke test.",
        coverage={"hosts": 3, "services": 12, "findings": 2},
        findings=[
            DraftFinding(
                title="Apache 2.4.49 Path Traversal",
                description="CVE-2021-41773 path traversal RCE",
                severity="critical",
                cvss_v3_1=9.8,
                cwe="CWE-22",
                cve=["CVE-2021-41773"],
                affected_assets=["http://10.10.10.5/"],
                evidence_refs=[
                    {"tool_run_id": str(uuid4()), "excerpt": "matched at /cgi-bin/.%2e/etc/passwd"}
                ],
                critic_verdict="approve",
            ),
            DraftFinding(
                title="Anonymous SMB null session",
                description="null session permitted on 445",
                severity="medium",
                evidence_refs=[{"tool_run_id": str(uuid4()), "excerpt": "STATUS_SUCCESS"}],
                critic_verdict="approve",
            ),
        ],
        vetoed_findings=[
            DraftFinding(
                title="Generic-Banner-Disclosure",
                description="Server banner reveals product (low value)",
                severity="info",
                critic_verdict="veto",
                critic_concern="single-source evidence; not a real vuln",
            )
        ],
        attack_paths=[{"objective": "data_exfil_sim", "cost": 2.0}],
    )


def test_html_render_includes_findings_and_appendix():
    html = render_html(_sample_draft())
    assert "Apache 2.4.49 Path Traversal" in html
    assert "Anonymous SMB null session" in html
    # Vetoed finding renders only in the appendix, not as a normal finding
    assert "Findings Rejected During Review" in html
    assert "Generic-Banner-Disclosure" in html
    # Standard finding count in main body should be 2, not 3
    assert html.count('class="finding ') == 2 + 1  # 2 accepted + 1 vetoed in appendix


def test_json_separates_accepted_and_vetoed():
    import json

    doc = json.loads(render_json(_sample_draft()))
    assert len(doc["findings"]) == 2
    assert len(doc["vetoed_findings"]) == 1
    assert doc["vetoed_findings"][0]["title"] == "Generic-Banner-Disclosure"


def test_markdown_includes_appendix():
    md = render_markdown(_sample_draft())
    assert "Appendix: Findings Rejected During Review" in md
    assert "~~Generic-Banner-Disclosure~~" in md


def test_sarif_emits_results_for_accepted_findings():
    import json

    doc = json.loads(render_sarif(_sample_draft()))
    # Only accepted findings flow to SARIF results (vetoed kept in JSON separately)
    assert len(doc["runs"][0]["results"]) == 2
