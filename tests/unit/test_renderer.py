from __future__ import annotations

import json
from uuid import uuid4

from agents.report import DraftFinding, DraftReport
from reports.renderer import (
    render_html,
    render_json,
    render_markdown,
    render_sarif,
)


def _draft() -> DraftReport:
    return DraftReport(
        engagement_id=uuid4(),
        title="Test Engagement",
        executive_summary="Two findings on host 10.10.10.5.",
        coverage={"hosts": 1, "services": 4, "findings": 2},
        findings=[
            DraftFinding(
                title="Apache Path Traversal",
                description="Apache 2.4.49 path traversal RCE",
                severity="critical",
                cvss_v3_1=9.8,
                cwe="CWE-22",
                cve=["CVE-2021-41773"],
                affected_assets=["http://10.10.10.5/"],
                evidence_refs=[
                    {"tool_run_id": str(uuid4()), "excerpt": "matched at /cgi-bin/.%2e/etc/passwd"}
                ],
                remediation="Upgrade to Apache 2.4.51+",
                references=["https://nvd.nist.gov/vuln/detail/CVE-2021-41773"],
                critic_verdict="approve",
            ),
            DraftFinding(
                title="Anonymous SMB",
                description="null session permitted",
                severity="medium",
                evidence_refs=[{"tool_run_id": str(uuid4()), "excerpt": "STATUS_SUCCESS"}],
                critic_verdict="approve",
            ),
        ],
        attack_paths=[{"objective": "data_exfil_sim", "cost": 2.0}],
    )


def test_html_contains_findings():
    html = render_html(_draft())
    assert "Apache Path Traversal" in html
    assert "CVE-2021-41773" in html
    assert "severity-critical" in html
    assert "Anonymous SMB" in html


def test_html_escapes_user_content():
    draft = _draft()
    draft.findings[0].title = "<script>alert(1)</script>"
    html = render_html(draft)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_round_trip():
    md = render_markdown(_draft())
    assert "## Findings" in md
    assert "[CRITICAL] Apache Path Traversal" in md
    assert "CVE-2021-41773" in md


def test_json_is_valid_and_complete():
    raw = render_json(_draft())
    doc = json.loads(raw)
    assert doc["title"] == "Test Engagement"
    assert len(doc["findings"]) == 2


def test_sarif_structure():
    raw = render_sarif(_draft())
    doc = json.loads(raw)
    assert doc["version"] == "2.1.0"
    assert len(doc["runs"]) == 1
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "Serenade"
    assert len(run["results"]) == 2
    crit = next(r for r in run["results"] if "CRITICAL" not in r["message"]["text"].upper())
    assert crit["level"] in ("error", "warning", "note")
    # affected_assets surfaces as locations
    apache_result = next(r for r in run["results"] if r["ruleId"] == "CWE-22")
    assert (
        apache_result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == "http://10.10.10.5/"
    )


def test_empty_draft_renders():
    draft = DraftReport(engagement_id=uuid4(), title="Empty", coverage={})
    assert "Empty" in render_html(draft)
    assert "No findings" in render_html(draft)
    assert "No findings" in render_markdown(draft)
