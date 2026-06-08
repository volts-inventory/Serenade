"""R2 #E — report renderer "Pipeline regressions" section.

Covers:
- The section appears in HTML + Markdown when ``DraftReport.regressions``
  is non-empty.
- The section is omitted entirely (no empty heading, no table) when no
  regressions were recorded — the common case for most engagements.
- Each row surfaces target_stage, trigger_kind, rationale, and the
  council verdict.
- The HTML render escapes attacker-controlled rationale text (same
  defense-in-depth as the findings render path; see test_renderer).
"""

from __future__ import annotations

from uuid import uuid4

from agents.report import DraftFinding, DraftReport, RegressionAppendix
from reports.renderer import render_html, render_markdown


def _draft_with_regressions() -> DraftReport:
    return DraftReport(
        engagement_id=uuid4(),
        title="Regression Demo",
        executive_summary="One finding plus two regressions.",
        coverage={"hosts": 1, "findings": 1},
        findings=[
            DraftFinding(
                title="HTTP service banner",
                description="apache disclosed",
                severity="info",
                evidence_refs=[{"tool_run_id": str(uuid4()), "excerpt": "Server: apache/2.4"}],
                critic_verdict="approve",
            ),
        ],
        regressions=[
            RegressionAppendix(
                target_stage="recon",
                trigger_kind="new_subnet_discovered",
                rationale=(
                    "Lateral movement uncovered creds reachable to 10.0.5.0/24 "
                    "which was not in initial scope coverage."
                ),
                council_decision="approve",
                council_rationale="novelty + budget headroom",
                requested_at="2024-05-21T11:30:00+00:00",
            ),
            RegressionAppendix(
                target_stage="enumeration",
                trigger_kind="new_service_class_discovered",
                rationale="New SMB port revealed by triage stage; re-enum required.",
                council_decision="approve",
                council_rationale="approve",
                requested_at="2024-05-21T12:00:00+00:00",
            ),
        ],
    )


def _draft_without_regressions() -> DraftReport:
    return DraftReport(
        engagement_id=uuid4(),
        title="Clean Run",
        coverage={"hosts": 1, "findings": 0},
    )


# --------------------------------------------------------------------------- #
# HTML                                                                         #
# --------------------------------------------------------------------------- #


def test_html_includes_pipeline_regressions_section_when_present() -> None:
    html = render_html(_draft_with_regressions())
    assert "Pipeline Regressions" in html
    # Stage codes appear as <code>…</code> entries in the table.
    assert "recon" in html
    assert "enumeration" in html
    # Trigger and rationale render verbatim (the renderer escapes
    # HTML special characters; the strings above contain none).
    assert "new_subnet_discovered" in html
    assert "10.0.5.0/24" in html


def test_html_omits_pipeline_regressions_section_when_empty() -> None:
    """Most engagements never trigger a regression. The section MUST NOT
    render as an empty heading + empty table — that would clutter every
    customer-facing report.
    """
    html = render_html(_draft_without_regressions())
    assert "Pipeline Regressions" not in html


def test_html_escapes_user_supplied_rationale() -> None:
    """Same defense-in-depth as the findings render path. A malicious
    tool output or a buggy agent could land hostile text in the rationale
    field; the renderer must escape it before injection into the HTML
    <td>.
    """
    draft = _draft_with_regressions()
    draft.regressions[0].rationale = '<script>alert("xss")</script>'
    html = render_html(draft)
    assert '<script>alert("xss")</script>' not in html
    assert "&lt;script&gt;" in html


# --------------------------------------------------------------------------- #
# Markdown                                                                     #
# --------------------------------------------------------------------------- #


def test_markdown_includes_pipeline_regressions_section_when_present() -> None:
    md = render_markdown(_draft_with_regressions())
    assert "## Pipeline Regressions" in md
    # Stage codes show up in the table.
    assert "recon" in md
    assert "enumeration" in md
    assert "new_subnet_discovered" in md
    assert "10.0.5.0/24" in md


def test_markdown_omits_pipeline_regressions_section_when_empty() -> None:
    md = render_markdown(_draft_without_regressions())
    assert "## Pipeline Regressions" not in md


def test_markdown_table_row_count_matches_regressions() -> None:
    """One header row + one separator row + one row per regression.
    Sanity-check the markdown table structure so a template tweak
    doesn't silently drop rows.
    """
    md = render_markdown(_draft_with_regressions())
    # The table starts at "| Target stage" and ends before the next
    # blank line / heading. Count pipe-led lines after the header.
    in_table = False
    rows = 0
    for line in md.splitlines():
        if line.startswith("| Target stage"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|---") or line.strip() == "":
                if line.strip() == "":
                    break
                continue
            if line.startswith("|"):
                rows += 1
    assert rows == 2
