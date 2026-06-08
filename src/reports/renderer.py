"""Render `DraftReport` (from `agents/report.py`) into HTML, PDF, Markdown,
JSON, SARIF.

PDF render uses WeasyPrint if available; otherwise the HTML is the deliverable.
Templates live in `reports/templates/`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader

from agents.report import DraftReport

log = structlog.get_logger(__name__)


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _env(*, autoescape_for_html: bool = True) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        # Default-on autoescape: tool output can contain attacker-controlled
        # strings (finding titles, evidence excerpts), so we escape unless a
        # specific renderer disables it (markdown does).
        autoescape=autoescape_for_html,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_html(draft: DraftReport) -> str:
    tpl = _env(autoescape_for_html=True).get_template("report.html.jinja")
    return tpl.render(draft=draft, generated_at=_now_iso())


def render_markdown(draft: DraftReport) -> str:
    # Markdown isn't HTML; autoescape would HTML-encode characters that markdown
    # treats as literal. The downstream markdown→HTML renderer (if any) is
    # responsible for its own sanitization.
    tpl = _env(autoescape_for_html=False).get_template("report.md.jinja")
    return tpl.render(draft=draft, generated_at=_now_iso())


def render_json(draft: DraftReport) -> str:
    # sort_keys=True so a customer-side hash of report.json is stable
    # across Python versions and dict-insertion-order refactors. Same
    # canonicalisation discipline as the audit-log exports.
    return json.dumps(draft.model_dump(), indent=2, sort_keys=True, default=str)


def render_sarif(draft: DraftReport) -> str:
    """SARIF v2.1.0 — for findings that map cleanly (one rule per template_id)."""
    runs = [
        {
            "tool": {
                "driver": {
                    "name": "Serenade",
                    "informationUri": "https://serenade.local",
                    "version": "0.3.0",
                    "rules": _sarif_rules(draft),
                }
            },
            "results": _sarif_results(draft),
            "automationDetails": {"id": f"serenade/{draft.engagement_id}"},
        }
    ]
    return json.dumps(
        {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": runs,
        },
        indent=2,
        sort_keys=True,  # Deterministic ordering for downstream SARIF indexers
        default=str,
    )


def render_pdf(draft: DraftReport) -> bytes:
    """PDF via WeasyPrint. Caller is responsible for handling ImportError if
    the optional dep isn't installed."""
    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "weasyprint is required for PDF rendering; pip install weasyprint"
        ) from e
    html = render_html(draft)
    return HTML(string=html).write_pdf()  # type: ignore[no-any-return]


def render_all(draft: DraftReport, out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    html_path = out / "report.html"
    html_path.write_text(render_html(draft))
    artifacts["html"] = html_path

    md_path = out / "report.md"
    md_path.write_text(render_markdown(draft))
    artifacts["markdown"] = md_path

    json_path = out / "report.json"
    json_path.write_text(render_json(draft))
    artifacts["json"] = json_path

    sarif_path = out / "report.sarif"
    sarif_path.write_text(render_sarif(draft))
    artifacts["sarif"] = sarif_path

    try:
        pdf_path = out / "report.pdf"
        pdf_path.write_bytes(render_pdf(draft))
        artifacts["pdf"] = pdf_path
    except RuntimeError as e:
        log.warning("renderer.pdf_skipped", err=str(e))

    return artifacts


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


_SEVERITY_TO_SARIF_LEVEL = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}


def _sarif_rules(draft: DraftReport) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for f in draft.findings:
        rid = (f.cwe or f.title)[:128]
        if rid in seen:
            continue
        seen[rid] = {
            "id": rid,
            "name": f.title[:120],
            "shortDescription": {"text": f.title},
            "fullDescription": {"text": f.description[:1000]},
            "defaultConfiguration": {"level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity, "note")},
        }
    return list(seen.values())


def _sarif_results(draft: DraftReport) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in draft.findings:
        rid = (f.cwe or f.title)[:128]
        result = {
            "ruleId": rid,
            "level": _SEVERITY_TO_SARIF_LEVEL.get(f.severity, "note"),
            "message": {"text": f.description[:2000]},
            "properties": {
                "severity": f.severity,
                "cvss_v3_1": f.cvss_v3_1,
                "cve": f.cve,
                "critic_verdict": f.critic_verdict,
            },
        }
        if f.affected_assets:
            result["locations"] = [
                {"physicalLocation": {"artifactLocation": {"uri": a}}} for a in f.affected_assets
            ]
        out.append(result)
    return out
