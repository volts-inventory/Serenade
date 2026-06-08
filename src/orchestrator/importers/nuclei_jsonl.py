"""nuclei JSONL importer."""

from __future__ import annotations

from uuid import UUID

from tools.nuclei_ import _parse as parse_nuclei_jsonl
from worldstate.schema import (
    EvidenceRef,
    Finding,
    Severity,
    deterministic_node_id,
    natural_key_for_finding,
)

_SEVERITY_MAP = {
    "info": Severity.INFO,
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


def parse_nuclei_jsonl_text(
    jsonl_text: str, *, engagement_id: UUID, tool_run_id: UUID
) -> list[Finding]:
    """Convert nuclei JSONL output into Finding nodes."""
    out: list[Finding] = []
    for f in parse_nuclei_jsonl(jsonl_text):
        sev = _SEVERITY_MAP.get(f.severity.lower(), Severity.INFO)
        cves = f.classification.get("cve-id", []) if f.classification else []
        cwes = f.classification.get("cwe-id", []) if f.classification else []
        title = f.name or f.template_id
        # Deterministic id keyed on (template, matched URL).
        # Re-importing the same JSONL file (e.g. on retry) writes the same
        # Finding ids so MERGE-on-id dedupes instead of inserting copies.
        out.append(
            Finding(
                id=deterministic_node_id(
                    "Finding",
                    engagement_id,
                    natural_key_for_finding(
                        category=f.template_id or "nuclei",
                        affected_node_id=None,
                        title=f"{title}|{f.matched_url}",
                    ),
                ),
                engagement_id=engagement_id,
                provenance=[tool_run_id],
                title=title,
                description=f.description or f"nuclei template {f.template_id} matched",
                severity=sev,
                cve=cves,
                cwe=cwes[0] if cwes else None,
                evidence=[
                    EvidenceRef(
                        tool_run_id=tool_run_id,
                        excerpt=f"{f.template_id} matched at {f.matched_url}",
                    )
                ],
                references=[f.matched_url] if f.matched_url else [],
            )
        )
    return out
