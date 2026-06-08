"""nuclei adapter — template-driven vulnerability scanner.

Reads JSONL output from nuclei and emits typed findings.
"""

from __future__ import annotations

import json
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from orchestrator.evidence import EvidenceStore
from sandbox.runner import SandboxRunner
from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class NucleiScanInput(BaseModel):
    targets: list[str] = Field(min_length=1)
    severity: list[Literal["info", "low", "medium", "high", "critical"]] = Field(
        default_factory=lambda: ["medium", "high", "critical"]
    )

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v: list[str]) -> list[str]:
        # nuclei targets are passed as `-u <value>`; a leading-dash value would
        # be parsed as a nuclei flag (e.g. `-` = stdin). Reject (sandbox-09).
        for t in v:
            if t.startswith("-"):
                raise ValueError(
                    f"target {t!r} must not start with '-' (would be parsed as "
                    "a nuclei flag)"
                )
        return v
    tags: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list, description="Template paths or IDs")
    rate_limit: int = Field(default=50, ge=1, le=1000)
    concurrency: int = Field(default=25, ge=1, le=200)

    @field_validator("templates")
    @classmethod
    def _check_templates(cls, v: list[str]) -> list[str]:
        # `-t` template values are NOT scope-checked egress targets, but a
        # REMOTE template URL would pull+execute template logic against an
        # arbitrary destination, bypassing the scope gate (sandbox-02). Reject
        # any remote URL; allow only local template IDs/paths (no scheme, no
        # leading dash so it can't act as a nuclei flag — sandbox-09).
        for t in v:
            low = t.lower()
            if low.startswith(("http://", "https://", "ftp://", "//")):
                raise ValueError(
                    f"remote template URLs are not allowed (got {t!r}); use a "
                    "curated local template id/path"
                )
            if t.startswith("-"):
                raise ValueError(
                    f"template {t!r} must not start with '-' (would be parsed "
                    "as a nuclei flag)"
                )
        return v


class NucleiFinding(BaseModel):
    template_id: str
    name: str
    severity: str
    matched_url: str
    host: str | None = None
    description: str | None = None
    classification: dict[str, list[str]] = Field(default_factory=dict)
    extracted_results: list[str] = Field(default_factory=list)


class NucleiOutput(BaseModel):
    findings: list[NucleiFinding] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    command: str


def _build_executor() -> ToolExecutor:
    from orchestrator.settings import get_settings

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    return ToolExecutor(SandboxRunner(), evidence)


class NucleiScan(Tool[NucleiScanInput, NucleiOutput]):
    name = "nuclei_scan"
    description = (
        "Run nuclei templates against targets. Returns matched findings with "
        "template id, severity, matched URL, and any extracted data. Use after "
        "httpx_probe identifies live web services."
    )
    stage = Stage.ENUMERATION
    risk_class = RiskClass.ENUM
    Input: ClassVar[type[BaseModel]] = NucleiScanInput
    Output: ClassVar[type[BaseModel]] = NucleiOutput

    def extract_targets(self, input: NucleiScanInput) -> list[str]:
        # Targets are the scan targets; templates that happen to be URLs are
        # NOT egress — nuclei reads them as template-source URLs which we
        # control via the curated registry. If a future input allows
        # arbitrary remote templates that's a separate concern.
        return list(input.targets)

    def summarize_for_model(
        self, output: NucleiOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        counts: dict[str, int] = {}
        for f in output.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        # Order so highest severity shows first.
        order = ("critical", "high", "medium", "low", "info", "unknown")
        sev_summary = ", ".join(
            f"{k}={counts[k]}" for k in order if counts.get(k)
        ) or "none"
        # Highlight up to 2 highest-severity templates by name.
        ranked = sorted(
            output.findings,
            key=lambda f: order.index(f.severity) if f.severity in order else len(order),
        )
        top = "; ".join(
            f"{f.severity}:{f.template_id} @ {f.matched_url}"[:120]
            for f in ranked[:2]
        )
        summary = (
            f"{self.name}: {len(output.findings)} findings ({sev_summary}). "
            f"top: {top or 'none'}. persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: NucleiScanInput, ctx: ToolRunContext) -> ToolResult[NucleiOutput]:
        # Stamp X-Serenade-Engagement on outbound requests so mitmproxy can
        # bucket per-engagement (producer side; the proxy is the consumer).
        eng_header = f"X-Serenade-Engagement: {ctx.engagement_id}"
        argv = [
            "nuclei",
            "-jsonl",
            "-silent",
            "-no-color",
            "-disable-update-check",
            "-severity",
            ",".join(input.severity),
            "-rate-limit",
            str(input.rate_limit),
            "-c",
            str(input.concurrency),
            "-H",
            eng_header,
        ]
        for t in input.targets:
            argv += ["-u", t]
        for tg in input.tags:
            argv += ["-tags", tg]
        for tp in input.templates:
            argv += ["-t", tp]
        executor = _build_executor()
        result, stdout_id, _ = await executor.run_and_capture(
            argv, ctx=ctx, tool_name=self.name, timeout_seconds=1800
        )
        findings = _parse(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=NucleiOutput(
                findings=findings,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


def _parse(jsonl: str) -> list[NucleiFinding]:
    out: list[NucleiFinding] = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        info = doc.get("info", {})
        cls = info.get("classification") or {}
        # Normalize classification fields that nuclei emits as comma-separated strings
        classification: dict[str, list[str]] = {}
        for k, v in cls.items():
            if isinstance(v, str):
                classification[k] = [s.strip() for s in v.split(",") if s.strip()]
            elif isinstance(v, list):
                classification[k] = [str(x) for x in v]
        extracted = doc.get("extracted-results") or doc.get("extracted_results") or []
        out.append(
            NucleiFinding(
                template_id=str(doc.get("template-id") or doc.get("template_id") or ""),
                name=str(info.get("name") or ""),
                severity=str(info.get("severity") or "info"),
                matched_url=str(doc.get("matched-at") or doc.get("matched_at") or ""),
                host=str(doc.get("host")) if doc.get("host") else None,
                description=str(info.get("description") or "") or None,
                classification=classification,
                extracted_results=[str(x) for x in extracted],
            )
        )
    return out
