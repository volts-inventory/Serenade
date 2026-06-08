"""searchsploit adapter — query the ExploitDB mirror by service + version.

This is the discovery tier from ARCHITECTURE.md §8. Output is metadata only —
EDB scripts are never executed from here. ExploitAgent uses curated PoCs only.
"""

from __future__ import annotations

import json
from datetime import date
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from orchestrator.evidence import EvidenceStore
from sandbox.runner import SandboxRunner
from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class SearchsploitInput(BaseModel):
    query: str = Field(min_length=2, description='Service+version, e.g. "vsftpd 2.3.4"')
    platform: Literal["linux", "windows", "multiple", "php", "jsp", "aspx", "unix", ""] = ""
    type: Literal["remote", "local", "webapps", "dos", "shellcode", ""] = ""


class SearchsploitCandidate(BaseModel):
    edb_id: int
    title: str
    date_published: date | None = None
    author: str | None = None
    platform: str | None = None
    type: str | None = None
    verified: bool = False
    path: str | None = None
    url: str | None = None


class SearchsploitOutput(BaseModel):
    candidates: list[SearchsploitCandidate] = Field(default_factory=list)
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


class SearchsploitLookup(Tool[SearchsploitInput, SearchsploitOutput]):
    name = "searchsploit_lookup"
    description = (
        "Look up ExploitDB candidates for a service+version pair. Returns "
        "metadata only — these are CANDIDATES for review, never auto-executed. "
        "ExploitAgent only runs from the curated PoC library."
    )
    stage = Stage.VULN_TRIAGE
    risk_class = RiskClass.RECON  # read-only lookup; not even network egress
    Input: ClassVar[type[BaseModel]] = SearchsploitInput
    Output: ClassVar[type[BaseModel]] = SearchsploitOutput

    def extract_targets(self, input: SearchsploitInput) -> list[str]:
        # searchsploit reads from a local sqlite mirror; no network targets.
        return []

    def summarize_for_model(
        self, output: SearchsploitOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        verified = sum(1 for c in output.candidates if c.verified)
        ranked = sorted(
            output.candidates,
            key=lambda c: (c.date_published.year if c.date_published else 0, c.verified),
            reverse=True,
        )
        top = "; ".join(
            f"EDB-{c.edb_id} ({c.date_published.year if c.date_published else '?'}): {c.title[:50]}"
            for c in ranked[:3]
        )
        summary = (
            f"{self.name}: {len(output.candidates)} candidates ({verified} verified). "
            f"top by year: {top or 'none'}. persisted={persisted}. "
            f"tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: SearchsploitInput, ctx: ToolRunContext
    ) -> ToolResult[SearchsploitOutput]:
        argv = ["searchsploit", "--json", "--disable-colour"]
        if input.platform:
            argv += ["--platform", input.platform]
        # `--` ends option parsing so a query term beginning with `-` (e.g.
        # `-m`, `--nmap`) is treated as a literal search term, not a
        # searchsploit flag (sandbox-09 argument injection).
        argv += ["--", *input.query.split()]
        executor = _build_executor()
        result, stdout_id, _ = await executor.run_and_capture(
            argv, ctx=ctx, tool_name=self.name, timeout_seconds=60
        )
        candidates = _parse(result.stdout.decode("utf-8", errors="replace"), filter_type=input.type)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=SearchsploitOutput(
                candidates=candidates,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


def _parse(stdout: str, filter_type: str = "") -> list[SearchsploitCandidate]:
    if not stdout.strip():
        return []
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    out: list[SearchsploitCandidate] = []
    for entry in doc.get("RESULTS_EXPLOIT") or []:
        if filter_type and entry.get("Type", "").lower() != filter_type:
            continue
        edb_id_raw = entry.get("EDB-ID") or entry.get("EDB_ID") or "0"
        try:
            edb_id = int(edb_id_raw)
        except (TypeError, ValueError):
            continue
        d: date | None = None
        date_str = entry.get("Date_Published") or entry.get("Date")
        if isinstance(date_str, str) and date_str:
            try:
                d = date.fromisoformat(date_str[:10])
            except ValueError:
                d = None
        out.append(
            SearchsploitCandidate(
                edb_id=edb_id,
                title=str(entry.get("Title") or ""),
                date_published=d,
                author=entry.get("Author"),
                platform=entry.get("Platform"),
                type=entry.get("Type"),
                verified=bool(entry.get("Verified") in (True, "1", 1, "true")),
                path=entry.get("Path"),
                url=entry.get("URL"),
            )
        )
    return out
