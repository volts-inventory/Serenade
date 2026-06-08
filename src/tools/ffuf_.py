"""ffuf adapter — directory and parameter fuzzing."""

from __future__ import annotations

import json
import shlex
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from orchestrator.evidence import EvidenceStore
from sandbox.runner import SandboxRunner
from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class FfufDirBruteInput(BaseModel):
    url: str = Field(description="Base URL with FUZZ placeholder if not auto-appended")
    wordlist: Literal["common", "dirb-common"] = "common"
    extensions: list[str] = Field(default_factory=list)
    rate_limit: int = Field(default=40, ge=1, le=500)
    match_codes: list[int] = Field(default_factory=lambda: [200, 204, 301, 302, 307, 401, 403])


class FfufHit(BaseModel):
    url: str
    status: int
    length: int
    words: int
    lines: int
    content_type: str | None = None


class FfufOutput(BaseModel):
    hits: list[FfufHit] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    command: str


_WORDLISTS = {
    "common": "/usr/share/wordlists/common.txt",
    "dirb-common": "/usr/share/wordlists/dirb-common.txt",
}


def _build_executor() -> ToolExecutor:
    from orchestrator.settings import get_settings

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    return ToolExecutor(SandboxRunner(), evidence)


class FfufDirBrute(Tool[FfufDirBruteInput, FfufOutput]):
    name = "ffuf_dir_brute"
    description = (
        "Brute-force directories and files under a base URL. Append FUZZ to the "
        "URL, or omit and the tool appends /FUZZ automatically. Returns matched "
        "paths with status, size, content-type."
    )
    stage = Stage.ENUMERATION
    risk_class = RiskClass.ENUM
    Input: ClassVar[type[BaseModel]] = FfufDirBruteInput
    Output: ClassVar[type[BaseModel]] = FfufOutput

    def extract_targets(self, input: FfufDirBruteInput) -> list[str]:
        return [input.url]

    def summarize_for_model(
        self, output: FfufOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        by_code: dict[int, int] = {}
        for h in output.hits:
            by_code[h.status] = by_code.get(h.status, 0) + 1
        codes_summary = ", ".join(f"{c}={n}" for c, n in sorted(by_code.items())) or "none"
        interesting = [h for h in output.hits if h.status in (200, 301, 302, 401)]
        top = ", ".join(h.url for h in interesting[:5]) or "none"
        summary = (
            f"{self.name}: {len(output.hits)} hits ({codes_summary}). "
            f"notable paths: {top}. persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: FfufDirBruteInput, ctx: ToolRunContext) -> ToolResult[FfufOutput]:
        url = input.url if "FUZZ" in input.url else input.url.rstrip("/") + "/FUZZ"
        wordlist = _WORDLISTS[input.wordlist]
        # Per-tool-run-id temp file; avoid /tmp races in any remaining
        # shared-sandbox scenarios. With per-call containers /tmp is already
        # unique, but defense-in-depth is cheap.
        out_path = f"/tmp/ffuf-{ctx.tool_run_id}.json"
        # X-Serenade-Engagement header for proxy bucketing.
        eng_header = f"X-Serenade-Engagement: {ctx.engagement_id}"
        argv = [
            "ffuf",
            "-u",
            url,
            "-w",
            wordlist,
            "-of",
            "json",
            "-o",
            out_path,
            "-mc",
            ",".join(str(c) for c in input.match_codes),
            "-rate",
            str(input.rate_limit),
            "-H",
            eng_header,
            "-s",  # silent
        ]
        if input.extensions:
            argv += ["-e", ",".join(input.extensions)]
        executor = _build_executor()
        # Run ffuf and immediately cat the JSON in the same container by
        # using a shell wrapper. With per-call sandbox containers, separate
        # `docker exec` calls land in different containers.
        shell_argv = [
            "/bin/sh",
            "-c",
            shlex.join(argv) + " ; cat " + shlex.quote(out_path),
        ]
        result, stdout_id, _ = await executor.run_and_capture(
            shell_argv, ctx=ctx, tool_name=self.name, timeout_seconds=900
        )
        # ffuf -s emits no stdout; the cat output is the JSON document.
        hits = _parse(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=FfufOutput(
                hits=hits,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


def _parse(text: str) -> list[FfufHit]:
    if not text.strip():
        return []
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    results = doc.get("results") or []
    out: list[FfufHit] = []
    for r in results:
        out.append(
            FfufHit(
                url=str(r.get("url") or r.get("input", {}).get("FUZZ", "")),
                status=int(r.get("status") or 0),
                length=int(r.get("length") or 0),
                words=int(r.get("words") or 0),
                lines=int(r.get("lines") or 0),
                content_type=r.get("content-type") or r.get("content_type"),
            )
        )
    return out
