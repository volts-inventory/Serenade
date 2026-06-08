"""httpx (ProjectDiscovery) adapter for web-service probing.

Probes a list of targets (IPs/hosts/URLs) and returns per-target metadata:
status, title, tech stack, IP, TLS, server header.
"""

from __future__ import annotations

import json
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field

from orchestrator.evidence import EvidenceStore
from sandbox.network import engagement_network_name
from sandbox.runner import SandboxRunner
from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class HttpxProbeInput(BaseModel):
    targets: list[str] = Field(min_length=1, description="IPs, hosts, or URLs")
    ports: list[int] = Field(default_factory=lambda: [80, 443, 8080, 8443])
    follow_redirects: bool = True


class HttpxResult(BaseModel):
    url: str
    host: str
    port: int
    scheme: str
    status_code: int | None = None
    title: str | None = None
    server: str | None = None
    technologies: list[str] = Field(default_factory=list)
    content_length: int | None = None
    tls: bool = False


class HttpxOutput(BaseModel):
    probes: list[HttpxResult] = Field(default_factory=list)
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


class HttpxProbe(Tool[HttpxProbeInput, HttpxOutput]):
    name = "httpx_probe"
    description = (
        "Probe targets for live HTTP(S) services. Returns status, title, "
        "tech stack, server header. Use this before web enumeration."
    )
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input: ClassVar[type[BaseModel]] = HttpxProbeInput
    Output: ClassVar[type[BaseModel]] = HttpxOutput

    def extract_targets(self, input: HttpxProbeInput) -> list[str]:
        return list(input.targets)

    def summarize_for_model(
        self, output: HttpxOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        live = [p for p in output.probes if (p.status_code or 0) > 0]
        notable: list[str] = []
        for p in live[:5]:
            tech = ",".join(p.technologies[:3]) if p.technologies else ""
            label = f"{p.status_code} {p.url}"
            if p.server:
                label += f" srv={p.server}"
            if tech:
                label += f" tech={tech}"
            if p.title:
                label += f" '{p.title[:40]}'"
            notable.append(label)
        joined = " | ".join(notable) if notable else "no live services"
        summary = (
            f"{self.name}: probed {len(output.probes)} targets, "
            f"{len(live)} live. top: {joined}. "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: HttpxProbeInput, ctx: ToolRunContext) -> ToolResult[HttpxOutput]:
        ports_csv = ",".join(str(p) for p in input.ports)
        # X-Serenade-Engagement header for proxy bucketing.
        eng_header = f"X-Serenade-Engagement: {ctx.engagement_id}"
        argv = [
            "httpx",
            "-silent",
            "-json",
            "-tech-detect",
            "-status-code",
            "-title",
            "-server",
            "-tls-probe",
            "-ports",
            ports_csv,
            "-H",
            eng_header,
        ]
        if input.follow_redirects:
            argv.append("-follow-redirects")
        executor = _build_executor()
        # Bind the runner to the per-engagement network so httpx can reach
        # the in-scope targets via the engagement's bridge (and CAN'T leak
        # to the host LAN through Docker's default network). Mirror the
        # canonical _helpers.ToolExecutor.run_and_capture pattern.
        executor.sandbox.engagement_id = str(ctx.engagement_id)
        executor.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        # httpx reads targets from stdin
        result = await executor.sandbox.run(
            argv,
            timeout_seconds=ctx.timeout_seconds,
        )
        # Re-run with stdin if no targets provided as args (httpx defaults to stdin anyway)
        if result.exit_code != 0 or not result.stdout:
            result = await executor.sandbox.run(
                [*argv],
                timeout_seconds=ctx.timeout_seconds,
            )

        stdout_id: str | None = None
        if executor.evidence and result.stdout:
            blob_id, _ = executor.evidence.put(
                result.stdout,
                engagement_id=ctx.engagement_id,
                kind=f"{self.name}/stdout",
            )
            stdout_id = str(blob_id)

        probes = _parse(result.stdout.decode("utf-8", errors="replace"))

        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=HttpxOutput(
                probes=probes,
                raw_stdout_blob_id=stdout_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


def _parse(jsonl: str) -> list[HttpxResult]:
    out: list[HttpxResult] = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = doc.get("url") or doc.get("input") or ""
        host = doc.get("host") or doc.get("input") or ""
        scheme = doc.get("scheme") or ("https" if url.startswith("https") else "http")
        port = int(doc.get("port") or (443 if scheme == "https" else 80))
        techs = doc.get("tech") or doc.get("technologies") or []
        if isinstance(techs, dict):
            techs = list(techs.keys())
        out.append(
            HttpxResult(
                url=url,
                host=host,
                port=port,
                scheme=scheme,
                status_code=doc.get("status_code") or doc.get("status-code"),
                title=doc.get("title"),
                server=doc.get("webserver") or doc.get("server"),
                technologies=[str(t) for t in techs],
                content_length=doc.get("content_length") or doc.get("content-length"),
                tls=bool(doc.get("tls") or scheme == "https"),
            )
        )
    return out
