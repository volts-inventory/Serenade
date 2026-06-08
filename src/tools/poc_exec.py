"""CuratedPoCExec — the only path that runs a curated PoC.

Because it's a Tool subclass, it flows through BaseAgent.step()'s canonical
loop: OPA policy gate → PreExploitCritic gate → sandbox execution. There is
no other code path that invokes a PoC.

Safety invariants:
- All curated PoC execution flows through this Tool — there is no bypass path.
- `dispatch()` enforces `can_auto_run` + preconditions internally, so callers
  cannot skip the gates.
- PoC sandbox execution is via SandboxRunner; PoC modules never exec in the
  worker process.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field

from agents.poc_runtime import PoCDispatcher, PoCExecutionContext, PoCLibrary, PoCResult
from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage


class CuratedPoCExecInput(BaseModel):
    poc_id: str = Field(min_length=1, description="ID of the curated PoC in pocs/<id>/")
    target_host: str = Field(min_length=1, description="In-scope target IP or hostname")
    target_port: int | None = Field(default=None, ge=0, le=65535)
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters matching the PoC's parameters.schema.json"
    )
    candidate_id: str | None = Field(
        default=None,
        description="ExploitCandidate UUID this attempt originates from, if any",
    )


class CuratedPoCExecOutput(BaseModel):
    outcome: str
    summary: str
    error_message: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    poc_id: str
    manifest_sha256: str
    candidate_id: str | None = None


# Module-level holders so the Tool subclass instantiates cleanly without
# constructor args (it's discovered via `build_default_registry()`). The
# worker bundle wires concrete instances at startup via `configure()`.
_LIBRARY: PoCLibrary | None = None
_DISPATCHER: PoCDispatcher | None = None


def configure(library: PoCLibrary, dispatcher: PoCDispatcher) -> None:
    """Wire the library + dispatcher singletons at worker startup."""
    global _LIBRARY, _DISPATCHER
    _LIBRARY = library
    _DISPATCHER = dispatcher


class CuratedPoCExec(Tool[CuratedPoCExecInput, CuratedPoCExecOutput]):
    name = "curated_poc_exec"
    description = (
        "Execute a curated PoC from pocs/<id>/ against an in-scope target. "
        "ONLY way to run an exploit. Runs ENTIRELY inside the sandbox. "
        "Subject to OPA policy + PreExploitCritic gates + (when implemented) "
        "operator approval. PoCs from EDB/Metasploit/nuclei discovery are NOT "
        "valid inputs to this tool — they must be promoted to the curated "
        "library first."
    )
    stage = Stage.EXPLOIT
    risk_class = RiskClass.EXPLOIT
    destructive = False  # Per-PoC manifest flag governs; dispatcher enforces.
    Input: ClassVar[type[BaseModel]] = CuratedPoCExecInput
    Output: ClassVar[type[BaseModel]] = CuratedPoCExecOutput

    def extract_targets(self, input: CuratedPoCExecInput) -> list[str]:
        return [input.target_host]

    def summarize_for_model(
        self, output: CuratedPoCExecOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # do NOT inline `output.summary` / `output.error_message`
        # into the planner-visible string. A curated PoC's summary field is
        # PoC-author-controlled free-form text — credential-extraction PoCs
        # (e.g. shadow-dump, SAM-dump, .htpasswd grab) routinely write
        # extracted secrets into the summary. The full summary is preserved
        # in the ToolResult and persisted to evidence; the planner gets
        # outcome + artifact key count only.
        artifact_count = len(output.artifacts)
        err_state = "yes" if output.error_message else "no"
        cand = f" candidate={output.candidate_id}" if output.candidate_id else ""
        summary = (
            f"{self.name}: poc={output.poc_id} outcome={output.outcome} "
            f"error={err_state} artifact_count={artifact_count}{cand} "
            f"(summary/error body redacted from planner context). "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: CuratedPoCExecInput, ctx: ToolRunContext
    ) -> ToolResult[CuratedPoCExecOutput]:
        from datetime import datetime

        if _DISPATCHER is None or _LIBRARY is None:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=CuratedPoCExecOutput(
                    outcome="error",
                    summary="curated_poc_exec not configured",
                    error_message="dispatcher_not_configured",
                    poc_id=input.poc_id,
                    manifest_sha256="",
                    candidate_id=input.candidate_id,
                ),
                exit_code=1,
                started_at=iso(datetime.now(UTC)),
                ended_at=iso(datetime.now(UTC)),
            )

        loaded = _LIBRARY.get(input.poc_id)
        if loaded is None:
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=CuratedPoCExecOutput(
                    outcome="error",
                    summary=f"PoC not in library: {input.poc_id}",
                    error_message="manifest_not_loaded",
                    poc_id=input.poc_id,
                    manifest_sha256="",
                    candidate_id=input.candidate_id,
                ),
                exit_code=1,
                started_at=iso(datetime.now(UTC)),
                ended_at=iso(datetime.now(UTC)),
            )

        poc_ctx = PoCExecutionContext(
            engagement_id=ctx.engagement_id,
            poc_id=input.poc_id,
            manifest_sha256=loaded.manifest_sha256,
            target_host=input.target_host,
            target_port=input.target_port,
        )
        # The engagement context the dispatcher needs is normally injected by
        # the orchestrator at engagement-start time. For tool-invoked PoC
        # execution, the BaseAgent.step injects the engagement dict into
        # ctx.extra_env. If absent, fail closed.
        engagement_for_dispatch: dict[str, Any] = {}
        # extra_env on ToolRunContext is currently typed dict[str, str];
        # engagement dict is conveyed via a JSON-encoded string key.
        import json as _json

        eng_json = ctx.extra_env.get("__engagement_json__")
        if eng_json:
            try:
                engagement_for_dispatch = _json.loads(eng_json)
            except Exception:
                engagement_for_dispatch = {}

        started = datetime.now(UTC)
        result: PoCResult = await _DISPATCHER.dispatch(
            poc_id=input.poc_id,
            params=input.params,
            ctx=poc_ctx,
            engagement=engagement_for_dispatch,
        )
        ended = datetime.now(UTC)

        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=CuratedPoCExecOutput(
                outcome=result.outcome,
                summary=result.summary,
                error_message=result.error_message,
                artifacts=result.artifacts,
                poc_id=input.poc_id,
                manifest_sha256=loaded.manifest_sha256,
                candidate_id=input.candidate_id,
            ),
            exit_code=0 if result.outcome in ("success", "skipped") else 1,
            started_at=iso(started),
            ended_at=iso(ended),
        )
