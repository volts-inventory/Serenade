from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from tools.base import Tool, ToolRegistry, ToolResult, ToolRunContext, build_default_registry
from worldstate.schema import RiskClass, Stage


def test_default_registry_registers_edb_exec_runnable() -> None:
    """``edb_exploit_exec`` must be registered (so the agent can actually use
    it) and must NOT be blanket-destructive — marking the whole tool
    destructive would make the policy gate DENY every EDB call, killing the
    feature. Destructive EDB modules (DoS) are gated per-module inside the
    tool; the destructive *gate* itself is exercised via the curated-PoC path
    (``curated_poc_exec``), which is also registered."""
    reg = build_default_registry()
    names = {t.name for t in reg.all()}
    assert "edb_exploit_exec" in names, "edb_exploit_exec must be registered so the agent can use it"
    assert reg.get("edb_exploit_exec").destructive is False, "edb must not be blanket-destructive"
    assert "curated_poc_exec" in names, "curated PoC path (destructive gate) must be registered"


class _NopIn(BaseModel):
    targets: list[str]


class _NopOut(BaseModel):
    count: int


class _NopTool(Tool[_NopIn, _NopOut]):
    name = "nop"
    description = "no-op tool used in tests"
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input = _NopIn
    Output = _NopOut

    async def run(self, input: _NopIn, ctx: ToolRunContext) -> ToolResult[_NopOut]:
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=_NopOut(count=len(input.targets)),
            exit_code=0,
            started_at="2024-01-01T00:00:00Z",
            ended_at="2024-01-01T00:00:01Z",
        )

    # R3 #I1 Bug 7: summarize_for_model is now @abstractmethod so any
    # concrete subclass (including test stubs) must provide one or
    # Tool() instantiation fails with TypeError.
    def summarize_for_model(
        self, output: _NopOut, *, tool_run_id: UUID, persisted: int
    ) -> str:
        return f"nop ok: count={output.count} persisted={persisted}"

    def extract_targets(self, input: _NopIn) -> list[str]:
        return list(input.targets)


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(_NopTool())
    with pytest.raises(ValueError):
        reg.register(_NopTool())


def test_for_stage_filters():
    reg = ToolRegistry()
    reg.register(_NopTool())
    assert len(reg.for_stage(Stage.RECON)) == 1
    assert reg.for_stage(Stage.EXPLOIT) == []


@pytest.mark.asyncio
async def test_input_schema_is_json_schema():
    t = _NopTool()
    schema = t.input_schema()
    assert schema["type"] == "object"
    assert "targets" in schema["properties"]


@pytest.mark.asyncio
async def test_tool_run_passes_context():
    t = _NopTool()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="dev",
    )
    res = await t.run(_NopIn(targets=["a", "b"]), ctx)
    assert res.output.count == 2
    assert res.exit_code == 0
