"""Tool adapter base class.

Every tool is a typed Pydantic adapter — never an LLM-generated shell command.
Tools are registered with a `name`, `stage`, and `risk_class` so the OPA
policy engine can gate calls before execution.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from worldstate.schema import RiskClass, Stage

_TIn = TypeVar("_TIn", bound=BaseModel)
_TOut = TypeVar("_TOut", bound=BaseModel)

# Per-tool summary fed back to the planner. Hard cap so an overlong
# summary cannot blow the context budget. The agent base layer enforces
# the cap defensively, but tools SHOULD stay within it.
TOOL_SUMMARY_MAX_CHARS = 500


class ToolRunContext(BaseModel):
    """Per-invocation context the orchestrator hands to a tool."""

    tool_run_id: UUID
    engagement_id: UUID
    sandbox_id: str
    timeout_seconds: int = 300
    extra_env: dict[str, str] = {}


class ToolResult(BaseModel, Generic[_TOut]):
    tool_run_id: UUID
    name: str
    output: _TOut
    raw_stdout_blob_id: UUID | None = None
    raw_stderr_blob_id: UUID | None = None
    exit_code: int
    started_at: str
    ended_at: str


class Tool(abc.ABC, Generic[_TIn, _TOut]):
    name: ClassVar[str]
    description: ClassVar[str]
    stage: ClassVar[Stage]
    risk_class: ClassVar[RiskClass]
    destructive: ClassVar[bool] = False
    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]

    def input_schema(self) -> dict[str, Any]:
        return self.Input.model_json_schema()

    def output_schema(self) -> dict[str, Any]:
        return self.Output.model_json_schema()

    @classmethod
    def new_run_id(cls) -> UUID:
        return uuid4()

    def extract_targets(self, input: _TIn) -> list[str]:
        """Return all network-reachable targets in this tool call for scope checking.

        Each Tool subclass MUST override this so the author of a new tool
        is forced to consider scope. The default raises so an unimplemented
        override is a loud failure, not a silent scope bypass — preferable
        to a brittle field-name heuristic that might miss a target.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.extract_targets must be implemented — scope check would silently pass otherwise"
        )

    @abc.abstractmethod
    def summarize_for_model(
        self, output: _TOut, *, tool_run_id: UUID, persisted: int
    ) -> str:
        """Return a <=500-char human-readable summary of the interesting
        deltas in ``output`` to feed back into the multi-turn agent loop.

        Feeding the entire ``output.model_dump_json()`` back to the planner
        each turn — up to 4096 chars per tool call — across a 15-turn loop
        with 2-3 tools per turn is hundreds of KB of mostly-noise JSON. This
        hook lets each tool emit ONLY the signal the next planner turn
        actually needs (new hosts, notable findings, severity counts),
        pointing to the ledger for the rest.

        Subclasses MUST override. A naive default ("enumerate list-shaped
        fields, print their lengths") is actively misleading for tools whose
        output is a scalar outcome + a list of rows: the planner sees
        ``rows=N`` and never the outcome string, so a "tool returned 5 rows
        but all FAILED" call looks indistinguishable from a successful
        5-row read.

        Declared ``@abc.abstractmethod`` so a tool that forgets to override
        fails at class instantiation (loud, at registry build time) instead
        of mid-step (silent until runtime). Abstractness is enforced
        structurally by Python's ABC machinery.

        Subclasses SHOULD return a string <= ``TOOL_SUMMARY_MAX_CHARS``
        (500). The agent layer hard-truncates anything longer and emits
        an ``agent.tool_summary_overflow`` warning.
        """

    @abc.abstractmethod
    async def run(self, input: _TIn, ctx: ToolRunContext) -> ToolResult[_TOut]:
        """Execute the tool. Implementations MUST:
        - validate scope before any network egress (the orchestrator already did,
          but defense-in-depth);
        - never accept arbitrary shell strings — only structured inputs;
        - write raw stdout/stderr to the evidence store and reference the blob IDs."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name]

    def all(self) -> list[Tool[Any, Any]]:
        return list(self._tools.values())

    def for_stage(self, stage: Stage) -> list[Tool[Any, Any]]:
        return [t for t in self._tools.values() if t.stage == stage]


def build_default_registry() -> ToolRegistry:
    """Import-and-register the built-in tools.

    Tool modules are imported lazily so the test surface stays small and so
    that a tool with optional binaries doesn't break registry construction.
    Missing tool modules raise ImportError, which is caught and logged.
    """
    reg = ToolRegistry()
    _safe_register(reg, "tools.nmap_", ["NmapServiceScan", "NmapHostDiscovery"])
    _safe_register(reg, "tools.httpx_", ["HttpxProbe"])
    _safe_register(reg, "tools.nuclei_", ["NucleiScan"])
    _safe_register(reg, "tools.ffuf_", ["FfufDirBrute"])
    _safe_register(reg, "tools.searchsploit_", ["SearchsploitLookup"])
    _safe_register(reg, "tools.poc_exec", ["CuratedPoCExec"])
    # EDB execution tier. Registered so the agent's `edb_exploit_exec`
    # proposals (see agents/exploit.py) reach the policy gate. The tool is
    # NOT blanket-destructive: non-destructive scripts (remote/local/privesc/
    # webapp/bypass) run in the sandbox (auto under autonomous via
    # edb_auto_exploit, approval-gated in guided); destructive DoS modules are
    # gated per-module inside run() behind destructive_opt_in.
    _safe_register(reg, "tools.edb_exec", ["EdbExec"])
    _safe_register(reg, "tools.netexec_", ["NetExec"])
    _safe_register(reg, "tools.impacket_", ["ImpacketKerberoast", "ImpacketASRepRoast"])
    _safe_register(reg, "tools.bloodhound_", ["BloodHoundCollect", "BloodHoundQuery"])
    _safe_register(reg, "tools.linpeas_", ["LinPEAS"])
    _safe_register(reg, "tools.msfrpc_", ["MetasploitExploit"])
    _safe_register(reg, "tools.windows_via_wmi", ["SharpHoundViaWmi", "RubeusViaWmi"])
    _safe_register(reg, "tools.mimikatz_via_worker", ["MimikatzViaWorker"])
    return reg


def _safe_register(reg: ToolRegistry, module_path: str, class_names: list[str]) -> None:
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return
    for cn in class_names:
        cls = getattr(mod, cn, None)
        if cls is None:
            continue
        try:
            reg.register(cls())
        except Exception:
            # Tool may need optional deps not present in test env
            pass
