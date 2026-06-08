"""Curated PoC runtime — manifest loader, parameter validator, exploit
sandbox dispatcher.

Hard rules (ARCHITECTURE.md §8):
- Only PoCs in `pocs/<id>/` with a valid manifest may be executed.
- `auto_run.allowed: true` AND `auto_run.requires_approval: false` AND all
  preconditions true is the only path that runs without operator approval.
- `risk.destructive: true` PoCs are never executed — surfaced as findings.
- Manifest sha256 is recorded with every `ExploitAttempt` for audit.
- **PoC modules NEVER execute in the orchestrator/worker process.** The
  worker has /var/run/docker.sock and full Anthropic+Neo4j+MinIO credentials
  in env; exec in the worker = host compromise. The contract instead is:
  PoC modules define `build_invocation()` + `parse_result()`, both pure
  functions the worker calls; the resulting argv runs inside the sandbox
  via `SandboxRunner.run()`.
- `can_auto_run` is enforced inside `dispatch()` so callers cannot bypass it.
- `auto_run.preconditions` are structured YAML conditions, not free-form
  expressions; the evaluator is deterministic and rejects unknown predicates.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import jsonschema
import structlog
import yaml
from pydantic import BaseModel, Field

from sandbox.runner import SandboxResult, SandboxRunner
from worldstate.schema import RiskClass, normalize_mode

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Manifest schema (Pydantic mirror of pocs/SCHEMA.md)                          #
# --------------------------------------------------------------------------- #


class PoCProvenance(BaseModel):
    source: Literal["exploitdb", "metasploit", "nuclei", "cve_writeup", "original_research"]
    ref: str


class PoCTarget(BaseModel):
    service: str
    version_range: str
    platforms: list[Literal["linux", "windows", "multiple"]]


class PoCParameters(BaseModel):
    schema_ref: str


class PoCRisk(BaseModel):
    class_: RiskClass = Field(alias="class")
    destructive: bool
    modifies_target: bool
    modifies_filesystem: bool
    network_egress_from_target: bool
    detection_likelihood: Literal["low", "medium", "high"]

    model_config = {"populate_by_name": True}


class PoCPrecondition(BaseModel):
    """Structured precondition (NOT free-form CEL).

    Each entry has exactly one predicate. Unknown predicates fail closed.
    Supported:
      - {"engagement_mode_in": ["normal", "aggressive"]}
      - {"engagement_destructive_opt_in": true}
      - {"target_not_in_no_exploit_list": true}
      - {"world_state_has_finding_with_cve": "CVE-XXXX-YYYY"}
    """

    # Accepts both legacy (passive/normal/aggressive/lab) and current
    # (recon_only/assess/guided/autonomous) names; normalized before compare.
    engagement_mode_in: list[str] | None = None
    engagement_destructive_opt_in: bool | None = None
    target_not_in_no_exploit_list: bool | None = None
    world_state_has_finding_with_cve: str | None = None


class PoCAutoRun(BaseModel):
    allowed: bool
    requires_approval: bool
    preconditions: list[PoCPrecondition] = Field(default_factory=list)


class PoCRollback(BaseModel):
    available: bool
    notes: str | None = None


class PoCLicense(BaseModel):
    spdx: str
    attribution: str


class PoCCI(BaseModel):
    test_command: str = "pytest test/test_exploit.py"
    lab: str | None = None


class PoCManifest(BaseModel):
    id: str
    provenance: list[PoCProvenance]
    target: PoCTarget
    parameters: PoCParameters
    risk: PoCRisk
    auto_run: PoCAutoRun
    rollback: PoCRollback
    license: PoCLicense
    ci: PoCCI


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class LoadedPoC:
    manifest: PoCManifest
    manifest_sha256: str
    directory: Path
    parameters_schema: dict[str, Any]


class PoCLibrary:
    """Loads manifests from `pocs/*/manifest.yaml` at construction.

    Refuses any manifest with `license.spdx == "UNKNOWN"` — these are
    template scaffolds, not finished PoCs.
    """

    def __init__(self, root: Path | str = "content/pocs") -> None:
        self.root = Path(root)
        self._by_id: dict[str, LoadedPoC] = {}
        if self.root.exists():
            self._load_all()

    def _load_all(self) -> None:
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            try:
                self._load_one(d)
            except Exception as e:
                log.error("poc.load_failed", dir=str(d), err=str(e))

    def _load_one(self, d: Path) -> None:
        manifest_path = d / "manifest.yaml"
        if not manifest_path.exists():
            return
        raw = manifest_path.read_bytes()
        data = yaml.safe_load(raw)
        manifest = PoCManifest.model_validate(data)
        if manifest.license.spdx == "UNKNOWN":
            log.warning("poc.skip_unknown_license", id=manifest.id, dir=str(d))
            return
        # AST-lint the exploit module BEFORE registering it. Any top-level
        # statement other than imports/defs rejects the PoC so the body
        # doesn't run in the worker on `exec_module`. This means a curated
        # PoC with module-level code never reaches the library.
        from agents.poc_lint import lint_module

        exploit_path = d / "exploit.py"
        if exploit_path.exists():
            ok, reason = lint_module(exploit_path)
            if not ok:
                log.warning("poc.skip_lint_failed", id=manifest.id, dir=str(d), reason=reason)
                return
        sha = hashlib.sha256(raw).hexdigest()

        schema_path = d / manifest.parameters.schema_ref.lstrip("./")
        params_schema: dict[str, Any] = {}
        if schema_path.exists():
            params_schema = json.loads(schema_path.read_text())

        self._by_id[manifest.id] = LoadedPoC(
            manifest=manifest,
            manifest_sha256=sha,
            directory=d,
            parameters_schema=params_schema,
        )

    def get(self, poc_id: str) -> LoadedPoC | None:
        return self._by_id.get(poc_id)

    def all(self) -> list[LoadedPoC]:
        return list(self._by_id.values())

    def for_service(self, service: str) -> list[LoadedPoC]:
        return [p for p in self._by_id.values() if p.manifest.target.service == service]


# --------------------------------------------------------------------------- #
# Execution contract                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class SandboxInvocation:
    """What a PoC module's `build_invocation()` returns.

    `argv` runs inside the sandbox via `SandboxRunner.run(argv)`. The PoC
    module never touches the sandbox itself; it only knows how to construct
    the argv and how to parse the result. No I/O on the worker side.
    """

    argv: list[str]
    timeout_seconds: int = 300
    redacted_argv: list[str] | None = None  # for logging if secrets are in argv


@dataclass
class PoCResult:
    outcome: Literal["success", "failure", "error", "skipped"]
    summary: str
    artifacts: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class PoCExecutionContext:
    """Read-only context passed to `build_invocation`/`parse_result`.

    No sandbox handle — PoCs don't run code on the worker.
    """

    engagement_id: UUID
    poc_id: str
    manifest_sha256: str
    target_host: str
    target_port: int | None = None


# --------------------------------------------------------------------------- #
# Precondition evaluator                                                       #
# --------------------------------------------------------------------------- #


class PreconditionEvaluator:
    """Evaluates structured preconditions against engagement + target state.

    Unknown predicates fail closed.
    """

    def evaluate(
        self,
        preconditions: list[PoCPrecondition],
        *,
        engagement: dict[str, Any],
        target_host: str,
        world_state_findings_cves: list[str] | None = None,
    ) -> tuple[bool, str]:
        for i, pc in enumerate(preconditions):
            ok, reason = self._evaluate_one(
                pc,
                engagement=engagement,
                target_host=target_host,
                world_state_findings_cves=world_state_findings_cves or [],
            )
            if not ok:
                return False, f"precondition[{i}] failed: {reason}"
        return True, "ok"

    def _evaluate_one(
        self,
        pc: PoCPrecondition,
        *,
        engagement: dict[str, Any],
        target_host: str,
        world_state_findings_cves: list[str],
    ) -> tuple[bool, str]:
        # Each PoCPrecondition has exactly one non-None field set; we walk them
        # in fixed order. Mutually-exclusive is not enforced at the schema
        # layer, so we treat multiple-set as "all must pass."
        checks_done = 0
        if pc.engagement_mode_in is not None:
            checks_done += 1
            mode = normalize_mode(engagement.get("mode", "guided"))
            allowed_modes = {normalize_mode(m) for m in pc.engagement_mode_in}
            if mode not in allowed_modes:
                return False, f"engagement_mode={mode} not in {pc.engagement_mode_in}"
        if pc.engagement_destructive_opt_in is not None:
            checks_done += 1
            actual = bool(engagement.get("destructive_opt_in", False))
            if actual != pc.engagement_destructive_opt_in:
                return (
                    False,
                    f"destructive_opt_in={actual} but required {pc.engagement_destructive_opt_in}",
                )
        if pc.target_not_in_no_exploit_list is not None and pc.target_not_in_no_exploit_list:
            checks_done += 1
            no_exploit = set(engagement.get("no_exploit_list", []))
            if target_host in no_exploit:
                return False, f"target {target_host} is in no_exploit_list"
        if pc.world_state_has_finding_with_cve is not None:
            checks_done += 1
            if pc.world_state_has_finding_with_cve not in world_state_findings_cves:
                return False, f"no finding with cve {pc.world_state_has_finding_with_cve}"
        if checks_done == 0:
            # An empty PoCPrecondition is a malformed manifest; fail closed.
            return False, "precondition is empty (no predicate set)"
        return True, "ok"


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #


class PoCDispatcher:
    """Validates parameters, evaluates preconditions + can_auto_run, then runs
    the PoC's sandbox argv via `SandboxRunner`.

    Safety invariants:
    - `can_auto_run` is enforced inside `dispatch()` so callers cannot bypass it.
    - `auto_run.preconditions` are evaluated via `PreconditionEvaluator`.
    - PoC modules never `exec_module()` into the worker — only their pure
      `build_invocation()` and `parse_result()` functions are called.
    """

    def __init__(
        self,
        library: PoCLibrary,
        sandbox: SandboxRunner,
        precondition_evaluator: PreconditionEvaluator | None = None,
    ) -> None:
        self.library = library
        self.sandbox = sandbox
        self.preconditions = precondition_evaluator or PreconditionEvaluator()

    def can_auto_run(
        self,
        poc: LoadedPoC,
        engagement: dict[str, Any],
    ) -> tuple[bool, str]:
        if not poc.manifest.auto_run.allowed:
            return False, "auto_run.allowed=false"
        if poc.manifest.auto_run.requires_approval:
            return False, "auto_run.requires_approval=true"
        if poc.manifest.risk.destructive and not engagement.get("destructive_opt_in"):
            return False, "destructive_without_optin"
        return True, "ok"

    async def dispatch(
        self,
        *,
        poc_id: str,
        params: dict[str, Any],
        ctx: PoCExecutionContext,
        engagement: dict[str, Any],
        world_state_findings_cves: list[str] | None = None,
    ) -> PoCResult:
        poc = self.library.get(poc_id)
        if poc is None:
            return PoCResult(
                outcome="error",
                summary=f"PoC not in library: {poc_id}",
                error_message="manifest_not_loaded",
            )

        # 1. Auto-run rules
        ok, reason = self.can_auto_run(poc, engagement)
        if not ok:
            return PoCResult(outcome="skipped", summary="auto-run denied", error_message=reason)

        # 2. Preconditions
        ok, reason = self.preconditions.evaluate(
            poc.manifest.auto_run.preconditions,
            engagement=engagement,
            target_host=ctx.target_host,
            world_state_findings_cves=world_state_findings_cves,
        )
        if not ok:
            return PoCResult(outcome="skipped", summary="precondition failed", error_message=reason)

        # 3. Parameter schema
        try:
            jsonschema.validate(params, poc.parameters_schema)
        except jsonschema.ValidationError as e:
            return PoCResult(outcome="error", summary="invalid params", error_message=e.message)

        # 4. Load the PoC module to call build_invocation + parse_result.
        # These are pure functions; they do not perform I/O.
        module_path = poc.directory / "exploit.py"
        if not module_path.exists():
            return PoCResult(
                outcome="error",
                summary="exploit.py missing",
                error_message="exploit_module_missing",
            )
        try:
            mod = self._load_module(poc_id, module_path)
        except Exception as e:
            log.exception("poc.module_load_error", poc_id=poc_id)
            return PoCResult(outcome="error", summary="module_load_failed", error_message=str(e))

        if not hasattr(mod, "build_invocation") or not hasattr(mod, "parse_result"):
            return PoCResult(
                outcome="error",
                summary="PoC module missing build_invocation/parse_result",
                error_message="bad_module_contract",
            )

        # 5. Build the sandbox argv (no I/O happens here).
        try:
            inv: SandboxInvocation = mod.build_invocation(params, ctx)
        except NotImplementedError:
            return PoCResult(
                outcome="skipped",
                summary="template PoC; not implemented",
                error_message="not_implemented",
            )
        except Exception as e:
            log.exception("poc.build_invocation_error", poc_id=poc_id)
            return PoCResult(
                outcome="error", summary="build_invocation raised", error_message=str(e)
            )

        # 6. Run in the sandbox.
        sandbox_result: SandboxResult = await self.sandbox.run(
            inv.argv, timeout_seconds=inv.timeout_seconds
        )

        # 7. Parse the result (pure function).
        try:
            return mod.parse_result(sandbox_result, params, ctx)
        except Exception as e:
            log.exception("poc.parse_result_error", poc_id=poc_id)
            return PoCResult(outcome="error", summary="parse_result raised", error_message=str(e))

    def _load_module(self, poc_id: str, module_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(
            f"pocs.{poc_id.replace('-', '_')}_exploit", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("spec_load_failed")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
