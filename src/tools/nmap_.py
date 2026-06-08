"""nmap adapter.

Two tools: NmapHostDiscovery (ICMP/ARP/TCP-SYN sweeps) and NmapServiceScan
(TCP service+version detection with optional default scripts).

Parses nmap XML output via defusedxml — no python-libnmap.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal
from uuid import UUID

from defusedxml import ElementTree as ET
from pydantic import BaseModel, Field, field_validator


def _reject_leading_dash_targets(v: list[str]) -> list[str]:
    """A target value beginning with ``-`` would be parsed by the binary as a
    FLAG (e.g. ``-oN/work/out``, ``--script=...``), not a positional host —
    argument injection within the argv (sandbox-09). nmap has no reliable
    end-of-options ``--`` separator, so reject leading-dash targets outright."""
    for t in v:
        if t.startswith("-"):
            raise ValueError(
                f"target {t!r} must not start with '-' (would be parsed as an "
                "nmap flag, not a host)"
            )
    return v

from orchestrator.evidence import EvidenceStore
from sandbox.runner import SandboxRunner
from tools._helpers import ToolExecutor, iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage

# --------------------------------------------------------------------------- #
# Output models                                                               #
# --------------------------------------------------------------------------- #


class NmapPort(BaseModel):
    port: int
    proto: Literal["tcp", "udp"]
    state: str
    name: str | None = None
    product: str | None = None
    version: str | None = None
    extrainfo: str | None = None
    cpe: list[str] = Field(default_factory=list)


class NmapHost(BaseModel):
    ip: str
    state: str
    hostnames: list[str] = Field(default_factory=list)
    os_family: Literal["linux", "windows", "macos", "bsd", "network", "unknown"] = "unknown"
    os_detail: str | None = None
    mac: str | None = None
    ports: list[NmapPort] = Field(default_factory=list)


class NmapOutput(BaseModel):
    hosts: list[NmapHost] = Field(default_factory=list)
    raw_stdout_blob_id: str | None = None
    raw_stderr_blob_id: str | None = None
    command: str


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #


class NmapHostDiscoveryInput(BaseModel):
    targets: list[str] = Field(min_length=1)
    method: Literal["icmp", "tcp_syn", "arp"] = "icmp"

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v: list[str]) -> list[str]:
        return _reject_leading_dash_targets(v)


class NmapServiceScanInput(BaseModel):
    targets: list[str] = Field(min_length=1)
    ports: str = Field(
        default="top1000",
        description='"top1000" | "top100" | "<n>" | "<a-b>" | "<comma-list>"',
    )
    scripts: Literal["default", "vuln", "none"] = "default"
    timing: int = Field(default=3, ge=0, le=5)
    udp: bool = False

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v: list[str]) -> list[str]:
        return _reject_leading_dash_targets(v)

    @field_validator("ports")
    @classmethod
    def _check_ports(cls, v: str) -> str:
        # `ports` flows into `-p <spec>`; a leading dash or metachar could act
        # as a flag. Restrict to the documented port-spec alphabet.
        spec = v.strip()
        if spec in ("top1000", "top100"):
            return spec
        if not re.match(r"^[0-9,\-]+$", spec):
            raise ValueError(
                "ports must be 'top1000', 'top100', or a digit/comma/dash "
                "port spec (e.g. '80,443' or '1-1024')"
            )
        return spec


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #


def _build_executor() -> ToolExecutor:
    # Lazy: settings/evidence wired by orchestrator at runtime; tests can
    # subclass and inject. For now, construct on first use with the worker's
    # process-wide singletons.
    from orchestrator.settings import get_settings

    settings = get_settings()
    try:
        evidence = EvidenceStore(settings)
    except Exception:
        evidence = None  # type: ignore[assignment]
    return ToolExecutor(SandboxRunner(), evidence)


class NmapHostDiscovery(Tool[NmapHostDiscoveryInput, NmapOutput]):
    name = "nmap_host_discovery"
    description = "Sweep targets for live hosts via ICMP, TCP-SYN, or ARP."
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input: ClassVar[type[BaseModel]] = NmapHostDiscoveryInput
    Output: ClassVar[type[BaseModel]] = NmapOutput

    def extract_targets(self, input: NmapHostDiscoveryInput) -> list[str]:
        return list(input.targets)

    def summarize_for_model(
        self, output: NmapOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        return _nmap_summary(self.name, output, tool_run_id=tool_run_id, persisted=persisted)

    async def run(
        self, input: NmapHostDiscoveryInput, ctx: ToolRunContext
    ) -> ToolResult[NmapOutput]:
        flags: list[str] = ["-sn", "-oX", "-"]
        if input.method == "tcp_syn":
            flags = ["-sn", "-PS22,80,443,3389,445", "-oX", "-"]
        elif input.method == "arp":
            flags = ["-sn", "-PR", "-oX", "-"]
        argv = ["nmap", *flags, *input.targets]
        executor = _build_executor()
        result, stdout_id, stderr_id = await executor.run_and_capture(
            argv, ctx=ctx, tool_name=self.name
        )
        hosts = _parse_nmap_xml(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=NmapOutput(
                hosts=hosts,
                raw_stdout_blob_id=stdout_id,
                raw_stderr_blob_id=stderr_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            raw_stderr_blob_id=UUID(stderr_id) if stderr_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


class NmapServiceScan(Tool[NmapServiceScanInput, NmapOutput]):
    name = "nmap_service_scan"
    description = (
        "Service + version detection on TCP (and optional UDP) ports. "
        "Use after host discovery to enumerate open services."
    )
    stage = Stage.RECON
    risk_class = RiskClass.RECON
    Input: ClassVar[type[BaseModel]] = NmapServiceScanInput
    Output: ClassVar[type[BaseModel]] = NmapOutput

    def extract_targets(self, input: NmapServiceScanInput) -> list[str]:
        return list(input.targets)

    def summarize_for_model(
        self, output: NmapOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        return _nmap_summary(self.name, output, tool_run_id=tool_run_id, persisted=persisted)

    async def run(self, input: NmapServiceScanInput, ctx: ToolRunContext) -> ToolResult[NmapOutput]:
        flags: list[str] = ["-sV", f"-T{input.timing}", "-Pn"]
        if input.udp:
            flags = ["-sUV", f"-T{input.timing}", "-Pn"]
        if input.scripts == "default":
            flags.append("-sC")
        elif input.scripts == "vuln":
            flags += ["--script", "vuln"]
        flags += _port_flag(input.ports)
        flags += ["-oX", "-"]
        argv = ["nmap", *flags, *input.targets]
        executor = _build_executor()
        result, stdout_id, stderr_id = await executor.run_and_capture(
            argv, ctx=ctx, tool_name=self.name, timeout_seconds=900
        )
        hosts = _parse_nmap_xml(result.stdout.decode("utf-8", errors="replace"))
        return ToolResult(
            tool_run_id=ctx.tool_run_id,
            name=self.name,
            output=NmapOutput(
                hosts=hosts,
                raw_stdout_blob_id=stdout_id,
                raw_stderr_blob_id=stderr_id,
                command=" ".join(argv),
            ),
            raw_stdout_blob_id=UUID(stdout_id) if stdout_id else None,
            raw_stderr_blob_id=UUID(stderr_id) if stderr_id else None,
            exit_code=result.exit_code,
            started_at=iso(result.started_at),
            ended_at=iso(result.ended_at),
        )


def _port_flag(spec: str) -> list[str]:
    spec = spec.strip()
    if spec == "top1000":
        return ["--top-ports", "1000"]
    if spec == "top100":
        return ["--top-ports", "100"]
    return ["-p", spec]


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #


def _parse_nmap_xml(xml_text: str) -> list[NmapHost]:
    """Parse nmap XML output into typed NmapHost list. Tolerant of partial output."""
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[NmapHost] = []
    for h in root.findall("host"):
        status = h.find("status")
        state = status.get("state", "unknown") if status is not None else "unknown"
        ip: str | None = None
        mac: str | None = None
        for addr in h.findall("address"):
            atype = addr.get("addrtype")
            if atype in ("ipv4", "ipv6") and ip is None:
                ip = addr.get("addr")
            if atype == "mac":
                mac = addr.get("addr")
        if not ip:
            continue
        hostnames = [hn.get("name", "") for hn in h.findall("hostnames/hostname") if hn.get("name")]
        os_family, os_detail = _parse_os(h)
        ports = _parse_ports(h)
        out.append(
            NmapHost(
                ip=ip,
                state=state,
                hostnames=hostnames,
                os_family=os_family,
                os_detail=os_detail,
                mac=mac,
                ports=ports,
            )
        )
    return out


def _parse_os(h: Any) -> tuple[Any, str | None]:
    os_elem = h.find("os/osmatch")
    if os_elem is None:
        return ("unknown", None)
    name = os_elem.get("name", "")
    lower = name.lower()
    family = "unknown"
    if "windows" in lower:
        family = "windows"
    elif "linux" in lower:
        family = "linux"
    elif "mac" in lower or "darwin" in lower:
        family = "macos"
    elif "bsd" in lower:
        family = "bsd"
    elif "ios" in lower or "router" in lower or "switch" in lower:
        family = "network"
    return (family, name or None)


def _nmap_summary(
    tool_name: str, output: NmapOutput, *, tool_run_id: UUID, persisted: int
) -> str:
    """Compact summary of an nmap scan: hosts up, total open ports, and
    up to 3 most-interesting services (named, with product/version)."""
    hosts_up = sum(1 for h in output.hosts if h.state == "up")
    open_ports: list[tuple[str, NmapPort]] = []
    for h in output.hosts:
        for p in h.ports:
            if p.state == "open":
                open_ports.append((h.ip, p))
    # Rank ports by "interestingness": named, has product/version > everything else.
    def _score(item: tuple[str, NmapPort]) -> int:
        _, p = item
        s = 0
        if p.product:
            s += 4
        if p.version:
            s += 2
        if p.name and p.name not in ("unknown", ""):
            s += 1
        return s

    top = sorted(open_ports, key=_score, reverse=True)[:3]
    top_str = ", ".join(
        f"{p.port}/{p.proto} {p.name or '?'} {p.product or ''} {p.version or ''}".strip()
        for _, p in top
    ) or "none"
    summary = (
        f"{tool_name}: {len(output.hosts)} hosts ({hosts_up} up), "
        f"{len(open_ports)} open ports. top: {top_str}. "
        f"persisted={persisted}. tool_run_id={tool_run_id}"
    )
    if len(summary) > 500:
        summary = summary[:485] + "... [truncated]"
    return summary


def _parse_ports(h: Any) -> list[NmapPort]:
    out: list[NmapPort] = []
    ports_elem = h.find("ports")
    if ports_elem is None:
        return out
    for p in ports_elem.findall("port"):
        try:
            port_num = int(p.get("portid", "0"))
        except ValueError:
            continue
        proto = p.get("protocol", "tcp")
        if proto not in ("tcp", "udp"):
            continue
        state_elem = p.find("state")
        state = state_elem.get("state", "unknown") if state_elem is not None else "unknown"
        svc = p.find("service")
        name = svc.get("name") if svc is not None else None
        product = svc.get("product") if svc is not None else None
        version = svc.get("version") if svc is not None else None
        extrainfo = svc.get("extrainfo") if svc is not None else None
        cpes = [c.text for c in (svc.findall("cpe") if svc is not None else []) if c.text]
        out.append(
            NmapPort(
                port=port_num,
                proto=proto,  # type: ignore[arg-type]
                state=state,
                name=name,
                product=product,
                version=version,
                extrainfo=extrainfo,
                cpe=cpes,
            )
        )
    return out
