"""R3 #F.5: Per-tool ``summarize_for_model`` overrides.

Phase F.5 replaced the per-tool JSON dump fed back into the multi-turn
agent loop with a deterministic per-tool summary of interesting output.
These tests lock in:

* every concrete Tool subclass overrides ``summarize_for_model`` (the
  default impl is too coarse for the planner-visible context);
* each override returns a string <=``TOOL_SUMMARY_MAX_CHARS`` (500)
  that mentions ``tool_run_id`` and does NOT carry the entire output
  JSON dump.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

import tools as tools_pkg
from tools.base import TOOL_SUMMARY_MAX_CHARS, Tool


# --------------------------------------------------------------------------- #
# Subclass discovery                                                          #
# --------------------------------------------------------------------------- #


def _walk_tool_subclasses() -> list[type[Tool[Any, Any]]]:
    """Import-walk every ``tools.*`` module and return concrete Tool
    subclasses. Uses module discovery (NOT `Tool.__subclasses__`) so a
    not-yet-imported tool can't silently bypass the override test."""
    discovered: dict[str, type[Tool[Any, Any]]] = {}
    for mod_info in pkgutil.iter_modules(tools_pkg.__path__):
        if mod_info.name.startswith("_") or mod_info.name == "base":
            continue
        mod = importlib.import_module(f"tools.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, Tool) or obj is Tool:
                continue
            if inspect.isabstract(obj):
                continue
            # Must declare the class-level `name` (concrete tool).
            if not getattr(obj, "name", None):
                continue
            discovered[obj.name] = obj
    return list(discovered.values())


ALL_TOOL_CLASSES = _walk_tool_subclasses()


def test_discovery_picks_up_known_tools() -> None:
    """Smoke check: the walker found the canonical Phase F tools.
    Belt-and-braces in case ``tools.base.build_default_registry`` and
    the parametrized discovery walk drift apart."""
    names = {c.name for c in ALL_TOOL_CLASSES}
    # A representative subset that must always be present.
    for required in {
        "nmap_host_discovery",
        "nmap_service_scan",
        "httpx_probe",
        "nuclei_scan",
        "ffuf_dir_brute",
        "searchsploit_lookup",
        "netexec",
        "impacket_kerberoast",
        "impacket_asreproast",
        "bloodhound_collect",
        "bloodhound_query",
        "linpeas",
        "curated_poc_exec",
        "metasploit_exploit",
        "operator_exec",
        "sharphound_via_wmi",
        "rubeus_via_wmi",
        "mimikatz_via_worker",
    }:
        assert required in names, f"tool walk missed {required!r}"


@pytest.mark.parametrize("tool_cls", ALL_TOOL_CLASSES, ids=lambda c: c.name)
def test_every_tool_overrides_summarize_for_model(tool_cls: type[Tool[Any, Any]]) -> None:
    """Force the author of a new tool to write a per-tool summary.

    R3 #F.5: the default ``Tool.summarize_for_model`` enumerates list
    field lengths — fine for sanity, useless for the planner. Every
    concrete tool MUST override so the next-turn context carries real
    signal. This test is the structural enforcement of that contract."""
    assert tool_cls.summarize_for_model is not Tool.summarize_for_model, (
        f"{tool_cls.__name__}.summarize_for_model must be overridden — "
        f"the default impl reports only field-list lengths and is too "
        f"coarse for the multi-turn planner."
    )


# --------------------------------------------------------------------------- #
# Per-tool representative-output helpers                                       #
# --------------------------------------------------------------------------- #


def _mk_nmap_output(field_class: type[BaseModel]) -> BaseModel:
    from tools.nmap_ import NmapHost, NmapOutput, NmapPort

    host = NmapHost(
        ip="10.0.0.5",
        state="up",
        hostnames=["acme.local"],
        os_family="linux",
        os_detail="Ubuntu 22.04",
        ports=[
            NmapPort(port=22, proto="tcp", state="open", name="ssh", product="OpenSSH", version="8.2"),
            NmapPort(port=443, proto="tcp", state="open", name="https", product="nginx", version="1.18"),
        ],
    )
    return NmapOutput(hosts=[host], command="nmap ...")


def _mk_httpx_output() -> BaseModel:
    from tools.httpx_ import HttpxOutput, HttpxResult

    return HttpxOutput(
        probes=[
            HttpxResult(
                url="http://10.0.0.5/",
                host="10.0.0.5",
                port=80,
                scheme="http",
                status_code=200,
                title="Acme",
                server="nginx/1.18",
                technologies=["WordPress", "PHP"],
            ),
            HttpxResult(
                url="https://10.0.0.5/admin",
                host="10.0.0.5",
                port=443,
                scheme="https",
                status_code=401,
                server="nginx",
            ),
        ],
        command="httpx ...",
    )


def _mk_nuclei_output() -> BaseModel:
    from tools.nuclei_ import NucleiFinding, NucleiOutput

    return NucleiOutput(
        findings=[
            NucleiFinding(
                template_id="CVE-2024-1234",
                name="nginx RCE",
                severity="high",
                matched_url="http://10.0.0.5/",
            ),
            NucleiFinding(
                template_id="exposed-panel",
                name="wp-admin panel exposed",
                severity="medium",
                matched_url="http://10.0.0.5/wp-admin",
            ),
            NucleiFinding(
                template_id="tech-detect:wp",
                name="WordPress detected",
                severity="info",
                matched_url="http://10.0.0.5/",
            ),
        ],
        command="nuclei ...",
    )


def _mk_ffuf_output() -> BaseModel:
    from tools.ffuf_ import FfufHit, FfufOutput

    return FfufOutput(
        hits=[
            FfufHit(url="http://x/admin", status=200, length=100, words=5, lines=5),
            FfufHit(url="http://x/api/v1/users", status=200, length=200, words=10, lines=10),
            FfufHit(url="http://x/backup.zip", status=200, length=5000, words=0, lines=0),
            FfufHit(url="http://x/.git/config", status=403, length=0, words=0, lines=0),
        ],
        command="ffuf ...",
    )


def _mk_searchsploit_output() -> BaseModel:
    from tools.searchsploit_ import SearchsploitCandidate, SearchsploitOutput

    return SearchsploitOutput(
        candidates=[
            SearchsploitCandidate(
                edb_id=1234,
                title="nginx 1.18 RCE",
                date_published=date(2024, 1, 1),
                verified=True,
            ),
            SearchsploitCandidate(
                edb_id=9999,
                title="nginx path traversal",
                date_published=date(2023, 6, 1),
            ),
        ],
        command="searchsploit ...",
    )


def _mk_netexec_output() -> BaseModel:
    from tools.netexec_ import NetExecAuthRow, NetExecOutput

    return NetExecOutput(
        rows=[
            NetExecAuthRow(
                target="10.0.0.5",
                protocol="smb",
                status="success",
                admin=True,
                domain="ACME",
                username="svc-backup",
                raw="...",
            ),
            NetExecAuthRow(
                target="10.0.0.6",
                protocol="smb",
                status="success",
                admin=False,
                username="guest",
                raw="...",
            ),
            NetExecAuthRow(
                target="10.0.0.7",
                protocol="smb",
                status="failure",
                username="guest",
                raw="...",
            ),
        ],
        command="netexec ...",
        action="list_shares",
    )


def _mk_kerberoast_output() -> BaseModel:
    from tools.impacket_ import KerberoastOutput, KerberoastTicket

    return KerberoastOutput(
        tickets=[
            KerberoastTicket(
                spn="HTTP/web.acme.local",
                sam_name="svc-web",
                hash_format="krb5tgs",
                truncated_hash_preview="$krb5tgs$23$*svc-web$ACME$HTTP/web.ac",
            ),
            KerberoastTicket(
                spn="MSSQLSvc/db.acme.local:1433",
                sam_name="svc-sql",
                hash_format="krb5tgs",
                truncated_hash_preview="$krb5tgs$23$*svc-sql$ACME$MSSQLSvc/db",
            ),
        ],
        command="GetUserSPNs ...",
    )


def _mk_asrep_output() -> BaseModel:
    from tools.impacket_ import ASRepHash, ASRepRoastOutput

    return ASRepRoastOutput(
        hashes=[
            ASRepHash(sam_name="alice", hash_format="krb5asrep", truncated_hash_preview="...."),
            ASRepHash(sam_name="bob", hash_format="krb5asrep", truncated_hash_preview="...."),
        ],
        command="GetNPUsers ...",
    )


def _mk_bloodhound_collect_output() -> BaseModel:
    from tools.bloodhound_ import BloodHoundCollectOutput

    return BloodHoundCollectOutput(
        zip_blob_id=str(uuid4()),
        files_collected=["users.json", "groups.json", "computers.json"],
        command="bloodhound-python ...",
    )


def _mk_bloodhound_query_output() -> BaseModel:
    from tools.bloodhound_ import BloodHoundQueryOutput

    return BloodHoundQueryOutput(
        rows=[
            {"u.name": "svc-backup", "path_length": 2, "target": "DOMAIN ADMINS"},
            {"u.name": "alice", "path_length": 3, "target": "DOMAIN ADMINS"},
        ],
        row_count=2,
    )


def _mk_linpeas_output() -> BaseModel:
    from tools.linpeas_ import LinPEASFinding, LinPEASOutput

    return LinPEASOutput(
        findings=[
            LinPEASFinding(
                section="suid",
                severity="red",
                detail="/usr/local/bin/backup setuid root",
                raw_excerpt="...",
            ),
            LinPEASFinding(
                section="kernel",
                severity="yellow",
                detail="kernel 5.10",
                raw_excerpt="...",
            ),
        ],
        raw_blob_id=str(uuid4()),
        sections_scanned=["suid", "sudo", "kernel", "cron"],
        exit_code=0,
    )


def _mk_poc_output() -> BaseModel:
    from tools.poc_exec import CuratedPoCExecOutput

    return CuratedPoCExecOutput(
        outcome="success",
        summary="nginx exploit landed; got reverse shell",
        artifacts={"stdout_blob": str(uuid4()), "screenshot": str(uuid4())},
        poc_id="nginx-rce-cve-2024-1234",
        manifest_sha256="a" * 64,
        candidate_id=str(uuid4()),
    )


def _mk_edb_output() -> BaseModel:
    from tools.edb_exec import EdbExecOutput

    return EdbExecOutput(
        outcome="success",
        summary="ExploitDB script landed; got command execution",
        edb_id=51234,
        edb_title="Apache 2.4.49 - Path Traversal & RCE",
        edb_type="webapps",
        exit_code=0,
        stdout_excerpt="uid=33(www-data) gid=33(www-data)",
        candidate_id=str(uuid4()),
    )


def _mk_msf_output() -> BaseModel:
    from tools.msfrpc_ import MetasploitExploitOutput

    return MetasploitExploitOutput(
        outcome="success",
        module_path="exploit/multi/http/log4shell_header_injection",
        session_id_in_msf=1,
        job_id=42,
        summary="msf module executed; session 1 opened",
    )


def _mk_operator_exec_output() -> BaseModel:
    from tools.operator_exec import OperatorExecOutput

    return OperatorExecOutput(
        exit_code=0,
        stdout="line1\nline2\n",
        stderr="",
        duration_seconds=1.23,
        truncated=False,
        target="10.0.0.5",
        justification="manual debugging of svc-backup PoC",
        bypass_critic=False,
    )


def _mk_sharphound_via_wmi_output() -> BaseModel:
    from tools.windows_via_wmi import SharpHoundViaWmiOutput

    return SharpHoundViaWmiOutput(
        exit_code=0,
        zip_blob_id=str(uuid4()),
        summary="sharphound via wmiexec on 10.0.0.5 (collection=Default); exit 0",
    )


def _mk_rubeus_via_wmi_output() -> BaseModel:
    from tools.windows_via_wmi import RubeusViaWmiOutput

    return RubeusViaWmiOutput(
        exit_code=0,
        stdout_excerpt="some rubeus output ...",
        summary="rubeus via wmiexec on 10.0.0.5: args='kerberoast /nowrap'; exit 0",
    )


def _mk_mimikatz_via_worker_output() -> BaseModel:
    from tools.mimikatz_via_worker import (
        MimikatzCredentialRow,
        MimikatzViaWorkerOutput,
    )

    return MimikatzViaWorkerOutput(
        exit_code=0,
        stdout_excerpt="...",
        stderr_excerpt="",
        summary="mimikatz ran; 2 credentials extracted",
        rows=[
            MimikatzCredentialRow(
                username="svc-backup",
                domain="ACME",
                password="P@ssw0rd!",
            ),
            MimikatzCredentialRow(
                username="alice",
                domain="ACME",
                ntlm="a" * 32,
            ),
        ],
    )


# Map tool.name -> factory that returns a representative Output.
_OUTPUT_FACTORIES: dict[str, Any] = {
    "nmap_host_discovery": lambda: _mk_nmap_output(object),
    "nmap_service_scan": lambda: _mk_nmap_output(object),
    "httpx_probe": _mk_httpx_output,
    "nuclei_scan": _mk_nuclei_output,
    "ffuf_dir_brute": _mk_ffuf_output,
    "searchsploit_lookup": _mk_searchsploit_output,
    "netexec": _mk_netexec_output,
    "impacket_kerberoast": _mk_kerberoast_output,
    "impacket_asreproast": _mk_asrep_output,
    "bloodhound_collect": _mk_bloodhound_collect_output,
    "bloodhound_query": _mk_bloodhound_query_output,
    "linpeas": _mk_linpeas_output,
    "curated_poc_exec": _mk_poc_output,
    "edb_exploit_exec": _mk_edb_output,
    "metasploit_exploit": _mk_msf_output,
    "operator_exec": _mk_operator_exec_output,
    "sharphound_via_wmi": _mk_sharphound_via_wmi_output,
    "rubeus_via_wmi": _mk_rubeus_via_wmi_output,
    "mimikatz_via_worker": _mk_mimikatz_via_worker_output,
}


@pytest.mark.parametrize("tool_cls", ALL_TOOL_CLASSES, ids=lambda c: c.name)
def test_summarize_for_model_contract(tool_cls: type[Tool[Any, Any]]) -> None:
    """For every concrete tool: build a representative Output instance,
    call ``summarize_for_model``, assert the result fits the contract
    (<=500 chars, mentions ``tool_run_id``, does NOT inline the raw
    JSON dump)."""
    factory = _OUTPUT_FACTORIES.get(tool_cls.name)
    if factory is None:
        pytest.fail(
            f"no representative-output factory for {tool_cls.name!r}; add one "
            "in tests/unit/test_tool_summarizers.py so the contract test covers "
            "this tool."
        )
    output = factory()
    tool = tool_cls()
    tool_run_id = uuid4()
    summary = tool.summarize_for_model(output, tool_run_id=tool_run_id, persisted=3)

    # Contract.
    assert isinstance(summary, str)
    assert summary, "summary must be non-empty"
    assert len(summary) <= TOOL_SUMMARY_MAX_CHARS, (
        f"{tool_cls.name} returned {len(summary)} chars (limit {TOOL_SUMMARY_MAX_CHARS})"
    )
    assert str(tool_run_id) in summary, (
        f"{tool_cls.name}: summary must include tool_run_id pointer"
    )
    # Must not be the raw model_dump_json (the regression Phase F.5 fixes).
    assert summary != output.model_dump_json()
    # And shouldn't contain a giant `{...}` literal that screams "I just dumped JSON".
    # Cheap heuristic: full JSON dumps contain key-value separators heavily.
    if hasattr(output, "model_dump_json"):
        full_json = output.model_dump_json()
        if len(full_json) > 200:
            # Long JSON; the summary must NOT be a copy of it.
            assert full_json not in summary
