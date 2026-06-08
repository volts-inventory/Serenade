"""R3 #F.5-audit: secret-redaction regression tests for per-tool summaries.

Phase F.5 replaced JSON dumps of tool output with per-tool
``summarize_for_model`` strings fed back into the planner each turn. The
audit pass that landed alongside that work scrubbed several leaks where
the summary was inlining operator-controlled or target-controlled bytes
(stdout of operator_exec, PoC author summaries, mimikatz `args` reflection,
linpeas `[+] detail` lines, Rubeus stdout, kerberos tickets, etc.).

This file locks in the no-leak guarantee for every high-risk tool by
stuffing UNIQUE SENTINEL STRINGS into every plausible secret-bearing
field of the Output schema, calling ``summarize_for_model``, and
asserting NONE of those sentinels appear in the summary string. A
failing sentinel name points directly at the leaking field.

If a new tool is added that handles credential material, add a case
here BEFORE the tool ships — it's much cheaper than catching the leak
in production traffic.
"""

from __future__ import annotations

from uuid import uuid4

import pytest


# --------------------------------------------------------------------------- #
# Sentinel registry                                                           #
# --------------------------------------------------------------------------- #
#
# Each sentinel is unique per (tool, field) pair so a failing assertion
# message identifies the leaking field by name. Using mixed credential
# classes (plaintext, ntlm, krb5tgs, ssh privkey, bearer token, AWS key)
# so a category regex hit also identifies the leak class.

PWD = "SECRET_PWD_CAFEBABE_{}"
NTLM = "SECRET_NTLM_DEADBEEF_{}"
KRB5TGS = "$krb5tgs$23$*SECRET_KRBTGS_FEEDFACE_{}*"
KRB5ASREP = "$krb5asrep$23$SECRET_ASREP_BAADF00D_{}@REALM:"
SSH_PRIV = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "SECRET_SSHKEY_C0FFEE42_{}\n"
    "-----END OPENSSH PRIVATE KEY-----"
)
BEARER = "Bearer SECRET_BEARER_BABE5AFE_{}"
AWS_KEY = "AKIASECRETAWS00{}DEADBEEF"
COOKIE = "session=SECRET_COOKIE_8BADF00D_{}"


def _all_sentinels(values: list[str]) -> list[str]:
    """Flatten and return every sentinel actually planted into an Output."""
    return [v for v in values if v]


# --------------------------------------------------------------------------- #
# impacket_kerberoast                                                          #
# --------------------------------------------------------------------------- #


def test_impacket_kerberoast_summary_redacts_hash_preview() -> None:
    from tools.impacket_ import (
        ImpacketKerberoast,
        KerberoastOutput,
        KerberoastTicket,
    )

    planted = [
        KRB5TGS.format("ticket_preview_a"),
        KRB5TGS.format("ticket_preview_b"),
    ]
    out = KerberoastOutput(
        tickets=[
            KerberoastTicket(
                spn="HTTP/web.acme.local",
                sam_name="svc-web",
                hash_format="krb5tgs",
                truncated_hash_preview=planted[0],
            ),
            KerberoastTicket(
                spn="MSSQLSvc/db.acme.local:1433",
                sam_name="svc-sql",
                hash_format="krb5tgs",
                truncated_hash_preview=planted[1],
            ),
        ],
        command="GetUserSPNs.py ...",
    )
    summary = ImpacketKerberoast().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=2
    )
    for s in _all_sentinels(planted):
        assert s not in summary, (
            f"impacket_kerberoast summary leaked sentinel {s!r} — "
            f"truncated_hash_preview MUST stay out of planner context "
            f"(it is the secret)."
        )
    # Headline signal still flows.
    assert "2" in summary
    assert "svc-web" in summary


# --------------------------------------------------------------------------- #
# impacket_asreproast                                                          #
# --------------------------------------------------------------------------- #


def test_impacket_asreproast_summary_redacts_hash_preview() -> None:
    from tools.impacket_ import (
        ASRepHash,
        ASRepRoastOutput,
        ImpacketASRepRoast,
    )

    planted = [
        KRB5ASREP.format("preview_alice"),
        KRB5ASREP.format("preview_bob"),
    ]
    out = ASRepRoastOutput(
        hashes=[
            ASRepHash(
                sam_name="alice",
                hash_format="krb5asrep",
                truncated_hash_preview=planted[0],
            ),
            ASRepHash(
                sam_name="bob",
                hash_format="krb5asrep",
                truncated_hash_preview=planted[1],
            ),
        ],
        command="GetNPUsers.py ...",
    )
    summary = ImpacketASRepRoast().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=2
    )
    for s in _all_sentinels(planted):
        assert s not in summary, (
            f"impacket_asreproast summary leaked sentinel {s!r} — "
            f"truncated_hash_preview MUST stay out of planner context."
        )


# --------------------------------------------------------------------------- #
# operator_exec                                                               #
# --------------------------------------------------------------------------- #


def test_operator_exec_summary_redacts_stdout_stderr() -> None:
    from tools.operator_exec import OperatorExec, OperatorExecOutput

    planted = {
        "stdout": PWD.format("operator_stdout"),
        "stderr": NTLM.format("operator_stderr"),
        "justification": "diagnose backup service " + COOKIE.format("operator_just"),
        "bypass_justification": BEARER.format("operator_bypassjust"),
    }
    out = OperatorExecOutput(
        exit_code=0,
        stdout=f"login: admin\npassword: {planted['stdout']}\n",
        stderr=f"warning: {planted['stderr']}",
        duration_seconds=1.0,
        truncated=False,
        target="10.0.0.5",
        justification=planted["justification"],
        bypass_critic=True,
        bypass_justification=planted["bypass_justification"],
    )
    summary = OperatorExec().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=1
    )
    for label, sentinel in planted.items():
        assert sentinel not in summary, (
            f"operator_exec leaked sentinel from {label!r} into planner "
            f"context (operator-supplied stdout/stderr/justification "
            f"routinely contains plaintext creds)."
        )


# --------------------------------------------------------------------------- #
# curated_poc_exec                                                            #
# --------------------------------------------------------------------------- #


def test_curated_poc_exec_summary_redacts_summary_and_error() -> None:
    from tools.poc_exec import CuratedPoCExec, CuratedPoCExecOutput

    planted = {
        "summary": "dumped /etc/shadow: root:" + PWD.format("poc_summary"),
        "error_message": "stack trace mentions " + NTLM.format("poc_error"),
    }
    out = CuratedPoCExecOutput(
        outcome="success",
        summary=planted["summary"],
        error_message=planted["error_message"],
        artifacts={"shadow_dump": "blob-id-1", "screenshot": "blob-id-2"},
        poc_id="cve-2024-1234",
        manifest_sha256="a" * 64,
        candidate_id=str(uuid4()),
    )
    summary = CuratedPoCExec().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=2
    )
    for label, sentinel in planted.items():
        assert sentinel not in summary, (
            f"curated_poc_exec leaked sentinel from {label!r}. "
            f"PoC summary/error_message are PoC-author-controlled and "
            f"credential-extraction PoCs commonly write extracted secrets "
            f"into these fields."
        )


# --------------------------------------------------------------------------- #
# metasploit_exploit                                                          #
# --------------------------------------------------------------------------- #


def test_metasploit_exploit_summary_redacts_error_message() -> None:
    from tools.msfrpc_ import MetasploitExploit, MetasploitExploitOutput

    planted_error = (
        "msfrpc auth failed: " + BEARER.format("msf_error") + " " + COOKIE.format("msf_cookie")
    )
    out = MetasploitExploitOutput(
        outcome="error",
        module_path="exploit/multi/http/log4shell_header_injection",
        session_id_in_msf=None,
        job_id=None,
        summary="msf module failed",
        error_message=planted_error,
    )
    summary = MetasploitExploit().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=0
    )
    for sentinel in (BEARER.format("msf_error"), COOKIE.format("msf_cookie")):
        assert sentinel not in summary, (
            f"metasploit_exploit leaked sentinel {sentinel!r} from "
            f"error_message — exception text can echo bearer tokens and "
            f"operator-supplied options."
        )


# --------------------------------------------------------------------------- #
# mimikatz_via_worker                                                         #
# --------------------------------------------------------------------------- #


def test_mimikatz_via_worker_summary_redacts_creds_and_args() -> None:
    from tools.mimikatz_via_worker import (
        MimikatzCredentialRow,
        MimikatzViaWorker,
        MimikatzViaWorkerOutput,
    )

    planted = {
        "row_password": PWD.format("mk_row_pwd"),
        "row_ntlm": "a" * 16 + "DEAD" + "0" * 12,  # 32-hex still, but unique tail
        # Real ntlm regex is /^[a-fA-F0-9]{32}$/ on input — Pydantic does
        # not validate the output schema's ntlm field shape, so we can
        # plant a recognizable sentinel directly.
        "row_ntlm_marker": NTLM.format("mk_row_ntlm"),
        "stdout_excerpt": PWD.format("mk_stdout") + " " + SSH_PRIV.format("mk_stdout_key"),
        "stderr_excerpt": NTLM.format("mk_stderr"),
        # `summary` is built in run() from `input.args!r`; an operator
        # passing /password:X or /ntlm:HASH would surface there. Plant a
        # synthetic one to lock that behavior down.
        "summary": "mimikatz args=['sekurlsa::pth', '/ntlm:" + NTLM.format("mk_args") + "']",
    }
    out = MimikatzViaWorkerOutput(
        exit_code=0,
        stdout_excerpt=planted["stdout_excerpt"],
        stderr_excerpt=planted["stderr_excerpt"],
        summary=planted["summary"],
        rows=[
            MimikatzCredentialRow(
                username="svc-backup",
                domain="ACME",
                password=planted["row_password"],
                ntlm=planted["row_ntlm_marker"],
            ),
        ],
    )
    summary = MimikatzViaWorker().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=1
    )
    for label, sentinel in planted.items():
        assert sentinel not in summary, (
            f"mimikatz_via_worker leaked sentinel from {label!r}. The "
            f"summarizer must surface only counts + usernames; the "
            f"password/ntlm/stdout fields are the secrets themselves."
        )


# --------------------------------------------------------------------------- #
# linpeas                                                                     #
# --------------------------------------------------------------------------- #


def test_linpeas_summary_redacts_finding_detail_and_excerpt() -> None:
    from tools.linpeas_ import LinPEAS, LinPEASFinding, LinPEASOutput

    planted = {
        "detail_red": (
            "/etc/.htpasswd contents: admin:"
            + PWD.format("linpeas_detail_red")
        ),
        "raw_excerpt_red": SSH_PRIV.format("linpeas_excerpt_red"),
        "detail_yellow": "AWS key " + AWS_KEY.format("LP1"),
        "raw_excerpt_yellow": BEARER.format("linpeas_excerpt_yellow"),
    }
    out = LinPEASOutput(
        findings=[
            LinPEASFinding(
                section="suid",
                severity="red",
                detail=planted["detail_red"],
                raw_excerpt=planted["raw_excerpt_red"],
            ),
            LinPEASFinding(
                section="cron",
                severity="yellow",
                detail=planted["detail_yellow"],
                raw_excerpt=planted["raw_excerpt_yellow"],
            ),
        ],
        raw_blob_id=str(uuid4()),
        sections_scanned=["suid", "sudo", "kernel", "cron"],
        exit_code=0,
    )
    summary = LinPEAS().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=2
    )
    for label, sentinel in planted.items():
        assert sentinel not in summary, (
            f"linpeas leaked sentinel from {label!r}. linpeas `detail` "
            f"and `raw_excerpt` fields contain file contents (.htpasswd, "
            f"ssh keys, AWS creds in env files) and must never be "
            f"inlined into the planner-visible summary."
        )


# --------------------------------------------------------------------------- #
# rubeus_via_wmi                                                              #
# --------------------------------------------------------------------------- #


def test_rubeus_via_wmi_summary_redacts_stdout_and_args() -> None:
    from tools.windows_via_wmi import RubeusViaWmi, RubeusViaWmiOutput

    planted = {
        "stdout_excerpt": (
            "[*] Action: Kerberoasting\n"
            + KRB5TGS.format("rubeus_stdout")
            + "\n[+] ticket dumped"
        ),
        # Summary built in run() from input.rubeus_args!r — an operator
        # passing /password:X or /rc4:HASH or /ticket:base64... would leak.
        "summary": (
            "rubeus via wmiexec on 10.0.0.5: args='kerberoast /rc4:"
            + NTLM.format("rubeus_args")
            + "'; exit 0"
        ),
    }
    out = RubeusViaWmiOutput(
        exit_code=0,
        stdout_excerpt=planted["stdout_excerpt"],
        summary=planted["summary"],
    )
    summary = RubeusViaWmi().summarize_for_model(
        out, tool_run_id=uuid4(), persisted=0
    )
    for label, sentinel in planted.items():
        assert sentinel not in summary, (
            f"rubeus_via_wmi leaked sentinel from {label!r}. Rubeus "
            f"stdout contains kerberos tickets/hashes; summary reflects "
            f"operator-supplied args which may include passwords or "
            f"ticket material."
        )


# --------------------------------------------------------------------------- #
# Coverage smoke: every high-risk tool listed above has a redaction case      #
# --------------------------------------------------------------------------- #


_HIGH_RISK_TOOLS = {
    "impacket_kerberoast",
    "impacket_asreproast",
    "operator_exec",
    "curated_poc_exec",
    "metasploit_exploit",
    "mimikatz_via_worker",
    "linpeas",
    "rubeus_via_wmi",
}


@pytest.mark.parametrize("tool_name", sorted(_HIGH_RISK_TOOLS))
def test_high_risk_tool_has_redaction_case(tool_name: str) -> None:
    """Locks in that THIS file defines a redaction test for every
    high-risk tool. If a new credential-bearing tool is added to
    ``_HIGH_RISK_TOOLS`` the matching ``test_<tool>_summary_redacts_*``
    function must exist. Reduces the chance of a future tool slipping
    through without a sentinel test."""
    import inspect
    import sys

    module = sys.modules[__name__]
    fn_names = {
        name
        for name, _ in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    }
    matches = [
        n for n in fn_names if n.startswith(f"test_{tool_name}_summary_redacts")
    ]
    assert matches, (
        f"high-risk tool {tool_name!r} has no redaction test in "
        f"test_summarizer_secret_redaction.py — add one before shipping."
    )
