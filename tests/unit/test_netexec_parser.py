from __future__ import annotations

from tools.netexec_ import NetExec, NetExecInput, _parse

_SAMPLE_NETEXEC_OUTPUT = """\
SMB    10.10.10.5    445   DC01     [*] Windows Server 2019 (name:DC01) (domain:LAB.LOCAL)
SMB    10.10.10.5    445   DC01     [+] LAB.LOCAL\\alice:Password123! (Pwn3d!)
SMB    10.10.10.6    445   WS01     [+] LAB.LOCAL\\alice:Password123!
SMB    10.10.10.7    445   WS02     [-] LAB.LOCAL\\alice:Password123! STATUS_LOGON_FAILURE
"""


def test_parses_success_and_failure():
    rows = _parse(_SAMPLE_NETEXEC_OUTPUT, "smb")
    assert len(rows) == 3
    statuses = [(r.target, r.status, r.admin) for r in rows]
    assert ("10.10.10.5", "success", True) in statuses
    assert ("10.10.10.6", "success", False) in statuses
    assert ("10.10.10.7", "failure", False) in statuses


def test_skips_informational_lines():
    rows = _parse(_SAMPLE_NETEXEC_OUTPUT, "smb")
    # The [*] info line is not an auth result
    assert all(r.status in ("success", "failure") for r in rows)


def test_strips_ansi_color():
    raw = "SMB    10.0.0.5    445   DC01     \x1b[32m[+]\x1b[0m LAB\\u:p"
    rows = _parse(raw, "smb")
    assert len(rows) == 1
    assert rows[0].status == "success"


def test_extract_targets():
    t = NetExec()
    inp = NetExecInput(targets=["10.0.0.5", "10.0.0.6"], protocol="smb")
    assert t.extract_targets(inp) == ["10.0.0.5", "10.0.0.6"]


def test_input_rejects_bad_nt_hash():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NetExecInput(targets=["10.0.0.5"], nt_hash="not-a-hash")
