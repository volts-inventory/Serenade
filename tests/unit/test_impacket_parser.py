from __future__ import annotations

from tools.impacket_ import _parse_asrep, _parse_kerberoast

_KERBEROAST_OUTPUT = """\
Impacket v0.11.0 - Copyright 2023 Fortra
ServicePrincipalName  Name           MemberOf  PasswordLastSet      LastLogon
--------------------  -------------  --------  -------------------  --------
HTTP/web.lab.local    svc_web                  2024-01-15 12:00:00  -
$krb5tgs$23$*svc_web$LAB.LOCAL$HTTP/web.lab.local*$abc123def456...truncated
$krb5tgs$23$*svc_sql$LAB.LOCAL$MSSQLSvc/sql.lab.local:1433*$789xyz...truncated
"""


def test_parses_kerberoast_tickets():
    tickets = _parse_kerberoast(_KERBEROAST_OUTPUT)
    assert len(tickets) == 2
    svc_web = next(t for t in tickets if t.sam_name == "svc_web")
    assert svc_web.spn == "HTTP/web.lab.local"
    assert svc_web.hash_format == "krb5tgs"
    svc_sql = next(t for t in tickets if t.sam_name == "svc_sql")
    assert "MSSQLSvc" in svc_sql.spn


_ASREP_OUTPUT = """\
Impacket v0.11.0 - Copyright 2023 Fortra
[*] Querying lab.local for users...
$krb5asrep$23$alice@LAB.LOCAL:abc123...truncated_hash_body
$krb5asrep$23$bob@LAB.LOCAL:def456...truncated
"""


def test_parses_asrep_hashes():
    hashes = _parse_asrep(_ASREP_OUTPUT)
    assert len(hashes) == 2
    names = {h.sam_name for h in hashes}
    assert names == {"alice", "bob"}
    assert all(h.hash_format == "krb5asrep" for h in hashes)


def test_empty_input_safe():
    assert _parse_kerberoast("") == []
    assert _parse_asrep("") == []
    assert _parse_kerberoast("no hashes here") == []
