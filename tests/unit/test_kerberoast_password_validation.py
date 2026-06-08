"""R3 #L2 — Finding 1: Kerberoast (and wmiexec) password must reject the
impacket auth-token separators ``@`` and ``:``, plus NUL/newline.

Impacket's ``GetUserSPNs.py`` parses the single argv token
``domain/user[:password]@dc-ip`` left-to-right on those three characters. A
password with an embedded ``@`` would be silently re-interpreted as a
hostname, sending credentials to an out-of-scope target. NUL/newline corrupt
the redacted-argv log line. Mirrors the BloodHound password validator with
the impacket-syntax additions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# --------------------------------------------------------------------------- #
# KerberoastInput                                                              #
# --------------------------------------------------------------------------- #


def test_kerberoast_password_rejects_at_sign():
    """``@`` in a password would mis-route auth to a different host."""
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            password="hunter2@evil-host",
        )


def test_kerberoast_password_rejects_colon():
    """``:`` in a password would split impacket's user:password token."""
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            password="hun:ter2",
        )


def test_kerberoast_password_rejects_nul():
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            password="hunter\x002",
        )


def test_kerberoast_password_rejects_newline():
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            password="hunter\n2",
        )


def test_kerberoast_password_rejects_carriage_return():
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            password="hunter\r2",
        )


def test_kerberoast_password_accepts_clean_password():
    """A password with no impacket separators should pass."""
    from tools.impacket_ import KerberoastInput

    inp = KerberoastInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice",
        password="Sufficient1ySalty!",
    )
    assert inp.password == "Sufficient1ySalty!"


def test_kerberoast_password_none_still_allowed():
    """Hash-only auth (``nt_hash`` set, no password) must continue to work."""
    from tools.impacket_ import KerberoastInput

    inp = KerberoastInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice",
        nt_hash="aad3b435b51404eeaad3b435b51404ee",
    )
    assert inp.password is None


# --------------------------------------------------------------------------- #
# SharpHoundViaWmiInput / RubeusViaWmiInput                                    #
# --------------------------------------------------------------------------- #


def test_sharphound_via_wmi_password_rejects_at_sign():
    from tools.windows_via_wmi import SharpHoundViaWmiInput

    with pytest.raises(ValidationError):
        SharpHoundViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="bad@pw",
        )


def test_rubeus_via_wmi_password_rejects_colon():
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="bad:pw",
            rubeus_args="kerberoast /nowrap",
        )


def test_rubeus_via_wmi_args_rejects_semicolon():
    """``rubeus_args`` is interpolated into ``cmd.exe /c ...`` — ``;`` would
    chain a second command."""
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="Passw0rd1",
            rubeus_args="; calc.exe",
        )


def test_rubeus_via_wmi_args_rejects_ampersand():
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="Passw0rd1",
            rubeus_args="kerberoast & whoami",
        )


def test_rubeus_via_wmi_args_rejects_dollar_paren():
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="Passw0rd1",
            rubeus_args="kerberoast $(id)",
        )


def test_rubeus_via_wmi_args_rejects_redirect():
    from tools.windows_via_wmi import RubeusViaWmiInput

    with pytest.raises(ValidationError):
        RubeusViaWmiInput(
            target="10.0.0.5",
            domain="lab.local",
            username="admin",
            password="Passw0rd1",
            rubeus_args="kerberoast > /tmp/out",
        )


def test_rubeus_via_wmi_args_accepts_canonical_rubeus_flags():
    """``/kerberoast /nowrap /domain:lab.local /user:alice /rc4:hex`` must
    all pass — Rubeus's documented flag syntax."""
    from tools.windows_via_wmi import RubeusViaWmiInput

    inp = RubeusViaWmiInput(
        target="10.0.0.5",
        domain="lab.local",
        username="admin",
        password="Passw0rd1",
        rubeus_args="kerberoast /nowrap /domain:lab.local /user:alice.svc",
    )
    assert inp.rubeus_args.startswith("kerberoast")
