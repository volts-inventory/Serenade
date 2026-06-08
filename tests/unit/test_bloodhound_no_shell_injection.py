"""R3 #L1 — BloodHound + Impacket adapters must not shell-interpolate
operator-controlled fields, and msfrpc TLS verification must be configurable.

Three bugs from the polish-loop's security findings:

1. ``tools/bloodhound_.py:BloodHoundCollect`` previously built a ``/bin/sh -c``
   command by f-string interpolating ``input.username``, ``input.password``,
   ``input.domain``, ``input.dc_ip`` into the shell text. A username like
   ``a;rm -rf /`` would inject. Fix: validate at the Pydantic layer (FQDN/IP
   regex, no-shell-metas username) and pass operator fields via *env vars*
   referenced as ``"$VAR"`` in the wrapper — never interpolated into shell.

2. ``tools/impacket_.py:ImpacketASRepRoast`` had the same pattern on
   ``input.domain`` and ``input.dc_ip``. Same fix.

3. ``tools/msfrpc_.py`` hard-coded ``httpx.AsyncClient(verify=False)``. Fix:
   ``MSFRPC_CA_BUNDLE`` setting (PEM path) takes precedence; ``MSFRPC_TLS_VERIFY``
   bool secondary; default unverified-with-warning preserves dev UX.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# --------------------------------------------------------------------------- #
# Bug 1 — BloodHoundCollect: Pydantic rejects shell metacharacters             #
# --------------------------------------------------------------------------- #


def test_bloodhound_username_rejects_shell_metachars():
    """Operator username with ``;`` (or any shell meta) must 422 at Pydantic."""
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="a;rm -rf /",
        )


def test_bloodhound_username_rejects_backtick():
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="a`whoami`",
        )


def test_bloodhound_username_rejects_dollar_paren():
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="a$(id)",
        )


def test_bloodhound_username_accepts_typical_ad_samaccountname():
    """AD sAMAccountNames with dots, dashes, dollar signs (machine accounts)
    and ``@`` (UPN-style) must still pass."""
    from tools.bloodhound_ import BloodHoundCollectInput

    inp = BloodHoundCollectInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="svc.adsync-01$",
    )
    assert inp.username == "svc.adsync-01$"

    inp2 = BloodHoundCollectInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice@lab.local",
    )
    assert inp2.username == "alice@lab.local"


def test_bloodhound_domain_rejects_shell_metachars():
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local;cat /etc/passwd",
            dc_ip="10.0.0.5",
            username="alice",
        )


def test_bloodhound_dc_ip_rejects_non_ip():
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local",
            dc_ip="10.0.0.5;rm -rf /",
            username="alice",
        )


def test_bloodhound_dc_ip_accepts_ipv6():
    from tools.bloodhound_ import BloodHoundCollectInput

    inp = BloodHoundCollectInput(
        domain="lab.local",
        dc_ip="fe80::1",
        username="alice",
    )
    assert inp.dc_ip == "fe80::1"


def test_bloodhound_collection_method_rejects_metachars():
    from tools.bloodhound_ import BloodHoundCollectInput

    with pytest.raises(ValidationError):
        BloodHoundCollectInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="alice",
            collection_method="All;rm -rf /",
        )


# --------------------------------------------------------------------------- #
# Bug 1 — BloodHoundCollect runtime: operator fields land as env vars, never  #
# spliced into the shell text                                                  #
# --------------------------------------------------------------------------- #


class _CapturingSandbox:
    """SandboxRunner stand-in that captures argv + kwargs of each .run() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.engagement_id: str | None = None

    async def run(self, argv, **kwargs):
        from datetime import UTC, datetime

        from sandbox.runner import SandboxResult

        self.calls.append((list(argv), dict(kwargs)))
        return SandboxResult(
            exit_code=0,
            stdout=b"",
            stderr=b"",
            duration_seconds=0.01,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_bloodhound_collect_passes_argv_list_not_string(monkeypatch):
    """The executor must be called with a *list* argv; operator strings must
    appear ONLY in the ``env`` dict, never substituted into the shell text."""
    from uuid import uuid4

    from tools import bloodhound_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self):
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(bloodhound_, "_build_executor", lambda: _FakeExecutor())

    tool = bloodhound_.BloodHoundCollect()
    inp = bloodhound_.BloodHoundCollectInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice",
        password="hunter2",
    )
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.calls, "sandbox.run was not invoked"
    argv, kwargs = sandbox.calls[0]
    # argv is a list (not a single string)
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)
    # Operator-controlled fields must NEVER appear as substrings in the shell
    # command text. They flow through env vars only.
    shell_text = argv[-1]  # /bin/sh -c <wrapper>
    assert "alice" not in shell_text, (
        f"username leaked into shell text: {shell_text!r}"
    )
    assert "hunter2" not in shell_text, "password leaked into shell text"
    assert "lab.local" not in shell_text, "domain leaked into shell text"
    assert "10.0.0.5" not in shell_text, "dc_ip leaked into shell text"
    # The env kwarg carries them
    env = kwargs.get("env") or {}
    assert env.get("BH_USER") == "alice"
    assert env.get("BH_DOMAIN") == "lab.local"
    assert env.get("BH_DC_IP") == "10.0.0.5"
    assert env.get("BH_PASSWORD") == "hunter2"


@pytest.mark.asyncio
async def test_bloodhound_collect_redacted_argv_hides_password(monkeypatch):
    """``redacted_argv`` (the logged form) must not contain the password."""
    from uuid import uuid4

    from tools import bloodhound_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self):
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(bloodhound_, "_build_executor", lambda: _FakeExecutor())
    tool = bloodhound_.BloodHoundCollect()
    inp = bloodhound_.BloodHoundCollectInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice",
        password="hunter2",
    )
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    argv, kwargs = sandbox.calls[0]
    red = kwargs.get("redacted_argv")
    assert red is not None
    assert isinstance(red, list)
    red_text = " ".join(red)
    assert "hunter2" not in red_text
    assert "***REDACTED***" in red_text


# --------------------------------------------------------------------------- #
# Bug 1 (sweep) — ImpacketASRepRoast: same fix shape                          #
# --------------------------------------------------------------------------- #


def test_impacket_asreproast_domain_rejects_shell_metachars():
    from tools.impacket_ import ASRepRoastInput

    with pytest.raises(ValidationError):
        ASRepRoastInput(
            domain="lab.local;rm",
            dc_ip="10.0.0.5",
            users_file_content="alice\n",
        )


def test_impacket_asreproast_dc_ip_rejects_non_ip():
    from tools.impacket_ import ASRepRoastInput

    with pytest.raises(ValidationError):
        ASRepRoastInput(
            domain="lab.local",
            dc_ip="not.an.ip;evil",
            users_file_content="alice\n",
        )


@pytest.mark.asyncio
async def test_impacket_asreproast_no_operator_string_in_shell_text(monkeypatch):
    """Operator domain + dc_ip must flow via env, not as splices."""
    from uuid import uuid4

    from tools import impacket_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self):
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(impacket_, "_build_executor", lambda: _FakeExecutor())
    tool = impacket_.ImpacketASRepRoast()
    inp = impacket_.ASRepRoastInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        users_file_content="alice\nbob\n",
    )
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    argv, kwargs = sandbox.calls[0]
    assert isinstance(argv, list)
    shell_text = argv[-1]
    assert "lab.local" not in shell_text
    assert "10.0.0.5" not in shell_text
    env = kwargs.get("env") or {}
    assert env.get("ASREP_DOMAIN") == "lab.local"
    assert env.get("ASREP_DC_IP") == "10.0.0.5"


# --------------------------------------------------------------------------- #
# Bug 1 (sweep) — ImpacketKerberoast input validation                          #
# --------------------------------------------------------------------------- #


def test_impacket_kerberoast_username_rejects_shell_metachars():
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="lab.local",
            dc_ip="10.0.0.5",
            username="a;rm -rf /",
        )


def test_impacket_kerberoast_domain_rejects_non_fqdn():
    from tools.impacket_ import KerberoastInput

    with pytest.raises(ValidationError):
        KerberoastInput(
            domain="not a domain!",
            dc_ip="10.0.0.5",
            username="alice",
        )


# --------------------------------------------------------------------------- #
# Bug 2 — msfrpc TLS verification is configurable                             #
# --------------------------------------------------------------------------- #


def test_msfrpc_resolves_ca_bundle_path(monkeypatch):
    """When ``MSFRPC_CA_BUNDLE`` is set, the resolver returns the path."""
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    s = Settings(  # type: ignore[call-arg]
        msfrpc_ca_bundle="/etc/ssl/certs/ca.crt",
    )
    monkeypatch.setattr(msfrpc_, "get_settings", lambda: s, raising=False)
    # The resolver imports get_settings from orchestrator.settings — patch
    # that module's binding so it picks up our override.
    import orchestrator.settings as _settings

    monkeypatch.setattr(_settings, "get_settings", lambda: s)
    get_settings.cache_clear()
    try:
        v = msfrpc_._resolve_msfrpc_tls_verify()
    finally:
        get_settings.cache_clear()
    assert v == "/etc/ssl/certs/ca.crt"


def test_msfrpc_resolves_verify_true_when_set(monkeypatch):
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    s = Settings(  # type: ignore[call-arg]
        msfrpc_tls_verify=True,
    )
    import orchestrator.settings as _settings

    monkeypatch.setattr(_settings, "get_settings", lambda: s)
    get_settings.cache_clear()
    try:
        v = msfrpc_._resolve_msfrpc_tls_verify()
    finally:
        get_settings.cache_clear()
    assert v is True


def test_msfrpc_default_unverified_logs_warning(monkeypatch, capsys):
    """Default config (no CA bundle, verify off) returns False AND warns.

    structlog's default ConsoleRenderer writes to stdout, so we use capsys.
    """
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    s = Settings()  # type: ignore[call-arg]
    import orchestrator.settings as _settings

    monkeypatch.setattr(_settings, "get_settings", lambda: s)
    get_settings.cache_clear()
    try:
        v = msfrpc_._resolve_msfrpc_tls_verify()
    finally:
        get_settings.cache_clear()
    assert v is False
    captured = capsys.readouterr()
    log_text = captured.out + captured.err
    assert "tls_verify_disabled" in log_text or "TLS verification" in log_text


def test_msfrpc_settings_has_ca_bundle_field():
    """The ``Settings`` model exposes the new fields with the documented defaults."""
    from orchestrator.settings import Settings

    s = Settings()  # type: ignore[call-arg]
    assert s.msfrpc_tls_verify is False  # documented default
    assert s.msfrpc_ca_bundle is None  # documented default
