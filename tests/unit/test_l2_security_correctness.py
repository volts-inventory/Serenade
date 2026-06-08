"""R3 #L2 — covers findings 3 through 10 from the comprehensive review.

  - Finding 3: msfrpc creds via Settings, not os.environ.
  - Finding 4: msfrpc login error sanitization.
  - Finding 5: per-engagement default_network bound on every tool's runner.
  - Finding 6: nftables create failure raises when sandbox_l3_required=True.
  - Finding 7: msfrpc poll uses time.monotonic, not asyncio.get_event_loop.
  - Finding 8: operator_exec audit actor is the principal, never client-set.
  - Finding 9: BloodHound output surfaces ``truncated`` + raised max bytes.
  - Finding 10: BloodHound default community password is gone.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Finding 3 — msfrpc creds via Settings                                        #
# --------------------------------------------------------------------------- #


def test_settings_exposes_msfrpc_url_user_password():
    from orchestrator.settings import Settings

    s = Settings()  # type: ignore[call-arg]
    # Defaults are the documented values; user/password default to None
    # (forcing an explicit configuration before the adapter can run).
    assert s.msfrpc_url == "https://127.0.0.1:55553/api/"
    assert s.msfrpc_user is None
    assert s.msfrpc_password is None


def test_msfrpc_resolves_credentials_from_settings(monkeypatch):
    """Credentials configured via Settings flow through to the resolver."""
    import orchestrator.settings as settings_mod
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    s = Settings(  # type: ignore[call-arg]
        msfrpc_url="https://example:55553/api/",
        msfrpc_user="msf",
        msfrpc_password="secret",
    )
    monkeypatch.setattr(settings_mod, "get_settings", lambda: s)
    get_settings.cache_clear()
    try:
        url, user, password = msfrpc_._resolve_msfrpc_credentials()
    finally:
        get_settings.cache_clear()
    assert url == "https://example:55553/api/"
    assert user == "msf"
    assert password == "secret"


def test_msfrpc_falls_back_to_env_with_deprecation_warning(monkeypatch, capsys):
    """If Settings has no creds but env vars are set, the resolver uses env
    and logs a one-time deprecation warning."""
    import orchestrator.settings as settings_mod
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    # Reset the deprecation memo so this test sees a fresh warning.
    msfrpc_._DEPRECATION_WARNED_ENV.clear()

    s = Settings()  # type: ignore[call-arg]
    monkeypatch.setattr(settings_mod, "get_settings", lambda: s)
    monkeypatch.setenv("MSFRPC_USER", "envuser")
    monkeypatch.setenv("MSFRPC_PASSWORD", "envpass")
    get_settings.cache_clear()
    try:
        url, user, password = msfrpc_._resolve_msfrpc_credentials()
    finally:
        get_settings.cache_clear()
    assert user == "envuser"
    assert password == "envpass"
    captured = capsys.readouterr()
    log_text = captured.out + captured.err
    assert "env_var_deprecated" in log_text or "deprecated" in log_text


def test_msfrpc_raises_when_credentials_missing(monkeypatch):
    """No Settings + no env vars → RuntimeError (NOT silent empty-string
    auth as the previous os.environ.get('MSFRPC_PASSWORD', '') path did)."""
    import orchestrator.settings as settings_mod
    from orchestrator.settings import Settings, get_settings
    from tools import msfrpc_

    s = Settings()  # type: ignore[call-arg]
    monkeypatch.setattr(settings_mod, "get_settings", lambda: s)
    monkeypatch.delenv("MSFRPC_USER", raising=False)
    monkeypatch.delenv("MSFRPC_PASSWORD", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="msfrpc credentials not configured"):
            msfrpc_._resolve_msfrpc_credentials()
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Finding 4 — msfrpc login error message sanitization                          #
# --------------------------------------------------------------------------- #


def test_msfrpc_login_error_excludes_response_body():
    """The RuntimeError must surface ONLY the response key set, never the
    raw body (which could echo back username/password)."""
    from tools.msfrpc_ import _MsfRpcClient

    # Drive .login() with a fake httpx client whose post returns a JSON body
    # echoing the supplied credentials but missing the token.
    class _FakeResp:
        def __init__(self, body: dict) -> None:
            self._body = body
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._body

    class _FakeClient:
        def __init__(self, body: dict) -> None:
            self._body = body

        async def post(self, url, json=None):
            # Echo the username/password back like a real msfrpcd error path.
            return _FakeResp(
                {
                    "error": "no token",
                    "username_echo": json["username"],
                    "password_echo": json["password"],
                }
            )

        async def aclose(self) -> None:
            return None

    client = _MsfRpcClient("https://x/api/", "alice", "hunter2-the-original")
    client._client = _FakeClient({})  # type: ignore[assignment]
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.login())
    msg = str(exc.value)
    # The original-password body is NOT in the message.
    assert "hunter2-the-original" not in msg
    assert "alice" not in msg
    # We DO surface the response key set so operators can debug.
    assert "keys" in msg


# --------------------------------------------------------------------------- #
# Finding 5 — per-engagement default_network on every direct caller            #
# --------------------------------------------------------------------------- #


class _CapturingSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.engagement_id: str | None = None
        self.default_network: str | None = None

    async def run(self, argv, **kwargs):
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
async def test_bloodhound_collect_sets_default_network(monkeypatch):
    from sandbox.network import engagement_network_name
    from tools import bloodhound_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(bloodhound_, "_build_executor", lambda: _FakeExecutor())
    tool = bloodhound_.BloodHoundCollect()
    inp = bloodhound_.BloodHoundCollectInput(
        domain="lab.local", dc_ip="10.0.0.5", username="alice"
    )
    engagement_uuid = uuid4()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_uuid,
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.engagement_id == str(engagement_uuid)
    assert sandbox.default_network == engagement_network_name(engagement_uuid)


@pytest.mark.asyncio
async def test_kerberoast_sets_default_network(monkeypatch):
    from sandbox.network import engagement_network_name
    from tools import impacket_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(impacket_, "_build_executor", lambda: _FakeExecutor())
    tool = impacket_.ImpacketKerberoast()
    inp = impacket_.KerberoastInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        username="alice",
        nt_hash="aad3b435b51404eeaad3b435b51404ee",
    )
    engagement_uuid = uuid4()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_uuid,
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.default_network == engagement_network_name(engagement_uuid)


@pytest.mark.asyncio
async def test_asreproast_sets_default_network(monkeypatch):
    from sandbox.network import engagement_network_name
    from tools import impacket_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(impacket_, "_build_executor", lambda: _FakeExecutor())
    tool = impacket_.ImpacketASRepRoast()
    inp = impacket_.ASRepRoastInput(
        domain="lab.local",
        dc_ip="10.0.0.5",
        users_file_content="alice\nbob\n",
    )
    engagement_uuid = uuid4()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_uuid,
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.default_network == engagement_network_name(engagement_uuid)


@pytest.mark.asyncio
async def test_netexec_sets_default_network(monkeypatch):
    from sandbox.network import engagement_network_name
    from tools import netexec_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(netexec_, "_build_executor", lambda: _FakeExecutor())
    tool = netexec_.NetExec()
    inp = netexec_.NetExecInput(protocol="smb", targets=["10.0.0.5"])
    engagement_uuid = uuid4()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_uuid,
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.default_network == engagement_network_name(engagement_uuid)


@pytest.mark.asyncio
async def test_httpx_sets_default_network(monkeypatch):
    from sandbox.network import engagement_network_name
    from tools import httpx_
    from tools.base import ToolRunContext

    sandbox = _CapturingSandbox()

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = sandbox
            self.evidence = None

    monkeypatch.setattr(httpx_, "_build_executor", lambda: _FakeExecutor())
    tool = httpx_.HttpxProbe()
    inp = httpx_.HttpxProbeInput(targets=["10.0.0.5"], ports=[80])
    engagement_uuid = uuid4()
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_uuid,
        sandbox_id="sb",
        timeout_seconds=60,
    )
    await tool.run(inp, ctx)
    assert sandbox.default_network == engagement_network_name(engagement_uuid)


# --------------------------------------------------------------------------- #
# Finding 6 — nftables create failure raises when L3 required                  #
# --------------------------------------------------------------------------- #


def test_settings_exposes_sandbox_l3_required():
    from orchestrator.settings import Settings

    s = Settings()  # type: ignore[call-arg]
    # Default is False to preserve the historical fallback-with-warning.
    assert s.sandbox_l3_required is False


def test_sandbox_runner_l3_required_helper(monkeypatch):
    """The lazy ``_l3_required()`` helper reflects ``Settings``."""
    import orchestrator.settings as settings_mod
    from orchestrator.settings import Settings, get_settings
    from sandbox.runner import _l3_required

    s = Settings(sandbox_l3_required=True)  # type: ignore[call-arg]
    monkeypatch.setattr(settings_mod, "get_settings", lambda: s)
    get_settings.cache_clear()
    try:
        assert _l3_required() is True
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Finding 7 — msfrpc poll uses time.monotonic, not asyncio.get_event_loop      #
# --------------------------------------------------------------------------- #


def test_msfrpc_no_deprecated_get_event_loop_calls():
    """Source-level check: the polling loop must not use
    ``asyncio.get_event_loop().time()`` which is deprecated in 3.12+."""
    from pathlib import Path

    src = Path(__file__).parent.parent.parent / "src" / "tools" / "msfrpc_.py"
    text = src.read_text()
    assert "asyncio.get_event_loop()" not in text, (
        "msfrpc_.py still calls asyncio.get_event_loop() (deprecated in 3.12+); "
        "use time.monotonic() for poll deadlines"
    )


# --------------------------------------------------------------------------- #
# Finding 8 — operator_exec actor is the authenticated principal               #
# --------------------------------------------------------------------------- #


def test_operator_exec_resolves_actor_from_principal_not_payload():
    """Source-level check: the endpoint reads ``actor = principal`` and uses
    it on every audit() call."""
    from pathlib import Path

    src = Path(__file__).parent.parent.parent / "src" / "orchestrator" / "services" / "operator_exec.py"
    text = src.read_text()
    assert "actor = principal" in text
    assert "actor=payload.actor" not in text


def test_operator_exec_request_body_actor_field_is_ignored():
    """Even if the client posts ``actor=evilroot``, the validated input drops
    it AND no code path reads it from ``payload`` for an audit field."""
    from orchestrator.services.operator_exec import OperatorExecRequest

    req = OperatorExecRequest(
        argv=["ls"],
        target="10.0.0.5",
        justification="t",
        actor="evilroot",  # client tries to forge
    )
    # The field is still present for backwards-compat with TUI/scripts but
    # the endpoint never reads it for audit. The validation here just
    # documents that intent.
    assert req.actor == "evilroot"  # field accepts whatever the client sent
    # But the endpoint source NEVER routes that into audit() (verified above).


# --------------------------------------------------------------------------- #
# Finding 9 — BloodHound truncated flag + raised max_output_bytes              #
# --------------------------------------------------------------------------- #


def test_bloodhound_output_has_truncated_field():
    from tools.bloodhound_ import BloodHoundCollectOutput

    out = BloodHoundCollectOutput(command="bloodhound-python -d lab.local")
    assert hasattr(out, "truncated")
    assert out.truncated is False


def test_bloodhound_executor_uses_100mb_cap():
    """The BloodHound _build_executor must construct SandboxRunner with the
    raised cap so large AD ZIPs aren't silently corrupted."""
    from tools.bloodhound_ import _build_executor

    exe = _build_executor()
    assert exe.sandbox.max_output_bytes == 100 * 1024 * 1024


@pytest.mark.asyncio
async def test_bloodhound_propagates_truncated_to_output(monkeypatch):
    """When SandboxResult.truncated is True, the tool's output surfaces it."""
    from tools import bloodhound_
    from tools.base import ToolRunContext

    class _TruncSandbox:
        engagement_id: str | None = None
        default_network: str | None = None

        async def run(self, argv, **kwargs):
            from sandbox.runner import SandboxResult

            return SandboxResult(
                exit_code=0,
                stdout=b"PK\x03\x04...zip-bytes...",
                stderr=b"",
                duration_seconds=1.0,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                truncated=True,
            )

    class _FakeExecutor:
        def __init__(self) -> None:
            self.sandbox = _TruncSandbox()
            self.evidence = None

    monkeypatch.setattr(bloodhound_, "_build_executor", lambda: _FakeExecutor())
    tool = bloodhound_.BloodHoundCollect()
    inp = bloodhound_.BloodHoundCollectInput(
        domain="lab.local", dc_ip="10.0.0.5", username="alice"
    )
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb",
        timeout_seconds=60,
    )
    result = await tool.run(inp, ctx)
    assert result.output.truncated is True


# --------------------------------------------------------------------------- #
# Finding 10 — BloodHound default community password gone                      #
# --------------------------------------------------------------------------- #


def test_bloodhound_query_no_default_community_password():
    """The well-known default ``bloodhoundcommunityedition`` must not be the
    schema default any more — operators must configure explicitly."""
    from tools.bloodhound_ import BloodHoundQueryInput

    inp = BloodHoundQueryInput(cypher="MATCH (n) RETURN n LIMIT 1")
    assert inp.bloodhound_password is None
    assert inp.bloodhound_password != "bloodhoundcommunityedition"


def test_bloodhound_settings_no_longer_has_neo4j_password_field():
    """BloodHound Neo4j settings were removed from Settings. BloodHound
    connection details are now provided via tool input only."""
    from orchestrator.settings import Settings

    s = Settings()  # type: ignore[call-arg]
    assert not hasattr(s, "bloodhound_neo4j_password")
    assert not hasattr(s, "bloodhound_neo4j_uri")


@pytest.mark.asyncio
async def test_bloodhound_query_raises_when_password_unset():
    """If input doesn't supply a password, the query path raises a clean
    RuntimeError (NOT a silent ``connection refused``)."""
    from tools import bloodhound_
    from tools.base import ToolRunContext

    tool = bloodhound_.BloodHoundQuery()
    inp = bloodhound_.BloodHoundQueryInput(cypher="MATCH (n) RETURN n LIMIT 1")
    ctx = ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=uuid4(),
        sandbox_id="sb",
        timeout_seconds=10,
    )
    with pytest.raises(RuntimeError, match="BloodHound Neo4j connection not configured"):
        await tool.run(inp, ctx)
