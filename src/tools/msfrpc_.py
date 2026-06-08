"""Metasploit RPC adapter.

Wraps msfrpcd over HTTP. Operator runs msfrpcd
separately (`msfrpcd -P <password> -S` for SSL) and configures the
connection via ``Settings.msfrpc_url`` / ``msfrpc_user`` / ``msfrpc_password``
(pydantic-settings reads the same env-var names but normalises types and
honours the ``get_settings`` lru_cache). Direct ``os.environ`` reads still
work for backwards compatibility but emit a one-time deprecation warning.

Currently exposes one tool, `MetasploitExploit` — runs a named module
against a target with parameters and waits for the session/result.
Detection-and-trigger workflow; the operator must approve every call.

The tool's risk_class is EXPLOIT, so `critic_required_for_risk_classes`
fires the PreExploitCritic gate. Auto-execution is impossible without the
critic approve.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC
from typing import ClassVar, Literal
from uuid import UUID

import httpx
import structlog
from pydantic import BaseModel, Field

from tools._helpers import iso
from tools.base import Tool, ToolResult, ToolRunContext
from worldstate.schema import RiskClass, Stage

log = structlog.get_logger(__name__)


_DEPRECATION_WARNED_ENV: set[str] = set()


def _resolve_msfrpc_credentials() -> tuple[str, str, str]:
    """Resolve (base_url, username, password) for msfrpcd from Settings.

    Precedence:
    1. ``Settings.msfrpc_*`` values via ``get_settings()`` (pydantic-settings
       normalises types and the ``lru_cache`` avoids per-call re-parsing).
    2. Backwards-compat: if a Settings value is unset (None for credentials)
       AND the equivalent env var is set, use the env var and log a one-time
       deprecation warning per var so operators see the migration path.

    Raises ``RuntimeError`` when no username/password is configured by either
    path — better than silently authenticating as empty.
    """
    import os

    from orchestrator.settings import get_settings

    s = get_settings()
    base_url = s.msfrpc_url
    if base_url == "https://127.0.0.1:55553/api/" and os.environ.get("MSFRPC_URL"):
        _warn_env_deprecated_once("MSFRPC_URL")
        base_url = os.environ["MSFRPC_URL"]

    username = s.msfrpc_user
    if username is None and os.environ.get("MSFRPC_USER"):
        _warn_env_deprecated_once("MSFRPC_USER")
        username = os.environ["MSFRPC_USER"]

    password = s.msfrpc_password
    if password is None and os.environ.get("MSFRPC_PASSWORD"):
        _warn_env_deprecated_once("MSFRPC_PASSWORD")
        password = os.environ["MSFRPC_PASSWORD"]

    if not username or not password:
        raise RuntimeError(
            "msfrpc credentials not configured; set MSFRPC_USER and "
            "MSFRPC_PASSWORD via Settings (preferred) or env var"
        )
    return base_url, username, password


def _warn_env_deprecated_once(var: str) -> None:
    """Emit a one-time structlog warning that ``var`` should move to Settings."""
    if var in _DEPRECATION_WARNED_ENV:
        return
    _DEPRECATION_WARNED_ENV.add(var)
    log.warning(
        "msfrpc.env_var_deprecated",
        var=var,
        advice=(
            f"reading {var} from os.environ directly is deprecated; "
            "configure via Settings (pydantic-settings reads the same env var "
            "but normalises types and honours get_settings() lru_cache)"
        ),
    )


def _resolve_msfrpc_tls_verify() -> bool | str:
    """Resolve the httpx ``verify=`` argument for the msfrpc client.

    Precedence (highest first):
    1. ``MSFRPC_CA_BUNDLE`` env / settings value — used as a path to a PEM
       bundle; httpx verifies the cert against that bundle.
    2. ``MSFRPC_TLS_VERIFY=True`` — verifies against the system CA store.
    3. Otherwise: ``False`` (no verification) and a structlog warning is
       emitted so operators see the gap in their logs.

    Returns the value to pass as ``httpx.AsyncClient(verify=...)``: a bool
    or a CA-bundle path string.
    """
    from orchestrator.settings import get_settings

    s = get_settings()
    if s.msfrpc_ca_bundle:
        return s.msfrpc_ca_bundle
    if s.msfrpc_tls_verify:
        return True
    log.warning(
        "msfrpc.tls_verify_disabled",
        advice=(
            "msfrpc TLS verification disabled; set MSFRPC_CA_BUNDLE to a PEM "
            "path (or MSFRPC_TLS_VERIFY=true with a system-trusted cert) for "
            "a CA-pinned trust path"
        ),
    )
    return False


class MetasploitExploitInput(BaseModel):
    module_path: str = Field(
        min_length=1,
        description=(
            "Module path, e.g. 'exploit/multi/http/log4shell_header_injection'. "
            "MUST be a real MSF module path; the adapter does not auto-discover."
        ),
    )
    rhosts: list[str] = Field(min_length=1, description="Target hosts (in-scope)")
    rport: int | None = Field(default=None, ge=1, le=65535)
    payload: str | None = Field(
        default=None,
        description="Payload module path, e.g. 'cmd/unix/reverse'. If omitted MSF picks a default.",
    )
    lhost: str | None = None
    lport: int | None = Field(default=None, ge=1, le=65535)
    extra_options: dict[str, str | int | bool] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=180, ge=10, le=1800)


class MetasploitExploitOutput(BaseModel):
    outcome: Literal["success", "failure", "error", "skipped"]
    module_path: str
    session_id_in_msf: int | None = None
    job_id: int | None = None
    summary: str
    error_message: str | None = None


class MetasploitExploit(Tool[MetasploitExploitInput, MetasploitExploitOutput]):
    name = "metasploit_exploit"
    description = (
        "Execute a Metasploit exploit module against an in-scope target via "
        "msfrpcd. Subject to OPA + PreExploitCritic gates. The operator must "
        "approve. Only operator-curated modules should be invoked."
    )
    stage = Stage.EXPLOIT
    risk_class = RiskClass.EXPLOIT
    destructive = False
    Input: ClassVar[type[BaseModel]] = MetasploitExploitInput
    Output: ClassVar[type[BaseModel]] = MetasploitExploitOutput

    def extract_targets(self, input: MetasploitExploitInput) -> list[str]:
        return list(input.rhosts)

    def summarize_for_model(
        self, output: MetasploitExploitOutput, *, tool_run_id: UUID, persisted: int
    ) -> str:
        # `output.summary` is built deterministically in
        # run() from module path + job/session IDs (no secrets). But
        # `output.error_message` is `str(e)[:500]` of an arbitrary
        # exception — httpx errors can echo URL fragments, msf payload
        # parsing errors might echo back operator-supplied option values
        # (LHOST/LPORT/credentials in extra_options). Surface only the
        # presence flag; full text persists in the ToolResult + evidence.
        err_state = "yes" if output.error_message else "no"
        summary = (
            f"{self.name}: module={output.module_path} outcome={output.outcome}, "
            f"msf_job={output.job_id}, msf_session={output.session_id_in_msf} "
            f"error={err_state}. {output.summary[:120]} "
            f"persisted={persisted}. tool_run_id={tool_run_id}"
        )
        if len(summary) > 500:
            summary = summary[:485] + "... [truncated]"
        return summary

    async def run(
        self, input: MetasploitExploitInput, ctx: ToolRunContext
    ) -> ToolResult[MetasploitExploitOutput]:
        from datetime import datetime

        started = datetime.now(UTC)
        # ``client`` is bound only AFTER credential resolution succeeds; the
        # ``finally`` below guards against an unbound name when
        # ``_resolve_msfrpc_credentials`` raises.
        client: _MsfRpcClient | None = None
        try:
            # Settings-first resolution: env-var backwards-compat lives in
            # ``_resolve_msfrpc_credentials`` (with a one-time deprecation
            # warning per var). Raises RuntimeError on missing creds; we
            # keep the call inside the same try block as ``client.login()``
            # so the operator-facing return path is a clean
            # ``MetasploitExploitOutput(outcome="error", error_message=...)``
            # rather than a raised exception that bubbles to the caller.
            base_url, user, password = _resolve_msfrpc_credentials()

            client = _MsfRpcClient(base_url, user, password)
            await client.login()
            options: dict[str, str | int | bool] = {
                "RHOSTS": ",".join(input.rhosts),
                **input.extra_options,
            }
            if input.rport is not None:
                options["RPORT"] = input.rport
            if input.payload:
                options["PAYLOAD"] = input.payload
            if input.lhost is not None:
                options["LHOST"] = input.lhost
            if input.lport is not None:
                options["LPORT"] = input.lport

            job_id, msf_session_id = await client.execute_exploit(
                input.module_path, options, timeout_seconds=input.timeout_seconds
            )
            outcome = "success" if msf_session_id is not None else "failure"
            summary = f"msf module {input.module_path} executed; job {job_id}; " + (
                f"session {msf_session_id} opened" if msf_session_id else "no session"
            )
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MetasploitExploitOutput(
                    outcome=outcome,  # type: ignore[arg-type]
                    module_path=input.module_path,
                    session_id_in_msf=msf_session_id,
                    job_id=job_id,
                    summary=summary,
                ),
                exit_code=0 if outcome == "success" else 1,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        except Exception as e:
            ended = datetime.now(UTC)
            return ToolResult(
                tool_run_id=ctx.tool_run_id,
                name=self.name,
                output=MetasploitExploitOutput(
                    outcome="error",
                    module_path=input.module_path,
                    summary="msfrpc client error",
                    error_message=str(e)[:500],
                ),
                exit_code=1,
                started_at=iso(started),
                ended_at=iso(ended),
            )
        finally:
            if client is not None:
                await client.close()


class _MsfRpcClient:
    """Minimal async msfrpcd HTTP client.

    msfrpcd uses MessagePack as the wire format on its native binary
    interface, but exposes JSON via the `/api/v1/...` REST endpoint in
    recent versions. We use the REST endpoint since httpx already speaks
    HTTP and no msgpack dep is needed.
    """

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token: str | None = None
        # TLS verification is resolved from settings: a CA bundle path wins
        # over a bare bool, and the unverified default emits a startup
        # warning so the operator notices the gap. See
        # _resolve_msfrpc_tls_verify above.
        self._client = httpx.AsyncClient(timeout=30.0, verify=_resolve_msfrpc_tls_verify())

    async def close(self) -> None:
        await self._client.aclose()

    async def login(self) -> None:
        resp = await self._client.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": self.username, "password": self.password},
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("data", {}).get("token") or data.get("token")
        if not token:
            # Do NOT inline the response body — msfrpcd has historically been
            # observed to echo the supplied username (and in some error paths
            # the password) back in its error JSON. ``error_message`` flows
            # into ``ToolResult.error_message[:500]`` which lands in audit
            # logs. Surface only the response key set + a sanitized status.
            keys = sorted(data.keys()) if isinstance(data, dict) else ["<non-dict>"]
            raise RuntimeError(
                f"msfrpc login: no token in response (keys: {keys})"
            )
        self.token = token

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("msfrpc client not logged in")
        return {"Authorization": f"Bearer {self.token}"}

    async def execute_exploit(
        self,
        module_path: str,
        options: dict[str, str | int | bool],
        *,
        timeout_seconds: int,
    ) -> tuple[int | None, int | None]:
        """Execute the module; return (job_id, msf_session_id)."""
        resp = await self._client.post(
            f"{self.base_url}/api/v1/modules/{module_path}",
            headers=self._headers(),
            json={"action": "execute", "options": options},
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        job_id = data.get("job_id")
        if job_id is None:
            return None, None
        # Poll for completion. ``time.monotonic`` (loop-independent and not
        # subject to NTP jumps) replaces the deprecated ``get_event_loop`` calls.
        # Python 3.12+ emits a DeprecationWarning when ``get_event_loop`` is
        # invoked outside a running loop; ``monotonic`` sidesteps the warning
        # and is cheaper besides.
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            jobs_resp = await self._client.get(
                f"{self.base_url}/api/v1/jobs", headers=self._headers()
            )
            jobs_resp.raise_for_status()
            jobs = {j["id"]: j for j in jobs_resp.json().get("data", [])}
            if job_id not in jobs:
                # Job completed; check sessions
                sess_resp = await self._client.get(
                    f"{self.base_url}/api/v1/sessions", headers=self._headers()
                )
                sess_resp.raise_for_status()
                sessions = sess_resp.json().get("data", [])
                # Most recent session wins (MSF assigns ascending IDs)
                if sessions:
                    return job_id, max(s["id"] for s in sessions)
                return job_id, None
            await asyncio.sleep(2)
        return job_id, None  # timed out
