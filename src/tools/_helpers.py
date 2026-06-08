"""Shared helpers for tool adapters: sandbox exec + evidence storage."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from orchestrator.evidence import EvidenceStore
from sandbox.network import engagement_network_name
from sandbox.runner import SandboxResult, SandboxRunner
from tools.base import ToolRunContext


class ToolExecutor:
    """Sandwich: sandbox exec + push raw outputs to evidence store + return blob IDs.

    The SandboxRunner stamps the engagement_id into the container name AND
    attaches it to the per-engagement Docker network. This is what makes
    ARCHITECTURE.md §7 L2 isolation actually work.
    """

    def __init__(self, sandbox: SandboxRunner, evidence: EvidenceStore | None) -> None:
        self.sandbox = sandbox
        self.evidence = evidence

    async def run_and_capture(
        self,
        argv: Sequence[str],
        *,
        ctx: ToolRunContext,
        tool_name: str,
        timeout_seconds: int | None = None,
        env: dict[str, str] | None = None,
        redacted_argv: Sequence[str] | None = None,
    ) -> tuple[SandboxResult, str | None, str | None]:
        """Run a sandboxed command; persist stdout/stderr; return (result, stdout_blob_id, stderr_blob_id).

        Blob IDs are strings; None if evidence store is unavailable (dev/test).

        ``env`` is passed through to the runner's ``-e`` mechanism so untrusted
        values can be supplied to a sandboxed process WITHOUT shell-string
        interpolation (the docker runner sets them on the container env, never
        via a ``k="v"`` shell prefix).
        """
        # Bind the engagement to the runner so it spawns containers on the
        # per-engagement network with the engagement-stamped name. Cheap to
        # do per-call; keeps the runner stateless across engagements.
        self.sandbox.engagement_id = str(ctx.engagement_id)
        self.sandbox.default_network = engagement_network_name(ctx.engagement_id)
        result = await self.sandbox.run(
            argv,
            timeout_seconds=timeout_seconds or ctx.timeout_seconds,
            env=env,
            redacted_argv=redacted_argv,
        )
        stdout_id: str | None = None
        stderr_id: str | None = None
        if self.evidence and result.stdout:
            blob_id, _ = self.evidence.put(
                result.stdout,
                engagement_id=ctx.engagement_id,
                kind=f"{tool_name}/stdout",
            )
            stdout_id = str(blob_id)
        if self.evidence and result.stderr:
            blob_id, _ = self.evidence.put(
                result.stderr,
                engagement_id=ctx.engagement_id,
                kind=f"{tool_name}/stderr",
            )
            stderr_id = str(blob_id)
        return result, stdout_id, stderr_id


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()
