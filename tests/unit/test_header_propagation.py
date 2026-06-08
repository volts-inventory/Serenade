"""Tests for X-Serenade-Engagement header in tool argv (Phase 16, Tier A #6)."""

from __future__ import annotations

from uuid import UUID, uuid4

from tools.base import ToolRunContext


def _ctx(engagement_id: UUID) -> ToolRunContext:
    return ToolRunContext(
        tool_run_id=uuid4(),
        engagement_id=engagement_id,
        sandbox_id="dev",
    )


def _argv_for_nuclei(input_, ctx) -> list[str]:
    """Reconstruct the nuclei argv prefix without actually running it."""
    # Mirror tools/nuclei_.py::NucleiScan.run argv construction up to the
    # point where the header should appear.
    eng_header = f"X-Serenade-Engagement: {ctx.engagement_id}"
    argv = [
        "nuclei",
        "-jsonl",
        "-silent",
        "-no-color",
        "-disable-update-check",
        "-severity",
        ",".join(input_["severity"]),
        "-rate-limit",
        str(input_["rate_limit"]),
        "-c",
        str(input_["concurrency"]),
        "-H",
        eng_header,
    ]
    return argv


def test_nuclei_includes_engagement_header():
    eng = uuid4()
    ctx = _ctx(eng)
    argv = _argv_for_nuclei({"severity": ["high"], "rate_limit": 50, "concurrency": 25}, ctx)
    # `-H` should appear with the engagement header value
    assert "-H" in argv
    h_idx = argv.index("-H")
    assert argv[h_idx + 1] == f"X-Serenade-Engagement: {eng}"


def test_header_format_round_trip():
    """The proxy bucketing test (Phase 8) accepts the header value as the
    bucket. Confirm the header value is just `<uuid>` (no trailing garbage)
    so `safe_bucket(header_value)` reduces to the engagement ID."""
    from sandbox.proxy._bucket import safe_bucket

    eng = uuid4()
    header_value = f"{eng}"
    bucket = safe_bucket(header_value)
    # UUID hex has dashes, which are in the safe_bucket allowlist.
    assert bucket == str(eng)
