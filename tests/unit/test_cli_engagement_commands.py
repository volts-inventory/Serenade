"""R3 #M2: tests for ``serenade engagement watch`` and ``summary``.

These CLI commands are operator-facing local observability surfaces:
they hit the orchestrator's existing SSE stream + summary endpoint and
render coloured output. The tests here pin the parser wiring, the
summary renderer, the event-summary helper, and the URL-resolution
fallback. Network IO is stubbed via httpx ``MockTransport``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


# --------------------------------------------------------------------------- #
# Parser wiring                                                                #
# --------------------------------------------------------------------------- #


def test_parser_recognises_engagement_watch_subcommand() -> None:
    from cli.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["engagement", "watch", "abc-123"])
    assert args.subcommand == "engagement"
    assert args.engagement_subcommand == "watch"
    assert args.engagement_id == "abc-123"
    assert args.filter_kinds is None


def test_parser_recognises_engagement_summary_subcommand() -> None:
    from cli.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["engagement", "summary", "abc-123", "--format", "json"])
    assert args.engagement_subcommand == "summary"
    assert args.engagement_id == "abc-123"
    assert args.format == "json"


def test_parser_watch_accepts_filter_flag() -> None:
    from cli.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["engagement", "watch", "abc-123", "--filter", "tool.completed,approval.queued"]
    )
    assert args.filter_kinds == "tool.completed,approval.queued"


# --------------------------------------------------------------------------- #
# URL resolution                                                              #
# --------------------------------------------------------------------------- #


def test_default_orchestrator_url_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cli.main import _default_orchestrator_url

    monkeypatch.setenv("SERENADE_ORCHESTRATOR_URL", "http://example.com:9999/")
    assert _default_orchestrator_url() == "http://example.com:9999"


def test_default_orchestrator_url_falls_back_to_localhost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("SERENADE_ORCHESTRATOR_URL", raising=False)
    # Point ENV_PATH at a non-existent file so the helper takes the
    # Settings default branch.
    from cli import main as cli_main

    monkeypatch.setattr(cli_main, "ENV_PATH", tmp_path / "no-env")
    url = cli_main._default_orchestrator_url()
    assert url.startswith("http://")
    assert "8000" in url


# --------------------------------------------------------------------------- #
# Event-line summary helper                                                   #
# --------------------------------------------------------------------------- #


def test_summarize_tool_completed() -> None:
    from cli.main import _summarize_event

    s = _summarize_event(
        "tool.completed",
        {"tool": "nuclei_scan", "stage": "recon", "exit_code": 0, "nodes_written": 7},
    )
    assert "nuclei_scan" in s
    assert "recon" in s
    assert "exit=0" in s
    assert "nodes=7" in s


def test_summarize_budget_warning_kind() -> None:
    from cli.main import _summarize_event

    s = _summarize_event(
        "engagement.budget_warning", {"kind": "token", "usage": 0.85}
    )
    assert "token" in s
    assert "0.85" in s


def test_color_for_kind_red_on_escalate() -> None:
    from cli.main import _color_for_kind

    assert _color_for_kind("agent.escalate_human") == "red"
    assert _color_for_kind("engagement.budget_warning") == "red"
    assert _color_for_kind("tool.completed") == "green"
    assert _color_for_kind("tool.suppressed") == "amber"
    assert _color_for_kind("approval.queued") == "blue"
    assert _color_for_kind("regression.requested") == "magenta"


# --------------------------------------------------------------------------- #
# Summary renderer                                                             #
# --------------------------------------------------------------------------- #


def test_render_summary_text_includes_all_counters() -> None:
    from cli.main import _render_summary_text

    payload = {
        "tool_calls_total": 10,
        "tool_calls_auto_approved": 4,
        "tool_calls_critic_vetoed": 1,
        "tool_calls_human_approved_gated": 2,
        "tool_calls_ungated": 3,
        "suppressed_tools_count": 1,
        "llm_cost_usd": 0.4523,
        "tool_cost_usd": 0.0,
        "wallclock_minutes_used": 12.5,
        "regressions_used": 1,
        "findings_count": 3,
    }
    out = _render_summary_text(payload)
    # Counters present.
    assert "10" in out
    assert "0.4523" in out
    assert "12.5" in out
    assert "Findings:" in out
    assert "Suppressed:" in out
    # Engagement summary header.
    assert "Engagement summary" in out


# --------------------------------------------------------------------------- #
# Summary command (network stubbed via MockTransport)                          #
# --------------------------------------------------------------------------- #


def test_summary_json_format_runs_and_prints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``serenade engagement summary <id> --format json`` writes JSON to stdout."""
    import json

    fake_payload = {
        "tool_calls_total": 5,
        "tool_calls_auto_approved": 2,
        "tool_calls_critic_vetoed": 0,
        "tool_calls_human_approved_gated": 1,
        "tool_calls_ungated": 2,
        "suppressed_tools_count": 0,
        "llm_cost_usd": 0.0,
        "tool_cost_usd": 0.0,
        "wallclock_minutes_used": 5.0,
        "regressions_used": 0,
        "findings_count": 0,
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "/summary" in str(request.url)
        return httpx.Response(200, json=fake_payload)

    # Patch httpx.Client to use the MockTransport so the test can exercise
    # the live JSON-render branch without a real HTTP server.
    real_client = httpx.Client

    def _patched_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(_handler))

    monkeypatch.setattr(httpx, "Client", _patched_client)

    from cli.main import cmd_engagement_summary

    rc = cmd_engagement_summary(
        engagement_id="00000000-0000-0000-0000-000000000000",
        fmt="json",
        base_url="http://127.0.0.1:8000",
    )
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["tool_calls_total"] == 5


def test_summary_http_error_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Server 404 → CLI prints an error line and returns rc=1."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="engagement not found")

    real_client = httpx.Client

    def _patched_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(_handler))

    monkeypatch.setattr(httpx, "Client", _patched_client)

    from cli.main import cmd_engagement_summary

    rc = cmd_engagement_summary(
        engagement_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
        fmt="text",
        base_url="http://127.0.0.1:8000",
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "404" in captured.err or "404" in captured.out
