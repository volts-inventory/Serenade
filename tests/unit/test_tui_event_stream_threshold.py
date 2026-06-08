"""R3 #M2.5: TUI event_stream renders the three threshold-warning kinds
with the same per-kind summary strings + glyphs + colours that the web UI
uses, so an operator watching the engagement on a Textual/RichLog client
sees the same signal as one watching the browser.

Pure-function coverage — the EventStreamPanel widget itself requires a
Textual app context to instantiate, but its helpers ``_summarize_threshold``,
``_kind_style``, ``_threshold_glyph``, and ``_is_threshold_kind`` are
public-by-convention (underscore-prefixed but module-level), so we test
them directly. If the widget's _write() ever diverges from these helpers,
the visual divergence would be obvious in manual smoke testing.
"""

from __future__ import annotations

from tui.widgets.event_stream import (  # type: ignore[import-not-found]
    _is_threshold_kind,
    _kind_style,
    _summarize_threshold,
    _threshold_glyph,
)


# --------------------------------------------------------------------------- #
# Kind predicates                                                             #
# --------------------------------------------------------------------------- #


def test_is_threshold_kind_true_for_all_three_warning_kinds() -> None:
    assert _is_threshold_kind("engagement.budget_warning") is True
    assert _is_threshold_kind("engagement.veto_rate_warning") is True
    assert _is_threshold_kind("engagement.suppression_spread") is True


def test_is_threshold_kind_false_for_unrelated_kinds() -> None:
    for k in (
        "stage.start",
        "tool.completed",
        "tool.suppressed",
        "regression.requested",
        "approval.decided",
        "engagement.aborted",  # unrelated engagement event
        "engagement.budget",  # not the warning suffix
        "",
    ):
        assert _is_threshold_kind(k) is False, k


# --------------------------------------------------------------------------- #
# Rich colour style mapping                                                   #
# --------------------------------------------------------------------------- #


def test_kind_style_budget_warning_is_amber() -> None:
    # Yellow1 is the Rich amber-equivalent style we picked to mirror the
    # web UI's amber palette for budget/suppression warnings.
    assert _kind_style("engagement.budget_warning") == "b yellow1"


def test_kind_style_suppression_spread_is_amber() -> None:
    assert _kind_style("engagement.suppression_spread") == "b yellow1"


def test_kind_style_veto_rate_warning_is_red() -> None:
    # Red — the council rejecting more than half the recent tool calls is
    # higher-severity than budget warnings.
    assert _kind_style("engagement.veto_rate_warning") == "b red3"


# --------------------------------------------------------------------------- #
# Per-kind glyph                                                              #
# --------------------------------------------------------------------------- #


def test_threshold_glyph_per_kind() -> None:
    assert _threshold_glyph("engagement.budget_warning") == "⚠️"
    assert _threshold_glyph("engagement.veto_rate_warning") == "🚫"
    assert _threshold_glyph("engagement.suppression_spread") == "🔇"


# --------------------------------------------------------------------------- #
# Summary string projection                                                   #
# --------------------------------------------------------------------------- #


def test_summarize_threshold_token_budget() -> None:
    """Mirrors the UI's "Token budget at 82% used" formatting."""
    s = _summarize_threshold(
        "engagement.budget_warning",
        {"kind": "token", "usage": 0.82, "used": 820_000, "budget": 1_000_000},
    )
    assert s == "Token budget at 82% used"


def test_summarize_threshold_wallclock_budget_with_hours_fragment() -> None:
    """When elapsed_ns + window_ns are populated, the summary surfaces
    the X-of-Y hours fragment so the operator sees concrete time."""
    s = _summarize_threshold(
        "engagement.budget_warning",
        {
            "kind": "wallclock",
            "usage": 0.82,
            "elapsed_ns": int(6 * 3.6e12),
            "window_ns": int(8 * 3.6e12),
        },
    )
    # "6.0 of 8.0 hours" — float formatting with one decimal.
    assert s == "Wallclock at 82% used (6.0 of 8.0 hours)"


def test_summarize_threshold_wallclock_without_hours_fragment() -> None:
    """A payload missing the elapsed/window fields still renders a usable
    summary — just drops the hours-of-hours suffix rather than breaking."""
    s = _summarize_threshold(
        "engagement.budget_warning",
        {"kind": "wallclock", "usage": 0.95},
    )
    assert s == "Wallclock at 95% used"


def test_summarize_threshold_veto_rate() -> None:
    """Critic-veto-rate warning surfaces percent + window size."""
    s = _summarize_threshold(
        "engagement.veto_rate_warning",
        {"rate": 0.6, "window": 10, "vetoes": 6},
    )
    assert s == "Critic veto rate 60% over last 10 verdicts"


def test_summarize_threshold_suppression_spread_singular_pluralisation() -> None:
    """``1 tool`` vs ``N tools`` — the plural-s is important for an honest
    operator-facing string."""
    one = _summarize_threshold(
        "engagement.suppression_spread",
        {"stage": "exploit", "new_count": 1},
    )
    assert one == "1 tool suppressed in stage exploit"
    three = _summarize_threshold(
        "engagement.suppression_spread",
        {"stage": "exploit", "new_count": 3},
    )
    assert three == "3 tools suppressed in stage exploit"


def test_summarize_threshold_suppression_spread_falls_back_to_count() -> None:
    """``new_count`` is preferred; when absent, we fall back to ``count``
    (current total). Matches the EventsTab.tsx fallback chain."""
    s = _summarize_threshold(
        "engagement.suppression_spread",
        {"stage": "lateral", "count": 2},
    )
    assert s == "2 tools suppressed in stage lateral"
