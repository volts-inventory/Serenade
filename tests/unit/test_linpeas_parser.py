"""Tests for the linpeas output parser (Phase 18, Tier B #7)."""

from __future__ import annotations

from tools.linpeas_ import _parse_linpeas

_SAMPLE = """\
\x1b[1;31m[+] SUID files\x1b[0m: found
/usr/bin/find — exploitable via GTFOBins
extra context here
\x1b[1;33m[+] Sudo entries\x1b[0m: ALL=(ALL) NOPASSWD: /bin/bash
\x1b[1;31m[+] Kernel exploits\x1b[0m: linux 4.4.0 — CVE-2017-1000112 likely
unrelated junk
\x1b[1;31m[+] Cron jobs\x1b[0m: writable script at /etc/cron.daily/backup.sh
"""


def test_parser_picks_up_section_hits():
    findings = _parse_linpeas(_SAMPLE, ["suid", "sudo", "kernel", "cron"])
    sections = {f.section for f in findings}
    assert "suid_files" in sections
    assert "kernel_exploits" in sections
    assert "cron_jobs" in sections


def test_parser_filters_by_section():
    findings = _parse_linpeas(_SAMPLE, ["suid"])
    sections = {f.section for f in findings}
    assert sections == {"suid_files"}


def test_parser_extracts_details():
    findings = _parse_linpeas(_SAMPLE, ["suid"])
    # `detail` is what's after the colon on the [+] line; `raw_excerpt`
    # includes surrounding context.
    assert any("found" in f.detail for f in findings)
    assert any("GTFOBins" in f.raw_excerpt for f in findings)


def test_parser_handles_empty_input():
    assert _parse_linpeas("", ["suid"]) == []


def test_parser_handles_no_red_hits():
    assert _parse_linpeas("just some boring linpeas chatter", ["suid"]) == []
