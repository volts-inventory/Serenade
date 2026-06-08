"""Tests for the mitmproxy addon's per-engagement bucketing (audit-fix #19).

Tests the pure `safe_bucket` helper directly so we don't need mitmproxy
installed in the unit-test environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import from sandbox/proxy/_bucket.py without going through the mitmproxy addon module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "sandbox" / "proxy"))

from _bucket import safe_bucket  # type: ignore[import-not-found]


def test_safe_bucket_strips_path_traversal():
    assert safe_bucket("../../etc/passwd") == "etcpasswd"


def test_safe_bucket_blocks_header_injection():
    # Newline and colon are stripped; the remaining alphanumerics/dashes/underscores stay
    assert safe_bucket("eng-1\nX-Other: foo") == "eng-1X-Otherfoo"
    # No control characters or path separators survive
    assert "\n" not in safe_bucket("a\nb")
    assert "/" not in safe_bucket("a/b")
    assert ":" not in safe_bucket("a:b")


def test_safe_bucket_empty_becomes_unattributed():
    assert safe_bucket("") == "unattributed"
    assert safe_bucket("   ") == "unattributed"
    assert safe_bucket("///") == "unattributed"


def test_safe_bucket_truncates_long_input():
    assert len(safe_bucket("a" * 200)) == 64


def test_safe_bucket_accepts_valid_engagement_id():
    assert safe_bucket("eng_12345-abcd") == "eng_12345-abcd"


def test_safe_bucket_rejects_unicode_homoglyphs():
    # Verification audit fix #7: full-width digits and letters are
    # alphanumeric per Unicode but must not be allowed.
    assert safe_bucket("ＡＢＣ123") == "123"  # ASCII digits survive; full-width letters don't
    assert safe_bucket("аbcd") == "bcd"  # Cyrillic 'а' (U+0430) stripped, ASCII 'bcd' kept
