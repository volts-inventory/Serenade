"""Pure helper for the proxy addon — importable without mitmproxy."""

from __future__ import annotations

# ASCII allowlist, not isalnum() (which is unicode-aware and lets full-width
# digits/letters through). Engagement IDs are UUIDs in practice, so ASCII-only
# is strictly correct.
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def safe_bucket(name: str) -> str:
    """Whitelist ASCII alphanumerics + dash + underscore; cap at 64 chars."""
    return "".join(c for c in name if c in _ALLOWED)[:64] or "unattributed"
