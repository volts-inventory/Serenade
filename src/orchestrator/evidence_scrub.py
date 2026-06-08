"""Best-effort credential scrubbing for evidence blobs.

Tool stdout/stderr from NetExec, secretsdump, linpeas, etc. can contain
cleartext passwords, NT hashes, JWT tokens, AWS keys. Without scrubbing,
those land in MinIO and end up referenced from findings and rendered into
the report.

This is best-effort defense; the actual cred values still exist in
`Credential` nodes (where they're typed and access-controlled). The
scrubber is for the raw blob copy.

Returns the (possibly-modified) bytes and a `bool` indicating whether
anything was redacted. Caller persists `redacted=True` on the EvidenceRow.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[bytes], bytes]] = [
    # NTLM hash (LM:NT format)
    (re.compile(rb"\b[a-fA-F0-9]{32}:[a-fA-F0-9]{32}\b"), b"[REDACTED-NTLM]"),
    # AWS access key
    (re.compile(rb"\bAKIA[0-9A-Z]{16}\b"), b"[REDACTED-AWS-AK]"),
    (
        re.compile(rb"\b(?:ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[0-9A-Z]{16}\b"),
        b"[REDACTED-AWS-AK]",
    ),
    # JWT
    (re.compile(rb"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), b"[REDACTED-JWT]"),
    # GitHub tokens (crypto-02): variable-length body covering the full modern
    # prefix set. Classic PATs are 36 chars but fine-grained PATs
    # (``github_pat_...``) are longer and contain underscores, so the old
    # ``ghp_{36}`` anchor missed them. {20,255} + ``_`` in the class catches
    # classic, fine-grained, OAuth, app and refresh tokens.
    (
        re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,255}\b"),
        b"[REDACTED-GHTOKEN]",
    ),
    # JSON / quoted credential keys (crypto-01): ``"password": "hunter2"`` and
    # friends. The bare ``key=value`` form below does NOT match these because
    # the char after the key word is a quote, not a separator. This pattern
    # tolerates an optional quote around the key and a quoted value, so the
    # most common machine-readable shape (JSON tool output) is redacted.
    # Replacement keeps the (quoted) key and stubs the value.
    #
    # cryptoN-04: the value class is escape-aware — ``(?:\\.|[^"\\])*`` consumes
    # any backslash-escaped char (incl. ``\"``) as a unit, so a JSON value like
    # ``"pa\"ss"`` is matched whole and the entire secret is redacted instead of
    # leaking the tail after the inner quote. Each input char belongs to exactly
    # one alternative (``\\.`` requires a leading backslash; ``[^"\\]`` forbids
    # it), so the quantifier is unambiguous and linear — no catastrophic
    # backtracking / ReDoS.
    (
        re.compile(
            rb"(?i)([\"']?(?:password|passwd|pwd|secret|token|api[_-]?key|nt_hash|ntlm|"
            rb"aes256_key|private_key|access[_-]?key|client[_-]?secret)[\"']?\s*[:=]\s*)"
            rb"\"(?:\\.|[^\"\\])*\""
        ),
        b"\\1\"[REDACTED]\"",
    ),
    (
        re.compile(
            rb"(?i)([\"']?(?:password|passwd|pwd|secret|token|api[_-]?key|nt_hash|ntlm|"
            rb"aes256_key|private_key|access[_-]?key|client[_-]?secret)[\"']?\s*[:=]\s*)"
            rb"'(?:\\.|[^'\\])*'"
        ),
        b"\\1'[REDACTED]'",
    ),
    # password=, passwd=, pwd= bare forms (case-insensitive). Runs after the
    # quoted forms so JSON values are handled by the patterns above.
    (
        re.compile(
            rb"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|nt_hash|ntlm|"
            rb"aes256_key|access[_-]?key|client[_-]?secret)\s*[:=]\s*\S+"
        ),
        b"\\1=[REDACTED]",
    ),
    # Authorization: Bearer ...
    (re.compile(rb"(?i)Authorization:\s*Bearer\s+\S+"), b"Authorization: Bearer [REDACTED]"),
    # Authorization: Basic ...
    (re.compile(rb"(?i)Authorization:\s*Basic\s+\S+"), b"Authorization: Basic [REDACTED]"),
    # PEM private key body (single-line marker; not robust to multi-line but cheap)
    (
        re.compile(
            rb"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]{20,5000}?-----END [A-Z ]+PRIVATE KEY-----"
        ),
        b"[REDACTED-PRIVATE-KEY]",
    ),
]


def scrub(data: bytes) -> tuple[bytes, bool]:
    """Apply all redaction patterns; return (scrubbed_bytes, was_modified)."""
    if not data:
        return data, False
    modified = False
    out = data
    for pat, repl in _PATTERNS:
        new, n = pat.subn(repl, out)
        if n > 0:
            modified = True
            out = new
    return out, modified
