from __future__ import annotations

import time

from orchestrator.evidence_scrub import scrub


def test_scrubs_ntlm_hash():
    raw = b"aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0 user1"
    out, modified = scrub(raw)
    assert modified
    assert b"[REDACTED-NTLM]" in out
    assert b"aad3b435" not in out


def test_scrubs_password_kv():
    raw = b"INFO user=alice password=Sup3rS3cret123 attempting auth"
    out, modified = scrub(raw)
    assert modified
    assert b"Sup3rS3cret123" not in out
    assert b"password=[REDACTED]" in out


def test_scrubs_authorization_bearer():
    raw = b"GET /api HTTP/1.1\r\nAuthorization: Bearer eyJ...long.token.here.x\r\n"
    out, modified = scrub(raw)
    assert modified
    assert b"Authorization: Bearer [REDACTED]" in out
    assert b"eyJ...long" not in out


def test_scrubs_aws_access_key():
    raw = b"export AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF"
    out, modified = scrub(raw)
    assert modified
    assert b"AKIA1234567890ABCDEF" not in out


def test_scrubs_jwt_standalone():
    raw = b"raw: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36 next"
    out, modified = scrub(raw)
    assert modified
    assert b"[REDACTED-JWT]" in out
    assert b"eyJhbGciOi" not in out


def test_scrubs_jwt_in_token_kv():
    """token=<jwt> hits the token= pattern first; either redaction is fine."""
    raw = b"token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36"
    out, modified = scrub(raw)
    assert modified
    assert b"eyJhbGciOi" not in out


def test_scrubs_pem_private_key():
    raw = (
        b"prelude\n"
        b"-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxxxxxxxxxxxxxxxxxxxxxxx\n"
        b"-----END RSA PRIVATE KEY-----\nepilogue"
    )
    out, modified = scrub(raw)
    assert modified
    assert b"[REDACTED-PRIVATE-KEY]" in out
    assert b"MIIEpAIBAAK" not in out


def test_unchanged_when_no_secrets():
    raw = b"nothing sensitive here, just plain log output"
    out, modified = scrub(raw)
    assert not modified
    assert out == raw


def test_empty_input_safe():
    out, modified = scrub(b"")
    assert out == b""
    assert not modified


def test_scrubs_escaped_quote_in_json_value_double():
    """cryptoN-04: a JSON value containing a backslash-escaped quote
    (``"pa\\"ss"``) must be redacted whole, with no cleartext tail surviving."""
    raw = rb'{"password": "pa\"ss"}'
    out, modified = scrub(raw)
    assert modified
    assert b'"password": "[REDACTED]"' in out
    # No fragment of the secret may survive (the old bug left ``ss`` behind).
    # The whole value (incl. the escaped quote) is gone -> nothing but the
    # redacted key/value remains.
    assert out == b'{"password": "[REDACTED]"}'


def test_scrubs_escaped_quote_in_json_value_single():
    raw = rb"{'secret': 'pa\'ss'}"
    out, modified = scrub(raw)
    assert modified
    assert out == b"{'secret': '[REDACTED]'}"


def test_scrubs_multiple_escapes_in_value():
    raw = rb'"api_key": "a\"b\"c\"d"'
    out, modified = scrub(raw)
    assert modified
    # The entire escaped-quote-laden value collapses to the single stub.
    assert out == b'"api_key": "[REDACTED]"'


def test_escape_aware_value_pattern_no_redos():
    """cryptoN-04: confirm the escape-aware value class is linear-time and does
    not catastrophically backtrack on an adversarial non-terminating value."""
    # A long run of escaped quotes with no closing delimiter is the classic
    # ReDoS trigger for naive escape patterns. The bounded, unambiguous class
    # must handle it in well under a second.
    raw = b'"password": "' + (rb"\"" * 20000)
    start = time.monotonic()
    scrub(raw)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"scrub took {elapsed:.3f}s -> possible ReDoS"
