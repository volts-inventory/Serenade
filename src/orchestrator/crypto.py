"""Credential-secret encryption and the per-install master key.

Credential secret fields (passwords, NT hashes, AES keys, SSH key refs) are
Fernet-encrypted whenever the world-state graph is serialized
(``InMemoryGraphRepo.to_json``) and decrypted on the way back in
(``from_json``). The live in-memory graph stays plaintext (single-process,
trusted RAM) so tools can use credential literals; only the *serialized*
copy carries ciphertext.

This is now load-bearing: the world-state graph is snapshotted to the
``world_state_snapshots`` table after agent steps and rehydrated on resume
(``orchestrator.world_state_store``). Two layers apply at rest:
``world_state_store`` Fernet-encrypts the WHOLE snapshot blob (so secrets in
ANY node/field -- Finding evidence, Session fields, raw excerpts -- are
covered), and ``to_json`` additionally field-encrypts Credential secret
fields so the serialized graph is credential-safe on any path. The separate
LLM exposure -- secrets reaching the model in prompt slices -- is closed by
slice redaction in ``worldstate.queries``.

The Fernet key lives at ``<data_dir>/.master_key`` (default
``~/.serenade/.master_key``) with mode 0600, generated on first use. The
same key also backs the audit-log HMAC chain.
"""

from __future__ import annotations

import hmac
import os
from hashlib import sha256
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

# Fernet tokens are url-safe base64 starting with this prefix (version byte
# 0x80 -> "gAAAAA"). Exposed only as a test/inspection convenience -- it is
# deliberately NOT used to decide whether to encrypt, because a plaintext
# value can legitimately start with the same bytes (which previously caused
# such values to be stored in cleartext).
_FERNET_PREFIX = "gAAAAA"

# Cache of Fernet instances keyed by the resolved master-key path, so
# repeated calls don't re-read the key file -- and so a change of
# ``data_dir`` (tests, multi-engagement) uses the right key instead of
# reusing the first one ever loaded.
_fernet_by_path: dict[str, Fernet] = {}


def master_key_path() -> Path:
    """Return the path to the master key file (``<data_dir>/.master_key``)."""
    # Imported lazily so a test monkeypatch of get_settings / env is honoured
    # and so importing this module doesn't force settings resolution.
    from orchestrator.settings import get_settings

    return Path(get_settings().data_dir).expanduser() / ".master_key"


def load_or_create_key() -> bytes:
    """Load the master key, generating it (mode 0600) if absent."""
    path = master_key_path()
    if path.exists():
        return path.read_bytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # O_EXCL so we fail rather than clobber if a racing process created it;
    # 0600 so only the owner can read the key.
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def audit_hmac(canonical: bytes) -> str:
    """HMAC-SHA256 of ``canonical`` keyed off the raw master key.

    This is the at-rest integrity primitive for the audit-log hash chain.
    It uses the *raw* master-key bytes directly (NOT a Fernet wrapper) so
    the keyed digest is a plain ``hmac.new(key, msg, sha256).hexdigest()``
    that any holder of the master key can reproduce. The key is read
    through :func:`load_or_create_key`, which carries its own cache; this
    helper adds no cache of its own.
    """
    return hmac.new(load_or_create_key(), canonical, sha256).hexdigest()


def _get_fernet() -> Fernet:
    key_path = str(master_key_path())
    fernet = _fernet_by_path.get(key_path)
    if fernet is None:
        fernet = Fernet(load_or_create_key())
        _fernet_by_path[key_path] = fernet
    return fernet


def reset_cache() -> None:
    """Clear the cached Fernet instances (test helper)."""
    _fernet_by_path.clear()


def is_encrypted(value: object) -> bool:
    """Best-effort check whether ``value`` looks like a Fernet token.

    For inspection/tests only -- never used to gate encryption (see module
    note on the prefix-collision hazard).
    """
    return isinstance(value, str) and value.startswith(_FERNET_PREFIX)


def encrypt(plaintext: str | None) -> str | None:
    """Encrypt ``plaintext`` to a Fernet token string. ``None`` passes through.

    Always encrypts -- callers (``to_json``) operate on the live in-memory
    graph whose attrs are guaranteed plaintext, so there is no double-encrypt
    path to guard against, and we must never skip encryption based on the
    value's content.
    """
    if plaintext is None:
        return None
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt(token: str | None) -> str | None:
    """Decrypt a Fernet token back to plaintext. ``None`` passes through.

    A value that is not a valid Fernet token (e.g. legacy plaintext from a
    pre-encryption serialization) is returned unchanged rather than raising,
    so an old blob round-trips instead of crashing.
    """
    if token is None:
        return None
    try:
        plaintext = _get_fernet().decrypt(token.encode("utf-8"))
    except InvalidToken:
        return token
    return plaintext.decode("utf-8")
