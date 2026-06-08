"""Credential-protection regression tests (tracking issue #112, P0 #1).

Proves the two protections for ``Credential`` secret fields:

  * **At rest** -- secret fields are Fernet-encrypted in the serialized
    graph (the SQLite checkpoint never holds plaintext); the round-trip
    back to an in-memory graph is lossless.
  * **In LLM-facing slices** -- secret fields are replaced with the
    literal ``[ENCRYPTED]`` before reaching a prompt.

The master key is pointed at a tmp dir via the ``DATA_DIR`` env var so no
real ``~/.serenade/.master_key`` is created or read.
"""

from __future__ import annotations

import os
import stat
from uuid import uuid4

import pytest

from orchestrator import crypto
from orchestrator.settings import get_settings
from worldstate.neo4j_repo import CREDENTIAL_SECRET_FIELDS, InMemoryGraphRepo
from worldstate.queries import (
    REDACTED_PLACEHOLDER,
    credentials_with_known_hosts,
    redact_credential_secrets,
)
from worldstate.schema import Credential, Edge, EdgeKind, Host


@pytest.fixture
def crypto_tmp(monkeypatch, tmp_path):
    """Point the master key at a tmp dir and reset all caches."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    crypto.reset_cache()
    yield tmp_path
    get_settings.cache_clear()
    crypto.reset_cache()


def test_master_key_created_with_mode_0600(crypto_tmp) -> None:
    path = crypto.master_key_path()
    assert not path.exists()

    key = crypto.load_or_create_key()
    assert isinstance(key, bytes) and key
    assert path.exists()

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    # Idempotent: second call reads the same key, doesn't regenerate.
    assert crypto.load_or_create_key() == key


def test_encrypt_decrypt_round_trip(crypto_tmp) -> None:
    secret = "S3cr3t-P@ssw0rd!"
    token = crypto.encrypt(secret)

    assert token is not None
    assert token != secret
    assert token.startswith("gAAAAA")  # Fernet token prefix
    assert crypto.decrypt(token) == secret


def test_encrypt_decrypt_none_passthrough(crypto_tmp) -> None:
    assert crypto.encrypt(None) is None
    assert crypto.decrypt(None) is None


def test_plaintext_with_fernet_prefix_is_still_encrypted(crypto_tmp) -> None:
    # Regression (issue #112 critic): a plaintext that happens to start with
    # the Fernet prefix must NOT be mistaken for ciphertext and stored in
    # cleartext. encrypt() always encrypts.
    sneaky = "gAAAAAlooks-like-a-token-but-is-a-password"
    token = crypto.encrypt(sneaky)
    assert token != sneaky
    assert crypto.decrypt(token) == sneaky


def test_decrypt_non_token_passes_through(crypto_tmp) -> None:
    # Legacy / non-Fernet values round-trip instead of raising InvalidToken.
    assert crypto.decrypt("not-a-fernet-token") == "not-a-fernet-token"
    assert crypto.decrypt("gAAAAAnot-really-valid") == "gAAAAAnot-really-valid"


def test_fernet_cache_keyed_by_data_dir(monkeypatch, tmp_path) -> None:
    # Regression (issue #112 critic): the Fernet cache must key off the
    # resolved master-key path, so a different data_dir uses a different key
    # rather than reusing the first one ever loaded.
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("DATA_DIR", str(dir_a))
    get_settings.cache_clear()
    crypto.reset_cache()
    key_a = crypto.load_or_create_key()

    monkeypatch.setenv("DATA_DIR", str(dir_b))
    get_settings.cache_clear()
    # Deliberately do NOT reset the Fernet cache -- the path key must do it.
    key_b = crypto.load_or_create_key()

    assert key_a != key_b
    assert (dir_a / ".master_key").exists()
    assert (dir_b / ".master_key").exists()
    get_settings.cache_clear()
    crypto.reset_cache()


def test_secret_fields_derived_from_schema_marker() -> None:
    # The encrypt/redact field list is the single source of truth derived
    # from Credential fields marked secret=True; a new secret field added
    # with the marker is covered automatically.
    assert set(CREDENTIAL_SECRET_FIELDS) == {
        "password",
        "nt_hash",
        "aes256_key",
        "ssh_private_key_ref",
    }
    for name in CREDENTIAL_SECRET_FIELDS:
        extra = Credential.model_fields[name].json_schema_extra
        assert isinstance(extra, dict) and extra.get("secret") is True


def _make_credential(eid, password=None, nt_hash=None):
    return Credential(
        id=uuid4(),
        engagement_id=eid,
        provenance=[uuid4()],
        username="administrator",
        domain="corp.example.com",
        source="secretsdump",
        password=password,
        nt_hash=nt_hash,
        valid=True,
    )


@pytest.mark.asyncio
async def test_at_rest_encryption_round_trip(crypto_tmp) -> None:
    eid = uuid4()
    password = "TopSecretPlaintext123"
    nt_hash = "aad3b435b51404eeaad3b435b51404ee"
    repo = InMemoryGraphRepo()
    cred = _make_credential(eid, password=password, nt_hash=nt_hash)
    await repo.upsert_node(cred)

    serialized = repo.to_json()

    # On-disk JSON must NOT leak plaintext secrets...
    assert password not in serialized
    assert nt_hash not in serialized
    # ...and must contain Fernet ciphertext.
    assert "gAAAAA" in serialized

    # The live in-memory node was NOT mutated by serialization.
    live = repo.G.nodes[str(cred.id)]
    assert live["password"] == password
    assert live["nt_hash"] == nt_hash

    # Round-trip back: rehydrated graph holds plaintext again.
    rehydrated = InMemoryGraphRepo.from_json(serialized)
    node = rehydrated.G.nodes[str(cred.id)]
    assert node["password"] == password
    assert node["nt_hash"] == nt_hash


@pytest.mark.asyncio
async def test_at_rest_none_secrets_stay_none(crypto_tmp) -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    cred = _make_credential(eid, password="onlypassword", nt_hash=None)
    await repo.upsert_node(cred)

    rehydrated = InMemoryGraphRepo.from_json(repo.to_json())
    node = rehydrated.G.nodes[str(cred.id)]
    assert node["password"] == "onlypassword"
    assert node["nt_hash"] is None
    assert node["aes256_key"] is None


def test_redact_credential_secrets_helper() -> None:
    node = {
        "username": "admin",
        "domain": "corp",
        "source": "kerberoast",
        "password": "plaintext-pw",
        "nt_hash": "deadbeef",
        "aes256_key": None,
        "ssh_private_key_ref": "blob://ref",
        "valid": True,
    }
    redacted = redact_credential_secrets(node)

    # Secrets present -> [ENCRYPTED]; None preserved.
    for field in CREDENTIAL_SECRET_FIELDS:
        if node[field] is not None:
            assert redacted[field] == REDACTED_PLACEHOLDER
        else:
            assert redacted[field] is None
    # Plaintext never present in the redacted copy.
    assert "plaintext-pw" not in redacted.values()
    assert "deadbeef" not in redacted.values()
    # Non-secret fields untouched; original not mutated.
    assert redacted["username"] == "admin"
    assert node["password"] == "plaintext-pw"


@pytest.mark.asyncio
async def test_credentials_with_known_hosts_redacts_secrets(crypto_tmp) -> None:
    eid = uuid4()
    repo = InMemoryGraphRepo()
    password = "DoNotLeakMe"
    nt_hash = "ffffffffffffffffffffffffffffffff"
    cred = _make_credential(eid, password=password, nt_hash=nt_hash)
    host = Host(
        id=uuid4(),
        engagement_id=eid,
        provenance=[uuid4()],
        ip="10.0.0.5",
    )
    await repo.upsert_node(cred)
    await repo.upsert_node(host)
    await repo.upsert_edge(
        Edge(
            kind=EdgeKind.HAS_SESSION_ON,
            src_id=cred.id,
            dst_id=host.id,
            engagement_id=eid,
            provenance=[uuid4()],
        )
    )

    result = await credentials_with_known_hosts(repo, eid)
    assert len(result) == 1
    cred_out = result[0]["credential"]

    assert cred_out["password"] == REDACTED_PLACEHOLDER
    assert cred_out["nt_hash"] == REDACTED_PLACEHOLDER
    # Plaintext never surfaces in the slice.
    assert password not in str(result)
    assert nt_hash not in str(result)
    # Non-secret fields still present for the planner.
    assert cred_out["username"] == "administrator"
