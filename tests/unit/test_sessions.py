"""Unit tests for the SessionManager primitives (Phase 18, Tier B #7)."""

from __future__ import annotations

from datetime import UTC

import pytest

from sandbox.sessions import SessionManager, _build_session_run_argv
from sandbox.network import engagement_network_name


def test_ssh_command_includes_strict_host_key_off():
    """Pentest targets won't be in known_hosts; the manager must not block."""
    mgr = SessionManager()
    argv = mgr._ssh_command("10.0.0.5", 22, "alice", "secret", "whoami")
    s = " ".join(argv)
    assert "StrictHostKeyChecking=no" in s
    assert "UserKnownHostsFile=/dev/null" in s
    assert "alice@10.0.0.5" in argv
    assert "whoami" in argv


def test_ssh_command_uses_sshpass_when_password_set():
    mgr = SessionManager()
    argv = mgr._ssh_command("10.0.0.5", 22, "alice", "p@ss", "whoami")
    assert argv[0] == "sshpass"
    assert "p@ss" in argv


def test_ssh_command_omits_sshpass_when_key_only():
    mgr = SessionManager()
    argv = mgr._ssh_command("10.0.0.5", 22, "alice", None, "whoami")
    assert argv[0] == "ssh"
    assert "sshpass" not in argv
    assert "BatchMode=yes" in " ".join(argv)


def test_ssh_command_respects_custom_port():
    mgr = SessionManager()
    argv = mgr._ssh_command("10.0.0.5", 2222, "alice", None, "id")
    assert "-p" in argv
    assert "2222" in argv


# --- sandbox-10: a session container ALWAYS attaches an explicit network ---


def test_session_run_argv_always_has_network():
    from uuid import uuid4

    eid = uuid4()
    net = engagement_network_name(eid)
    argv = _build_session_run_argv(
        container_name="serenade-ses-x",
        env_args=["-e", f"ENGAGEMENT_ID={eid}"],
        network=net,
        image="serenade/sandbox:dev",
    )
    assert "--network" in argv
    assert argv[argv.index("--network") + 1] == net
    # Hardening preserved.
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--cap-add=NET_RAW" in argv


def test_session_run_argv_rejects_empty_network():
    """Defence-in-depth: no code path may spawn a session on the default bridge."""
    with pytest.raises(ValueError):
        _build_session_run_argv(
            container_name="c",
            env_args=[],
            network="",
            image="img",
        )


def test_session_manager_lock_dict_is_plain_dict():
    # async-10: per-session lock dict is bounded (popped on close), not a
    # defaultdict.
    mgr = SessionManager()
    assert not hasattr(mgr._locks, "default_factory")


@pytest.mark.asyncio
async def test_exec_unknown_session_raises():
    from uuid import uuid4

    mgr = SessionManager()
    with pytest.raises(KeyError):
        await mgr.exec(uuid4(), "whoami")


def test_list_open_filters_by_engagement():
    from datetime import datetime
    from uuid import uuid4

    from sandbox.sessions import SessionHandle

    mgr = SessionManager()
    e1, e2 = uuid4(), uuid4()
    s1 = SessionHandle(
        id=uuid4(),
        container_name="c1",
        type="ssh",
        target_host="x",
        engagement_id=e1,
        opened_at=datetime.now(UTC),
    )
    s2 = SessionHandle(
        id=uuid4(),
        container_name="c2",
        type="ssh",
        target_host="y",
        engagement_id=e2,
        opened_at=datetime.now(UTC),
    )
    mgr._handles[s1.id] = s1
    mgr._handles[s2.id] = s2
    e1_only = mgr.list_open(e1)
    assert len(e1_only) == 1
    assert e1_only[0].container_name == "c1"
    assert len(mgr.list_open()) == 2
