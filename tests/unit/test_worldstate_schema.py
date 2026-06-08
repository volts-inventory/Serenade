from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from worldstate.schema import (
    Credential,
    EdgeKind,
    Engagement,
    EngagementMode,
    Finding,
    Host,
    Scope,
    Service,
    Severity,
    Stage,
    TimeWindow,
)


def _engagement_id():
    return uuid4()


def test_host_minimal_fields():
    eid = _engagement_id()
    h = Host(engagement_id=eid, ip="10.0.0.5", provenance=[uuid4()])
    assert h.kind == "Host"
    assert h.confidence == 1.0
    assert h.os_family == "unknown"
    assert h.engagement_id == eid


def test_service_requires_port_in_range():
    eid = _engagement_id()
    hid = uuid4()
    with pytest.raises(ValidationError):
        Service(engagement_id=eid, host_id=hid, port=70000, proto="tcp", provenance=[uuid4()])


def test_finding_severity_enum():
    eid = _engagement_id()
    f = Finding(
        engagement_id=eid,
        title="anonymous SMB",
        description="null session allowed",
        severity=Severity.MEDIUM,
        provenance=[uuid4()],
    )
    assert f.severity == Severity.MEDIUM


def test_engagement_serializes():
    eid = _engagement_id()
    e = Engagement(
        engagement_id=eid,
        id=eid,
        name="demo",
        scope=Scope(cidrs=["10.10.10.0/24"], domains=["lab.local"]),
        time_window=TimeWindow(
            start=datetime.now(UTC),
            end=datetime.now(UTC) + timedelta(hours=8),
        ),
        mode=EngagementMode.GUIDED,
        provenance=[uuid4()],
    )
    blob = e.model_dump(mode="json")
    assert blob["scope"]["cidrs"] == ["10.10.10.0/24"]
    assert blob["mode"] == "guided"


def test_credential_provenance_carried():
    eid = _engagement_id()
    tr = uuid4()
    c = Credential(engagement_id=eid, username="alice", source="smb_null", provenance=[tr])
    assert tr in c.provenance


def test_stage_enum_values():
    assert Stage.RECON.value == "recon"
    assert Stage.LATERAL.value == "lateral"


# Repo root: tests/unit/<file>.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# EdgeKind members that are part of the documented worldstate schema but have
# no production call sites yet (planned for later phases / out-of-band write
# paths). They are intentionally reserved; track them here so this test still
# catches accidental additions of new dead members. If you wire one up,
# remove it from this set so the meta-test starts enforcing it.
_RESERVED_EDGE_KINDS: frozenset[str] = frozenset(
    {
        "BELONGS_TO",
        "MEMBER_OF",
        "EVIDENCED_BY",
        "LEADS_TO",
        "PIVOTED_VIA",
        "YIELDED",
        "DISCOVERED_BY",
    }
)


@pytest.mark.parametrize("kind", list(EdgeKind), ids=lambda k: k.name)
def test_edge_kind_is_referenced(kind: EdgeKind) -> None:
    """Every non-reserved EdgeKind member must be referenced by a call site.

    Guards against dead enum entries (e.g. the obsolete ``HOSTS`` edge removed
    in K6). The defining ``EdgeKind.<NAME> = ...`` line in ``schema.py`` is
    excluded so that the member must have an external consumer to count.
    Members in ``_RESERVED_EDGE_KINDS`` are skipped — see that set's docstring.
    """
    if kind.name in _RESERVED_EDGE_KINDS:
        pytest.skip(f"EdgeKind.{kind.name} is a documented reserved member")
    needle = f"EdgeKind.{kind.name}"
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            needle,
            str(_REPO_ROOT / "src"),
            str(_REPO_ROOT / "tests"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Exclude the enum definition line itself; it isn't a real consumer.
    schema_path = str(_REPO_ROOT / "src" / "worldstate" / "schema.py")
    hits = [
        line
        for line in result.stdout.splitlines()
        if line and not (line.startswith(schema_path) and f"{kind.name} =" in line)
    ]
    assert hits, f"EdgeKind.{kind.name} has no references outside its definition"


def test_no_hosts_edge_kind() -> None:
    """Regression guard: the obsolete ``HOSTS`` edge kind must stay removed.

    Host->WebApp edges aren't modeled (WebApp is endpoint-anchored); the
    Host<->Service edge is ``RUNS_ON``. K6 removed ``HOSTS`` as dead weight.
    """
    assert "HOSTS" not in {k.name for k in EdgeKind}
