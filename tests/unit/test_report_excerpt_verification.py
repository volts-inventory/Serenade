"""Regression test for ws-08 / wsN-03: the report path must verify each
EvidenceRef ``excerpt`` against its backing blob and drop/flag any excerpt it
cannot confirm — a fabricated quote that cites a perfectly valid tool_run_id
must NOT reach the rendered report.

Before this fix ``ReportAgent._validate_evidence_refs`` only checked
tool_run_id existence (ws-03); ``verify_evidence_excerpts`` existed in
provenance.py but was never invoked, so a hallucinated excerpt passed every
gate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agents.report import DraftReport, ReportAgent
from agents.base import StageContext
from worldstate.schema import Stage


def _report_agent() -> ReportAgent:
    return ReportAgent(
        router=MagicMock(),
        repo=MagicMock(),
        registry=MagicMock(),
        policy=MagicMock(),
        sandbox=MagicMock(),
    )


def _ctx(eid) -> StageContext:
    return StageContext(
        engagement_id=eid,
        engagement={"scope": {"cidrs": ["10.0.0.0/16"]}},
        stage=Stage.REPORT,
    )


# The single blob backing every ref in these tests.
_BLOB = b"HTTP/1.1 200 OK\r\nServer: Apache/2.4.52\r\n\r\nadmin panel exposed\r\n"


@pytest.mark.asyncio
async def test_fabricated_excerpt_is_dropped_by_report_path():
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    blob_id = uuid4()
    tr_id = uuid4()

    async def _fetch(ref):
        # Same blob backs both refs, keyed only by presence of blob_id.
        return _BLOB if (ref.get("blob_id") if isinstance(ref, dict) else None) else None

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Exposed admin panel",
        description="...",
        severity="high",
        evidence_refs=[
            # Verbatim excerpt -> survives.
            {
                "tool_run_id": str(tr_id),
                "blob_id": str(blob_id),
                "excerpt": "admin panel exposed",
            },
            # Fabricated excerpt -> must be dropped (not in blob).
            {
                "tool_run_id": str(tr_id),
                "blob_id": str(blob_id),
                "excerpt": "root shell obtained via CVE-2099-0001",
            },
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    # tool_run_id existence check passes (no missing) so the only drop driver
    # is excerpt verification.
    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    excerpts = [r["excerpt"] for r in draft.findings[0].evidence_refs]
    assert "admin panel exposed" in excerpts
    assert "root shell obtained via CVE-2099-0001" not in excerpts
    assert len(draft.findings[0].evidence_refs) == 1


@pytest.mark.asyncio
async def test_excerpt_with_no_blob_is_kept_and_flagged_unverified():
    """ws3-01: a cited excerpt whose blob can't be fetched is UNVERIFIABLE,
    not fabricated. We must NOT drop it (production producers leave blob_id
    unset and synthesize excerpts). Keep it, flag it ``excerpt_unverified``."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    async def _fetch(ref):
        return None  # blob unavailable

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Claim",
        description="...",
        severity="medium",
        evidence_refs=[
            {"tool_run_id": str(uuid4()), "blob_id": str(uuid4()), "excerpt": "anything"}
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    assert len(draft.findings[0].evidence_refs) == 1
    ref = draft.findings[0].evidence_refs[0]
    assert ref["excerpt"] == "anything"
    assert ref.get("excerpt_unverified") is True


@pytest.mark.asyncio
async def test_production_shape_ref_no_blob_id_is_kept():
    """ws3-01 regression: the REAL producer shape — a non-empty synthesized
    excerpt and NO ``blob_id`` at all — must survive the report path. This is
    what web_enum / nuclei importer / exploit / privesc / loot all emit."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    # Default EvidenceStore-backed fetcher: no blob_id -> returns None.
    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Nuclei hit",
        description="...",
        severity="high",
        evidence_refs=[
            # web_enum / nuclei importer shape
            {
                "tool_run_id": str(uuid4()),
                "excerpt": "exposed-panel matched at https://10.0.0.5/admin",
            },
            # exploit / privesc / loot truncated-summary shape
            {
                "tool_run_id": str(uuid4()),
                "excerpt": "uid=0(root) gid=0(root) groups=0(root)"[:500],
            },
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    # Both refs kept (none dropped); both flagged unverified (no blob).
    refs = draft.findings[0].evidence_refs
    assert len(refs) == 2
    assert all(r.get("excerpt_unverified") is True for r in refs)


@pytest.mark.asyncio
async def test_realistic_multi_ref_report_retains_evidence():
    """A realistic multi-finding report (production-shaped refs, no blob_ids)
    must retain ALL its evidence after the report path runs — the ws3-01
    denial-of-evidence regression must not recur."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    from agents.report import DraftFinding

    findings = [
        DraftFinding(
            title="Exposed admin panel",
            description="...",
            severity="high",
            evidence_refs=[
                {"tool_run_id": str(uuid4()), "excerpt": "admin-login matched at https://h/a"},
            ],
        ),
        DraftFinding(
            title="Local privesc",
            description="...",
            severity="critical",
            evidence_refs=[
                {"tool_run_id": str(uuid4()), "excerpt": "sudo: writable script in PATH"},
                {"tool_run_id": str(uuid4()), "excerpt": "uid=0(root)"},
            ],
        ),
        DraftFinding(
            title="Loot",
            description="...",
            severity="medium",
            evidence_refs=[
                {"tool_run_id": str(uuid4()), "excerpt": "AWS_SECRET_ACCESS_KEY=AKIA..."},
            ],
        ),
    ]
    draft = DraftReport(engagement_id=eid, title="t", findings=findings)

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    total = sum(len(f.evidence_refs) for f in draft.findings)
    assert total == 4  # nothing dropped
    for f in draft.findings:
        assert len(f.evidence_refs) >= 1


@pytest.mark.asyncio
async def test_blob_fetched_at_most_once_per_surviving_ref():
    """ws4-02: the blob backing an excerpt-bearing ref must be resolved at most
    ONCE per validation pass — once for verification, then reused (cached) for
    the verified/unverified stamping. Previously it was fetched twice, doubling
    the EvidenceStore instantiation + engagement-dir glob on every report."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    calls: dict[str, int] = {}

    async def _fetch(ref):
        blob_id = ref.get("blob_id") if isinstance(ref, dict) else None
        calls[str(blob_id)] = calls.get(str(blob_id), 0) + 1
        return _BLOB if blob_id else None

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    blob_a = uuid4()
    blob_b = uuid4()
    finding = DraftFinding(
        title="Two backed quotes",
        description="...",
        severity="high",
        evidence_refs=[
            # Verbatim, fetchable blob -> survives, gets verified stamp.
            {"tool_run_id": str(uuid4()), "blob_id": str(blob_a), "excerpt": "admin panel exposed"},
            # No blob_id -> fetcher returns None -> kept + unverified.
            {"tool_run_id": str(uuid4()), "excerpt": "synthesized summary"},
            # Another fetchable blob, verbatim -> survives, verified.
            {"tool_run_id": str(uuid4()), "blob_id": str(blob_b), "excerpt": "Server: Apache/2.4.52"},
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    # Each distinct blob resolved at most once despite being needed by both the
    # verification pass and the stamping loop.
    assert calls.get(str(blob_a), 0) == 1
    assert calls.get(str(blob_b), 0) == 1
    # Nothing dropped; surviving blob-backed refs are verified, no-blob ref is not.
    refs = draft.findings[0].evidence_refs
    assert len(refs) == 3


@pytest.mark.asyncio
async def test_no_blob_ref_kept_marked_unverified_and_never_verified():
    """ws4-01: a no-blob ref (the production shape) must be KEPT, explicitly
    marked ``excerpt_unverified``, and must NOT carry ``excerpt_verified`` — the
    structured output a renderer/SARIF/JSON consumer sees must not imply the
    quote was blob-confirmed."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    async def _fetch(ref):
        return None  # no blob ever (production: no producer sets blob_id)

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Real-world finding",
        description="...",
        severity="high",
        evidence_refs=[
            {"tool_run_id": str(uuid4()), "excerpt": "matched at https://10.0.0.5/admin"},
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    refs = draft.findings[0].evidence_refs
    assert len(refs) == 1  # kept (ws3-01: never dropped on absence)
    ref = refs[0]
    assert ref.get("excerpt_unverified") is True
    # Structured output MUST NOT claim verification for an unconfirmable excerpt.
    assert "excerpt_verified" not in ref
    assert ref.get("excerpt_verified") is None


@pytest.mark.asyncio
async def test_unverifiable_excerpt_cannot_carry_stale_verified_flag():
    """ws4-01: even if an upstream producer (incorrectly) pre-stamped
    ``excerpt_verified`` on a ref whose blob can't be fetched, the report path
    must strip it so the structured output never labels an unverifiable excerpt
    as verified."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    async def _fetch(ref):
        return None  # unverifiable

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Producer lied about verification",
        description="...",
        severity="critical",
        evidence_refs=[
            {
                "tool_run_id": str(uuid4()),
                "excerpt": "hallucinated but flagged verified",
                # A producer that wrongly claims verification:
                "excerpt_verified": True,
            }
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    ref = draft.findings[0].evidence_refs[0]
    # Stale claim stripped; honest unverified marker applied.
    assert "excerpt_verified" not in ref
    assert ref.get("excerpt_unverified") is True


@pytest.mark.asyncio
async def test_blob_confirmed_excerpt_is_marked_verified_not_unverified():
    """A genuinely blob-confirmed excerpt (fetchable blob + verbatim match) is
    the ONLY case that may be labelled ``excerpt_verified`` — and it must not
    also carry the ``excerpt_unverified`` marker."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    async def _fetch(ref):
        return _BLOB if (ref.get("blob_id") if isinstance(ref, dict) else None) else None

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Genuinely backed",
        description="...",
        severity="high",
        evidence_refs=[
            {"tool_run_id": str(uuid4()), "blob_id": str(uuid4()), "excerpt": "admin panel exposed"},
        ],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    ref = draft.findings[0].evidence_refs[0]
    assert ref.get("excerpt_verified") is True
    assert "excerpt_unverified" not in ref


@pytest.mark.asyncio
async def test_ref_without_excerpt_is_kept():
    """A ref that carries no excerpt has nothing to fabricate -> kept."""
    eid = uuid4()
    agent = _report_agent()
    ctx = _ctx(eid)

    async def _fetch(ref):
        return None

    agent._blob_fetcher = _fetch

    from agents.report import DraftFinding

    finding = DraftFinding(
        title="Backed, no quote",
        description="...",
        severity="low",
        evidence_refs=[{"tool_run_id": str(uuid4()), "blob_id": str(uuid4())}],
    )
    draft = DraftReport(engagement_id=eid, title="t", findings=[finding])

    async def _none_missing(eid_, ids):
        return []

    with patch(
        "orchestrator.provenance.validate_evidence_refs",
        side_effect=_none_missing,
    ):
        await agent._validate_evidence_refs(ctx, draft)

    assert len(draft.findings[0].evidence_refs) == 1
