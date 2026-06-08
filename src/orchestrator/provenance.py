"""Provenance validator for InMemoryGraphRepo.

Every WorldStateNode's `provenance` is a list of `tool_run_id`s. Without a
backing check, a fabricated UUID would satisfy the format-level validator;
this module closes that loop by verifying each UUID exists in the
`tool_runs` table for the given engagement.

The same plumbing backs `EvidenceRef.tool_run_id` integrity: the report
builder can call `validate_evidence_refs()` before rendering.

Excerpt verification — HONEST-ASSURANCE NOTE (ws4-01)
-----------------------------------------------------
`verify_evidence_excerpts` / `verify_excerpt_against_blob` only EFFECTIVELY
catch a fabricated excerpt when the cited blob is fetchable, and a blob is only
fetchable when the producing `EvidenceRef` carries a `blob_id`. TODAY no
production producer sets `blob_id`, so for every real ref the blob is `None`,
the excerpt cannot be checked, and the ref is KEPT (never dropped — that was the
ws3-01 regression) and stamped `excerpt_unverified` by the caller. The check
therefore only actively engages for refs that cite a fetchable blob (tests / a
future blob-carrying producer). Making the "excerpt MUST come verbatim from
blob" contract actively enforced requires producers to populate `blob_id`
(tracked follow-up). Until then callers MUST present such excerpts as
unverified and MUST NOT imply they were blob-confirmed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select

from orchestrator.db import session_scope
from orchestrator.models import ToolRunRow

log = structlog.get_logger(__name__)


async def postgres_provenance_validator(engagement_id: UUID, tool_run_ids: list[UUID]) -> bool:
    """Return True iff every `tool_run_id` exists for this `engagement_id` in the database.

    Fails CLOSED (ws-03): if provenance cannot be verified — empty input or a
    DB error — the node is rejected rather than admitted. The previous
    fail-open behaviour opened a window during a transient DB outage where a
    fabricated `tool_run_id` (satisfying only the format-level "non-empty list"
    check) could be persisted, bypassing the "no findings without an evidence
    chain" invariant precisely when the same outage stopped `record_tool_run`
    from durably writing the cited id. Availability is preserved at a coarser
    layer (retry / quarantine) rather than by admitting unverifiable nodes.

    Note: the function name references Postgres for backwards compatibility
    with callers; the implementation is pure SQLAlchemy and works with any
    backend (currently SQLite).
    """
    if not tool_run_ids:
        return False
    try:
        async with session_scope() as s:
            stmt = select(ToolRunRow.id).where(
                ToolRunRow.engagement_id == engagement_id,
                ToolRunRow.id.in_(tool_run_ids),
            )
            rows = (await s.execute(stmt)).scalars().all()
            # The GUID column reads back as canonical ``str`` (see models.GUID),
            # but callers pass ``UUID`` objects. Normalise BOTH sides to ``str``
            # before the set difference, otherwise it is never empty and the
            # validator fails closed on every node — a latent bug unmasked once
            # the db-01 fix made tool_runs actually persist.
            found = {str(row) for row in rows}
            missing = {str(t) for t in tool_run_ids} - found
            if missing:
                log.warning(
                    "provenance.unknown_tool_runs",
                    engagement_id=str(engagement_id),
                    missing=[str(m) for m in missing],
                )
                return False
            return True
    except Exception as e:
        # Fail CLOSED (ws-03): an unverifiable provenance check rejects the
        # node. Admitting it would let a fabricated tool_run_id slip in during
        # the exact window the DB is down.
        log.warning(
            "provenance.validator_error",
            err=str(e),
            advice="operator_notify; provenance unverifiable -> node REJECTED (fail-closed)",
        )
        return False


async def validate_evidence_refs(engagement_id: UUID, tool_run_ids: list[UUID]) -> list[UUID]:
    """Return the subset of tool_run_ids that don't exist for this engagement.

    Callers (e.g. `ReportAgent`) decide what to do (drop the evidence ref,
    escalate to operator, etc.)."""
    if not tool_run_ids:
        return []
    try:
        async with session_scope() as s:
            stmt = select(ToolRunRow.id).where(
                ToolRunRow.engagement_id == engagement_id,
                ToolRunRow.id.in_(tool_run_ids),
            )
            # Normalise to ``str``: GUID columns read back as canonical strings
            # while callers pass ``UUID`` objects (see postgres_provenance_validator).
            found = {str(row) for row in (await s.execute(stmt)).scalars().all()}
            return [t for t in tool_run_ids if str(t) not in found]
    except Exception as e:
        log.warning("evidence.validate_error", err=str(e))
        return []


def verify_excerpt_against_blob(
    excerpt: str | None,
    blob: bytes | None,
    byte_offset: int | None = None,
) -> bool:
    """Return True unless there is POSITIVE evidence the ``excerpt`` was
    fabricated — i.e. a blob IS available and the excerpt demonstrably does not
    appear in it (ws-08, ws3-01).

    ``EvidenceRef.excerpt`` is documented as "MUST come verbatim from blob",
    but nothing checks it: an agent that cites a valid ``tool_run_id`` could
    attach an arbitrary fabricated quote to a Finding and pass every gate.

    Policy (ws3-01 — must NOT fail closed):
    * a ``None``/empty excerpt is trivially "verified" (nothing to fabricate);
    * with **no blob available** we cannot disprove the excerpt, so we return
      ``True`` and let the caller keep (optionally flag) the ref — production
      ``EvidenceRef`` producers attach a synthesized/truncated excerpt and
      leave ``blob_id`` unset, so fail-closed here would strip *all* real
      evidence. We only drop on affirmative mismatch, never on absence.
    * when a blob IS present, the excerpt must appear in it; only then can we
      return ``False`` (the excerpt is provably not from the cited blob).

    ``byte_offset`` (ws3-02): when given, we first look at the cited offset
    window, but if the excerpt isn't there we fall back to a whole-blob
    substring search. A correct excerpt cited at a slightly-off offset
    (e.g. the start of its containing line/record) is therefore NOT flagged.

    Comparison is on the UTF-8 (errors-replaced) decoding of the blob so a
    text excerpt matches a binary blob's printable region. Whitespace is not
    normalized — "verbatim" is taken literally.
    """
    if not excerpt:
        return True
    if blob is None:
        # No blob to compare against -> no positive evidence of tampering.
        # Keep the ref (the caller may mark it unverified). Do NOT fail closed.
        return True
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        # Undecodable blob -> can't disprove the excerpt; keep it.
        return True
    if byte_offset is not None and byte_offset >= 0:
        # Compare at the cited offset (decode the slice so multi-byte
        # boundaries don't spuriously fail a text excerpt). ws3-02: if the
        # offset window doesn't contain the excerpt, fall back to a whole-blob
        # substring check so a slightly-off (e.g. line/record-start) offset on
        # an otherwise-correct excerpt isn't flagged as fabricated.
        window = blob[byte_offset : byte_offset + len(excerpt.encode("utf-8")) + 4]
        if excerpt in window.decode("utf-8", errors="replace"):
            return True
        # fall through to whole-blob substring search (tolerant)
    return excerpt in text


async def verify_evidence_excerpts(
    refs: list[Any],
    blob_fetcher: Callable[[Any], Awaitable[bytes | None]],
) -> list[int]:
    """Return the indices of ``refs`` whose ``excerpt`` is PROVABLY fabricated
    — a blob was fetched and the excerpt does not appear in it (ws-08, ws3-01).

    ``refs`` is a list of ``EvidenceRef`` (or dicts with ``excerpt``/
    ``blob_id``/``byte_offset`` keys). ``blob_fetcher`` resolves a ref to its
    raw blob bytes (or ``None`` if unavailable); it is supplied by the caller
    so this module stays free of EvidenceStore / kind-resolution wiring.

    A ref with no ``excerpt`` is fine. A ref whose blob can't be fetched is
    fine too (we cannot disprove it — ws3-01: production producers leave
    ``blob_id`` unset and synthesize/truncate excerpts, so fail-closed here
    would strip all real evidence). Only a ref whose excerpt demonstrably does
    not match a *fetchable* blob is reported, so the caller can drop the
    fabricated quote before the finding is report-eligible.
    """
    bad: list[int] = []
    for i, ref in enumerate(refs):
        if isinstance(ref, dict):
            excerpt = ref.get("excerpt")
            byte_offset = ref.get("byte_offset")
        else:
            excerpt = getattr(ref, "excerpt", None)
            byte_offset = getattr(ref, "byte_offset", None)
        if not excerpt:
            continue
        try:
            blob = await blob_fetcher(ref)
        except Exception as e:
            log.warning("evidence.excerpt_fetch_error", err=str(e), index=i)
            blob = None
        if not verify_excerpt_against_blob(excerpt, blob, byte_offset):
            bad.append(i)
    return bad
