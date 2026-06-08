"""PDF rendering for the audit log export.

This module renders the client-deliverable shape: a paginated PDF of the
audit log. There are now two layered integrity mechanisms:

* **At-rest hash chain (source of truth).** Every ``AuditLogRow`` carries
  a per-engagement ``seq`` / ``prev_hash`` / master-key-HMAC ``row_hash``
  written by ``orchestrator.ledger`` (issue #112, P0 #2). Before
  rendering we call ``ledger.verify_audit_chain`` and stamp the result
  (VERIFIED / TAMPERED at seq N) at the top of the PDF, and surface each
  row's stored ``seq`` / ``prev_hash`` / ``row_hash`` so a downstream
  consumer holding the master key can independently re-verify the chain.
  This is the authoritative tamper-evidence: tampering (edit / delete /
  reorder) breaks the chain and is reported here.

* **Export-time transport MAC.** The per-row + whole-document
  ``SERENADE_AUDIT_EXPORT_SECRET`` HMAC is retained as a transport
  signature so the PDF can be checked offline without the master key.
  It is layered on top of the at-rest chain, not a competing source of
  truth — when the two could disagree, the at-rest chain wins.

Both export-time signatures use the same key — ``SERENADE_AUDIT_EXPORT_SECRET``
(or fall back to ``SERENADE_SESSION_SECRET``). HMAC-SHA256.
"""

from __future__ import annotations

import hmac
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _audit_secret() -> str:
    return (
        os.environ.get("SERENADE_AUDIT_EXPORT_SECRET")
        or os.environ.get("SERENADE_SESSION_SECRET")
        or ""
    )


def _sign(secret: str, data: bytes) -> str:
    if not secret:
        return ""
    return hmac.new(secret.encode("utf-8"), data, sha256).hexdigest()


def _canonical_line(row: dict[str, Any]) -> bytes:
    """sort_keys=True for deterministic canonicalisation across Python
    versions / refactors. Both the JSONL exporter and this PDF renderer use
    the same encoding so the same HMAC verifies either deliverable."""
    return json.dumps(row, sort_keys=True, default=str).encode("utf-8")


async def verify_chain_for_export(engagement_id: str | None) -> dict[str, Any]:
    """Run the at-rest hash-chain verifier and return a render-ready banner.

    Delegates to ``orchestrator.ledger.verify_audit_chain`` (the master-key
    HMAC chain — the source of truth) and shapes the result into the dict
    the template stamps at the top of the PDF.
    """
    from orchestrator import ledger

    result = await ledger.verify_audit_chain(engagement_id)
    if result.ok:
        return {"verified": True, "label": "VERIFIED", "detail": "", "reason": "ok"}
    return {
        "verified": False,
        "label": f"TAMPERED (at seq {result.first_bad_seq})",
        "detail": result.reason,
        "reason": result.reason,
    }


def _canonical_scope(engagement_id: Any) -> str | None:
    """Normalize an engagement id the SAME way the chain canonicalization does
    (``audit_canonical_bytes``: ``str(x) if x else None``), so the header scope
    and a row's ``engagement_id`` compare on equal footing (cryptoN-02)."""
    return str(engagement_id) if engagement_id else None


def verify_rendered_rows(
    rows: list[dict[str, Any]],
    engagement_id: Any = None,
) -> dict[str, Any]:
    """Verify the hash chain over the EXACT rows being rendered (crypto-05).

    The banner must attest to the rows on the page, not to a separately-loaded
    DB chain. We recompute each row's master-key HMAC from its own fields and
    check it against the stored ``row_hash``, that ``prev_hash`` links to the
    predecessor's ``row_hash``, and that ``seq`` is contiguous from 1 with no
    gaps/dups — using the SAME canonicalization the writer/verifier use.

    ``engagement_id`` (when supplied) is the scope stamped into the PDF header;
    every rendered row must belong to it or the page would stamp VERIFIED under
    the wrong header (cryptoN-02). A genuine, internally-consistent chain from a
    DIFFERENT engagement is therefore downgraded to UNVERIFIED.

    Returns a banner dict. A row set that is a filtered/paginated/reordered
    subset, a different engagement, that lacks the chain columns, or that
    carries an un-canonicalizable ``ts`` (cryptoN-03) cannot re-link and is
    stamped PARTIAL/UNVERIFIED — never a misleading VERIFIED.
    """
    from orchestrator import crypto
    from orchestrator.ledger import audit_canonical_bytes

    if not rows:
        # Nothing to attest; don't claim VERIFIED over an empty page.
        return {
            "verified": False,
            "label": "UNVERIFIED",
            "detail": "no rows to verify",
            "reason": "empty",
        }

    def _bad(seq: Any, reason: str) -> dict[str, Any]:
        return {
            "verified": False,
            "label": f"UNVERIFIED (rows do not form an intact chain at seq {seq})",
            "detail": reason,
            "reason": reason,
        }

    header_scope = _canonical_scope(engagement_id)
    expected_seq = 1
    prev_row_hash: str | None = None
    for r in rows:
        seq = r.get("seq")
        row_hash = r.get("row_hash")
        prev_hash = r.get("prev_hash")
        if seq is None or row_hash is None:
            return _bad(seq, "missing chain columns (seq/row_hash) on rendered row")
        # cryptoN-02: bind the rendered rows to the header's engagement. A
        # genuine chain from a different engagement carries valid hashes for
        # ITS scope, so the integrity checks below would pass — only this
        # comparison stops it stamping VERIFIED under the wrong header.
        if _canonical_scope(r.get("engagement_id")) != header_scope:
            return _bad(
                seq,
                "engagement-mismatch (row belongs to a different engagement than the header)",
            )
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            return _bad(seq, "non-integer seq on rendered row")
        if seq_int == expected_seq - 1:
            return _bad(seq_int, "seq-dup")
        if seq_int != expected_seq:
            return _bad(expected_seq, "seq-gap (rows are a subset or reordered)")
        if seq_int == 1:
            if prev_hash is not None:
                return _bad(seq_int, "genesis-prev-hash")
        elif prev_hash != prev_row_hash:
            return _bad(seq_int, "broken-link (rows are a subset or reordered)")

        # cryptoN-03: ``audit_canonical_bytes`` expects a ``datetime`` ``ts``
        # (it reads ``ts.tzinfo``). Rows crossing a JSON/API boundary carry
        # ``ts`` as a string (or other non-datetime). Rather than crash the
        # whole render with an AttributeError, treat any canonicalization
        # failure as a verification failure and degrade to UNVERIFIED.
        try:
            canonical = audit_canonical_bytes(
                seq=seq_int,
                engagement_id=r.get("engagement_id"),
                actor=r.get("actor"),
                action=r.get("action"),
                payload=r.get("payload"),
                ts=r.get("ts"),
                prev_hash=prev_hash,
            )
        except (AttributeError, TypeError, ValueError) as e:
            return _bad(seq_int, f"uncanonicalizable row (bad ts/field type: {e})")
        if not hmac.compare_digest(crypto.audit_hmac(canonical), str(row_hash)):
            return _bad(seq_int, "hash-mismatch (row content does not match its row_hash)")

        prev_row_hash = str(row_hash)
        expected_seq += 1

    return {"verified": True, "label": "VERIFIED", "detail": "", "reason": "ok"}


def render_audit_log_pdf(
    rows: list[dict[str, Any]],
    engagement_id: str | None = None,
    chain_status: dict[str, Any] | None = None,
) -> bytes:
    """Render the audit log to PDF bytes via WeasyPrint.

    crypto-05: the banner attests to the EXACT ``rows`` being rendered, not to
    a separately-loaded DB chain. We always re-verify the supplied ``rows``
    here (:func:`verify_rendered_rows`). A caller-supplied ``chain_status`` is
    only honoured when it agrees with that row-level result; if the
    whole-chain verdict says VERIFIED but the rendered rows are a
    filtered/paginated/reordered subset (or a different engagement), we stamp
    the row-level UNVERIFIED verdict rather than the misleading whole-chain
    one. Each row's stored ``seq`` / ``prev_hash`` / ``row_hash`` is surfaced
    so a holder of the master key can independently re-verify.

    Raises RuntimeError if WeasyPrint isn't available (caller maps to 503).
    """
    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "weasyprint is required for PDF rendering; pip install weasyprint"
        ) from e

    # Verify the rows actually on the page (crypto-05), bound to the engagement
    # named in the header (cryptoN-02).
    rows_status = verify_rendered_rows(rows, engagement_id=engagement_id)
    if chain_status is None:
        # No external verdict supplied: the row-level result IS the banner.
        chain_status = rows_status
    elif chain_status.get("verified") and not rows_status.get("verified"):
        # Whole-chain says VERIFIED but the rendered rows don't re-link:
        # never let the page carry a VERIFIED stamp it can't back up.
        chain_status = rows_status
    elif not chain_status.get("verified") and rows_status.get("verified"):
        # External verdict is TAMPERED for the full chain; the rendered subset
        # is internally consistent but cannot upgrade a TAMPERED chain. Keep
        # the (more alarming) external verdict.
        pass

    secret = _audit_secret()
    # Per-row signatures.
    enriched: list[dict[str, Any]] = []
    doc_canonical_lines: list[bytes] = []
    for r in rows:
        line = _canonical_line(r)
        doc_canonical_lines.append(line)
        sig = _sign(secret, line)
        enriched.append(
            {
                **r,
                "payload_repr": json.dumps(r.get("payload") or {}, default=str, ensure_ascii=False)[
                    :240
                ],
                "sig": sig or "<unsigned: SERENADE_AUDIT_EXPORT_SECRET unset>",
                # At-rest chain columns surfaced for independent re-verification.
                "seq": r.get("seq"),
                "prev_hash": r.get("prev_hash"),
                "row_hash": r.get("row_hash"),
            }
        )

    # Whole-document signature: HMAC over the concatenation of canonical lines.
    doc_payload = b"\n".join(doc_canonical_lines)
    doc_sig = _sign(secret, doc_payload) or "<unsigned: SERENADE_AUDIT_EXPORT_SECRET unset>"

    tpl = _env().get_template("audit_log.html.jinja")
    html = tpl.render(
        rows=enriched,
        engagement_id=engagement_id,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        doc_signature=doc_sig,
        chain_status=chain_status,
    )
    return HTML(string=html).write_pdf()  # type: ignore[no-any-return]
