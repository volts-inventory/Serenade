"""Evidence store -- content-addressed blob storage on local filesystem.

Every tool stdout/stderr, screenshot, captured packet, etc. goes through here.
Findings reference evidence by `tool_run_id` + `blob_id`.

Blobs are stored at ``{data_dir}/evidence/{engagement_id}/{kind}/{blob_id}``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import structlog

from orchestrator.settings import Settings

log = structlog.get_logger(__name__)


class EvidenceStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.evidence_dir = Path(settings.data_dir).expanduser() / "evidence"

    async def ensure_dir(self) -> None:
        """Create the top-level evidence directory (replaces ensure_bucket)."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    # Keep the old name as an alias so any startup code calling
    # ensure_bucket() continues to work without changes.
    ensure_bucket = ensure_dir

    def put(
        self,
        data: bytes,
        *,
        engagement_id: UUID,
        kind: str,
        content_type: str = "application/octet-stream",
        scrub_credentials: bool = True,
    ) -> tuple[UUID, str]:
        """Store a blob; return (blob_id, sha256_hex).

        Scrubs common credential patterns by default. Pass
        ``scrub_credentials=False`` only for evidence that *intentionally*
        stores creds (e.g. operator-attached authorization artifact). The
        scrubbed copy is what's stored; the sha256 is over the scrubbed bytes.
        """
        from orchestrator.evidence_scrub import scrub

        modified = False
        if scrub_credentials:
            data, modified = scrub(data)
        blob_id = uuid4()
        sha256 = hashlib.sha256(data).hexdigest()

        blob_path = self.evidence_dir / str(engagement_id) / kind / str(blob_id)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(data)

        log.info(
            "evidence.put",
            blob_id=str(blob_id),
            path=str(blob_path),
            bytes=len(data),
            sha256=sha256,
            redacted=modified,
        )
        return blob_id, sha256

    def get(self, engagement_id: UUID, kind: str, blob_id: UUID) -> bytes:
        """Read a blob from the local filesystem."""
        blob_path = self.evidence_dir / str(engagement_id) / kind / str(blob_id)
        return blob_path.read_bytes()
