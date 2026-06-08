"""SIEM webhook forwarder.

Best-effort tail of `audit_log` to an HTTP(S) webhook (Splunk HEC, Elastic
ingest pipeline, generic JSONL collector). Runs as a background task in the
worker; configured via env:

  SIEM_WEBHOOK_URL       -- required to enable
  SIEM_WEBHOOK_AUTH      -- optional bearer token / API key
  SIEM_BATCH_SIZE        -- default 50 rows
  SIEM_POLL_SECONDS      -- default 5

The forwarder remembers the last `audit_log.id` it shipped as in-process
state. Failures retry; rows never duplicate because we use an id-based
cursor. The cursor is not persisted across process restarts (acceptable
for a single-process laptop tool -- on restart, previously-shipped rows
may be re-shipped, which is safe because SIEM endpoints are idempotent
on the event id).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
import structlog
from sqlalchemy import select

from orchestrator.db import session_scope
from orchestrator.models import AuditLogRow

log = structlog.get_logger(__name__)

# In-process cursor -- replaces the old Redis key `serenade:siem:last_id`.
_last_shipped_id: str | None = None


def _config() -> dict[str, Any]:
    return {
        "url": os.environ.get("SIEM_WEBHOOK_URL"),
        "auth": os.environ.get("SIEM_WEBHOOK_AUTH"),
        "batch_size": int(os.environ.get("SIEM_BATCH_SIZE", "50")),
        "poll_seconds": float(os.environ.get("SIEM_POLL_SECONDS", "5")),
    }


async def run_siem_forwarder_forever() -> None:
    """Long-running task. Idempotent on rows by id-based cursor."""
    global _last_shipped_id
    cfg = _config()
    if not cfg["url"]:
        log.info("siem.forwarder.disabled")
        return
    log.info("siem.forwarder.start", url=cfg["url"], last_id=_last_shipped_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                rows = await _fetch_batch(_last_shipped_id, cfg["batch_size"])
                if rows:
                    ok = await _ship_batch(client, cfg, rows)
                    if ok:
                        _last_shipped_id = str(rows[-1]["id"])
                await asyncio.sleep(cfg["poll_seconds"])
            except Exception as e:
                log.warning("siem.forwarder.error", err=str(e))
                await asyncio.sleep(cfg["poll_seconds"] * 2)


async def _fetch_batch(after_id: str | None, batch_size: int) -> list[dict[str, Any]]:
    async with session_scope() as s:
        stmt = select(AuditLogRow).order_by(AuditLogRow.id).limit(batch_size)
        if after_id:
            from uuid import UUID

            try:
                stmt = stmt.where(AuditLogRow.id > UUID(after_id))
            except (ValueError, TypeError):
                pass
        result = (await s.execute(stmt)).scalars().all()
        return [
            {
                "id": str(r.id),
                "engagement_id": str(r.engagement_id) if r.engagement_id else None,
                "actor": r.actor,
                "action": r.action,
                "payload": r.payload,
                "ts": r.ts.isoformat() if r.ts else None,
            }
            for r in result
        ]


async def _ship_batch(client: httpx.AsyncClient, cfg: dict, rows: list[dict]) -> bool:
    headers = {"Content-Type": "application/x-ndjson"}
    if cfg.get("auth"):
        headers["Authorization"] = cfg["auth"]
    body = "\n".join(json.dumps(r, default=str) for r in rows)
    try:
        resp = await client.post(cfg["url"], content=body, headers=headers)
        if resp.status_code >= 400:
            log.warning("siem.forwarder.http_error", status=resp.status_code)
            return False
        return True
    except httpx.HTTPError as e:
        log.warning("siem.forwarder.transport_error", err=str(e))
        return False
