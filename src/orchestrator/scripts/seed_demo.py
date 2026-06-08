"""Seed a demo engagement against the bundled DVWA target."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import structlog

from orchestrator.db import create_all, init_engine, session_scope
from orchestrator.models import EngagementRow

log = structlog.get_logger(__name__)


async def main() -> None:
    init_engine()
    await create_all()
    auth_blob = "DEMO ENGAGEMENT AUTHORIZATION — local DVWA target only".encode()
    sha = hashlib.sha256(auth_blob).hexdigest()
    now = datetime.now(UTC)
    async with session_scope() as s:
        row = EngagementRow(
            name="demo: DVWA",
            scope={
                "cidrs": ["172.30.0.0/24"],
                "domains": ["dvwa.local"],
                "urls": ["http://dvwa.local"],
                "excluded_cidrs": [],
                "excluded_domains": [],
                "excluded_paths": [],
            },
            objectives=["xss_demo", "sqli_demo"],
            mode="guided",
            time_window_start=now,
            time_window_end=now + timedelta(hours=2),
            token_budget=500_000,
            wallclock_hours=2,
            authorization_artifact_sha256=sha,
            status="draft",
        )
        s.add(row)
        await s.flush()
        log.info("seed.created", engagement_id=str(row.id))
        print(f"Demo engagement id: {row.id}")


if __name__ == "__main__":
    asyncio.run(main())
