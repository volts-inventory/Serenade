"""Importers for prior-engagement data.

Each importer parses a file format (nmap XML, nuclei JSONL, NetExec JSON)
into world-state nodes with `provenance=[import_run_id]`. The import is
recorded as a synthetic ToolRunRow with `tool="import:{kind}"` so the
provenance referential-integrity check passes.
"""

from __future__ import annotations
