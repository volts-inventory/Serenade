"""Smoke test: every shipped PoC under content/pocs/ loads cleanly into PoCLibrary."""

from __future__ import annotations

from pathlib import Path

from agents.poc_lint import lint_module
from agents.poc_runtime import PoCLibrary
from tools.destructive_scan import scan_destructive

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POCS_DIR = REPO_ROOT / "content" / "pocs"


def test_all_shipped_pocs_pass_lint():
    """Every exploit.py under content/pocs/<id>/ (excluding _template) must pass AST lint."""
    failures: list[tuple[str, str]] = []
    for entry in sorted(POCS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        exploit_py = entry / "exploit.py"
        if not exploit_py.exists():
            continue
        ok, reason = lint_module(exploit_py)
        if not ok:
            failures.append((entry.name, reason))
    assert not failures, f"PoCs failing lint: {failures}"


def test_destructive_pocs_declare_destructive():
    """A shipped PoC whose exploit.py performs destructive operations (file
    removal/overwrite, disk wipe, DoS) MUST declare ``risk.destructive: true``
    so the destructive_opt_in gate covers it — file removal is destructive.
    Guards against an author shipping a data-deleting PoC unflagged."""
    lib = PoCLibrary(POCS_DIR)
    by_id = {p.manifest.id: p for p in lib.all()}
    offenders: list[tuple[str, list[str]]] = []
    for entry in sorted(POCS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        exploit_py = entry / "exploit.py"
        if not exploit_py.exists():
            continue
        hits = scan_destructive(exploit_py.read_text(encoding="utf-8", errors="replace"))
        poc = by_id.get(entry.name)
        if hits and poc is not None and not poc.manifest.risk.destructive:
            offenders.append((entry.name, hits))
    assert not offenders, (
        "PoCs with destructive operations but risk.destructive=false: " + repr(offenders)
    )


def test_library_loads_real_pocs():
    """PoCLibrary loads every shipped non-template PoC; license isn't UNKNOWN."""
    lib = PoCLibrary(POCS_DIR)
    loaded_ids = {p.manifest.id for p in lib.all()}
    # We've shipped these by name:
    expected = {
        "example-banner-check",
        "log4shell-CVE-2021-44228",
        "spring4shell-CVE-2022-22965",
    }
    assert expected.issubset(loaded_ids), (
        f"missing PoCs: {expected - loaded_ids}; library has: {loaded_ids}"
    )
    for poc in lib.all():
        assert poc.manifest.license.spdx != "UNKNOWN"
        assert poc.manifest_sha256, f"{poc.manifest.id} missing sha256"


def test_log4shell_auto_run_is_disabled():
    """Defense-in-depth: real PoCs ship with auto_run.allowed=false."""
    lib = PoCLibrary(POCS_DIR)
    log4shell = lib.get("log4shell-CVE-2021-44228")
    assert log4shell is not None
    assert log4shell.manifest.auto_run.allowed is False
    assert log4shell.manifest.auto_run.requires_approval is True


def test_spring4shell_marked_modifies_target():
    """Spring4Shell is conservative about side effects."""
    lib = PoCLibrary(POCS_DIR)
    s4s = lib.get("spring4shell-CVE-2022-22965")
    assert s4s is not None
    assert s4s.manifest.risk.modifies_target is True
    assert s4s.manifest.risk.modifies_filesystem is True
