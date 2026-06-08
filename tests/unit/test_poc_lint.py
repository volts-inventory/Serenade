"""Tests for the PoC AST lint (verification-audit fix #4)."""

from __future__ import annotations

from pathlib import Path

from agents.poc_lint import lint_module


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exploit.py"
    p.write_text(body)
    return p


def test_imports_and_defs_ok(tmp_path: Path):
    src = '''"""docstring"""
from __future__ import annotations
import os
from typing import Any

def build_invocation(params, ctx):
    return None

class Helper:
    pass

async def helper_async():
    pass
'''
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert ok, reason


def test_rejects_top_level_call(tmp_path: Path):
    src = "import os\nos.system('rm -rf /')\n"
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert not ok
    assert "Expr" in reason or "disallowed" in reason


def test_rejects_top_level_assignment(tmp_path: Path):
    src = "BAD = open('/etc/passwd').read()\n"
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert not ok
    assert "Assign" in reason or "disallowed" in reason


def test_rejects_top_level_if(tmp_path: Path):
    src = "if True:\n    print('side effect')\n"
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert not ok
    assert "if" in reason.lower()


def test_rejects_top_level_main_guard(tmp_path: Path):
    src = "if __name__ == '__main__':\n    print('not allowed either')\n"
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert not ok
    assert "if" in reason.lower()


def test_syntax_error_rejected(tmp_path: Path):
    src = "def broken(:\n    pass\n"
    p = _write(tmp_path, src)
    ok, reason = lint_module(p)
    assert not ok
    assert "syntax_error" in reason


def test_existing_template_passes_lint():
    """The shipped content/pocs/_template/exploit.py must pass the lint we just shipped."""
    p = (
        Path(__file__).resolve().parent.parent.parent
        / "content"
        / "pocs"
        / "_template"
        / "exploit.py"
    )
    ok, reason = lint_module(p)
    assert ok, f"_template/exploit.py failed lint: {reason}"


def test_existing_example_passes_lint():
    """The shipped reference PoC must pass."""
    p = (
        Path(__file__).resolve().parent.parent.parent
        / "content"
        / "pocs"
        / "example-banner-check"
        / "exploit.py"
    )
    ok, reason = lint_module(p)
    assert ok, f"example-banner-check/exploit.py failed lint: {reason}"
