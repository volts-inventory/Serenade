"""AST-level lint for curated PoC modules.

The dispatcher calls `spec.loader.exec_module` on `pocs/<id>/exploit.py`.
Module-level statements run in the worker process, which has the Docker
socket and Anthropic credentials. To prevent a malicious or careless PoC
from running arbitrary code on import, we lint each module and reject
anything other than imports + function/class definitions at the top level.

`PoCLibrary` calls `lint_module()` for every entry at load time; modules that
fail the lint are not registered.
"""

from __future__ import annotations

import ast
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


# Statement nodes that are always safe at module top level:
# imports, function/class definitions, type aliases, docstrings (Expr/Constant str),
# `from __future__ import ...`.
_ALLOWED_TOP_LEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def lint_module(source_path: Path) -> tuple[bool, str]:
    """Return (ok, reason). `ok=True` iff every top-level node is import/def/docstring."""
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"read_failed: {e}"
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as e:
        return False, f"syntax_error: {e.msg} at line {e.lineno}"
    for node in tree.body:
        if isinstance(node, _ALLOWED_TOP_LEVEL):
            continue
        # Module/function docstring: `Expr(Constant(value=str))`
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        # Module-level type alias (PEP 695) — safe pure value.
        if hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            continue
        # Module-level `UPPERCASE = <literal>` constant assignments.
        # Allow only when the LHS is a single Name matching ALL_CAPS_OR_UNDERSCORE
        # and the RHS is a pure literal (Constant/Tuple/List/Dict/Set of constants).
        if isinstance(node, ast.Assign) and _is_safe_const_assign(node):
            continue
        return False, (
            f"disallowed top-level statement {type(node).__name__} at line {node.lineno}: "
            "PoC modules must contain only imports, function/class defs, a docstring, "
            "and UPPERCASE = <literal> constants"
        )
    # Also disallow `if __name__ == '__main__':` block — that's executable code
    # at top level even though it's inside an If. Walk for it.
    for node in tree.body:
        if isinstance(node, ast.If):
            return False, f"top-level `if` not allowed in PoC module at line {node.lineno}"
    return True, "ok"


def _is_safe_const_assign(node: ast.Assign) -> bool:
    """True iff `node` is `UPPERCASE_NAME = <pure literal>`."""
    if len(node.targets) != 1:
        return False
    tgt = node.targets[0]
    if not isinstance(tgt, ast.Name):
        return False
    name = tgt.id
    if not (name.isupper() or all(c.isupper() or c == "_" or c.isdigit() for c in name)):
        return False
    if not name or (not name[0].isalpha() and name[0] != "_"):
        return False
    return _is_pure_literal(node.value)


def _is_pure_literal(value: ast.AST) -> bool:
    """True iff value is a Constant, or a Tuple/List/Set/Dict of pure literals.
    Function calls, attribute access, comprehensions, etc. are rejected."""
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_pure_literal(e) for e in value.elts)
    if isinstance(value, ast.Dict):
        return all(
            (k is None or _is_pure_literal(k)) and _is_pure_literal(v)
            for k, v in zip(value.keys, value.values)
        )
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        return _is_pure_literal(value.operand)
    return False
