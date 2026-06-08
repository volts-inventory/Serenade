"""R3 #H2 Bug 4 + R3 #I1 Bug 7 — every concrete ``Tool`` subclass must
override ``summarize_for_model``.

The default in ``tools.base.Tool`` used to enumerate list-shaped fields
and report their lengths — actively misleading for tools returning
``outcome: str`` + ``rows: list``: the planner saw ``rows=N`` and
never the outcome string. ("5 rows returned, all FAILED" looked
indistinguishable from "5 rows returned successfully".)

R3 #H2 made the default raise ``NotImplementedError``; that caught a
missing override at exec time. R3 #I1 Bug 7 promotes the hook to
``@abc.abstractmethod`` so a tool that forgets the override fails at
class **instantiation** — far earlier in the lifecycle. This test
locks the contract in by walking every concrete ``Tool`` subclass
exposed in the default registry and asserting each one provides its
own ``summarize_for_model``, AND that an abstract subclass cannot be
instantiated.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from pydantic import BaseModel

from tools.base import Tool, build_default_registry


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #


def _concrete_tool_subclasses() -> list[type[Tool[Any, Any]]]:
    """Walk ``src/tools`` and return every concrete (non-abstract)
    ``Tool[*]`` subclass we ship. Anything that's importable + a Tool
    subclass + not ``Tool`` itself + not abstract qualifies."""
    import tools

    out: list[type[Tool[Any, Any]]] = []
    seen: set[type] = set()
    pkg_path = tools.__path__
    for _finder, modname, _ispkg in pkgutil.iter_modules(pkg_path):
        full = f"tools.{modname}"
        try:
            mod = importlib.import_module(full)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj in seen:
                continue
            if obj is Tool:
                continue
            if not issubclass(obj, Tool):
                continue
            if inspect.isabstract(obj):
                continue
            seen.add(obj)
            out.append(obj)
    return out


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_subclass_without_summarize_for_model_cannot_instantiate() -> None:
    """R3 #I1 Bug 7: a Tool subclass that forgets to override
    ``summarize_for_model`` must fail at class instantiation, not mid-
    step. Python's ``abc.ABC`` machinery raises ``TypeError`` when
    instantiation is attempted on a class with unimplemented abstract
    methods; the error names ``summarize_for_model`` so the author of
    the new tool sees the contract violation in plain text."""

    class _DummyOutput(BaseModel):
        rows: list[int] = []

    # Build a minimal Tool subclass that does NOT override
    # summarize_for_model. The CLASS is fine; the INSTANCE is not.
    class _LeakyTool(Tool[BaseModel, _DummyOutput]):
        name = "leaky"
        description = "test-only"
        from worldstate.schema import RiskClass, Stage

        stage = Stage.RECON
        risk_class = RiskClass.RECON
        Input = BaseModel
        Output = _DummyOutput

        async def run(self, input: Any, ctx: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

        def extract_targets(self, input: Any) -> list[str]:  # pragma: no cover
            return []

    with pytest.raises(TypeError) as exc_info:
        _LeakyTool()
    assert "summarize_for_model" in str(exc_info.value), (
        f"abstract-method error must name the missing method: {exc_info.value!r}"
    )


def test_every_concrete_tool_overrides_summarize_for_model() -> None:
    """Forcing function: every Tool subclass shipped in ``src/tools``
    overrides ``summarize_for_model`` so it isn't silently inheriting
    the NotImplementedError default."""
    subclasses = _concrete_tool_subclasses()
    assert subclasses, "no Tool subclasses discovered — sanity check failed"
    missing: list[str] = []
    for cls in subclasses:
        qn = cls.summarize_for_model.__qualname__
        # An overridden method's ``__qualname__`` starts with the
        # subclass name (e.g. ``NmapServiceScan.summarize_for_model``).
        # The base method's qualname is ``Tool.summarize_for_model``.
        if qn.startswith("Tool."):
            missing.append(cls.__name__)
    assert not missing, (
        f"these Tool subclasses inherit the default summarize_for_model "
        f"and would raise NotImplementedError at exec time: {missing}"
    )


def test_default_registry_tools_have_overrides() -> None:
    """The tools the worker actually ships (build_default_registry)
    must each override summarize_for_model. This is the per-instance
    counterpart to the class-walk test above — catches a regression
    where someone copies a class but forgets the override."""
    reg = build_default_registry()
    missing: list[str] = []
    for tool in reg.all():
        qn = type(tool).summarize_for_model.__qualname__
        if qn.startswith("Tool."):
            missing.append(type(tool).__name__)
    assert not missing, (
        f"default-registry tools missing summarize_for_model override: {missing}"
    )
