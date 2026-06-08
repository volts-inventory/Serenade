"""Per-agent expert personas (Phase 60).

Locks in two contracts:

1. ``system_prompt(persona)`` returns the universal hard rules PLUS the
   per-agent persona, in that order. An empty persona returns just the
   universal rules — backwards-compatible.
2. Every stage agent (Recon, WebEnum, NetEnum, VulnTriage, Exploit,
   PrivEsc, Lateral, Loot, Report) declares a non-empty
   ``system_persona``. New agents added later get the test failure as
   a forcing function to write a real persona instead of inheriting
   an empty string from BaseAgent.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SERENADE_AUTH_ALLOW_DEV_FALLBACK", "1")
os.environ.setdefault("SERENADE_SESSION_SECRET", "x" * 32)


# --------------------------------------------------------------------------- #
# system_prompt composition                                                    #
# --------------------------------------------------------------------------- #


def test_system_prompt_with_empty_persona_returns_universal_rules_only() -> None:
    """An agent that hasn't set a persona still gets the universal
    rules — the persona is additive, never replaces."""
    from llm.prompts.base import SYSTEM_PREFIX, system_prompt

    out = system_prompt("")
    assert out == SYSTEM_PREFIX
    # Sanity: universal rule fragments are still in the output
    assert "Hard rules" in out
    assert "in scope" in out


def test_system_prompt_appends_persona_after_universal_rules() -> None:
    """The persona is appended below the universal block, not prepended.
    The universal rules cache prefix needs to come first."""
    from llm.prompts.base import SYSTEM_PREFIX, system_prompt

    persona = "PERSONA-SENTINEL-XYZ"
    out = system_prompt(persona)
    assert out.startswith(SYSTEM_PREFIX)
    assert persona in out
    # Universal block must precede the persona block
    assert out.index("Hard rules") < out.index(persona)


# --------------------------------------------------------------------------- #
# Every agent declares a non-empty persona                                     #
# --------------------------------------------------------------------------- #


_ALL_AGENT_CLASSES = [
    "ReconAgent",
    "WebEnumAgent",
    "NetEnumAgent",
    "VulnTriageAgent",
    "ExploitAgent",
    "PrivEscAgent",
    "LateralAgent",
    "LootAgent",
    "ReportAgent",
]


@pytest.mark.parametrize("cls_name", _ALL_AGENT_CLASSES)
def test_each_agent_has_non_empty_persona(cls_name: str) -> None:
    """Forcing function: a stage agent ships an expert persona, not an
    inherited empty default. If you add a new stage and skip the
    persona, this test fails."""
    import importlib

    # Module name matches the agent's lowercase class minus 'Agent'
    # (ReconAgent → recon, VulnTriageAgent → vuln_triage, etc.)
    rename = {
        "ReconAgent": "recon",
        "WebEnumAgent": "web_enum",
        "NetEnumAgent": "net_enum",
        "VulnTriageAgent": "vuln_triage",
        "ExploitAgent": "exploit",
        "PrivEscAgent": "privesc",
        "LateralAgent": "lateral",
        "LootAgent": "loot",
        "ReportAgent": "report",
    }
    mod = importlib.import_module(f"agents.{rename[cls_name]}")
    cls = getattr(mod, cls_name)
    persona = cls.system_persona
    assert persona, f"{cls_name}.system_persona is empty — write one"
    # Personas should be at least a few hundred chars — anything shorter
    # is almost certainly a placeholder.
    assert len(persona) >= 200, (
        f"{cls_name}.system_persona is suspiciously short "
        f"({len(persona)} chars); ensure it actually encodes role expertise"
    )


def test_personas_are_distinct_across_agents() -> None:
    """Each agent's persona is unique to its role. If two agents share
    a persona by accident, the test catches it."""
    import importlib

    rename = {
        "ReconAgent": "recon",
        "WebEnumAgent": "web_enum",
        "NetEnumAgent": "net_enum",
        "VulnTriageAgent": "vuln_triage",
        "ExploitAgent": "exploit",
        "PrivEscAgent": "privesc",
        "LateralAgent": "lateral",
        "LootAgent": "loot",
        "ReportAgent": "report",
    }
    personas: set[str] = set()
    for cls_name in _ALL_AGENT_CLASSES:
        mod = importlib.import_module(f"agents.{rename[cls_name]}")
        cls = getattr(mod, cls_name)
        personas.add(cls.system_persona)
    assert len(personas) == len(_ALL_AGENT_CLASSES), (
        "Two or more agents share an identical persona string"
    )
