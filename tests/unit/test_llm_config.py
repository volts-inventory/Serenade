"""LocalBackend per-CallSite routing.

The legacy ``orchestrator.llm_config`` module (encryption / is_local /
apply_to_settings / snapshot helpers) was removed in the rewrite. The
only surviving contract here is that ``LocalBackend`` resolves a
distinct model per ``CallSite``.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# LocalBackend per-CallSite routing                                            #
# --------------------------------------------------------------------------- #


def test_local_backend_per_callsite_routing() -> None:
    """Construct a LocalBackend with distinct per-site model names and
    verify each CallSite resolves to its own model."""
    from unittest.mock import patch

    from llm.local_backend import LocalBackend
    from llm.router import CallSite
    from orchestrator.settings import Settings

    s = Settings(
        local_base_url="http://localhost:11434/v1",
        local_model_planner="planner-model",
        local_model_parser="parser-model",
        local_model_critic="critic-model",
        local_model_synthesizer="synth-model",
        local_model_triage="triage-model",
    )

    # Patch AsyncOpenAI to avoid real HTTP connections
    with patch("llm.local_backend.AsyncOpenAI"):
        backend = LocalBackend(s)

    assert backend._model_map[CallSite.PLANNER.value] == "planner-model"
    assert backend._model_map[CallSite.PARSER.value] == "parser-model"
    assert backend._model_map[CallSite.CRITIC.value] == "critic-model"
    assert backend._model_map[CallSite.SYNTHESIZER.value] == "synth-model"
    assert backend._model_map[CallSite.TRIAGE.value] == "triage-model"

    # Unknown call site falls back to default (planner)
    assert backend._default_model == "planner-model"
