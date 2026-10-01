"""Step 3 verification: the skill capability must be a working three-role seam.

These tests cover the definition/provider/consumer split directly: the provider
discovers the real ``skills/`` directory, the definition searches and loads
bounded bodies, and the consumers are registered tools whose visibility follows
the explicit harness switch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import tool_registry
from core.harness_skills import FilesystemSkillProvider, SkillCapability, default_capability
from core.tools_impl import register_core_tools, visible_schemas


REPO_SKILLS = Path(__file__).resolve().parents[1] / "skills"


@pytest.fixture(autouse=True)
def _registered():
    register_core_tools()
    yield


def test_provider_discovers_the_real_skill_library():
    records = FilesystemSkillProvider(REPO_SKILLS).list()
    assert len(records) > 50
    names = {record.name for record in records}
    assert {"adni-skill", "fmri-skill", "eeg-skill"} <= names
    assert all(record.skill_md.exists() for record in records)


def test_definition_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        SkillCapability([])


def test_search_ranks_name_matches_ahead_of_body_matches():
    capability = default_capability(REPO_SKILLS)
    hits = capability.search("adni bids")
    assert hits, "expected a match for a named dataset"
    assert hits[0].name == "adni-skill"


def test_search_with_no_keywords_returns_nothing():
    capability = default_capability(REPO_SKILLS)
    assert capability.search("   ") == []


def test_load_is_bounded_and_reports_a_next_offset():
    capability = default_capability(REPO_SKILLS)
    first = capability.load("adni-skill", max_chars=200)
    assert first["success"] is True
    assert len(first["content"]) == 200
    assert first["truncated"] is True
    assert first["next_offset"] == 200
    second = capability.load("adni-skill", max_chars=200, offset=first["next_offset"])
    assert second["content"] != first["content"]
    whole = capability.load("adni-skill", max_chars=10_000_000)
    assert whole["truncated"] is False
    assert whole["content"].startswith("---")


def test_load_unknown_skill_is_an_error_result_not_an_exception():
    result = default_capability(REPO_SKILLS).load("definitely-not-a-skill")
    assert result["success"] is False
    assert result["error_type"] == "unknown_skill"


def test_load_requires_a_name():
    result = default_capability(REPO_SKILLS).load("  ")
    assert result["success"] is False
    assert result["error_type"] == "invalid_skill_name"


def test_consumers_are_registered():
    assert tool_registry.get("search_skills") is not None
    assert tool_registry.get("read_skill") is not None


def test_skill_tools_are_hidden_unless_the_harness_switch_is_on(monkeypatch):
    monkeypatch.delenv("NEUROCLAW_HARNESS_SKILL_TOOLS", raising=False)
    default_names = {item["function"]["name"] for item in visible_schemas()}
    assert "search_skills" not in default_names
    assert "read_skill" not in default_names
    monkeypatch.setenv("NEUROCLAW_HARNESS_SKILL_TOOLS", "1")
    harness_names = {item["function"]["name"] for item in visible_schemas()}
    assert {"search_skills", "read_skill"} <= harness_names


def test_read_skill_tool_loads_through_the_consumer(monkeypatch):
    monkeypatch.setenv("NEUROCLAW_HARNESS_SKILL_TOOLS", "1")

    class Session:
        workspace = Path.cwd()
        _tool_events: list = []

    session = Session()
    result = tool_registry.dispatch(session, "read_skill", {"name": "adni-skill"}, Path.cwd())
    assert result["success"] is True
    assert result["name"] == "adni-skill"
    assert session._tool_events[-1]["tool"] == "read_skill"


def test_search_skills_tool_reports_no_match_helpfully():
    class Session:
        workspace = Path.cwd()
        _tool_events: list = []

    session = Session()
    result = tool_registry.dispatch(session, "search_skills", {"query": "zzzz-nonexistent-zzzz"}, Path.cwd())
    assert result["success"] is False
    assert result["error_type"] == "no_skill_match"
    assert result["skill_count"] > 50


def _prompt_with(tmp_path: Path, monkeypatch, *, enabled: bool, benchmark: bool, no_skill: bool) -> str:
    from core.agent import main as agent_main

    if enabled:
        monkeypatch.setenv("NEUROCLAW_HARNESS_SKILL_TOOLS", "1")
    else:
        monkeypatch.delenv("NEUROCLAW_HARNESS_SKILL_TOOLS", raising=False)
    agent = agent_main.AgentSession.__new__(agent_main.AgentSession)
    agent.workspace = tmp_path
    agent.env = {"llm_backend": {"provider": "openai", "model": "offline-test"}, "model": "offline-test"}
    agent.history = []
    agent.benchmark_mode = benchmark
    agent.no_skill_mode = no_skill
    agent._memory_store = None
    agent.instruction_sources = []
    return agent._build_system_prompt([])


def test_catalog_is_injected_only_when_the_switch_is_on(tmp_path, monkeypatch):
    off = _prompt_with(tmp_path, monkeypatch, enabled=False, benchmark=False, no_skill=False)
    on = _prompt_with(tmp_path, monkeypatch, enabled=True, benchmark=False, no_skill=False)
    assert "[Skill Catalog]" not in off
    assert "[Skill Catalog]" in on
    assert "read_skill" in on


def test_catalog_is_absent_in_benchmark_and_no_skill_modes(tmp_path, monkeypatch):
    benchmark = _prompt_with(tmp_path, monkeypatch, enabled=True, benchmark=True, no_skill=False)
    no_skill = _prompt_with(tmp_path, monkeypatch, enabled=True, benchmark=False, no_skill=True)
    assert "[Skill Catalog]" not in benchmark
    assert "[Skill Catalog]" not in no_skill
