"""Step 1 verification: the registry surface must match the loop's old surface.

The point of the registry is that nothing observable changes. These tests freeze
the four inline schemas by name, required fields and property keys, confirm the
registry is the single source they are read from, and check that dispatch routes
by name and never raises when a tool body fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import tool_registry, tools_impl


EXPECTED = {
    "run_shell_command": {
        "required": ["command"],
        "properties": {"command", "timeout_sec"},
    },
    "inspect_local_path": {
        "required": ["path"],
        "properties": {"path"},
    },
    "read_workspace_file": {
        "required": ["path"],
        "properties": {"path", "max_chars", "offset", "paper_start", "paper_count"},
    },
    "spawn_subagent": {
        "required": ["task"],
        "properties": {"task", "persona", "skills_filter", "mode"},
    },
    # Step 3 added the skill consumers. They are registered here but stay out of
    # the visible surface unless NEUROCLAW_HARNESS_SKILL_TOOLS is on.
    "search_skills": {
        "required": ["query"],
        "properties": {"query", "limit"},
    },
    "read_skill": {
        "required": ["name"],
        "properties": {"name", "offset"},
    },
}


@pytest.fixture(autouse=True)
def _registered():
    tools_impl.register_core_tools()
    yield


def test_the_expected_tools_are_registered():
    assert set(tool_registry.names()) == set(EXPECTED)


def test_the_four_loop_tools_are_registered_without_the_skill_gate():
    loop_tools = {"run_shell_command", "inspect_local_path", "read_workspace_file", "spawn_subagent"}
    assert loop_tools <= set(tool_registry.names())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_schema_matches_original_shape(name: str):
    spec = tool_registry.get(name)
    assert spec is not None
    schema = spec.parameters
    assert schema["type"] == "object"
    assert set(schema["required"]) == set(EXPECTED[name]["required"])
    assert set(schema["properties"]) == EXPECTED[name]["properties"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_schemas_are_provider_ready(name: str):
    payload = {item["function"]["name"]: item for item in tool_registry.schemas()}
    entry = payload[name]
    assert entry["type"] == "function"
    assert entry["function"]["description"].strip()
    json.dumps(entry)  # must be JSON serializable for the API call


def test_schemas_include_and_exclude_narrow_without_mutating_registry():
    only = tool_registry.schemas(include=["read_workspace_file"])
    assert [item["function"]["name"] for item in only] == ["read_workspace_file"]
    hidden = tool_registry.schemas(exclude=["spawn_subagent"])
    assert "spawn_subagent" not in {item["function"]["name"] for item in hidden}
    assert set(tool_registry.names()) == set(EXPECTED)


def test_dispatch_unknown_tool_is_a_result_not_an_exception():
    result = tool_registry.dispatch(object(), "not_a_tool", {}, Path.cwd())
    assert result["success"] is False
    assert result["error_type"] == "unknown_tool"


def test_dispatch_converts_a_throwing_handler_into_an_error_result():
    tool_registry.REGISTRY["__boom"] = tool_registry.ToolSpec(
        name="__boom",
        description="test",
        parameters={"type": "object", "properties": {}},
        handler=lambda session, args, workspace: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    try:
        result = tool_registry.dispatch(object(), "__boom", {}, Path.cwd())
        assert result["success"] is False
        assert result["error_type"] == "tool_execution_error"
        assert "kaboom" in result["error"]
    finally:
        del tool_registry.REGISTRY["__boom"]


def test_duplicate_registration_is_refused():
    with pytest.raises(ValueError):
        tool_registry.register(tool_registry.ToolSpec(
            name="inspect_local_path",
            description="duplicate",
            parameters={"type": "object", "properties": {}},
            handler=lambda session, args, workspace: {},
        ))


def test_digest_is_stable_across_calls():
    assert tool_registry.canonical_schema_digest() == tool_registry.canonical_schema_digest()
