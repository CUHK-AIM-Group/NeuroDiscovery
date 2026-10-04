"""Exercise policy, side-effect and transcript boundaries through real consumers."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import tool_registry
from core.mcp_stdio import HarnessHost, call_tool
from core.tool_pipeline import ToolPipeline


def test_stages_order_and_frozen_authoritative_result(tmp_path):
    seen, published = [], []
    original = {"success": True, "data": {"items": [1]}}

    def before(call):
        seen.append("pre")
        with pytest.raises(TypeError):
            call.arguments["nested"]["value"] = "changed"

    def guard(call):
        seen.append("guard")

    def around(call, invoke):
        seen.append("execute_enter")
        value = invoke()
        seen.append("execute_leave")
        return value

    def body(args):
        seen.append("body")
        assert args == {"nested": {"value": "original"}}
        return original

    def post(call, value):
        seen.append("post")
        value["annotation"] = "checked"

    def observe(call, value):
        seen.append("result")
        with pytest.raises(TypeError):
            value["data"]["items"][0] = 2
        published.append(value)

    pipeline = ToolPipeline(pre_execute=(before,), execute=(around,), post_execute=(post,), result=(observe,))
    result = pipeline.run("sample", {"nested": {"value": "original"}}, tmp_path, body, guards=(guard,))
    assert seen == ["pre", "guard", "execute_enter", "body", "execute_leave", "post", "result"]
    original["data"]["items"].append(2)
    assert result["data"]["items"] == [1]
    assert "annotation" not in original
    assert published[0]["data"]["items"] == (1,)


@pytest.mark.parametrize("stage", ["pre_execute", "guards"])
def test_denial_cannot_be_reversed_or_reach_body(tmp_path, stage):
    executed, observed = [], []
    deny = lambda call: {"success": False, "error_type": "permission_denied", "error": "Denied"}
    promote = lambda call, value: {"success": True, "executed": True, "error_type": ""}
    pipeline = ToolPipeline(**{stage: (deny,)}, post_execute=(promote,), result=(lambda c, r: observed.append(r),))
    result = pipeline.run("mutate", {}, tmp_path, lambda a: executed.append(a))
    assert not executed
    assert result["success"] is False and result["executed"] is False
    assert result["error_type"] == observed[0]["error_type"] == "permission_denied"


def test_execute_wrapper_cannot_repeat_a_side_effect(tmp_path):
    calls = []

    def replay(call, invoke):
        invoke()
        return invoke()

    result = ToolPipeline(execute=(replay,)).run(
        "write", {}, tmp_path, lambda a: calls.append(1) or {"success": True})
    assert calls == [1]
    assert result["success"] is False and result["outcome_unknown"] is True


def test_result_observer_failure_does_not_change_other_observers_or_caller(tmp_path, caplog):
    seen = []

    def broken(call, result):
        result["success"] = False

    result = ToolPipeline(result=(broken, lambda c, r: seen.append(r["success"]))).run(
        "read", {}, tmp_path, lambda a: {"success": True})
    assert result == {"success": True}
    assert seen == [True]
    assert "Tool result observer failed" in caplog.text


@pytest.mark.parametrize("bad", [float("nan"), object(), {1: "lossy key"}])
def test_unsnapshotable_results_fail_without_reexecuting(tmp_path, bad):
    calls = []
    result = ToolPipeline().run("write", {}, tmp_path, lambda a: calls.append(1) or {"value": bad})
    assert result["success"] is False and result["outcome_unknown"] is True
    assert calls == [1]


def test_shell_business_failure_is_normalized_before_recording(tmp_path, monkeypatch):
    from core.tools_impl import register_core_tools
    from core.agent import main
    import threading

    register_core_tools()
    monkeypatch.setattr(main, "_run_shell_command", lambda **kw: {
        "success": True, "stdout": "ModuleNotFoundError: No module named 'synthetic_missing'", "stderr": ""})
    observed = []
    host = SimpleNamespace(_tool_events=[], _checkpoint_mgr=None, benchmark_mode=False,
                           _cancel_event=threading.Event(),
                           _tool_pipeline=ToolPipeline(result=(lambda c, r: observed.append(r),)))
    result = tool_registry.dispatch(host, "run_shell_command", {"command": "synthetic only"}, tmp_path)
    assert result["error_type"] == "missing_dependency"
    assert observed[0]["success"] is False
    assert host._tool_events[-1]["result"] == result
    assert host._tool_events[-1]["success"] is False


def test_mcp_and_direct_consume_the_same_pipeline(tmp_path, monkeypatch):
    observed = []
    pipeline = ToolPipeline(result=(lambda call, value: observed.append((call.name, value["success"])),))
    monkeypatch.setattr(tool_registry, "DEFAULT_PIPELINE", pipeline)
    from core.tools_impl import register_core_tools
    register_core_tools()
    # Use the real library and consumers; this performs no model or network calls.
    workspace = Path(__file__).resolve().parents[1]
    direct = tool_registry.dispatch(HarnessHost(workspace), "search_skills", {"query": "ADNI"}, workspace)
    mcp = call_tool("search_skills", {"query": "ADNI"}, workspace)
    assert json.loads(mcp["content"][0]["text"]) == direct
    assert observed == [("search_skills", True), ("search_skills", True)]


def test_mcp_name_exposure_does_not_authorize_shell_or_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURORUNTIME_MCP_TOOLS", "run_shell_command,read_workspace_file")
    for name, args in [("run_shell_command", {"command": "must not execute"}),
                       ("read_workspace_file", {"path": "../outside"})]:
        result = json.loads(call_tool(name, args, tmp_path)["content"][0]["text"])
        assert result["error_type"] == "permission_denied"
        assert result["executed"] is False


def test_registered_extension_is_visible_and_executes_through_agent_guard(tmp_path, monkeypatch):
    from core.agent.test_autoresearch_execution import session, reply, tool

    invoked, observed = [], []
    monkeypatch.setitem(tool_registry.REGISTRY, "synthetic_extension", tool_registry.ToolSpec(
        "synthetic_extension", "Synthetic extension", {"type": "object", "properties": {}},
        lambda s, a, w: invoked.append(1) or {"success": True, "message": "extension result"}))
    agent, calls = session(tmp_path, [reply(calls=[tool("synthetic_extension")]), reply("Finished")], mode="off")
    agent._tool_pipeline = ToolPipeline(result=(lambda c, r: observed.append(dict(r)),))
    assert agent._chat() == "Finished"
    assert invoked == [1]
    assert "synthetic_extension" in [s["function"]["name"] for s in calls[0]["tools"]]
    messages = calls[1]["messages"]
    returned = next(json.loads(m["content"]) for m in messages if m.get("role") == "tool")
    assert returned == observed[0]


def test_agent_permission_denial_is_also_a_final_pipeline_result(tmp_path, monkeypatch):
    from core.agent.test_autoresearch_execution import session, reply, tool

    observed = []
    agent, calls = session(tmp_path, [reply(calls=[tool("run_shell_command", command="must not execute")]),
                                     reply("Denied")], mode="off")
    monkeypatch.setattr(agent, "_tool_authorized", lambda *args: False)
    agent._tool_pipeline = ToolPipeline(result=(lambda c, r: observed.append(dict(r)),))
    assert agent._chat() == "Denied"
    assert observed[0]["error_type"] == "permission_denied"
    assert observed[0]["executed"] is False


def test_hidden_skill_tool_cannot_be_invoked_by_a_hallucinated_call(tmp_path, monkeypatch):
    from core.agent.test_autoresearch_execution import session, reply, tool

    monkeypatch.setenv("NEUROCLAW_HARNESS_SKILL_TOOLS", "0")
    call = tool("read_skill")
    call.function.arguments = json.dumps({"name": "adni-skill"})
    agent, calls = session(tmp_path, [reply(calls=[call]),
                                     reply("Unavailable")], mode="off")
    observed = []
    agent._tool_pipeline = ToolPipeline(result=(lambda c, r: observed.append(dict(r)),))
    assert agent._chat() == "Unavailable"
    assert observed[0]["error_type"] == "tool_not_available"
    assert observed[0]["executed"] is False
