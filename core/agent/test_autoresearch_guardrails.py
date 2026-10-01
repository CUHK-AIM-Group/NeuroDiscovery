import json
import os

import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun, iteration_limit


def test_default_budget_is_bounded(monkeypatch):
    monkeypatch.delenv("NEUROCLAW_MAX_TOOL_ITERATIONS", raising=False)
    assert iteration_limit() == 40


def test_changed_command_does_not_disguise_same_observation(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    run.state['correction_sent'] = True
    outcomes = [run.observe("run_shell_command", {"command": f"probe-{index}"},
                            {"success": True, "stdout": "unchanged", "returncode": 0})
                for index in range(12)]
    assert not any(outcomes[:-1])
    assert outcomes[-1]
    assert "closeout check" in run.closeout_prompt()
    assert "probe-" not in run.path.read_text()
    assert "unchanged" not in run.path.read_text()


def test_changing_stdout_is_not_research_progress(tmp_path):
    run = AutoResearchRun(tmp_path, "data")
    run.state['correction_sent'] = True
    for index in range(12):
        run.begin_iteration()
        stalled = run.observe("run_shell_command", {}, {"success": True, "stdout": str(index)})
    assert stalled
    assert run.state["no_progress_count"] == 12
    assert run.state["iterations"] == 12


def test_successful_probes_do_not_erase_repeated_failure(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    run.state['correction_sent'] = True
    for index in range(6):
        run.observe("read_workspace_file", {}, {"success": True, "content": str(index)})
        stalled = run.observe("run_shell_command", {}, {"success": False, "error": "same"})
    assert stalled


def test_receipt_reads_are_not_progress(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    run.state['correction_sent'] = True
    for index in range(12):
        stalled = run.observe("read_workspace_file", {"path": str(run.path)},
                              {"success": True, "content": str(index)})
    assert stalled


def test_budget_closeout_never_dispatches_extra_or_claims_success(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "3")
    (tmp_path / "input.txt").write_text("input")
    agent, calls = session(tmp_path, [reply(calls=[tool("read_workspace_file", path="input.txt")])] * 3)
    assert "budget_exhausted" in agent._chat()
    assert len(calls) == 3
    assert any("closeout check" in message.get("content", "") for message in calls[-1]["messages"])
    receipt = json.loads(agent._autoresearch_run.path.read_text())
    assert receipt["closeout"]["accepted"] is False
    assert receipt["closeout"]["iteration_limit"] == 3
    assert not receipt["artifacts"]


def test_legacy_unlimited_resume_cannot_reset_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROCLAW_MAX_TOOL_ITERATIONS", raising=False)
    previous = AutoResearchRun(tmp_path, "idea")
    previous.state.update(status="interrupted", iteration_limit=None, iterations=45)
    previous.save()
    run = AutoResearchRun(tmp_path, "idea")
    with pytest.raises(ValueError, match="exhausted"):
        run.inherit_checkpoint(previous.path.parent.name, "continue")


def test_shell_output_utf8_gbk_and_bad_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "AGENT_SHELL_STATUS_FILE", tmp_path / "status.json")
    text = "中文输出"
    for encoding in (["utf-8", "gbk"] if os.name == "nt" else ["utf-8"]):
        payload = text.encode(encoding).hex()
        result = main._run_shell_command(f'python -c "import sys; sys.stdout.buffer.write(bytes.fromhex(\'{payload}\'))"', tmp_path, timeout_sec=10)
        assert result["success"]
        assert result["stdout"] == text
    result = main._run_shell_command('python -c "import sys; sys.stdout.buffer.write(bytes([255]))"', tmp_path, timeout_sec=10)
    assert not result["success"]
    assert result["error_type"] == "output_decode_error"
    assert result["stdout"]
    json.dumps(result)


def test_missing_output_never_decodes_as_success():
    assert main._decode_shell_output(None) == ("", True)
    assert main._decode_shell_output(b"") == ("", False)


def test_utf16_bom_output():
    assert main._decode_shell_output("中文".encode("utf-16")) == ("中文", False)


def test_repeated_successes_stop_agent_with_partial_receipt(tmp_path):
    (tmp_path / "input.txt").write_text("unchanged")
    agent, calls = session(tmp_path, [reply(calls=[tool("read_workspace_file", path="input.txt")])] * 13)
    assert "stalled" in agent._chat()
    assert len(calls) == 12
    assert agent.autoresearch_state["closeout"]["accepted"] is False


def test_final_iteration_cannot_skip_independent_review(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "2")
    (tmp_path / "result.md").write_text("Synthetic result")
    agent, calls = session(tmp_path, [reply(calls=[tool("read_workspace_file", path="result.md")]),
                                    reply(calls=[tool("finish_autoresearch", **completion())])])
    agent.env["autoresearch_independent_review"] = True
    assert "budget_exhausted" in agent._chat()
    assert len(calls) == 2
    assert agent.autoresearch_state["artifacts"]
    assert agent.autoresearch_state["closeout"]["accepted"] is False


def test_recovery_of_unlimited_legacy_run_has_no_fresh_budget(tmp_path):
    agent, calls = session(tmp_path, [])
    agent._recovery_state = {"iterations": 50, "iteration_limit": None, "state_path": str(tmp_path / "old" / "run.json")}
    assert "budget_exhausted" in agent._chat()
    assert calls == []
    assert agent.autoresearch_state["iteration_limit"] == 40


@pytest.mark.parametrize("remaining", [0, 1])
def test_context_overflow_retry_consumes_shared_budget(tmp_path, monkeypatch, remaining):
    import core.session.context
    error = RuntimeError("synthetic overflow")
    error.code = "context_length_exceeded"
    agent, calls = session(tmp_path, [error, reply("ok")])
    run = AutoResearchRun(tmp_path, "idea")
    run.state["iterations"] = run.limit - remaining
    agent._autoresearch_run = run
    monkeypatch.setattr(core.session.context, "compact_context", lambda *args, **kwargs: {"status": "compacted"})
    if remaining:
        assert agent._context_chat_create("offline", [], []).choices[0].message.content == "ok"
    else:
        with pytest.raises(RuntimeError, match="budget"):
            agent._context_chat_create("offline", [], [])
    assert len(calls) == 1 + remaining
    assert run.state["iterations"] == run.limit
