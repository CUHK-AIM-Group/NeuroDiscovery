"""Deterministic runtime tests: synthetic model replies, temporary files, no provider calls."""

from copy import deepcopy
import json
import hashlib
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest

from core.agent import main
from core.autoresearch import build_autoresearch_scope_prompt
from core.autoresearch_runtime import AutoResearchRun, iteration_limit


def reply(text="", calls=()):
    return NS(choices=[NS(message=NS(content=text, tool_calls=list(calls)))], usage=None)


def tool(name, **args):
    return NS(id=f"call_{name}", type="function", function=NS(name=name, arguments=json.dumps(args)))


def completion(**overrides):
    return dict(status="completed", summary="Finished the scoped synthetic report.",
                artifacts=["result.md"], validation="Read back the nonempty synthetic report.",
                evidence_ids=[1]) | overrides


def test_shell_python_resolves_to_active_runtime(tmp_path):
    result = main._run_shell_command('python -c "import sys; print(sys.executable)"', tmp_path, timeout_sec=10)
    assert result["success"]
    from pathlib import Path
    assert Path(result["stdout"].strip()).resolve() == Path(sys.executable).resolve()


@pytest.mark.parametrize("accepted", [True, False])
def test_separate_readonly_verifier_gates_completion(tmp_path, monkeypatch, accepted):
    def execute(**kwargs):
        (tmp_path / "result.md").write_text("Synthetic verified result", encoding="utf-8")
        return {"success": True, "stdout": "Inspected synthetic result"}
    monkeypatch.setattr(main, "_run_shell_command", execute)
    agent, calls = session(tmp_path, [
        reply(calls=[tool("run_shell_command", command="synthetic")]),
        reply(calls=[tool("finish_autoresearch", **completion())]),
        reply(json.dumps({"accepted": accepted, "reason": "Offline independent verdict"})),
    ])
    agent.env["autoresearch_independent_review"] = True
    agent._chat()
    assert agent.autoresearch_state["status"] == ("completed" if accepted else "review_required")
    assert len(calls) == 3
    assert "tools" not in calls[-1]
    assert len(calls[-1]["messages"]) == 2
    assert "separate read-only" in calls[-1]["messages"][0]["content"]
    assert agent.autoresearch_state["independent_review"]["accepted"] is accepted


def test_steering_is_applied_before_next_model_dispatch(tmp_path):
    agent, calls = session(tmp_path, [reply("Finished with updated instruction")], mode="off")
    agent._instruction_lock = threading.Lock()
    agent._pending_instructions = []
    agent.steer("Use the revised output directory")
    assert "updated" in agent._chat()
    assert calls[0]["messages"][-1]["content"] == "Use the revised output directory"
    assert not agent._pending_instructions
    agent.request_cancel()
    with pytest.raises(ValueError, match="stopping"):
        agent.steer("Do not accept")


def test_long_artifact_is_reviewed_completely(tmp_path):
    content = '中文证据🙂' * 3000 + '\nFINAL EVIDENCE AND LIMITATIONS'
    (tmp_path / "result.md").write_bytes(content.encode('utf-8'))
    agent, calls = session(tmp_path, [reply('{"accepted": true, "reason": "Claimed pass"}')])
    run = AutoResearchRun(tmp_path, "data")
    run.state["artifacts"] = [str(tmp_path / "result.md")]
    assert agent._review_autoresearch(run, "offline-test")["accepted"] is True
    assert len(calls) == 1
    packet = json.loads(calls[0]['messages'][1]['content'])
    assert packet['artifacts'][0]['content'] == content
    assert packet['artifacts'][0]['partial'] is False


def test_resume_uses_new_receipt_without_inheriting_execution_credit(tmp_path):
    previous = AutoResearchRun(tmp_path, "data")
    previous.state.update(objective="Original task", summary="Partial result", status="interrupted")
    previous.save()
    frozen = previous.path.read_bytes()
    (tmp_path / "result.md").write_text("Offline output")
    agent, calls = session(tmp_path, [
        reply(calls=[tool("inspect_local_path", path="result.md")]),
        reply(calls=[tool("finish_autoresearch", **completion())]),
    ])
    agent.resume_autoresearch_id = previous.path.parent.name
    agent._chat()
    assert agent.autoresearch_state["parent_run"] == previous.path.parent.name
    assert agent.autoresearch_state["objective"] == "Original task"
    assert len(agent.autoresearch_state["evidence"]) == 1
    assert "Do not replay earlier tools blindly" in calls[0]["messages"][-1]["content"]
    assert previous.path.read_bytes() == frozen


@pytest.fixture(autouse=True)
def default_cap(monkeypatch):
    monkeypatch.delenv("NEUROCLAW_MAX_TOOL_ITERATIONS", raising=False)


def session(tmp_path, responses, mode="data"):
    # Bypass environment/credential loading and real checkpoint/memory initialization.
    if mode == "data":
        _install_synthetic_data_contract(tmp_path)
    agent = main.AgentSession.__new__(main.AgentSession)
    agent.workspace = tmp_path
    agent.env = {"llm_backend": {"provider": "openai", "model": "offline-test"}}
    agent.history = [{"role": "system", "content": "Offline test."}, {"role": "user", "content": "Produce the report."}]
    agent.autoresearch_mode = mode
    agent._autoresearch_run = None
    agent._cancel_event = threading.Event()
    agent.benchmark_mode = False
    agent.no_skill_mode = False
    agent._checkpoint_mgr = None
    agent._tool_events = []
    agent._last_token_usage = {}
    calls = []
    response_iter = iter(responses)

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        response = next(response_iter)
        if isinstance(response, Exception):
            raise response
        return response() if callable(response) else response

    agent._llm = NS(chat=NS(completions=NS(create=create)))
    return agent, calls


def _install_synthetic_data_contract(tmp_path):
    """Give data-mode lifecycle tests a valid local delivery binding."""
    def record(name, text, **extra):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}
    ids = record("synthetic_subject_ids.json", '["S1"]')
    train = record("synthetic_train_ids.json", '["S1"]')
    validation = record("synthetic_validation_ids.json", "[]")
    test = record("synthetic_test_ids.json", "[]")
    qc_checks = [{"name": "finite", "status": "passed",
                  "observed": 0, "criterion": "nonfinite_count == 0"}]
    manifest = {
        "contract": "neuroclaw.data.v2",
        "inputs": [record("synthetic_input.csv", "subject,value\nS1,1\n",
                           subject_ids=ids, subject_ids_sha256=ids["sha256"],
                           subject_count=1, subject_id_column="subject")],
        "outputs": [record("synthetic_output.csv", "subject,value\nS1,1\n",
                            subject_ids=ids, subject_ids_sha256=ids["sha256"],
                            subject_count=1, subject_id_column="subject",
                            shape=[1, 2], schema={"subject": "string", "value": "float"})],
        "split": {"strategy": "grouped", "group_key": "subject_id", "seed": 1,
                   "subject_ids": {"train": train, "validation": validation, "test": test},
                   "train_subject_ids_sha256": train["sha256"],
                   "validation_subject_ids_sha256": validation["sha256"],
                   "test_subject_ids_sha256": test["sha256"]},
        "transforms": [{"name": "identity", "parameters": {}, "fit_on": "none",
                        "source": record("synthetic_transform.py", "# identity\n")}],
        "qc": {"status": "passed", "checks": qc_checks,
               "report": record("synthetic_qc.json", json.dumps({"status": "passed", "checks": qc_checks}))},
    }
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_continues_beyond_eight_rounds_and_plan_only_replies(tmp_path):
    (tmp_path / "result.md").write_text("Synthetic deliverable; not scientific evidence.", encoding="utf-8")
    sequence = [reply("I will inspect the inputs; shall I continue?")]
    sequence += [reply(calls=[tool("inspect_local_path", path="result.md")]) for _ in range(9)]
    sequence += [reply(calls=[tool("inspect_research_progress", path="result.md", kind="deliverable")])]
    sequence += [reply("Next I will deliver the report."), reply(calls=[tool("finish_autoresearch", **completion())])]
    agent, calls = session(tmp_path, sequence)
    result = agent._chat()
    assert "completed" in result and "result.md" in result
    assert len(calls) == 13
    assert agent.autoresearch_state["iterations"] == 13
    assert agent.autoresearch_state["iteration_limit"] == 40
    assert len(agent.autoresearch_state["evidence"]) == 10
    assert len(agent.history) == 2  # ephemeral continuation prompts don't masquerade as user requests
    assert "AutoResearch continuation" in calls[1]["messages"][-1]["content"]
    assert json.loads(agent._autoresearch_run.path.read_text())["status"] == "completed"


def test_recoverable_failure_is_repaired_before_delivery(tmp_path, monkeypatch):
    outcomes = iter([{"success": False, "error_type": "command_failed", "error": "Missing local package"},
                     {"success": True, "stdout": "Synthetic computation passed"}])

    def execute(**kwargs):
        result = next(outcomes)
        if result["success"]:
            (tmp_path / "result.md").write_text("Synthetic output", encoding="utf-8")
        return result

    monkeypatch.setattr(main, "_run_shell_command", execute)
    agent, calls = session(tmp_path, [
        reply(calls=[tool("run_shell_command", command="synthetic initial attempt")]),
        reply(calls=[tool("run_shell_command", command="synthetic corrected attempt")]),
        reply(calls=[tool("finish_autoresearch", **completion(evidence_ids=[2]))]),
    ])
    assert "completed" in agent._chat()
    assert len(calls) == 3
    assert [e["success"] for e in agent.autoresearch_state["evidence"]] == [False, True]


def test_missing_data_is_a_genuine_evidenced_blocker(tmp_path):
    report = completion(status="blocked", artifacts=[], blocker_kind="missing_input",
                        missing_requirement="Supply a usable dataset path.",
                        attempted_recovery="Checked the only supplied path; there are no authorized alternate sources.",
                        validation="The supplied data path does not exist.")
    agent, calls = session(tmp_path, [reply(calls=[tool("inspect_local_path", path="missing_data")]),
                                     reply(calls=[tool("finish_autoresearch", **report)])])
    assert "blocked" in agent._chat()
    assert len(calls) == 2
    assert agent.autoresearch_state["missing_requirement"] == "Supply a usable dataset path."


def test_missing_artifact_cannot_be_claimed_complete_and_can_be_repaired(tmp_path, monkeypatch):
    def execute(**kwargs):
        (tmp_path / "result.md").write_text("Synthetic checked output", encoding="utf-8")
        return {"success": True, "stdout": "Created report"}
    monkeypatch.setattr(main, "_run_shell_command", execute)
    agent, calls = session(tmp_path, [
        reply(calls=[tool("finish_autoresearch", **completion(evidence_ids=[]))]),
        reply(calls=[tool("run_shell_command", command="synthetic repair")]),
        reply(calls=[tool("finish_autoresearch", **completion(evidence_ids=[2]))]),
    ])
    assert "completed" in agent._chat()
    assert "Deliverable is missing" in calls[1]["messages"][-1]["content"]


def test_explicit_cap_stops_without_extra_finalization_request(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "2")
    agent, calls = session(tmp_path, [reply(calls=[tool("inspect_local_path", path=".")])] * 2)
    assert "budget_exhausted" in agent._chat()
    assert len(calls) == 2


@pytest.mark.parametrize("bad", ["garbage", "0", "-1"])
def test_malformed_explicit_budget_does_not_enable_unlimited_calls(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", bad)
    agent, calls = session(tmp_path, [])
    assert "blocked" in agent._chat()
    assert calls == []


def test_normal_chat_retains_existing_eight_round_default(tmp_path):
    sequence = [reply(calls=[tool("inspect_local_path", path=".")])] * 8 + [reply("Ordinary final answer")]
    agent, calls = session(tmp_path, sequence, mode="off")
    assert agent._chat() == "Ordinary final answer"
    assert len(calls) == 9  # eight ordinary tool iterations plus legacy finalization
    assert agent.autoresearch_state is None
    assert not (tmp_path / ".neurodiscovery").exists()
    assert all(t["function"]["name"] != "finish_autoresearch" for t in calls[0]["tools"])


def test_repeated_text_only_model_refusal_is_incomplete_not_success(tmp_path):
    agent, calls = session(tmp_path, [reply("Shall I continue?")] * 4)
    assert "stalled" in agent._chat()
    assert len(calls) == 4
    assert not agent.autoresearch_state["artifacts"]


def test_identical_failed_commands_do_not_loop_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_run_shell_command", lambda **kwargs: {"success": False, "error": "Unchanged failure"})
    agent, calls = session(tmp_path, [reply(calls=[tool("run_shell_command", command="synthetic failure")])] * 12)
    assert "stalled" in agent._chat()
    assert len(calls) == 12


def test_cancel_during_response_prevents_all_subsequent_tool_dispatch(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(main, "_run_shell_command", lambda **kwargs: executed.append(kwargs))
    agent, calls = session(tmp_path, [lambda: (agent.request_cancel() or reply(calls=[tool("run_shell_command", command="never execute")]))])
    assert "cancelled" in agent._chat()
    assert not executed and len(calls) == 1


def test_cancel_between_batched_tools_prevents_the_second_tool(tmp_path, monkeypatch):
    executed = []
    def execute(**kwargs):
        executed.append(kwargs["command"])
        agent.request_cancel()
        return {"success": False, "error_type": "cancelled"}
    monkeypatch.setattr(main, "_run_shell_command", execute)
    agent, _ = session(tmp_path, [reply(calls=[tool("run_shell_command", command="first"), tool("run_shell_command", command="second")])])
    assert "cancelled" in agent._chat()
    assert executed == ["first"]


def test_cancelled_retry_does_not_call_provider():
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        main._retry_api_call("synthetic", lambda: pytest.fail("Unexpected dispatch"), cancel_event=event)


def test_cancellation_terminates_owned_shell(tmp_path, monkeypatch):
    # A harmless local sleep verifies cancellation, not any scientific/model-backed execution.
    monkeypatch.setattr(main, "AGENT_SHELL_STATUS_FILE", tmp_path / "shell_status.json")
    event = threading.Event()
    timer = threading.Timer(0.2, event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = main._run_shell_command(f'"{sys.executable}" -c "import time; time.sleep(30)"', tmp_path, cancel_event=event)
    finally:
        timer.cancel()
    assert result["error_type"] == "cancelled"
    assert time.monotonic() - started < 8
    assert not (tmp_path / "shell_status.json").exists()


@pytest.mark.parametrize("provider", ["local", "unconfigured-provider"])
def test_backend_without_tools_is_reported_not_faked(tmp_path, provider):
    agent, calls = session(tmp_path, [])
    agent.env["llm_backend"]["provider"] = provider
    assert "blocked" in agent._chat()
    assert not calls


def test_provider_failure_keeps_receipt_without_recording_exception_secrets(tmp_path):
    agent, calls = session(tmp_path, [ValueError("synthetic-private-value")])
    response = agent._chat()
    assert "interrupted" in response and len(calls) == 1
    assert "synthetic-private-value" not in agent._autoresearch_run.path.read_text()


@pytest.mark.parametrize("mode", ["data", "model", "idea", "end-to-end"])
def test_scope_prompts_require_persistence_with_scientific_safety(mode):
    prompt = build_autoresearch_scope_prompt(mode)
    for text in ("work continuously", "NOT a mandatory questionnaire", "Negative results", "strict novelty",
                 "cancellation", "explicit budgets", "frozen", "idea-only task", "real artifact paths", "do not execute components outside it"):
        assert text.lower() in prompt.lower()
    assert "Require explicit confirmation before dependency installation, long-running jobs" not in prompt


@pytest.mark.parametrize("artifact", ["missing.md", ".", "empty.md", "../outside.md"])
def test_delivery_requires_real_nonempty_workspace_files(tmp_path, artifact):
    (tmp_path / "empty.md").touch()
    run = AutoResearchRun(tmp_path, "data")
    run.observe("read_workspace_file", {}, {"success": True})
    assert not run.finish(completion(artifacts=[artifact]))["success"]
    assert run.state["status"] == "running"


def test_receipt_and_failed_finish_ids_are_not_delivery_evidence(tmp_path):
    run = AutoResearchRun(tmp_path, "idea")
    run.observe("finish_autoresearch", {}, {"success": False})
    assert not run.finish(completion(artifacts=[str(run.path)]))["success"]
    (tmp_path / "result.md").write_text("Negative/empty-selection report", encoding="utf-8")
    assert not run.finish(completion())["success"]
    run.observe("read_workspace_file", {}, {"success": True})
    assert not run.finish(completion(artifacts=[str(run.path)], evidence_ids=[2]))["success"]
    assert run.finish(completion(evidence_ids=[2]))["success"]


def test_missing_data_claim_requires_inspection_evidence(tmp_path):
    run = AutoResearchRun(tmp_path, "data")
    result = run.finish(completion(status="blocked", artifacts=[], evidence_ids=[], blocker_kind="missing_input",
                                  missing_requirement="Data path", attempted_recovery="No check yet"))
    assert not result["success"]


def test_explicit_large_budget_is_not_silently_clamped_to_twenty(monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "75")
    assert iteration_limit() == 75


def test_receipt_does_not_save_tool_arguments_or_output(tmp_path):
    run = AutoResearchRun(tmp_path, "data")
    run.observe("run_shell_command", {"command": "sensitive-command-placeholder"},
                {"success": True, "stdout": "sensitive-output-placeholder"})
    text = run.path.read_text()
    assert "sensitive-" not in text
    assert '"tool": "run_shell_command"' in text


def test_bad_tool_arguments_are_recoverable_not_an_interrupted_run(tmp_path):
    (tmp_path / "result.md").write_text("Synthetic checked output", encoding="utf-8")
    agent, calls = session(tmp_path, [
        reply(calls=[tool("run_shell_command", command="must not execute", timeout_sec="not-an-int")]),
        reply(calls=[tool("read_workspace_file", path="result.md")]),
        reply(calls=[tool("finish_autoresearch", **completion(evidence_ids=[2]))]),
    ])
    assert "completed" in agent._chat()
    assert "invalid_tool_input" in calls[1]["messages"][-1]["content"]


def test_safety_policy_block_is_not_bypassed_in_autoresearch(tmp_path):
    report = completion(status="blocked", artifacts=[], blocker_kind="authorization",
                        missing_requirement="A safe authorized alternative is required.",
                        attempted_recovery="The requested destructive shell command was rejected; no unsafe retry was made.")
    agent, calls = session(tmp_path, [
        reply(calls=[tool("run_shell_command", command="rm -rf synthetic-target")]),
        reply(calls=[tool("finish_autoresearch", **report)]),
    ])
    assert "blocked" in agent._chat()
    assert "safety_policy_block" in calls[1]["messages"][-1]["content"]
    assert agent._tool_events[0]["executed"] is False


def test_finish_cannot_share_a_batch_with_unobserved_execution(tmp_path, monkeypatch):
    (tmp_path / "result.md").write_text("Synthetic output", encoding="utf-8")
    agent, calls = session(tmp_path, [
        reply(calls=[tool("inspect_local_path", path="result.md"), tool("finish_autoresearch", **completion())]),
        reply(calls=[tool("finish_autoresearch", **completion())]),
    ])
    assert "completed" in agent._chat()
    assert "Call finish_autoresearch alone" in calls[1]["messages"][-1]["content"]


def test_idea_only_can_finish_without_local_patient_data(tmp_path):
    (tmp_path / "IDEA.md").write_text("Synthetic literature-only idea report; no patient data used.", encoding="utf-8")
    agent, calls = session(tmp_path, [
        reply(calls=[tool("read_workspace_file", path="IDEA.md")]),
        reply(calls=[tool("finish_autoresearch", **completion(artifacts=["IDEA.md"]))]),
    ], mode="idea")
    assert "completed" in agent._chat()
    assert len(calls) == 2


def test_end_to_end_exposes_idea_chain_and_ranking_tools(tmp_path):
    agent, calls = session(tmp_path, [reply("Continue with the full workflow.")], mode="end-to-end")
    agent._chat()
    names = {item["function"]["name"] for item in calls[0]["tools"]}
    assert {"generate_idea_hypotheses", "rank_idea_hypotheses"}.issubset(names)
