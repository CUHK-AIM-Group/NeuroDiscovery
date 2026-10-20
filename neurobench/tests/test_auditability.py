"""Fault-injection checks for experiment controls, receipts and cost accounting.

All model responses here are synthetic. Real tool consumers and SQLite/artifact
publication run unchanged; no network or external CLI invocation is allowed.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobench.runner import campaign, conditions, drivers, ledger, prompts, scorer_compat
from neurobench.runner.artifacts import file_sha256, freeze_bytes, sha256_text
from neurobench.runner.cli_drivers import parse_cli_events, parse_claude_result


ANSWER = ("## Solution Thinking\nThis is a synthetic engineering fixture with no scientific claim.\n\n"
          "## Commands Or Code\n```python\n# Proposed only; never run a benchmark workflow.\nprint('fixture')\n```\n")


def condition(driver="http_chat", enabled=True):
    return next(c for c in conditions.CONDITIONS.values()
                if c.driver == driver and c.harness_enabled == enabled)


@pytest.fixture
def task(tmp_path):
    text = "# Synthetic fixture\nPropose code only; this is not a scientific benchmark result.\n"
    path = tmp_path / "task.md"
    path.write_text(text, encoding="utf-8")
    return SimpleNamespace(case_id="T01", number=1, path=path, text=text, title="Synthetic fixture",
                           source_sha256=sha256_text(text))


def outcome(**changes):
    return dataclasses.replace(drivers.AttemptOutcome(
        True, "", "", "synthetic-session", ANSWER, 100, 20, 10, .01,
        uncached_input_tokens=70, cache_creation_input_tokens=10), **changes)


def completion(*, calls=None, usage=None, content=ANSWER):
    message = {"role": "assistant", "content": None if calls else content,
               "reasoning_content": "provider-specific synthetic field"}
    if calls:
        message["tool_calls"] = calls
    return {"id": "synthetic", "model": "returned-model", "usage": usage,
            "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}]}


def call(name, arguments, tool_id="c1"):
    return {"id": tool_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def fake_http(monkeypatch, replies, events_path):
    import openai
    requests = []
    remaining = iter(replies)

    def factory(**config):
        assert config["max_retries"] == 0

        def create(**request):
            # The request must be durable BEFORE a provider can receive it.
            rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            assert rows[-1]["event"] == "request"
            assert rows[-1]["payload"]["messages"] == request["messages"]
            requests.append(copy.deepcopy(request))
            reply = next(remaining)
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(model_dump=lambda **kw: copy.deepcopy(reply))

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               close=config["http_client"].close)

    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("NK", "synthetic-test-secret")
    return requests


def http_run(tmp_path, *, enabled=True, skills_dir=None, model="openai/gpt-5.6-sol"):
    return drivers.run_http_chat(prompt="Synthetic only", model=model, base_url="https://invalid.example/v1",
        provider="nvidia", harness_enabled=enabled, workspace=tmp_path, timeout_seconds=10,
        events_path=tmp_path / "events.jsonl", request_overrides={"omit_temperature": True}, skills_dir=skills_dir)


def test_http_records_every_model_visible_message_and_real_frozen_skill_result(tmp_path, monkeypatch):
    skills = tmp_path / "frozen" / "fixture-skill"
    skills.mkdir(parents=True)
    body = "---\nname: fixture-skill\ndescription: fixture methodology\n---\n# Frozen body\nFrozen guidance."
    (skills / "SKILL.md").write_text(body, encoding="utf-8")
    replies = [completion(calls=[call("search_skills", {"query": "fixture"})]),
               completion(calls=[call("read_skill", {"name": "fixture-skill"}, "c2")]), completion()]
    for n, reply in enumerate(replies, 1):
        reply["usage"] = {"prompt_tokens": 100*n, "completion_tokens": 10*n,
                          "prompt_tokens_details": {"cached_tokens": 20*n, "cache_creation_tokens": 10*n}}
    requests = fake_http(monkeypatch, replies, tmp_path / "events.jsonl")
    result = http_run(tmp_path, skills_dir=skills.parent)
    assert result.ok and result.usage_complete
    assert (result.input_tokens, result.uncached_input_tokens, result.cached_input_tokens,
            result.cache_creation_input_tokens, result.output_tokens) == (600, 420, 120, 60, 60)
    assert result.skills_loaded == ["fixture-skill"] and result.request_count == 3
    assert result.model_ids == ["returned-model"]
    assert all("temperature" not in r and r["max_completion_tokens"] == 12000 for r in requests)
    assert requests[1]["messages"][1]["reasoning_content"] == "provider-specific synthetic field"
    assert json.loads(requests[2]["messages"][-1]["content"])["content"] == body
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows].count("response") == 3
    assert [r["event"] for r in rows].count("tool_result") == 2


def test_baseline_forbidden_call_is_recorded_and_never_dispatched(tmp_path, monkeypatch):
    replies = [completion(calls=[call("read_skill", {"name": "adni-skill"})],
                          usage={"prompt_tokens": 9, "completion_tokens": 3})]
    requests = fake_http(monkeypatch, replies, tmp_path / "events.jsonl")
    monkeypatch.setattr(drivers, "_dispatch_tool", lambda *a, **k: pytest.fail("Baseline executed a tool"))
    result = http_run(tmp_path, enabled=False)
    assert not result.ok and result.retry_class == "policy"
    assert result.tool_events[0]["blocked"] and "tools" not in requests[0]


def test_timeout_preserves_prior_usage_and_never_retries_unknown_request(tmp_path, monkeypatch):
    import httpx
    import openai
    error = openai.APITimeoutError(request=httpx.Request("POST", "https://invalid.example/v1"))
    replies = [completion(calls=[call("search_skills", {"query": "adni"})],
                          usage={"prompt_tokens": 40, "completion_tokens": 8}), error]
    requests = fake_http(monkeypatch, replies, tmp_path / "events.jsonl")
    result = http_run(tmp_path)
    assert not result.ok and result.remote_outcome_unknown and not result.usage_complete
    assert len(requests) == 2 and (result.input_tokens, result.output_tokens) == (40, 8)
    assert result.retry_class == "timeout"


def test_length_stop_gets_one_logged_same_attempt_continuation(tmp_path, monkeypatch):
    partial = "## Solution Thinking\n" + ("A" * 80)
    first = completion(content=partial, usage={"prompt_tokens": 40, "completion_tokens": 120})
    first["choices"][0]["finish_reason"] = "length"
    second = completion(usage={"prompt_tokens": 60, "completion_tokens": 30})
    requests = fake_http(monkeypatch, [first, second], tmp_path / "events.jsonl")
    result = http_run(tmp_path, enabled=True)

    assert result.ok and result.request_count == 2
    assert result.runtime_settings["contract_recovery_attempts"] == 1
    assert result.runtime_settings["contract_recovery_tools"] == "disabled"
    assert result.response == ANSWER
    assert result.input_tokens == 100 and result.output_tokens == 150
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": partial}
    assert requests[1]["messages"][-1]["role"] == "user"
    assert "tools" not in requests[1]
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "contract_recovery" for row in rows)
    assert sum(row["event"] == "request" for row in rows) == 2


def test_length_recovery_preserves_partial_first_section_when_continuation_is_second_only(tmp_path, monkeypatch):
    partial = "## Solution Thinking\nA bounded explanation."
    second_only = "## Commands Or Code\n```python\nprint('fixture')\n```\n"
    first = completion(content=partial, usage={"prompt_tokens": 40, "completion_tokens": 120})
    first["choices"][0]["finish_reason"] = "length"
    second = completion(content=second_only, usage={"prompt_tokens": 60, "completion_tokens": 30})
    requests = fake_http(monkeypatch, [first, second], tmp_path / "events.jsonl")
    result = http_run(tmp_path, enabled=True)
    assert result.ok and "A bounded explanation." in result.response
    assert result.response.count("## Solution Thinking") == 1
    assert result.response.count("## Commands Or Code") == 1
    assert requests[1]["messages"][-2]["content"] == partial


@pytest.mark.parametrize("model", [
    "openai/openai/gpt-5.6-sol", "azure/anthropic/claude-opus-4-8",
    "xai/xai/grok-4.7", "gcp/google/gemini-3.1-pro-preview",
    "deepseek-v4-pro", "glm-5.3", "minimax-m3", "kimi-k3",
])
def test_http_requests_preserve_the_required_high_reasoning(tmp_path, monkeypatch, model):
    requests = fake_http(monkeypatch, [completion()], tmp_path / "events.jsonl")
    result = http_run(tmp_path, enabled=False, model=model)
    assert result.ok
    assert requests[0]["reasoning_effort"] == "high"
    assert result.reasoning_effort == "high"
    assert result.runtime_settings["request_overrides"]["reasoning_effort"] == "high"


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 10},
    {"prompt_tokens": -1, "completion_tokens": 3},
    {"prompt_tokens": True, "completion_tokens": 3},
    {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 11}}])
def test_incomplete_usage_cannot_be_published_as_a_success(tmp_path, monkeypatch, usage):
    fake_http(monkeypatch, [completion(usage=usage)], tmp_path / "events.jsonl")
    result = http_run(tmp_path)
    assert result.response == ANSWER and not result.usage_complete
    campaign.validate_outcome(result, condition())
    assert not result.ok and result.retry_class == "accounting" and not result.remote_outcome_unknown


def test_claude_counts_all_models_cache_creation_and_only_successful_skill_reads():
    rows = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "mcp__neuro_runtime__read_skill", "input": {"name": "loaded"}},
            {"type": "tool_use", "id": "b", "name": "mcp__neuro_runtime__read_skill", "input": {"name": "failed"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": '{"success":true}'}]},
            {"type": "tool_result", "tool_use_id": "b", "is_error": True, "content": '{"success":true}'}]}},
        {"type": "result", "result": ANSWER, "modelUsage": {
            "primary": {"inputTokens": 8, "cacheReadInputTokens": 89248, "cacheCreationInputTokens": 11916, "outputTokens": 5417},
            "auxiliary": {"inputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4, "outputTokens": 5}},
         "usage": {"input_tokens": 6, "output_tokens": 5399}}]
    result = parse_claude_result("\n".join(json.dumps(r) for r in rows))
    assert (result["input_tokens"], result["cached_input_tokens"], result["cache_creation_input_tokens"],
            result["uncached_input_tokens"], result["output_tokens"]) == (101181, 89251, 11920, 10, 5422)
    assert result["skills_loaded"] == ["loaded"] and len(result["tool_events"]) == 2
    assert result["usage_records"][0]["usage"]["input_tokens"] == 6


@pytest.mark.parametrize("parser,terminal", [
    (parse_cli_events, {"type": "turn.completed", "usage": {}}),
    (parse_claude_result, {"type": "result", "usage": {}}),
    (parse_claude_result, {"type": "result", "modelUsage": {"primary": {"inputTokens": 3}}}),
])
def test_native_cli_missing_usage_is_not_counted_as_complete(parser, terminal):
    assert not parser(json.dumps(terminal))["usage_complete"]


@pytest.mark.parametrize("enabled", [True, False])
def test_claude_launch_isolates_context_and_exposes_only_the_declared_tools(tmp_path, monkeypatch, enabled):
    from neurobench.runner import cli_drivers as cli
    monkeypatch.setenv("NK", "synthetic-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-inherited-token")
    monkeypatch.setenv("MAX_THINKING_TOKENS", "0")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
    monkeypatch.setattr(cli, "find_claude_cli", lambda: "synthetic.exe")
    config = cli.write_claude_mcp_config(tmp_path / "mcp.json", workspace=tmp_path,
                                        harness_enabled=enabled, skills_dir=tmp_path / "frozen")

    def capture(command, **kw):
        assert "--bare" in command and "--disable-slash-commands" in command
        assert command[command.index("--tools") + 1] == ""
        assert "ANTHROPIC_AUTH_TOKEN" not in kw["env"]
        assert kw["env"]["ANTHROPIC_API_KEY"] == "synthetic-key"
        assert command[command.index("--effort") + 1] == "high"
        assert "MAX_THINKING_TOKENS" not in kw["env"]
        assert kw["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
        assert ("--allowedTools" in command) is enabled
        servers = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]
        if enabled:
            assert set(servers) == {"neuro_runtime"}
            assert servers["neuro_runtime"]["env"]["NEURORUNTIME_MCP_SKILLS_DIR"] == str(tmp_path / "frozen")
        else:
            assert servers == {}
        return 0, "", json.dumps({"type": "result", "result": ANSWER,
            "usage": {"input_tokens": 4, "output_tokens": 7}}), ""

    monkeypatch.setattr(cli, "_capture", capture)
    result = cli.run_claude_code(prompt="synthetic", model="synthetic", base_url="https://invalid.example/v1",
        harness_enabled=enabled, workspace=tmp_path, output_root=tmp_path, case_id="T01", attempt=1,
        timeout_seconds=10, mcp_config_path=config)
    assert result.ok and result.usage_complete and result.runtime_settings["bare"]
    assert result.reasoning_effort == "high"


def test_every_attempt_retains_its_usage_and_unknown_is_never_redispatched(tmp_path, task):
    con = ledger.connect(tmp_path / "state.sqlite3")
    c = condition()
    ledger.seed_condition(con, c, [task], prompts={task.case_id: "frozen"})
    ledger.mark_started(con, c, task)
    assert ledger.mark_finished(con, c, task, outcome(ok=False, error="503", retry_class="transient"),
                                report={}, max_attempts=2) == "pending"
    ledger.mark_started(con, c, task)
    assert ledger.mark_finished(con, c, task, outcome(remote_outcome_unknown=True, ok=False,
        input_tokens=40, output_tokens=5, usage_complete=False), report={}, max_attempts=2) == "held_unknown"
    assert ledger.totals(con)["input_tokens"] == 140
    assert ledger.totals(con)["output_tokens"] == 15
    assert ledger.totals(con)["unknown_remote_attempts"] == 1
    assert con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
    with pytest.raises(RuntimeError, match="not dispatchable"):
        ledger.mark_started(con, c, task)
    with pytest.raises(RuntimeError, match="already settled"):
        ledger.mark_finished(con, c, task, outcome(), report={}, max_attempts=2)
    con.close()


def test_legacy_ledger_and_frozen_prompt_cannot_be_overwritten(tmp_path, task):
    old = tmp_path / "legacy.sqlite3"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE runs (condition TEXT)")
    con.commit()
    con.close()
    before = old.read_bytes()
    with pytest.raises(RuntimeError, match="historical evidence"):
        ledger.connect(old)
    assert old.read_bytes() == before
    con = ledger.connect(tmp_path / "new.sqlite3")
    c = condition()
    ledger.seed_condition(con, c, [task], prompts={task.case_id: "original"})
    with pytest.raises(RuntimeError, match="Frozen task/condition/prompt changed"):
        ledger.seed_condition(con, c, [task], prompts={task.case_id: "changed"})
    assert con.execute("SELECT prompt_sha256 FROM runs").fetchone()[0] == sha256_text("original")
    con.close()


def prepare(tmp_path, task, cs=None):
    root = tmp_path / "campaign"
    cs = cs or [condition()]
    campaign.freeze_manifest(root, cs, [task], campaign.parse_args([]))
    con = ledger.connect(root / "state.sqlite3")
    for c in cs:
        ledger.seed_condition(con, c, [task], prompts={task.case_id: prompts.render_prompt(task, harness_enabled=c.harness_enabled)})
    return root, con, cs[0]


@pytest.mark.parametrize("damage", ["none", "response", "receipt", "no_receipt"])
def test_crash_between_receipt_and_ledger_commit_is_recovered_only_with_bound_evidence(tmp_path, task, monkeypatch, damage):
    monkeypatch.setattr(campaign, "run_http_chat", lambda **kw: outcome())
    root, con, c = prepare(tmp_path, task)
    receipt = campaign._receipt_path(root, c, task, 1)
    ledger.mark_started(con, c, task, receipt_path=receipt)
    if damage != "no_receipt":
        result, report = campaign.run_one(output_root=root, condition=c, task=task,
            prompt=prompts.render_prompt(task, harness_enabled=c.harness_enabled), attempt=1, timeout_seconds=10)
        assert result.ok
        if damage == "response":
            Path(report["response_path"]).write_text("altered", encoding="utf-8")
        if damage == "receipt":
            record = json.loads(receipt.read_text(encoding="utf-8"))
            record["prompt_sha256"] = "wrong prompt"
            receipt.write_text(json.dumps(record), encoding="utf-8")
    ledger.recover_interrupted(con)
    expected = "success" if damage == "none" else "held_unknown"
    assert ledger.counts(con, c.key)[expected] == 1
    ledger.recover_interrupted(con)
    assert ledger.totals(con)["attempts"] == 1
    if damage == "none":
        assert ledger.totals(con)["input_tokens"] == 100
        assert scorer_compat.scoring._parse_benchmark_report_filename(Path(report["report_path"]))
    con.close()


def test_input_snapshot_preserves_bytes_and_detects_catalog_or_code_drift(tmp_path, task):
    root, con, c = prepare(tmp_path, task)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "core/skill_loader/loader.py" in manifest["source_sha256"]
    assert "core/tool_pipeline.py" in manifest["source_sha256"]
    for path, expected in manifest["source_sha256"].items():
        assert file_sha256(root / "source_snapshot" / path) == expected
    raw = b"windows\r\nline endings\r\n"
    freeze_bytes(tmp_path / "exact.bin", raw)
    assert (tmp_path / "exact.bin").read_bytes() == raw
    extra = root / "skill_snapshot" / "extra" / "SKILL.md"
    extra.parent.mkdir()
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen skill catalog changed"):
        campaign.verify_frozen_inputs(root)
    con.close()


def test_answer_contract_failure_is_preserved_without_report_or_retry(tmp_path, task, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "run_http_chat", lambda **kw: calls.append(1) or outcome(response="invalid shape"))
    root, con, c = prepare(tmp_path, task)
    campaign.run_condition(connection=con, output_root=root, condition=c, tasks=[task], workers=1,
                           timeout_seconds=10, max_attempts=3)
    assert calls == [1] and ledger.counts(con, c.key)["failed"] == 1
    row = con.execute("SELECT * FROM attempts").fetchone()
    assert row["retry_class"] == "contract" and row["input_tokens"] == 100
    report = json.loads(row["report_json"])
    assert not report["report_path"]
    assert Path(report["response_path"]).read_text(encoding="utf-8") == "invalid shape"
    con.close()


def test_dispatch_budget_and_resume_do_not_silently_move_to_the_next_tasks(tmp_path, task, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "run_http_chat", lambda **kw: calls.append(1) or outcome())
    root, con, c = prepare(tmp_path, task)
    second = SimpleNamespace(**(vars(task) | {"number": 2, "case_id": "T02"}))
    ledger.seed_condition(con, c, [second], prompts={second.case_id: prompts.render_prompt(second, harness_enabled=True)})
    for _ in range(2):
        campaign.run_condition(connection=con, output_root=root, condition=c, tasks=[task, second], workers=2,
            timeout_seconds=10, max_attempts=2, limit=1, budget_allows=lambda: ledger.totals(con)["attempts"] < 1)
    assert calls == [1]
    assert ledger.counts(con, c.key) == dict(pending=1, running=0, success=1, failed=0, held_unknown=0)
    con.close()


def test_only_one_runner_can_own_a_campaign_directory(tmp_path):
    with campaign.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another runner"):
            with campaign.campaign_lock(tmp_path):
                pytest.fail("Double dispatcher acquired the same ledger")
