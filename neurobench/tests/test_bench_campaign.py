"""Offline tests for the unified 10,000-run runner (no network, no model calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurobench.runner import campaign, conditions as conditions_mod  # noqa: E402
from neurobench.runner import ledger as ledger_mod  # noqa: E402
from neurobench.runner import prompts as prompts_mod  # noqa: E402
from neurobench.runner import scorer_compat  # noqa: E402
from neurobench.runner.cli_drivers import parse_claude_result, parse_cli_events  # noqa: E402
from neurobench.runner.drivers import AttemptOutcome, classify_retry  # noqa: E402


def test_condition_matrix_is_the_10000_run_design():
    matrix = conditions_mod.condition_matrix()
    assert matrix["group1"]["runs"] == 8000
    assert matrix["group2"]["runs"] == 2000
    assert matrix["total_runs"] == 10000
    assert matrix["total_conditions"] == 20
    assert len(conditions_mod.CONDITIONS) == 20
    assert all(c.reasoning_effort == "high" for c in conditions_mod.CONDITIONS.values())
    assert all(c.request_overrides["reasoning_effort"] == "high"
               for c in conditions_mod.CONDITIONS.values() if c.driver == "http_chat")


def test_every_report_model_is_unique_and_scorer_shaped():
    names = [condition.report_model for condition in conditions_mod.CONDITIONS.values()]
    assert len(names) == len(set(names))
    for name in names:
        assert name.endswith(conditions_mod.WITH_SKILLS_SUFFIX) or name.endswith(conditions_mod.NO_SKILLS_SUFFIX)
        assert "/" not in name and ":" not in name


def test_each_model_has_exactly_one_harness_arm_and_one_baseline_arm():
    for group in (1, 2):
        model_arms: dict = {}
        for condition in conditions_mod.CONDITIONS.values():
            if condition.group != group:
                continue
            key = (condition.model, condition.driver)
            model_arms.setdefault(key, set()).add(condition.harness_enabled)
        for key, arms in model_arms.items():
            assert arms == {True, False}, key


def test_scorer_discovers_all_500_tasks():
    report = scorer_compat.audit(REPO_ROOT / "neurobench" / "tasks")
    assert report["task_count"] == 500
    assert report["expected_500"] is True
    assert all(case_id.isdigit() for case_id in report["task_ids"])


def test_report_filenames_round_trip_through_the_scorer_parser():
    module = scorer_compat.scoring
    for condition in conditions_mod.CONDITIONS.values():
        stem = "17_%s_run1" % condition.report_model
        parsed = module._parse_benchmark_report_filename(Path(stem + ".md"))
        assert parsed is not None, condition.key
        case_id, model_name, run_index = parsed
        assert case_id == "17"
        assert model_name == condition.report_model
        assert run_index == 1


def test_ledger_seed_start_finish_roundtrip(tmp_path):
    connection = ledger_mod.connect(tmp_path / "state.sqlite3")

    class Task:
        case_id = "T01"
        number = 1
        path = REPO_ROOT / "neurobench" / "tasks" / "T01_academic_search" / "task.md"
        source_sha256 = "a" * 64

    condition = next(iter(conditions_mod.CONDITIONS.values()))
    assert ledger_mod.seed_condition(connection, condition, [Task()]) == 1
    # Seeding is idempotent so a resume never duplicates rows.
    assert ledger_mod.seed_condition(connection, condition, [Task()]) == 0
    assert ledger_mod.counts(connection, condition.key)["pending"] == 1

    ledger_mod.mark_started(connection, condition, Task())
    outcome = AttemptOutcome(
        ok=True, error="", retry_class="", session_id="s1", response="r",
        input_tokens=1, cached_input_tokens=0, output_tokens=2, elapsed_seconds=0.5,
        tool_events=[{"name": "search_skills", "ok": True}], skills_loaded=["adni-skill"],
    )
    status = ledger_mod.mark_finished(
        connection, condition, Task(), outcome,
        report={"response_sha256": "b", "report_path": "p", "report_sha256": "c",
                "events_path": "", "stderr_path": ""},
        max_attempts=3,
    )
    assert status == "success"
    row = connection.execute("SELECT * FROM runs WHERE condition=? AND case_id='T01'",
                             (condition.key,)).fetchone()
    assert row["status"] == "success"
    assert row["tool_event_count"] == 1
    assert json.loads(row["skills_loaded_json"]) == ["adni-skill"]


def test_failed_attempt_returns_to_pending_until_max_attempts(tmp_path):
    connection = ledger_mod.connect(tmp_path / "state.sqlite3")

    class Task:
        case_id = "T02"
        number = 2
        path = REPO_ROOT / "neurobench" / "tasks" / "T02_bids_organizer" / "task.md"
        source_sha256 = "a" * 64

    condition = next(iter(conditions_mod.CONDITIONS.values()))
    ledger_mod.seed_condition(connection, condition, [Task()])
    bad = AttemptOutcome(ok=False, error="boom", retry_class="transient", session_id="",
                         response="", input_tokens=0, cached_input_tokens=0,
                         output_tokens=0, elapsed_seconds=0.0)
    report = {"response_sha256": "", "report_path": "", "report_sha256": "",
              "events_path": "", "stderr_path": ""}
    for attempt in (1, 2):
        ledger_mod.mark_started(connection, condition, Task())
        assert ledger_mod.mark_finished(connection, condition, Task(), bad,
                                        report=report, max_attempts=3) == "pending"
    ledger_mod.mark_started(connection, condition, Task())
    assert ledger_mod.mark_finished(connection, condition, Task(), bad,
                                    report=report, max_attempts=3) == "failed"


def test_baseline_prompt_is_the_published_no_skills_contract():
    class Task:
        case_id = "T01"
        source_sha256 = "a" * 64
        text = "# Task\n\nbody\n"

    baseline = prompts_mod.render_prompt(Task(), harness_enabled=False)
    harness = prompts_mod.render_prompt(Task(), harness_enabled=True)
    assert prompts_mod.baseline_contract() in baseline
    assert "NeuroRuntime harness mode" in harness
    assert harness != baseline
    # The harness arm permits the skill tools instead of forbidding every tool.
    assert "search_skills" in harness
    assert "Do not call any tool." in baseline


def test_answer_contract_accepts_the_two_section_shape():
    good = ("## Solution Thinking\n" + "plan " * 12
            + "\n\n## Commands Or Code\n```bash\n" + "echo x\n" * 8 + "```\n")
    assert prompts_mod.validate_answer_contract(good) == ""
    assert prompts_mod.validate_answer_contract("no headings here") != ""


def test_leading_preamble_is_structurally_normalized_and_raw_text_is_retained():
    good = ("## Solution Thinking\n" + "plan " * 12
            + "\n\n## Commands Or Code\n```bash\n" + "echo x\n" * 8 + "```\n")
    raw = "I have the full skill guidance loaded.\n\n" + good
    normalized, kind = prompts_mod.normalize_answer_for_contract(raw)
    assert kind == "leading_preamble_stripped"
    assert normalized == good
    result = AttemptOutcome(True, "", "", "session", raw, 1, 0, 1, .01)
    condition = next(c for c in conditions_mod.CONDITIONS.values()
                     if c.driver == "http_chat" and c.harness_enabled)
    campaign.validate_outcome(result, condition)
    assert result.ok and result.response == good
    assert result.raw_response == raw
    assert result.runtime_settings["answer_normalization"] == "leading_preamble_stripped"
    assert ledger_mod.outcome_record(result).get("raw_response") is None


def test_codex_event_parser_dedupes_in_progress_and_completed_mcp_items():
    raw = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "sess-1"}),
        json.dumps({"type": "item.completed", "item": {
            "id": "item_1", "type": "mcp_tool_call", "server": "neuro_runtime",
            "tool": "read_skill", "arguments": {"name": "adni-skill"}, "status": "in_progress"}}),
        json.dumps({"type": "item.completed", "item": {
            "id": "item_1", "type": "mcp_tool_call", "server": "neuro_runtime",
            "tool": "read_skill", "arguments": {"name": "adni-skill"}, "status": "completed",
            "result": {"content": [{"type": "text", "text": "{\"success\": true}"}]}}}),
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}}),
    ])
    summary = parse_cli_events(raw)
    assert summary["session_id"] == "sess-1"
    assert len(summary["tool_events"]) == 1
    assert summary["tool_events"][0]["name"] == "neuro_runtime/read_skill"
    assert summary["tool_events"][0]["ok"] is True
    assert summary["skills_loaded"] == ["adni-skill"]
    assert (summary["input_tokens"], summary["cached_input_tokens"], summary["output_tokens"]) == (100, 40, 20)


def test_claude_result_parser_reads_usage_and_errors():
    payload = {
        "type": "result", "session_id": "abc", "is_error": False, "subtype": "success",
        "result": "answer", "usage": {"input_tokens": 11, "cache_read_input_tokens": 3, "output_tokens": 7},
    }
    summary = parse_claude_result(json.dumps(payload))
    assert summary["session_id"] == "abc"
    assert summary["response"] == "answer"
    assert (summary["input_tokens"], summary["cached_input_tokens"], summary["output_tokens"]) == (14, 3, 7)
    assert summary["uncached_input_tokens"] == 11
    assert summary["is_error"] is False

    error_payload = dict(payload, is_error=True, subtype="error_max_turns", api_error_status=429)
    summary = parse_claude_result(json.dumps(error_payload))
    assert summary["is_error"] is True


def test_retry_classification_separates_transient_from_contract():
    assert classify_retry("429 rate limit exceeded") == "rate_limit"
    assert classify_retry("request timed out after 900s") == "timeout"
    assert classify_retry("Answer must begin with ## Solution Thinking") == "contract"
    assert classify_retry("503 no available channel") == "transient"
    assert classify_retry("some unknown failure") == "unknown"
