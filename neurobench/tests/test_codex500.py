from __future__ import annotations

import json
from pathlib import Path


from neurobench.runner import codex500 as MODULE


def test_discovers_exactly_500_tasks() -> None:
    tasks = MODULE.discover_tasks(MODULE.DEFAULT_BENCHMARK_ROOT)
    assert len(tasks) == 500
    assert tasks[0].case_id == "T01"
    assert tasks[-1].case_id == "T500"


def test_condition_matrix_is_complete_and_report_names_are_unique() -> None:
    assert list(MODULE.CONDITIONS) == [
        "neuroclaw_gpt_5_6_sol_high",
        "codex_gpt_5_6_sol_high",
        "neuroclaw_gpt_5_5_high",
        "codex_gpt_5_5_high",
    ]
    observed = {
        (condition.model, condition.skills_enabled)
        for condition in MODULE.CONDITIONS.values()
    }
    assert observed == {
        ("gpt-5.6-sol", True),
        ("gpt-5.6-sol", False),
        ("gpt-5.5", True),
        ("gpt-5.5", False),
    }
    assert len({condition.report_model for condition in MODULE.CONDITIONS.values()}) == 4


def test_answer_contract_requires_exact_sections() -> None:
    good = "## Solution Thinking\n" + ("plan " * 12) + "\n## Commands Or Code\n```bash\n" + ("echo x\n" * 8) + "```"
    assert MODULE.validate_answer_contract(good) == ""
    bad = good + "\n## Extra\nno"
    assert "exactly two" in MODULE.validate_answer_contract(bad)


def test_event_parser_and_tool_policies() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "session-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "Get-Content -Raw 'skills\\fmri-skill\\SKILL.md'",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5},
                }
            ),
        ]
    )
    summary = MODULE.parse_codex_events(raw)
    assert summary["session_id"] == "session-1"
    assert summary["skill_files_read"] == ["fmri-skill"]
    with_skills = MODULE.CONDITIONS["neuroclaw_gpt_5_6_sol_high"]
    no_skills = MODULE.CONDITIONS["codex_gpt_5_5_high"]
    assert "forbidden tool activity" in MODULE.validate_tool_policy(with_skills, summary)
    assert "No-skills" in MODULE.validate_tool_policy(no_skills, summary)


def test_no_skill_empty_event_stream_passes_policy() -> None:
    summary = MODULE.parse_codex_events(
        json.dumps({"type": "thread.started", "thread_id": "session-2"})
    )
    no_skills = MODULE.CONDITIONS["codex_gpt_5_5_high"]
    assert MODULE.validate_tool_policy(no_skills, summary) == ""


def test_prompts_are_condition_separated() -> None:
    task = MODULE.discover_tasks(MODULE.DEFAULT_BENCHMARK_ROOT)[0]
    skills = MODULE.load_local_skills(MODULE.REPO_ROOT / "skills")
    candidates = MODULE.select_candidate_skills(task, skills)
    with_prompt = MODULE.render_prompt(
        task, MODULE.CONDITIONS["neuroclaw_gpt_5_6_sol_high"], candidates
    )
    without_prompt = MODULE.render_prompt(
        task, MODULE.CONDITIONS["codex_gpt_5_5_high"], candidates
    )
    with_prompt_gpt_5_5 = MODULE.render_prompt(
        task, MODULE.CONDITIONS["neuroclaw_gpt_5_5_high"], candidates
    )
    without_prompt_gpt_5_6 = MODULE.render_prompt(
        task, MODULE.CONDITIONS["codex_gpt_5_6_sol_high"], candidates
    )
    assert "preloading their complete frozen SKILL.md contents" in with_prompt
    assert "BEGIN PRELOADED NEUROCLAW SKILL" in with_prompt
    assert "Do not call any tool" in without_prompt
    assert "Frozen SKILL.md SHA-256:" in with_prompt
    assert "Frozen SKILL.md SHA-256:" not in without_prompt
    assert with_prompt_gpt_5_5 == with_prompt
    assert without_prompt_gpt_5_6 == without_prompt
    assert with_prompt != without_prompt


def test_manifest_condition_extension_is_strictly_additive() -> None:
    current = {
        key: MODULE.dataclasses.asdict(condition)
        for key, condition in MODULE.CONDITIONS.items()
    }
    diagonal = {
        key: current[key]
        for key in ("neuroclaw_gpt_5_6_sol_high", "codex_gpt_5_5_high")
    }
    assert MODULE.validate_additive_condition_extension(diagonal, current) == [
        "codex_gpt_5_6_sol_high",
        "neuroclaw_gpt_5_5_high",
    ]
    assert MODULE.validate_additive_condition_extension(current, current) == []

    changed = dict(diagonal)
    changed["codex_gpt_5_5_high"] = {
        **changed["codex_gpt_5_5_high"],
        "reasoning_effort": "low",
    }
    try:
        MODULE.validate_additive_condition_extension(changed, current)
    except RuntimeError as exc:
        assert "definitions changed" in str(exc)
    else:
        raise AssertionError("A changed frozen condition must be rejected")


def test_high_precision_skill_routing() -> None:
    tasks = MODULE.discover_tasks(MODULE.DEFAULT_BENCHMARK_ROOT)
    skills = MODULE.load_local_skills(MODULE.REPO_ROOT / "skills")
    by_id = {task.case_id: task for task in tasks}
    t250 = [skill.name for skill in MODULE.select_candidate_skills(by_id["T250"], skills)]
    t500 = [skill.name for skill in MODULE.select_candidate_skills(by_id["T500"], skills)]
    assert t250[0] in {"bids-organizer", "nibabel-skill"}
    assert "experiment-controller" in t500[:5]
    assert "statistical-ml" in t500[:5]


def test_condition_specific_sandboxes() -> None:
    workspace = MODULE.REPO_ROOT
    last_message = workspace / "last-message.md"
    for condition in MODULE.CONDITIONS.values():
        command = MODULE.build_codex_command(
            "codex",
            condition,
            workspace,
            last_message,
        )
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "sandbox_workspace_write.network_access=false" in command


def test_skill_snapshot_is_isolated_and_fingerprinted(tmp_path: Path) -> None:
    source_skills = MODULE.load_local_skills(MODULE.REPO_ROOT / "skills")[:2]
    mapped = MODULE.materialize_skill_snapshot(source_skills, tmp_path, write_files=True)
    assert all(skill.skill_md.is_file() for skill in mapped)
    assert all(tmp_path.resolve() in skill.skill_md.parents for skill in mapped)
    before = MODULE.cheap_workspace_fingerprint(tmp_path / "control" / "skill_snapshot")
    unexpected = tmp_path / "control" / "skill_snapshot" / "unexpected.txt"
    MODULE.atomic_write_text(unexpected, "mutation")
    after = MODULE.cheap_workspace_fingerprint(tmp_path / "control" / "skill_snapshot")
    assert before != after


def test_non_tool_error_event_is_not_counted_as_activity() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "session-3"}),
            json.dumps({"type": "item.completed", "item": {"type": "error", "message": "warning"}}),
        ]
    )
    summary = MODULE.parse_codex_events(raw)
    assert summary["tool_event_count"] == 0


def test_report_metadata_ignores_body_list_items() -> None:
    report = "\n".join(
        [
            "# Benchmark Report: T426_gpt-5.6-sol_withskills",
            "",
            "- Model: gpt-5.6-sol_withskills",
            "- Actual model: gpt-5.6-sol",
            "",
            "## Solution Thinking",
            "- Model: two-layer dense GCN",
            "",
            "## Commands Or Code",
            "```python",
            "pass",
            "```",
        ]
    )
    metadata = MODULE.parse_report_metadata(report)
    assert metadata["Model"] == "gpt-5.6-sol_withskills"
    assert metadata["Actual model"] == "gpt-5.6-sol"


def test_source_hash_matches_model_text_across_line_endings(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.md"
    crlf_path = tmp_path / "crlf.md"
    lf_path.write_bytes(b"# Task\n\nDo the work.\n")
    crlf_path.write_bytes(b"# Task\r\n\r\nDo the work.\r\n")

    assert MODULE.sha256_file(lf_path) != MODULE.sha256_file(crlf_path)
    assert MODULE.sha256_normalized_text_file(lf_path) == MODULE.sha256_normalized_text_file(
        crlf_path
    )
