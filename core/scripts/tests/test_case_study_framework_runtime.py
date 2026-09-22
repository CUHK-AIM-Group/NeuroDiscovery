from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from core.scripts import case_study_framework_runtime as framework_runtime
from core.scripts.biomni_case_study_batch_client import (
    enforce_blinded_resource_policy,
    parse_args as biomni_args,
)
from core.scripts.case_study_official_adapter_client import (
    _sciagents_final_artifact,
    compile_native_policy,
)
from core.scripts.run_case_study_framework_comparison import (
    _benchmark_external_outcomes,
    _external_validation_assignment,
    _complete_policy_matrix_present,
    build_task,
    chat_base_url,
    parse_args as framework_args,
)
from core.scripts.summarize_case_study_framework_comparison import (
    summarize_status_records,
)


ROOT = Path(__file__).resolve().parents[3]


def test_official_compiler_reads_csv_and_hash_candidate_ids(tmp_path: Path) -> None:
    candidate_id = "brain_age:0123456789abcdef01234567"
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "modality": "resting_state_fmri",
                "atlas": "schaefer_400_7net",
                "feature_family": "fc_edge_projection",
                "model": "ridge",
                "candidate_id": candidate_id,
            }
        ]
    ).to_csv(registry, index=False)

    policy, audit = compile_native_policy(
        method="virtual_lab",
        task={
            "case_study_id": "brain_age",
            "policy_schema_version": "case-study-search-policy.v1",
            "candidate_id_template": "brain_age:<24 hexadecimal characters>",
            "trial": 3,
            "n_anchors": 1,
            "public_registry_path": str(registry),
        },
        native_result={"summary": f"candidate_id: {candidate_id}"},
    )

    assert [row["candidate_id"] for row in policy["anchors"]] == [candidate_id]
    assert audit["invalid_exact_mentions"] == []


def test_task_prompt_exposes_only_public_factor_columns(tmp_path: Path) -> None:
    registry = pd.DataFrame(
        [
            {
                "modality": "resting_state_fmri",
                "atlas": "aal3_166",
                "feature_family": "fc_edge_projection",
                "model": "ridge",
                "candidate_id": "brain_age:0123456789abcdef01234567",
            }
        ]
    )
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    task = build_task(
        task="brain_age",
        method="ai_scientist_v2",
        trial=0,
        n_anchors=1,
        registry=registry,
        registry_path=registry_path,
    )

    assert "score_neurodiscovery" not in task["research_goal"]
    assert "kg_scoped_node_support" not in task["research_goal"]
    assert "brain_age:0123456789abcdef01234567" in task["research_goal"]
    assert task["native_retrieval_enabled"] is False
    assert task["open_coscientist_max_iterations"] == 0
    assert task["open_coscientist_overgeneration_factor"] == 1.0
    assert task["open_coscientist_tool_generation_enabled"] is False
    assert task["open_coscientist_debate_cohorts_enabled"] is True


def test_framework_benchmark_honours_registered_external_assignment(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_outcomes.csv"
    external.write_text("candidate_id,executable,validated\nx,false,false\n", encoding="utf-8")

    assert _benchmark_external_outcomes("disease_subtyping", tmp_path) is None
    assert _benchmark_external_outcomes("brain_age", tmp_path) == external
    external.unlink()
    with pytest.raises(FileNotFoundError, match="brain_age"):
        _benchmark_external_outcomes("brain_age", tmp_path)

    assert _external_validation_assignment("brain_age") == "required"
    assert _external_validation_assignment("disease_subtyping") == "registered_exempt"


def test_biomarker_task_compiles_exact_factor_rules(tmp_path: Path) -> None:
    registry = pd.DataFrame(
        [
            {
                "disease": "psychosis_SZ_SZA",
                "atlas": "schaefer_100_7net",
                "roi_index": "7",
                "anatomy": "insula",
                "feature_family": "amplitude",
                "feature": "roi_alff_proxy",
                "candidate_id": "biomarker_discovery:0123456789abcdef01234567",
            }
        ]
    )
    registry_path = tmp_path / "biomarker_registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    task = build_task(
        task="biomarker_discovery",
        method="biomni_native",
        trial=0,
        n_anchors=2,
        registry=registry,
        registry_path=registry_path,
    )
    assert task["policy_mode"] == "factor_rules"
    assert "score_neurodiscovery" not in task["research_goal"]
    policy, audit = compile_native_policy(
        method="biomni_native",
        task=task,
        native_result={
            "answer": """```json
            {"rules":[{"rank":1,"weight":0.9,"when":{"disease":"psychosis_SZ_SZA","feature_family":"amplitude"},"rationale":"test"}]}
            ```"""
        },
    )
    assert policy["anchors"] == []
    assert policy["rules"] == [
        {
            "weight": 0.9,
            "when": {
                "disease": "psychosis_SZ_SZA",
                "feature_family": "amplitude",
            },
            "rationale": "test",
        }
    ]
    assert audit["valid_unique_rules"] == 1


def test_sciagents_selects_rule_artifact_before_bare_termination() -> None:
    rule = {
        "rules": [
            {
                "rank": 1,
                "weight": 0.9,
                "when": {"disease": "psychosis_SZ_SZA"},
                "rationale": "test",
            }
        ]
    }
    artifact = json.dumps(rule) + "TERMINATE"
    history = [
        {"name": "critic", "content": artifact},
        {"name": "planner", "content": "TERMINATE"},
    ]

    assert _sciagents_final_artifact(history, factor_rule_mode=True) == artifact


def test_sciagents_compiler_recovers_rules_from_native_chat_history(
    tmp_path: Path,
) -> None:
    registry = pd.DataFrame(
        [
            {
                "disease": "psychosis_SZ_SZA",
                "atlas": "schaefer_100_7net",
                "roi_index": "7",
                "anatomy": "insula",
                "feature_family": "amplitude",
                "feature": "roi_alff_proxy",
                "candidate_id": "biomarker_discovery:0123456789abcdef01234567",
            }
        ]
    )
    registry_path = tmp_path / "registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True)
    task = build_task(
        task="biomarker_discovery",
        method="sciagents",
        trial=1,
        n_anchors=1,
        registry=registry,
        registry_path=registry_path,
    )
    artifact = json.dumps(
        {
            "rules": [
                {
                    "rank": 1,
                    "weight": 0.9,
                    "when": {"disease": "psychosis_SZ_SZA"},
                    "rationale": "test",
                }
            ]
        }
    ) + "TERMINATE"

    policy, audit = compile_native_policy(
        method="sciagents",
        task=task,
        native_result={
            "final_artifact": "TERMINATE",
            "chat_history": [
                {"name": "critic", "content": artifact},
                {"name": "planner", "content": "TERMINATE"},
            ],
        },
    )

    assert len(policy["rules"]) == 1
    assert audit["artifact_scope"] == (
        "sciagents_last_rule_artifact_from_chat_history"
    )


def test_chat_compatibility_url_is_derived_from_responses_root() -> None:
    assert chat_base_url("http://localhost:8080") == "http://localhost:8080/v1"
    assert chat_base_url("http://localhost:8080/v1") == "http://localhost:8080/v1"


def test_biomni_primary_comparison_disables_tool_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "biomni_case_study_batch_client.py",
            "--prompt",
            "prompt.txt",
            "--out",
            "out",
            "--workspace",
            "workspace",
            "--biomni-root",
            "biomni",
        ],
    )
    assert biomni_args().use_tool_retriever is False


def test_neuroruntime_prefers_named_handler_on_windows_path(tmp_path: Path) -> None:
    handler = tmp_path / "handler.py"
    handler.write_text(
        "from pathlib import Path\n\n"
        "def helper(value):\n"
        "    return {'wrong': value}\n\n"
        "def handler(input_data):\n"
        "    return {'seen': input_data['value']}\n",
        encoding="utf-8",
    )
    runtime_path = ROOT / "core/tool-runtime/runtime.py"
    spec = importlib.util.spec_from_file_location("test_neuroruntime", runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.ToolRuntime(timeout=30).run(handler, {"value": 7})

    assert result == {"success": True, "output": {"seen": 7}, "error": None}


def test_biomni_blinded_policy_selects_no_external_resources() -> None:
    class DummyAgent:
        selected_resources = None

        def update_system_prompt_with_selected_resources(self, resources) -> None:
            self.selected_resources = resources

    agent = DummyAgent()
    selected = enforce_blinded_resource_policy(agent)

    assert selected == {
        "tools": [],
        "data_lake": [],
        "libraries": [],
        "know_how": [],
    }
    assert agent.selected_resources == selected


def test_framework_status_summary_keeps_latest_trial_record() -> None:
    frame = summarize_status_records(
        [
            {
                "method": "brainpilot_native",
                "trial": 0,
                "status": "failed",
            },
            {
                "method": "brainpilot_native",
                "trial": 0,
                "status": "complete",
                "duration_seconds": 10.0,
                "valid_anchors": 8,
                "requested_anchors": 10,
            },
            {
                "method": "brainpilot_native",
                "trial": 1,
                "status": "complete",
                "duration_seconds": 14.0,
                "valid_anchors": 10,
                "requested_anchors": 10,
            },
        ],
        task="prognosis",
    )

    row = frame.iloc[0]
    assert row["completed_trials"] == 2
    assert row["failed_trials"] == 0
    assert row["duration_seconds_mean"] == 12.0
    assert row["duration_seconds_variance"] == 8.0
    assert row["valid_anchor_fraction_mean"] == 0.9


def test_framework_status_summary_counts_reused_trial_and_retains_runtime() -> None:
    frame = summarize_status_records(
        [
            {
                "method": "brainpilot_native",
                "trial": 0,
                "status": "complete",
                "duration_seconds": 9.0,
                "valid_anchors": 10,
                "requested_anchors": 10,
            },
            {
                "method": "brainpilot_native",
                "trial": 0,
                "status": "reused",
                "valid_anchors": 10,
            },
        ],
        task="brain_age",
    )

    row = frame.iloc[0]
    assert row["completed_trials"] == 1
    assert row["runtime_observed_trials"] == 1
    assert row["reused_trials"] == 1
    assert row["failed_trials"] == 0
    assert row["duration_seconds_mean"] == 9.0


def test_native_exact_id_prompt_removes_prior_registry_rows() -> None:
    first = "brain_age:0123456789abcdef01234567"
    second = "brain_age:89abcdef0123456701234567"
    task = {
        "policy_mode": "exact_candidate_ids",
        "research_goal": (
            "Blinded executable task.\n\n"
            "Only the following public factor values are executable:\n\n"
            f"- {first} :: atlas=aal3_166; model=ridge\n"
            f"- {second} :: atlas=schaefer_400_7net; model=svr\n\n"
            "Prioritize scientific plausibility."
        ),
    }

    prompt = framework_runtime._native_prompt(
        task,
        method="biomni_native",
        start_rank=2,
        end_rank=2,
        previous_ids=[first],
    )

    assert first not in prompt
    assert second in prompt
    assert "have been removed from the registry" in prompt


def test_native_exact_id_prompt_fails_closed_for_unknown_prior_id() -> None:
    task = {
        "policy_mode": "exact_candidate_ids",
        "research_goal": (
            "Only the following public factor values are executable:\n\n"
            "- brain_age:0123456789abcdef01234567 :: atlas=aal3_166"
        ),
    }

    try:
        framework_runtime._native_prompt(
            task,
            method="biomni_native",
            start_rank=2,
            end_rank=2,
            previous_ids=["brain_age:ffffffffffffffffffffffff"],
        )
    except ValueError as exc:
        assert "not present in the public registry" in str(exc)
    else:
        raise AssertionError("unknown prior IDs must fail closed")


def test_framework_artifact_write_retries_transient_share_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "nested" / "artifact.txt"
    original_write_text = Path.write_text
    calls = 0

    def flaky_write_text(self, data, *args, **kwargs):
        nonlocal calls
        if self == target:
            calls += 1
            if calls < 3:
                raise FileNotFoundError("temporary SMB path loss")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)
    monkeypatch.setattr(framework_runtime.time, "sleep", lambda _seconds: None)

    framework_runtime._write_text_with_retry(target, "complete")

    assert calls == 3
    assert target.read_text(encoding="utf-8") == "complete"


def test_framework_default_request_timeout_covers_router_retries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_case_study_framework_comparison.py"])

    args = framework_args()

    assert args.request_timeout_seconds == 1800
    assert args.timeout_seconds == 3600


def test_official_runtime_disables_sciagents_outer_retries(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setenv("SCIAGENTS_LLM_MAX_RETRIES", "20")
    monkeypatch.setattr(
        framework_runtime,
        "_repository_python",
        lambda _method, _repo: Path(sys.executable),
    )

    def capture_run_process(_command, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(framework_runtime, "_run_process", capture_run_process)
    framework_runtime._run_official(
        {
            "method": "sciagents",
            "output_dir": str(tmp_path / "out"),
            "task_path": str(tmp_path / "task.json"),
            "model": "deepseek-v4-pro",
            "chat_base_url": "http://127.0.0.1:18182/v1",
            "reasoning_effort": "high",
            "request_timeout_seconds": 1800,
            "timeout_seconds": 3600,
            "max_retries": 1,
        },
        "local-secret",
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SCIAGENTS_LLM_MAX_RETRIES"] == "0"
    assert env["CASE_STUDY_LLM_REQUEST_TIMEOUT"] == "1800"


def test_complete_policy_matrix_resume_skips_only_when_every_trial_exists(
    tmp_path: Path,
) -> None:
    args = type(
        "Args",
        (),
        {
            "root": tmp_path,
            "output_name": "frameworks",
            "tasks": ["brain_age", "prognosis"],
            "methods": ["virtual_lab", "sciagents"],
            "trials": 2,
        },
    )()
    expected = [
        tmp_path
        / task
        / "frameworks"
        / method
        / f"trial_{trial:02d}"
        / "search_policy.json"
        for task in args.tasks
        for method in args.methods
        for trial in range(args.trials)
    ]
    for path in expected[:-1]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert _complete_policy_matrix_present(args) is False

    expected[-1].parent.mkdir(parents=True, exist_ok=True)
    expected[-1].write_text("{}", encoding="utf-8")
    assert _complete_policy_matrix_present(args) is True
