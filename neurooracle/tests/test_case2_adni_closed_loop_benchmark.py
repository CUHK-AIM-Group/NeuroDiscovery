from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts import case2_official_baseline_experiment as baseline_experiment
from neurooracle.scripts.case2_closed_loop_luna_cli_worker import (
    build_provider_prompt,
    next_capture_attempt,
    validate_inner_response_schema,
)

from neurooracle.scripts.case2_closed_loop_luna_thread_gateway import (
    MODEL,
    ReplayServer,
    _response_artifacts,
    chat_response,
    responses_response,
    seal_response,
)
from neurooracle.scripts.evaluate_case2_adni_closed_loop_benchmark import (
    discrete_cumulative_gain_auc,
    ndcg_at_k,
    normalized_auc,
)
from neurooracle.scripts.materialize_case2_closed_loop_subgraph import materialize
from neurooracle.scripts.prepare_case2_adni_closed_loop_benchmark import (
    build_oracle,
    load_json,
    source_code_paths,
    validate_config,
)
from neurooracle.scripts.run_case2_adni_closed_loop_benchmark import (
    INVALID_PREFIX,
    _configure_frozen_baselines_root,
    _feedback_text,
    feedback_for_ids,
    parse_args as parse_run_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "neurooracle/configs/case2_adni_closed_loop_benchmark_v3_luna_seed1.json"


def _public() -> pd.DataFrame:
    rows = []
    for pathway in range(7):
        for marker in range(8):
            for outcome in ("ADAS13", "FAQ", "LDELTOTAL"):
                rows.append(
                    {
                        "candidate_id": f"score_{pathway}|pet|marker_{marker}|{outcome}",
                        "exposure": f"score_{pathway}",
                        "pathway_id": f"pathway_{pathway}",
                        "pathway_name": f"Pathway {pathway}",
                        "pathway_source": "synthetic",
                        "threshold_label": "p1em03",
                        "gene_count": str(pathway + 2),
                        "modality": "pet",
                        "marker": f"marker_{marker}",
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows).sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _formal(public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, candidate_id in enumerate(public["candidate_id"]):
        indirect_p = 0.01 if index < 15 else 0.5
        component_p = 0.01 if index < 13 else 0.5
        rows.append(
            {
                "candidate_id": candidate_id,
                "complete_case_n": 500,
                "a_path_std": 0.1,
                "a_path_hc3_p": component_p,
                "b_path_std": 0.2,
                "b_path_hc3_p": component_p,
                "indirect_effect_std": 0.02,
                "indirect_bootstrap_ci_lower": 0.001,
                "indirect_bootstrap_ci_upper": 0.04,
                "indirect_bootstrap_p": indirect_p,
                "indirect_bootstrap_q_family_8": 0.04 if index < 6 else 0.5,
                "indirect_bootstrap_q_global_168": 0.5,
                "analysis_status": "estimated",
            }
        )
    return pd.DataFrame(rows)


def test_closed_loop_protocol_is_full_universe_single_seed() -> None:
    config = load_json(CONFIG)
    validate_config(config)
    assert sum(config["sequential_design"]["main_round_batch_sizes"]) == 168
    assert config["sequential_design"]["trial_count"] == 1
    assert config["sequential_design"]["seed"] == 247629302
    assert config["model_backend"]["logical_model"] == "gpt-5.6-luna"
    assert config["model_backend"]["reasoning_effort"] == "max"
    assert config["model_backend"]["framework_process_retries"] == 1
    assert config["model_backend"]["execution_channel"] == "codex_thread_replay"
    assert config["model_backend"]["provider_api_calls"] is False
    assert config["model_backend"]["provider_keys_required"] is False
    assert config["model_backend"]["request_hash_lock"] is True
    assert config["model_backend"]["response_hash_lock"] is True
    assert config["model_backend"]["dispatch_delivery_mode"] == (
        "message-specified single read-only dispatch file"
    )
    assert config["model_backend"]["response_capture_mode"] == (
        "codex-cli output-last-message direct capture"
    )
    assert "CURRENT_DISPATCH.txt" in config["model_backend"]["dispatch_materialization_mode"]
    assert config["model_backend"]["provider_command_audit_required"] is True
    assert config["hybrid_resume"]["reused_method"] == "neurodiscovery"
    assert config["hybrid_resume"]["rerun_neurodiscovery"] is False
    parsed = parse_run_args(["--benchmark-root", str(Path("benchmark"))])
    assert parsed.max_retries == 1


def test_runner_binds_framework_to_frozen_baseline_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutable_default = tmp_path / "mutable-default"
    monkeypatch.setattr(baseline_experiment, "BASELINES_ROOT", mutable_default)
    configured = _configure_frozen_baselines_root(
        {"paths": {"baseline_repository_root": str(tmp_path)}}
    )
    assert configured == tmp_path.resolve()
    assert baseline_experiment.BASELINES_ROOT == tmp_path.resolve()
    with pytest.raises(ValueError, match="Frozen baseline repository root is missing"):
        _configure_frozen_baselines_root(
            {"paths": {"baseline_repository_root": str(tmp_path / "missing")}}
        )


def test_frozen_source_pins_include_local_runtime_import_closure() -> None:
    relative = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in source_code_paths(CONFIG)
    }
    assert "core/scripts/case_study_search_policy.py" in relative
    assert "core/scripts/case_study_neurodiscovery_policy.py" in relative
    assert "neurooracle/src/feedback_state.py" in relative
    assert "neurooracle/scripts/case2_closed_loop_luna_thread_gateway.py" in relative
    assert "neurooracle/scripts/case2_closed_loop_luna_cli_worker.py" in relative
    assert "neurooracle/configs/case2_luna_response_envelope.schema.json" in relative
    assert "neurooracle/scripts/start_case2_closed_loop_luna_cli_worker.ps1" in relative
    assert "neurooracle/data/case_study_reaudit/full_graph_v3/RUBRIC.md" in relative
    assert "neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md" in relative


def test_oracle_feedback_status_uses_no_q_value(tmp_path: Path) -> None:
    public = _public()
    formal_path = tmp_path / "formal.csv"
    _formal(public).to_csv(formal_path, index=False)
    oracle = build_oracle(formal_path, public)
    assert oracle["feedback_status"].eq("supported").sum() == 13
    indexed = oracle.set_index("candidate_id", drop=False)
    feedback = feedback_for_ids(indexed, public["candidate_id"].head(2).tolist())
    assert feedback
    assert not any("q_" in key or "_q" in key for row in feedback for key in row)
    text = _feedback_text(feedback)
    assert "bootstrap P=" in text
    assert "q=" not in text


def test_normalized_auc_uses_random_and_oracle_bounds() -> None:
    utility = np.linspace(2.0, 0.01, 168)
    perfect = normalized_auc(utility, utility, horizon=168)
    assert np.isclose(perfect["random_oracle_normalized_auc"], 1.0)
    reverse = normalized_auc(utility[::-1], utility, horizon=168)
    assert reverse["random_oracle_normalized_auc"] < 0
    assert discrete_cumulative_gain_auc(utility, horizon=168) > 0
    assert np.isclose(ndcg_at_k(utility, utility, 20), 1.0)


def test_invalid_action_has_zero_gain_and_consumes_horizon() -> None:
    utility = np.linspace(2.0, 0.01, 168)
    ideal = normalized_auc(utility, utility, horizon=168)
    penalized_values = np.concatenate(([0.0], utility[:-1]))
    penalized = normalized_auc(penalized_values, utility, horizon=168)
    assert penalized["raw_auc"] < ideal["raw_auc"]
    assert INVALID_PREFIX.startswith("__INVALID_")


def test_luna_thread_gateway_hash_locks_request_and_raw_answer(tmp_path: Path) -> None:
    config = load_json(CONFIG)
    backend = config["model_backend"]
    server = ReplayServer(
        ("127.0.0.1", 0),
        data_dir=tmp_path,
        thread_id=backend["codex_thread_id"],
        host_id=backend["codex_host_id"],
        timeout_seconds=0.1,
    )
    try:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Return JSON"}],
            "reasoning_effort": "max",
            "stream": False,
        }
        record = server.materialize_request("/v1/chat/completions", body)
        request_id = record["request_id"]
        dispatch = tmp_path / "dispatches" / f"{request_id}.txt"
        assert record["request_sha256"] in dispatch.read_text(encoding="utf-8")
        raw = tmp_path / "responses" / f"{request_id}.txt"
        raw.write_text(
            json.dumps(
                {
                    "message": {"role": "assistant", "content": "{\"ok\":true}"},
                    "finish_reason": "stop",
                }
            ),
            encoding="utf-8",
        )
        meta = seal_response(
            tmp_path,
            request_id,
            thread_id=backend["codex_thread_id"],
            host_id=backend["codex_host_id"],
            provider_turn_id="turn_test",
            provider_message_id="message_test",
        )
        envelope, sealed, raw_path, _ = _response_artifacts(tmp_path, record)
        assert sealed == meta
        assert raw_path == raw
        chat = chat_response(record, envelope)
        responses = responses_response(record, envelope)
        assert chat["model"] == MODEL
        assert chat["choices"][0]["message"]["content"] == "{\"ok\":true}"
        assert responses["output"][0]["content"][0]["text"] == "{\"ok\":true}"
        assert meta["manual_content_edit_after_provider_response"] is False
    finally:
        server.server_close()


def test_luna_worker_validates_inner_structured_response_before_sealing() -> None:
    record = {
        "request_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"candidate_ids": {"type": "array", "minItems": 2}},
                        "required": ["candidate_ids"],
                        "additionalProperties": False,
                    }
                },
            }
        }
    }
    valid = {
        "message": {
            "role": "assistant",
            "content": json.dumps({"candidate_ids": ["a", "b"]}),
            "tool_calls": [],
        },
        "finish_reason": "stop",
    }
    result = validate_inner_response_schema(record, valid)
    assert result["validated"] is True
    invalid = {
        "message": {
            "role": "assistant",
            "content": json.dumps({"candidate_ids": ["a"]}),
            "tool_calls": [],
        },
        "finish_reason": "stop",
    }
    with pytest.raises(Exception, match="is too short"):
        validate_inner_response_schema(record, invalid)


def test_luna_worker_prompt_constrains_inner_json_and_preserves_attempt_numbers(
    tmp_path: Path,
) -> None:
    dispatch = tmp_path / "dispatch.txt"
    dispatch.write_text("dispatch", encoding="utf-8")
    record = {
        "request_id": "luna_test",
        "request_sha256": "a" * 64,
        "request_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "required": ["first", "last"],
                    }
                },
            }
        },
    }
    prompt = build_provider_prompt(record, dispatch, attempt=3)
    assert "capture attempt 3" in prompt
    assert "最后一个右花括号后不得有任何字符" in prompt
    assert '["first", "last"]' in prompt
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "a1.json").write_text("{}", encoding="utf-8")
    (capture / "a2.json").write_text("{}", encoding="utf-8")
    (capture / "a2.jsonl").write_text("ignored", encoding="utf-8")
    assert next_capture_attempt(capture) == 3


def test_imported_replay_receipt_retains_source_thread(tmp_path: Path) -> None:
    old_thread = "00000000-0000-0000-0000-000000000001"
    new_thread = "00000000-0000-0000-0000-000000000002"
    source = ReplayServer(
        ("127.0.0.1", 0), data_dir=tmp_path, thread_id=old_thread,
        host_id="local", timeout_seconds=0.1,
    )
    try:
        record = source.materialize_request(
            "/v1/chat/completions",
            {"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        )
        request_id = record["request_id"]
        raw = tmp_path / "responses" / f"{request_id}.txt"
        raw.write_text(
            json.dumps({
                "message": {"role": "assistant", "content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }),
            encoding="utf-8",
        )
        seal_response(
            tmp_path, request_id, thread_id=old_thread, host_id="local",
            provider_turn_id="old_turn", provider_message_id="old_message",
        )
        envelope, _, raw_path, meta_path = _response_artifacts(tmp_path, record)
    finally:
        source.server_close()
    destination = ReplayServer(
        ("127.0.0.1", 0), data_dir=tmp_path, thread_id=new_thread,
        host_id="local", timeout_seconds=0.1,
    )
    try:
        destination.write_receipt(record, chat_response(record, envelope), raw_path, meta_path)
        receipt = json.loads(
            (tmp_path / "receipts" / f"{request_id}.json").read_text(encoding="utf-8")
        )
        assert receipt["thread_id"] == old_thread
        assert receipt["thread_id"] != new_thread
    finally:
        destination.server_close()


def test_case2_scoped_graph_materializer_is_result_blind(tmp_path: Path) -> None:
    claims = tmp_path / "claims.jsonl"
    rows = [
        {
            "id": "CLM:one", "subject_id": "GENE:APOE", "subject_name": "APOE",
            "predicate": "affects", "object_id": "IM:amyloid", "object_name": "amyloid PET",
            "confidence": 0.8, "raw_text": "APOE affects amyloid PET",
            "metadata": {"subject_type": "gene", "object_type": "imaging_marker"},
            "claim_case_study_ids": ["case2_pathway_mediation"],
        },
        {
            "id": "CLM:other", "subject_id": "A", "subject_name": "A",
            "predicate": "associated_with", "object_id": "B", "object_name": "B",
            "confidence": 0.5, "metadata": {}, "claim_case_study_ids": ["brain_age"],
        },
    ]
    claims.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # Reproduce the deep-path geometry of the benchmark while keeping the
    # final artifact below MAX_PATH.  The former verbose staging basename
    # crossed the boundary; the compact staging basename must not.
    output_parent = tmp_path
    while len(str(output_parent)) < 215:
        remaining = max(1, 215 - len(str(output_parent)) - 1)
        output_parent /= "u" * min(50, remaining)
    output = output_parent / "case2.json"
    manifest = materialize(claims, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["claim_count"] == 1
    assert set(payload["concepts"]) == {"GENE:APOE", "IM:amyloid", "CLM:one"}
    assert payload["edges"][0]["metadata"]["claim_id"] == "CLM:one"
    assert "indirect_bootstrap_p" not in output.read_text(encoding="utf-8")
    assert not list(output_parent.glob(".c2g_*"))
