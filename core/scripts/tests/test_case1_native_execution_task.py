from __future__ import annotations

from argparse import Namespace
import json

import numpy as np
import pandas as pd

from core.scripts.case1_native_execution_task import (
    RESULT_FIELDS,
    execute_neuroruntime,
    extract_json,
    score_results,
    select_candidate_ids,
    select_candidates,
)
from core.scripts.case1_exhaustive_v2 import ols_case_effect


def test_candidate_selection_does_not_use_effect_magnitude():
    rows = []
    for index in range(5):
        rows.append(
            {
                "modality": "fmri",
                "source": "atlas_multiatlas",
                "disease": f"disease_{index}",
                "feature": f"feature_{index}",
                "roi_index": index,
                "adjusted_residual_d": 0.1 + index,
            }
        )
    original = pd.DataFrame(rows)
    changed = original.copy()
    changed["adjusted_residual_d"] = [-9.0, 7.0, -5.0, 3.0, -1.0]

    selected_original = select_candidates(
        original, sources=["atlas_multiatlas"], n_candidates=3, seed=11
    )
    selected_changed = select_candidates(
        changed, sources=["atlas_multiatlas"], n_candidates=3, seed=11
    )

    assert selected_original["candidate_id"].tolist() == selected_changed[
        "candidate_id"
    ].tolist()


def test_score_results_requires_exact_complete_registered_submission(tmp_path):
    bundle = tmp_path / "bundle"
    hidden = bundle / "hidden"
    hidden.mkdir(parents=True)
    expected = {
        "candidate_id": "fmri|atlas|disease|feature|0",
        "n_case": 10,
        "n_control": 20,
        "adjusted_beta_case_minus_control": 0.2,
        "adjusted_beta_se": 0.1,
        "adjusted_t": 2.0,
        "p_value": 0.05,
        "adjusted_residual_d": 0.3,
    }
    (hidden / "gold_results.json").write_text(
        json.dumps({"results": [expected]}), encoding="utf-8"
    )
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "results": [
                    expected,
                    {
                        "candidate_id": "unknown",
                        **{field: 0 for field in RESULT_FIELDS},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "audit"

    score_results(
        Namespace(
            bundle_dir=bundle,
            result_file=result_file,
            method="test",
            trial=1,
            out_dir=out_dir,
            rtol=1e-4,
            atol=1e-7,
        )
    )

    summary = json.loads(
        (out_dir / "execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["successful_candidates"] == 1
    assert summary["execution_success_rate"] == 1.0
    assert summary["unknown_result_count"] == 1
    assert summary["complete_valid_submission"] is False


def test_extract_json_reassembles_brainpilot_stream_chunks():
    log = """[TEXT_MESSAGE_START] agent=principal
[TEXT_MESSAGE_CONTENT] {\"
[TEXT_MESSAGE_CONTENT] results
[TEXT_MESSAGE_CONTENT] \":[]}
[TEXT_MESSAGE_END] agent=principal
"""

    assert extract_json(log) == {"results": []}


def test_score_results_records_unparseable_submission(tmp_path):
    bundle = tmp_path / "bundle"
    hidden = bundle / "hidden"
    hidden.mkdir(parents=True)
    expected = {
        "candidate_id": "fmri|atlas|disease|feature|0",
        "n_case": 10,
        "n_control": 20,
        "adjusted_beta_case_minus_control": 0.2,
        "adjusted_beta_se": 0.1,
        "adjusted_t": 2.0,
        "p_value": 0.05,
        "adjusted_residual_d": 0.3,
    }
    (hidden / "gold_results.json").write_text(
        json.dumps({"results": [expected]}), encoding="utf-8"
    )
    result_file = tmp_path / "result.txt"
    result_file.write_text("not json", encoding="utf-8")
    out_dir = tmp_path / "audit"

    score_results(
        Namespace(
            bundle_dir=bundle,
            result_file=result_file,
            method="test",
            trial=2,
            out_dir=out_dir,
            rtol=1e-4,
            atol=1e-7,
        )
    )

    summary = json.loads(
        (out_dir / "execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["successful_candidates"] == 0
    assert summary["complete_valid_submission"] is False
    assert summary["parse_error"].startswith("ValueError:")


def test_select_candidate_ids_preserves_requested_order():
    frame = pd.DataFrame(
        [
            {
                "modality": "fmri",
                "source": "atlas_multiatlas",
                "disease": "disease",
                "feature": "feature",
                "roi_index": index,
                "adjusted_residual_d": 0.1,
            }
            for index in range(3)
        ]
    )
    requested = [
        "fmri|atlas_multiatlas|disease|feature|2",
        "fmri|atlas_multiatlas|disease|feature|0",
    ]

    selected = select_candidate_ids(frame, requested)

    assert selected["candidate_id"].tolist() == requested


def test_neuroruntime_executes_public_bundle_and_scores_exactly(tmp_path):
    bundle = tmp_path / "bundle"
    public_dir = bundle / "public"
    hidden_dir = bundle / "hidden"
    public_dir.mkdir(parents=True)
    hidden_dir.mkdir(parents=True)
    candidate = "fmri|atlas|disease|feature|0"
    frame = pd.DataFrame(
        {
            "candidate_id": [candidate] * 8,
            "subject_id": [f"s{index}" for index in range(8)],
            "case": [0, 0, 0, 0, 1, 1, 1, 1],
            "control": [1, 1, 1, 1, 0, 0, 0, 0],
            "value": [0.1, 0.2, 0.4, 0.5, 0.8, 0.9, 1.0, 1.2],
            "age": [20, 22, 24, 26, 21, 23, 25, 27],
        }
    )
    frame.to_csv(public_dir / "tcp_subject_level_features.csv", index=False)
    (public_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [{"candidate_id": candidate}],
                "covariate_columns": ["age"],
            }
        ),
        encoding="utf-8",
    )
    case_mask = frame["case"].eq(1).to_numpy()
    control_mask = frame["control"].eq(1).to_numpy()
    expected = ols_case_effect(
        frame["value"].to_numpy()[:, None],
        case_mask,
        control_mask,
        frame[["age"]],
    )
    gold = {
        "candidate_id": candidate,
        "n_case": int(expected["n_case"]),
        "n_control": int(expected["n_control"]),
        "adjusted_beta_case_minus_control": float(np.asarray(expected["beta"])[0]),
        "adjusted_beta_se": float(np.asarray(expected["se"])[0]),
        "adjusted_t": float(np.asarray(expected["t"])[0]),
        "p_value": float(np.asarray(expected["p"])[0]),
        "adjusted_residual_d": float(np.asarray(expected["residual_d"])[0]),
    }
    (hidden_dir / "gold_results.json").write_text(
        json.dumps({"results": [gold]}), encoding="utf-8"
    )
    result_file = tmp_path / "neuroruntime.json"
    execute_neuroruntime(
        Namespace(bundle_public=public_dir, out=result_file)
    )
    out_dir = tmp_path / "audit"
    score_results(
        Namespace(
            bundle_dir=bundle,
            result_file=result_file,
            method="neuroruntime",
            trial=0,
            out_dir=out_dir,
            rtol=1e-4,
            atol=1e-7,
        )
    )
    summary = json.loads(
        (out_dir / "execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete_valid_submission"] is True
