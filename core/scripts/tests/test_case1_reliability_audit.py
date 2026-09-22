from __future__ import annotations

import json

import pandas as pd

from core.scripts.case1_reliability_audit import (
    fresh_official_rows,
    neuroruntime_rows,
)


def test_fresh_official_rows_keeps_invalid_slots(tmp_path):
    trial = tmp_path / "sciagents" / "trial_03"
    trial.mkdir(parents=True)
    (trial / "task.json").write_text(
        json.dumps({"trial": 3, "n_anchors": 20}), encoding="utf-8"
    )
    (trial / "native_result.json").write_text("{}", encoding="utf-8")
    (trial / "search_policy.json").write_text("{}", encoding="utf-8")
    (trial / "native_compile_audit.json").write_text(
        json.dumps({"valid_unique_anchors": 17}), encoding="utf-8"
    )
    (trial / "adapter_attempt_01.duration.txt").write_text(
        "12.5\n", encoding="utf-8"
    )

    rows = fresh_official_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["first_attempt_workflow_success"] is True
    assert rows[0]["legal_unique_hypotheses"] == 17
    assert rows[0]["illegal_duplicate_or_missing"] == 3
    assert rows[0]["legal_rate"] == 0.85


def test_neuroruntime_rows_counts_nonfinite_as_failed(tmp_path):
    source = tmp_path / "all_tests.csv"
    pd.DataFrame(
        {
            "modality": ["fmri", "fmri"],
            "source": ["atlas_multiatlas", "atlas_multiatlas"],
            "feature": ["corr_mean", "corr_mean"],
            "adjusted_residual_d": [0.4, float("nan")],
            "p_value": [0.01, float("nan")],
            "q_fdr_global": [0.04, float("nan")],
        }
    ).to_csv(source, index=False)

    rows = neuroruntime_rows(source)
    total = rows[-1]

    assert total["attempted_hypotheses"] == 2
    assert total["completed_hypotheses"] == 1
    assert total["failed_or_nonfinite"] == 1
    assert total["completion_rate"] == 0.5
