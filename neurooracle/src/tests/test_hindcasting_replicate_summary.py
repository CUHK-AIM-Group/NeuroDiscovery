from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from neurooracle.scripts.summarize_hindcasting_replicates import (
    _assert_locked_primary_matrix,
    _holm_adjust_paired_rows,
    _summary_rows,
    _resolve_path,
    exact_sign_flip_p_value,
    summarize,
)


def _write_run(
    root: Path,
    *,
    method: str,
    seed: int,
    unique: int,
    recall: float,
) -> dict[str, object]:
    output = root / method / f"seed_{seed:02d}" / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "benchmark_status": "executable",
                "topk": {
                    "10": {
                        "observed": {
                            "unique_primary_discoveries": unique,
                            "future_pair_recall": recall,
                            "primary_hits": unique,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "method": method,
        "seed": seed,
        "case_study_id": "case1_transdiagnostic",
        "freeze_year": 2016,
        "future_start_year": 2017,
        "future_end_year": 2017,
        "metrics_path": str(output),
    }


def test_exact_sign_flip_is_directional() -> None:
    assert exact_sign_flip_p_value([1.0, 1.0, 1.0]) == pytest.approx(0.125)
    assert exact_sign_flip_p_value([-1.0, -1.0, -1.0]) == pytest.approx(1.0)
    assert exact_sign_flip_p_value([0.0, 0.0]) == pytest.approx(1.0)


def test_holm_adjustment_is_applied_within_endpoint() -> None:
    base = {
        "scope": "macro_case_study_equal",
        "case_study_id": "ALL",
        "freeze_year": "ALL",
        "future_start_year": "ALL",
        "future_end_year": "ALL",
        "k": 1000,
        "metric": "unique_primary_discoveries",
    }
    adjusted = _holm_adjust_paired_rows(
        [
            {
                **base,
                "comparison_method": "a",
                "p_reference_greater_exact_sign_flip": 0.01,
            },
            {
                **base,
                "comparison_method": "b",
                "p_reference_greater_exact_sign_flip": 0.04,
            },
        ]
    )

    by_method = {row["comparison_method"]: row for row in adjusted}
    assert by_method["a"]["p_holm_within_endpoint"] == pytest.approx(0.02)
    assert by_method["b"]["p_holm_within_endpoint"] == pytest.approx(0.04)
    assert by_method["a"]["holm_family_size"] == 2


def test_resolve_path_accepts_repository_relative_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    evaluation = repository / "data" / "evaluation"
    metrics = evaluation / "method" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(repository)

    resolved = _resolve_path("data/evaluation/method/metrics.json", evaluation)

    assert resolved == metrics.resolve()


def test_summary_reports_sample_variance_and_paired_difference(tmp_path: Path) -> None:
    runs = []
    for seed, reference_value in enumerate((3, 5, 7)):
        runs.append(
            _write_run(
                tmp_path,
                method="reference",
                seed=seed,
                unique=reference_value,
                recall=reference_value / 10,
            )
        )
        runs.append(
            _write_run(
                tmp_path,
                method="comparator",
                seed=seed,
                unique=reference_value - 1,
                recall=(reference_value - 1) / 10,
            )
        )
    (tmp_path / "evaluation_manifest.json").write_text(
        json.dumps({"runs": runs}), encoding="utf-8"
    )
    output = tmp_path / "summary"
    result = summarize(
        argparse.Namespace(
            evaluation_root=tmp_path,
            output_root=output,
            reference_method="reference",
            metrics=["unique_primary_discoveries"],
        )
    )

    assert result["n_observations"] == 6
    summaries = (output / "mean_variance_summary.csv").read_text(encoding="utf-8")
    assert "5.0,4.0" in summaries
    comparisons = (output / "paired_comparisons.csv").read_text(encoding="utf-8")
    assert "1.0,0.0,0.125" in comparisons


def test_primary_macro_weights_case_studies_equally_not_windows() -> None:
    rows = []
    for seed in (0, 1):
        for freeze in range(2016, 2021):
            rows.append(
                {
                    "method": "method",
                    "seed": seed,
                    "case_study_id": "five_windows",
                    "freeze_year": freeze,
                    "future_start_year": freeze + 1,
                    "future_end_year": freeze + 5,
                    "k": 100,
                    "metric": "unique_primary_discoveries",
                    "value": 10.0,
                }
            )
        for freeze in (2019, 2020):
            rows.append(
                {
                    "method": "method",
                    "seed": seed,
                    "case_study_id": "two_windows",
                    "freeze_year": freeze,
                    "future_start_year": freeze + 1,
                    "future_end_year": freeze + 5,
                    "k": 100,
                    "metric": "unique_primary_discoveries",
                    "value": 0.0,
                }
            )

    summary = _summary_rows(rows)
    equal_case = next(row for row in summary if row["scope"] == "macro_case_study_equal")
    flat_window = next(row for row in summary if row["scope"] == "macro_case_window")

    assert equal_case["mean"] == pytest.approx(5.0)
    assert equal_case["sample_variance"] == pytest.approx(0.0)
    assert flat_window["mean"] == pytest.approx(50.0 / 7.0)


def test_locked_primary_matrix_rejects_missing_window() -> None:
    primary = {
        ("case_a", 2016, 2017, 2021),
        ("case_b", 2017, 2018, 2022),
    }
    rows = [
        {
            "method": "method",
            "seed": 0,
            "case_study_id": "case_a",
            "freeze_year": 2016,
            "future_start_year": 2017,
            "future_end_year": 2021,
            "k": 100,
            "metric": "unique_primary_discoveries",
            "benchmark_status": "executable",
        }
    ]

    with pytest.raises(ValueError, match="incomplete locked primary matrix"):
        _assert_locked_primary_matrix(rows, primary)
