from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from neurooracle.scripts.compare_dynamic_closed_open_loop import compare


def _write_metrics(
    root: Path,
    method: str,
    values: tuple[int, ...],
    *,
    feedback_enabled: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metrics_by_run.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                "case_study_id",
                "freeze_year",
                "future_start_year",
                "future_end_year",
                "requested_k",
                "terminal_unique_primary_discoveries",
            ],
        )
        writer.writeheader()
        for seed, value in enumerate(values):
            writer.writerow(
                {
                    "method": method,
                    "seed": seed,
                    "case_study_id": "case1_transdiagnostic",
                    "freeze_year": 2016,
                    "future_start_year": 2017,
                    "future_end_year": 2018,
                    "requested_k": 100,
                    "terminal_unique_primary_discoveries": value,
                }
            )
    run_summaries = [
        {
            "seed": seed,
            "case_study_id": "case1_transdiagnostic",
            "freeze_year": 2016,
            "future_start_year": 2017,
            "future_end_year": 2018,
            "config": {
                "feedback_enabled": feedback_enabled,
                "max_executions": 100,
            },
            "supported_feedback_records": 1 if feedback_enabled else 0,
            "withheld_supported_outcomes": 0 if feedback_enabled else 1,
            "closed_loop_activated": feedback_enabled,
            "first_feedback_generation_round": 1 if feedback_enabled else None,
            "feedback_start_budget": 2,
            "temporal_isolation": {
                "terminal_labels_available_to_generator": False,
                "formal_kg_mutated": False,
            },
        }
        for seed in range(len(values))
    ]
    for seed in range(len(values)):
        run_dir = (
            root
            / f"seed_{seed:02d}"
            / "case1_transdiagnostic"
            / "kg2016_to_2017_2018"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        run_payload = run_summaries[seed]
        (run_dir / "run_manifest.json").write_text(
            json.dumps(run_payload), encoding="utf-8"
        )
        hypotheses = [
            {
                "id": f"H{rank}",
                "hypothesis_type": "bridge",
                "source_id": f"S{rank}",
                "target_id": f"T{rank}",
                "path": [
                    {
                        "from_id": f"S{rank}",
                        "relation_type": "associated_with",
                        "to_id": f"T{rank}",
                    }
                ],
            }
            for rank in range(2)
        ]
        (run_dir / "executed_hypotheses.json").write_text(
            json.dumps({"n_hypotheses": 2, "hypotheses": hypotheses}),
            encoding="utf-8",
        )
    manifest = {
        "snapshot_root": "snapshots",
        "future_claims": "claims.jsonl",
        "case_studies": ["case1_transdiagnostic"],
        "windows": [
            {
                "freeze_year": 2016,
                "future_start_year": 2017,
                "future_end_year": 2018,
            }
        ],
        "seeds": list(range(len(values))),
        "budgets": [100],
        "profile": {"name": "test"},
        "config": {
            "feedback_enabled": feedback_enabled,
            "max_executions": 100,
        },
        "runs": len(values),
        "run_summaries": run_summaries,
    }
    (root / "dynamic_closed_loop_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_compare_dynamic_closed_and_open_reports_paired_statistics(
    tmp_path: Path,
) -> None:
    closed = tmp_path / "closed"
    opened = tmp_path / "open"
    output = tmp_path / "comparison"
    _write_metrics(closed, "closed", (2, 3, 4), feedback_enabled=True)
    _write_metrics(opened, "open", (1, 2, 3), feedback_enabled=False)

    manifest = compare(
        argparse.Namespace(
            closed_root=closed,
            open_root=opened,
            output_root=output,
            metrics=["terminal_unique_primary_discoveries"],
        )
    )

    assert manifest["n_observations"] == 3
    rows = list(
        csv.DictReader(
            (output / "mean_variance_paired_summary.csv").open(
                "r", encoding="utf-8"
            )
        )
    )
    case_row = next(row for row in rows if row["scope"] == "case_window")
    assert float(case_row["closed_mean"]) == 3.0
    assert float(case_row["open_mean"]) == 2.0
    assert float(case_row["mean_paired_difference"]) == 1.0
    assert float(case_row["p_closed_greater_exact_sign_flip"]) == 0.125
    macro_row = next(
        row for row in rows if row["scope"] == "macro_case_study_equal"
    )
    assert float(macro_row["closed_mean"]) == 3.0
    assert float(macro_row["open_mean"]) == 2.0
    assert manifest["design_audit"]["status"] == "passed"
    assert (output / "design_audit.json").is_file()


def test_compare_rejects_non_feedback_configuration_difference(
    tmp_path: Path,
) -> None:
    closed = tmp_path / "closed"
    opened = tmp_path / "open"
    _write_metrics(closed, "closed", (2,), feedback_enabled=True)
    _write_metrics(opened, "open", (1,), feedback_enabled=False)
    path = opened / "dynamic_closed_loop_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["config"]["max_executions"] = 200
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="non-feedback configurations differ"):
        compare(
            argparse.Namespace(
                closed_root=closed,
                open_root=opened,
                output_root=tmp_path / "comparison",
                metrics=["terminal_unique_primary_discoveries"],
            )
        )


def test_compare_rejects_open_feedback_exposure(tmp_path: Path) -> None:
    closed = tmp_path / "closed"
    opened = tmp_path / "open"
    _write_metrics(closed, "closed", (2,), feedback_enabled=True)
    _write_metrics(opened, "open", (1,), feedback_enabled=False)
    path = opened / "dynamic_closed_loop_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["run_summaries"][0]["supported_feedback_records"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exposed supported feedback"):
        compare(
            argparse.Namespace(
                closed_root=closed,
                open_root=opened,
                output_root=tmp_path / "comparison",
                metrics=["terminal_unique_primary_discoveries"],
            )
        )


def test_compare_rejects_pre_feedback_execution_divergence(
    tmp_path: Path,
) -> None:
    closed = tmp_path / "closed"
    opened = tmp_path / "open"
    _write_metrics(closed, "closed", (2,), feedback_enabled=True)
    _write_metrics(opened, "open", (1,), feedback_enabled=False)
    path = (
        opened
        / "seed_00"
        / "case1_transdiagnostic"
        / "kg2016_to_2017_2018"
        / "executed_hypotheses.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hypotheses"][1]["target_id"] = "DIFFERENT"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="diverged before feedback"):
        compare(
            argparse.Namespace(
                closed_root=closed,
                open_root=opened,
                output_root=tmp_path / "comparison",
                metrics=["terminal_unique_primary_discoveries"],
            )
        )
