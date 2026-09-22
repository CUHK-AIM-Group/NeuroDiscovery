"""Aggregate generic case-study hindcasting runs into audit-friendly tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from neurooracle.src.case_studies import case_study_by_name


DEFAULT_ROOT = Path("neurooracle/data/experiments/hindcasting/matrix_current")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/kg*_to_*_*/hindcasting/metrics.json")):
        metrics = _load_json(path)
        case_study_id = str(metrics.get("case_study_id") or path.parents[2].name)
        for k, result in metrics.get("topk", {}).items():
            observed = result.get("observed") or {}
            random_baseline = result.get("random_same_hypothesis_pool") or {}
            rows.append(
                {
                    "case_study_id": case_study_id,
                    "validation_protocol": metrics.get("validation_protocol") or "hindcasting",
                    "method": metrics.get("method") or "unknown",
                    "benchmark_status": metrics.get("benchmark_status") or "legacy_unknown",
                    "benchmark_reason": metrics.get("benchmark_reason") or "",
                    "freeze_year": metrics.get("freeze_year"),
                    "future_start_year": metrics.get("future_start_year"),
                    "future_end_year": metrics.get("future_end_year"),
                    "k": int(k),
                    "n_hypotheses": metrics.get("n_hypotheses", 0),
                    "future_unique_pairs": (metrics.get("future_stats") or {}).get(
                        "future_unique_pairs", 0
                    ),
                    "primary_hits": observed.get("primary_hits", 0),
                    "primary_hit_rate": observed.get("primary_hit_rate", 0.0),
                    "unique_primary_discoveries": observed.get(
                        "unique_primary_discoveries", observed.get("primary_hits", 0)
                    ),
                    "unique_primary_discovery_rate": observed.get(
                        "unique_primary_discovery_rate", observed.get("primary_hit_rate", 0.0)
                    ),
                    "recovered_future_pairs": observed.get(
                        "recovered_future_pairs", observed.get("primary_hits", 0)
                    ),
                    "future_pair_recall": observed.get("future_pair_recall"),
                    "endpoint_hits": observed.get("endpoint_hits", 0),
                    "endpoint_hit_rate": observed.get("endpoint_hit_rate", 0.0),
                    "any_future_hits": observed.get("any_future_hits", 0),
                    "any_future_hit_rate": observed.get("any_future_hit_rate", 0.0),
                    "mean_path_edge_hit_rate": observed.get("mean_path_edge_hit_rate", 0.0),
                    "random_applicable": bool(random_baseline.get("applicable")),
                    "random_candidate_pool_size": random_baseline.get(
                        "candidate_pool_size", metrics.get("n_hypotheses", 0)
                    ),
                    "random_inapplicable_reason": random_baseline.get("reason", ""),
                    "random_mean_primary_hits": random_baseline.get("mean_primary_hits"),
                    "random_sd_primary_hits": random_baseline.get("sd_primary_hits"),
                    "p_primary_hits_ge_observed": random_baseline.get(
                        "p_primary_hits_ge_observed"
                    ),
                    "random_mean_unique_primary_discoveries": random_baseline.get(
                        "mean_unique_primary_discoveries"
                    ),
                    "random_variance_unique_primary_discoveries": random_baseline.get(
                        "variance_unique_primary_discoveries"
                    ),
                    "p_unique_primary_discoveries_ge_observed": random_baseline.get(
                        "p_unique_primary_discoveries_ge_observed"
                    ),
                    "random_mean_any_future_hits": random_baseline.get(
                        "mean_any_future_hits"
                    ),
                    "random_mean_endpoint_hits": random_baseline.get(
                        "mean_endpoint_hits"
                    ),
                    "metrics_path": str(path),
                }
            )
    return rows


def build_coverage_rows(
    root: Path,
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_path = root / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    planned = [
        str(item.get("id") or item.get("case_study_id") or "")
        for item in manifest.get("case_studies", [])
    ]
    run_by_id = {
        str(row.get("case_study_id") or ""): row
        for row in manifest.get("runs", [])
    }
    windows = manifest.get("windows", [])
    expected_freezes = {
        int(row["freeze_year"]) for row in windows if row.get("freeze_year") is not None
    }
    observed: dict[str, set[int]] = {}
    executable: dict[str, set[int]] = {}
    sparse: dict[str, set[int]] = {}
    non_executable: dict[str, set[int]] = {}
    for row in metrics:
        case_study_id = row["case_study_id"]
        freeze_year = int(row["freeze_year"])
        observed.setdefault(case_study_id, set()).add(freeze_year)
        status = str(row.get("benchmark_status") or "legacy_unknown")
        if status == "executable":
            executable.setdefault(case_study_id, set()).add(freeze_year)
        elif status == "sparse":
            sparse.setdefault(case_study_id, set()).add(freeze_year)
        elif status == "non_executable":
            non_executable.setdefault(case_study_id, set()).add(freeze_year)

    rows: list[dict[str, Any]] = []
    for case_study_id in planned:
        case = case_study_by_name(case_study_id)
        completed = observed.get(case_study_id, set())
        run = run_by_id.get(case_study_id, {})
        if run.get("status") == "failed":
            status = "failed"
        elif expected_freezes and completed == expected_freezes:
            status = "completed"
        elif completed:
            status = "partial"
        elif not run:
            status = "planned"
        else:
            status = str(run.get("status") or "missing")
        rows.append(
            {
                "case_study_id": case_study_id,
                "english_name": case.english_name,
                "status": status,
                "completed_windows": len(completed),
                "executable_windows": len(executable.get(case_study_id, set())),
                "sparse_windows": len(sparse.get(case_study_id, set())),
                "non_executable_windows": len(non_executable.get(case_study_id, set())),
                "expected_windows": len(expected_freezes),
                "returncode": run.get("returncode"),
                "log": run.get("log", ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    metrics = collect_metrics(args.root)
    coverage = build_coverage_rows(args.root, metrics)
    _write_csv(
        args.root / "hindcasting_metrics.csv",
        metrics,
        [
            "case_study_id",
            "validation_protocol",
            "method",
            "benchmark_status",
            "benchmark_reason",
            "freeze_year",
            "future_start_year",
            "future_end_year",
            "k",
            "n_hypotheses",
            "future_unique_pairs",
            "primary_hits",
            "primary_hit_rate",
            "unique_primary_discoveries",
            "unique_primary_discovery_rate",
            "recovered_future_pairs",
            "future_pair_recall",
            "endpoint_hits",
            "endpoint_hit_rate",
            "any_future_hits",
            "any_future_hit_rate",
            "mean_path_edge_hit_rate",
            "random_applicable",
            "random_candidate_pool_size",
            "random_inapplicable_reason",
            "random_mean_primary_hits",
            "random_sd_primary_hits",
            "p_primary_hits_ge_observed",
            "random_mean_unique_primary_discoveries",
            "random_variance_unique_primary_discoveries",
            "p_unique_primary_discoveries_ge_observed",
            "random_mean_any_future_hits",
            "random_mean_endpoint_hits",
            "metrics_path",
        ],
    )
    _write_csv(
        args.root / "coverage.csv",
        coverage,
        [
            "case_study_id",
            "english_name",
            "status",
            "completed_windows",
            "executable_windows",
            "sparse_windows",
            "non_executable_windows",
            "expected_windows",
            "returncode",
            "log",
        ],
    )
    summary = {
        "validation_protocol": "hindcasting",
        "metrics_rows": len(metrics),
        "case_studies": len(coverage),
        "completed": sum(row["status"] == "completed" for row in coverage),
        "top100_random_applicable_windows": sum(
            row["k"] == 100 and row["random_applicable"] for row in metrics
        ),
        "top100_random_inapplicable_windows": sum(
            row["k"] == 100 and not row["random_applicable"] for row in metrics
        ),
        "partial": [row["case_study_id"] for row in coverage if row["status"] == "partial"],
        "failed": [row["case_study_id"] for row in coverage if row["status"] == "failed"],
        "planned": [row["case_study_id"] for row in coverage if row["status"] == "planned"],
    }
    (args.root / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
