"""Aggregate sealed v4r1 hindcasting evaluations with the locked TEAS-5 metric."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from neurooracle.src.hindcasting_v4_feedback_boundary import sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = (
    ROOT
    / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/evaluation"
    / "evaluation_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/result_summary"
DISPLAY = {
    "neurodiscovery": "NeuroDiscovery",
    "sciagents": "SciAgents",
    "openscholar_rag": "OpenScholar-RAG",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(20260916)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_ci(values: Sequence[float]) -> tuple[float, float]:
    return _ci(values)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.evaluation_manifest)
    if manifest.get("status") != "complete":
        raise ValueError("v4r1 evaluation is not complete")
    if not (manifest.get("label_permutation_invariance") or {}).get(
        "all_discovery_sequences_unchanged"
    ):
        raise ValueError("label-permutation invariance gate did not pass")
    cell_rows: list[dict[str, Any]] = []
    for run in manifest.get("runs") or ():
        metrics_path = Path(str(run["metrics_path"]))
        if sha256_file(metrics_path) != str(run["metrics_sha256"]):
            raise ValueError(f"evaluated metrics changed: {metrics_path}")
        metrics = read_json(metrics_path)
        # The inherited evaluator's on-disk schema calls this mapping ``topk``.
        # Accept ``top_k`` as well for forward compatibility without changing
        # any locked values or recomputing evaluation labels.
        topk = metrics.get("topk") or metrics.get("top_k")
        if not isinstance(topk, dict):
            raise ValueError(f"missing top-k metrics mapping: {metrics_path}")
        for raw_k, record in topk.items():
            observed = record["observed"]
            discoveries = int(observed["unique_primary_discoveries"])
            cell_rows.append(
                {
                    "method": str(run["method"]),
                    "display_name": DISPLAY[str(run["method"])],
                    "task_id": str(run["case_study_id"]),
                    "freeze_year": int(run["freeze_year"]),
                    "seed": int(run["seed"]),
                    "k": int(raw_k),
                    "unique_exact_discoveries": discoveries,
                    "raw_primary_hit_slots": int(observed["primary_hits"]),
                    "teas5": 20.0 * min(discoveries, 5),
                    "future_pair_recall": observed.get("future_pair_recall"),
                    "mean_primary_lead_time": observed.get("mean_primary_lead_time"),
                }
            )
    methods = list(DISPLAY)
    budgets = sorted({int(row["k"]) for row in cell_rows})
    expected_cells = len(methods) * 2 * 5 * 10 * len(budgets)
    if len(cell_rows) != expected_cells:
        raise ValueError(f"aggregate matrix incomplete: {len(cell_rows)}/{expected_cells}")

    summary: list[dict[str, Any]] = []
    for method in methods:
        for k in budgets:
            subset = [row for row in cell_rows if row["method"] == method and row["k"] == k]
            teas = [float(row["teas5"]) for row in subset]
            discoveries = [float(row["unique_exact_discoveries"]) for row in subset]
            raw_hits = [float(row["raw_primary_hit_slots"]) for row in subset]
            low, high = _ci(teas)
            summary.append(
                {
                    "method": method,
                    "display_name": DISPLAY[method],
                    "k": k,
                    "cells": len(subset),
                    "teas5_mean": mean(teas),
                    "teas5_sd": stdev(teas) if len(teas) > 1 else 0.0,
                    "teas5_ci95_low": low,
                    "teas5_ci95_high": high,
                    "unique_exact_discoveries_mean": mean(discoveries),
                    "unique_exact_discoveries_sum": int(sum(discoveries)),
                    "raw_primary_hit_slots_mean": mean(raw_hits),
                    "cells_at_teas5_ceiling": sum(value >= 100.0 for value in teas),
                    "cells_with_zero_exact_discoveries": sum(value == 0.0 for value in discoveries),
                }
            )

    keyed = {
        (row["method"], row["task_id"], row["freeze_year"], row["seed"], row["k"]): row
        for row in cell_rows
    }
    paired: list[dict[str, Any]] = []
    for baseline in ("sciagents", "openscholar_rag"):
        for k in budgets:
            deltas: list[float] = []
            raw_deltas: list[float] = []
            for task_id in ("case1_transdiagnostic", "biomarker_discovery"):
                for year in range(2016, 2021):
                    for seed in range(10):
                        nd = keyed[("neurodiscovery", task_id, year, seed, k)]
                        other = keyed[(baseline, task_id, year, seed, k)]
                        deltas.append(float(nd["teas5"]) - float(other["teas5"]))
                        raw_deltas.append(
                            float(nd["unique_exact_discoveries"])
                            - float(other["unique_exact_discoveries"])
                        )
            low, high = _paired_ci(deltas)
            paired.append(
                {
                    "comparison": f"NeuroDiscovery - {DISPLAY[baseline]}",
                    "baseline": baseline,
                    "k": k,
                    "paired_cells": len(deltas),
                    "teas5_mean_difference": mean(deltas),
                    "teas5_difference_ci95_low": low,
                    "teas5_difference_ci95_high": high,
                    "exact_discoveries_mean_difference": mean(raw_deltas),
                    "neurodiscovery_wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "neurodiscovery_losses": sum(value < 0 for value in deltas),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cell_metrics.csv", cell_rows)
    write_csv(args.output / "method_summary.csv", summary)
    write_csv(args.output / "paired_comparisons.csv", paired)
    primary_rows = [row for row in summary if int(row["k"]) in {100, 1000}]
    by_method = {method: {} for method in methods}
    for row in primary_rows:
        by_method[str(row["method"])][int(row["k"])] = row
    markdown = [
        "# Hindcasting v4r1 comparison",
        "",
        "Primary metric: TEAS-5 = 20 × min(unique exact discoveries, 5), averaged over 100 locked task × cutoff × seed cells.",
        "",
        "| Method | TEAS-5 @100 | Exact discoveries @100 | TEAS-5 @1000 | Exact discoveries @1000 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        r100 = by_method[method][100]
        r1000 = by_method[method][1000]
        markdown.append(
            f"| {DISPLAY[method]} | {r100['teas5_mean']:.2f} | "
            f"{r100['unique_exact_discoveries_mean']:.3f} | "
            f"{r1000['teas5_mean']:.2f} | "
            f"{r1000['unique_exact_discoveries_mean']:.3f} |"
        )
    markdown.extend(
        [
            "",
            "The exact-discovery columns are the auditable underlying counts; TEAS-5 is only a bounded display transform.",
            "All five post-cutoff years are evaluation-only. The discovery stage made no LLM calls and NeuroDiscovery feedback came only from registered TCP computations.",
            "This is algorithmically blinded retrospective hindcasting, not strict historical resource availability, because the TCP datasets were released after the historical cutoffs.",
        ]
    )
    report_path = args.output / "REPORT.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    result = {
        "schema_version": "neurodiscovery-hindcasting-v4r1-aggregate.v1",
        "status": "complete",
        "created_at": utc_now(),
        "evaluation_manifest": str(args.evaluation_manifest.resolve()),
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest.resolve()),
        "metric": {
            "name": "TEAS-5",
            "formula": "20 * min(unique_exact_discoveries_at_K, 5)",
            "range": [0, 100],
            "aggregation_cells": 100,
        },
        "method_summary": summary,
        "paired_comparisons": paired,
        "artifacts": {
            "cell_metrics": str((args.output / "cell_metrics.csv").resolve()),
            "method_summary": str((args.output / "method_summary.csv").resolve()),
            "paired_comparisons": str((args.output / "paired_comparisons.csv").resolve()),
            "report": str(report_path.resolve()),
        },
    }
    atomic_json(args.output / "RESULTS.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("aggregate output root must be absent or empty")
    return args


def main() -> None:
    result = aggregate(parse_args())
    primary = [row for row in result["method_summary"] if row["k"] in {100, 1000}]
    print(json.dumps(primary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
