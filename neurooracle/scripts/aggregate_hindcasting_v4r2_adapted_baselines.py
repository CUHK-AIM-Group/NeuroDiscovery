"""Combine frozen v4r1 NeuroDiscovery with v4r2r1 adapted-baseline evaluations."""

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

from neurooracle.scripts.evaluate_hindcasting_v4 import _preflight
from neurooracle.src.hindcasting_v4_feedback_boundary import sha256_file


ROOT = Path(__file__).resolve().parents[2]
V4R1_ROOT = ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916"
V4R2_ROOT = ROOT / "neurooracle/data/hv4_runs/r2r1_adapted_baselines_20260916"
DEFAULT_V4R1_EVALUATION = V4R1_ROOT / "evaluation/evaluation_manifest.json"
DEFAULT_V4R1_RESULTS = V4R1_ROOT / "result_summary/RESULTS.json"
DEFAULT_V4R1_AUDIT = V4R1_ROOT / "result_summary/FINAL_AUDIT.json"
DEFAULT_V4R2_DISCOVERY = V4R2_ROOT / "discovery/discovery_manifest.json"
DEFAULT_V4R2_EVALUATION = V4R2_ROOT / "evaluation/evaluation_manifest.json"
DEFAULT_OUTPUT = V4R2_ROOT / "result_summary"
METHODS = ("neurodiscovery", "sciagents_adapted", "openscholar_rag_adapted")
DISPLAY = {
    "neurodiscovery": "NeuroDiscovery",
    "sciagents_adapted": "SciAgents-adapted",
    "openscholar_rag_adapted": "OpenScholar-RAG-adapted",
}
OLD_TO_NEW = {
    "sciagents_adapted": "sciagents",
    "openscholar_rag_adapted": "openscholar_rag",
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ci95(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260916)
    draws = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _metric_rows(manifest_path: Path, allowed_methods: set[str]) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"evaluation is incomplete: {manifest_path}")
    if not (manifest.get("label_permutation_invariance") or {}).get(
        "all_discovery_sequences_unchanged"
    ):
        raise ValueError(f"label-permutation gate failed: {manifest_path}")
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs") or ():
        method = str(run["method"])
        if method not in allowed_methods:
            continue
        metrics_path = Path(str(run["metrics_path"]))
        if sha256_file(metrics_path) != str(run["metrics_sha256"]):
            raise ValueError(f"metrics changed: {metrics_path}")
        metrics = read_json(metrics_path)
        topk = metrics.get("topk") or metrics.get("top_k")
        if not isinstance(topk, dict):
            raise ValueError(f"missing top-k mapping: {metrics_path}")
        for raw_k, record in topk.items():
            observed = record["observed"]
            discoveries = int(observed["unique_primary_discoveries"])
            rows.append(
                {
                    "method": method,
                    "display_name": DISPLAY.get(method, method),
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
    return rows


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"aggregate output must be absent or empty: {args.output}")
    new_rows = _metric_rows(
        args.v4r2_evaluation, {"sciagents_adapted", "openscholar_rag_adapted"}
    )
    old_all_rows = _metric_rows(
        args.v4r1_evaluation, {"neurodiscovery", "sciagents", "openscholar_rag"}
    )
    rows = [row for row in old_all_rows if row["method"] == "neurodiscovery"] + new_rows
    budgets = sorted({int(row["k"]) for row in rows})
    expected = len(METHODS) * 2 * 5 * 10 * len(budgets)
    if len(rows) != expected:
        raise ValueError(f"combined matrix incomplete: {len(rows)}/{expected}")

    summary: list[dict[str, Any]] = []
    for method in METHODS:
        for k in budgets:
            subset = [row for row in rows if row["method"] == method and row["k"] == k]
            teas = [float(row["teas5"]) for row in subset]
            exact = [float(row["unique_exact_discoveries"]) for row in subset]
            raw = [float(row["raw_primary_hit_slots"]) for row in subset]
            low, high = ci95(teas)
            summary.append(
                {
                    "method": method,
                    "display_name": DISPLAY[method],
                    "k": k,
                    "cells": len(subset),
                    "teas5_mean": mean(teas),
                    "teas5_sd": stdev(teas),
                    "teas5_ci95_low": low,
                    "teas5_ci95_high": high,
                    "unique_exact_discoveries_mean": mean(exact),
                    "unique_exact_discoveries_sum": int(sum(exact)),
                    "raw_primary_hit_slots_mean": mean(raw),
                    "cells_at_teas5_ceiling": sum(value >= 100 for value in teas),
                    "cells_with_zero_exact_discoveries": sum(value == 0 for value in exact),
                }
            )

    keyed = {
        (row["method"], row["task_id"], row["freeze_year"], row["seed"], row["k"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for baseline in METHODS[1:]:
        for k in budgets:
            deltas: list[float] = []
            exact_deltas: list[float] = []
            for task in ("case1_transdiagnostic", "biomarker_discovery"):
                for year in range(2016, 2021):
                    for seed in range(10):
                        nd = keyed[("neurodiscovery", task, year, seed, k)]
                        other = keyed[(baseline, task, year, seed, k)]
                        deltas.append(float(nd["teas5"]) - float(other["teas5"]))
                        exact_deltas.append(
                            float(nd["unique_exact_discoveries"])
                            - float(other["unique_exact_discoveries"])
                        )
            low, high = ci95(deltas)
            paired.append(
                {
                    "comparison": f"NeuroDiscovery - {DISPLAY[baseline]}",
                    "baseline": baseline,
                    "k": k,
                    "paired_cells": len(deltas),
                    "teas5_mean_difference": mean(deltas),
                    "teas5_difference_ci95_low": low,
                    "teas5_difference_ci95_high": high,
                    "exact_discoveries_mean_difference": mean(exact_deltas),
                    "neurodiscovery_wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "neurodiscovery_losses": sum(value < 0 for value in deltas),
                }
            )

    old_keyed = {
        (row["method"], row["task_id"], row["freeze_year"], row["seed"], row["k"]): row
        for row in old_all_rows
    }
    adaptation: list[dict[str, Any]] = []
    for new_method, old_method in OLD_TO_NEW.items():
        for k in budgets:
            differences: list[float] = []
            teas_differences: list[float] = []
            for task in ("case1_transdiagnostic", "biomarker_discovery"):
                for year in range(2016, 2021):
                    for seed in range(10):
                        new = keyed[(new_method, task, year, seed, k)]
                        old = old_keyed[(old_method, task, year, seed, k)]
                        differences.append(
                            float(new["unique_exact_discoveries"])
                            - float(old["unique_exact_discoveries"])
                        )
                        teas_differences.append(float(new["teas5"]) - float(old["teas5"]))
            low, high = ci95(teas_differences)
            adaptation.append(
                {
                    "method": new_method,
                    "old_v4r1_method": old_method,
                    "k": k,
                    "paired_cells": len(differences),
                    "exact_discovery_mean_change": mean(differences),
                    "teas5_mean_change": mean(teas_differences),
                    "teas5_change_ci95_low": low,
                    "teas5_change_ci95_high": high,
                    "improved_cells": sum(value > 0 for value in differences),
                    "unchanged_cells": sum(value == 0 for value in differences),
                    "worsened_cells": sum(value < 0 for value in differences),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cell_metrics.csv", rows)
    write_csv(args.output / "method_summary.csv", summary)
    write_csv(args.output / "paired_comparisons.csv", paired)
    write_csv(args.output / "adaptation_delta_vs_v4r1.csv", adaptation)

    by_method: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    for row in summary:
        if int(row["k"]) in {100, 1000}:
            by_method[str(row["method"])][int(row["k"])] = row
    markdown = [
        "# Hindcasting v4r2r1 adapted-baseline comparison",
        "",
        "NeuroDiscovery is the unchanged, frozen v4r1 full closed-loop result. Only the two baselines were regenerated.",
        "",
        "| Method | TEAS-5 @100 | Exact @100 | TEAS-5 @1000 | Exact @1000 | Zero cells @1000 | Ceiling cells @1000 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        r100 = by_method[method][100]
        r1000 = by_method[method][1000]
        markdown.append(
            f"| {DISPLAY[method]} | {r100['teas5_mean']:.2f} | "
            f"{r100['unique_exact_discoveries_mean']:.3f} | "
            f"{r1000['teas5_mean']:.2f} | "
            f"{r1000['unique_exact_discoveries_mean']:.3f} | "
            f"{r1000['cells_with_zero_exact_discoveries']} | "
            f"{r1000['cells_at_teas5_ceiling']} |"
        )
    markdown.extend(
        [
            "",
            "Interpretation: this is an exploratory corrective rerun because the v4r1 baseline results were inspected before the adapter defects were corrected. The adapted names must be retained; these runs are not native upstream reproductions.",
            "",
            "No contemporary LLM was used in historical discovery. Post-cutoff publications were withheld until every v4r2r1 baseline sequence was sealed. OpenScholar-RAG-adapted uses deterministic pseudo-relevance feedback and a fixed hypothesis compiler; SciAgents-adapted uses task-scoped graph/claim evidence and deterministic role decomposition.",
            "",
            "Post-hoc construct audit: all OpenScholar-RAG-adapted candidates retain retrieval citations and source-paper keys. All SciAgents-adapted candidates retain role scores and source-paper keys, but only 200/100,000 are literal connected two-hop bridges; 99,800 are separately paper-grounded endpoint compositions. Therefore neither adapted result is a native upstream-system performance estimate.",
            "",
            "Experimental-design audit framework: Kassis et al. (2026), *Scientific Critical Thinking*, arXiv:2609.00065.",
        ]
    )
    report = args.output / "REPORT.md"
    report.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    result = {
        "schema_version": "neurodiscovery-hindcasting-v4r2r1-adapted-aggregate.v1",
        "status": "complete",
        "created_at": utc_now(),
        "claim_status": "exploratory_corrective_rerun",
        "frozen_neurodiscovery": {
            "source_release": "v4r1_two_tasks_teas5_20260916",
            "evaluation_manifest": str(args.v4r1_evaluation.resolve()),
            "evaluation_manifest_sha256": sha256_file(args.v4r1_evaluation),
            "regenerated": False,
        },
        "adapted_baselines": {
            "evaluation_manifest": str(args.v4r2_evaluation.resolve()),
            "evaluation_manifest_sha256": sha256_file(args.v4r2_evaluation),
        },
        "metric": {
            "name": "TEAS-5",
            "formula": "20 * min(unique_exact_discoveries_at_K, 5)",
            "range": [0, 100],
            "aggregation_cells_per_method": 100,
        },
        "method_summary": summary,
        "paired_comparisons": paired,
        "adaptation_delta_vs_v4r1": adaptation,
        "artifacts": {
            "cell_metrics": str((args.output / "cell_metrics.csv").resolve()),
            "method_summary": str((args.output / "method_summary.csv").resolve()),
            "paired_comparisons": str((args.output / "paired_comparisons.csv").resolve()),
            "adaptation_delta": str((args.output / "adaptation_delta_vs_v4r1.csv").resolve()),
            "report": str(report.resolve()),
        },
    }
    atomic_json(args.output / "RESULTS.json", result)
    return result


def audit(args: argparse.Namespace, results: Mapping[str, Any]) -> dict[str, Any]:
    v4r1_audit = read_json(args.v4r1_audit)
    if v4r1_audit.get("status") != "passed":
        raise ValueError("frozen v4r1 NeuroDiscovery source audit is not passed")
    discovery, discovery_rows, sequence_hash = _preflight(args.v4r2_discovery)
    evaluation = read_json(args.v4r2_evaluation)
    gate = read_json(args.v4r2_evaluation.parent / "PRE_EVALUATION_GATE.json")
    invariance = read_json(
        args.v4r2_evaluation.parent / "LABEL_PERMUTATION_INVARIANCE.json"
    )
    if len(discovery_rows) != 200 or int(evaluation.get("run_count", -1)) != 200:
        raise ValueError("v4r2r1 discovery/evaluation matrix is not 200 cells")
    if discovery.get("llm_api_calls") != 0:
        raise ValueError("v4r2r1 discovery unexpectedly used an LLM API")
    if discovery.get("publication_evaluation_data_loaded") is not False:
        raise ValueError("v4r2r1 discovery reports evaluation access")
    if discovery.get("all_task_differentiation_gates_passed") is not True:
        raise ValueError("a task-conditioning gate failed")
    if not (
        gate.get("status") == "passed"
        and int(gate.get("sealed_cells", -1)) == 200
        and invariance.get("status") == "passed"
        and invariance.get("all_discovery_sequences_unchanged") is True
        and str(invariance["discovery_sequence_registry_before_sha256"])
        == str(invariance["discovery_sequence_registry_after_sha256"])
        == sequence_hash
    ):
        raise ValueError("v4r2r1 reveal-boundary audit failed")
    for row in discovery_rows:
        if Path(str(row["feedback_path"])).stat().st_size != 0:
            raise ValueError("adapted baseline consumed feedback")
        if row.get("feedback_activated") is not False:
            raise ValueError("adapted baseline feedback flag is not false")
    cells = csv_rows(Path(str(results["artifacts"]["cell_metrics"])))
    summaries = csv_rows(Path(str(results["artifacts"]["method_summary"])))
    if len(cells) != 2100 or len(summaries) != 21:
        raise ValueError("combined aggregate has the wrong shape")
    if any(
        not math.isclose(
            float(row["teas5"]),
            20.0 * min(int(row["unique_exact_discoveries"]), 5),
        )
        for row in cells
    ):
        raise ValueError("TEAS-5 reproduction failed")
    audit_result = {
        "schema_version": "neurodiscovery-hindcasting-v4r2r1-final-audit.v1",
        "status": "passed",
        "created_at": utc_now(),
        "scope": "frozen v4r1 NeuroDiscovery plus v4r2r1 adapted baselines",
        "matrix": {
            "v4r2_discovery_cells": len(discovery_rows),
            "v4r2_evaluation_cells": int(evaluation["run_count"]),
            "combined_metric_rows": len(cells),
            "methods": list(METHODS),
        },
        "no_leakage": {
            "llm_api_calls": 0,
            "all_v4r2_cells_sealed_before_evaluation": True,
            "label_permutation_invariance": True,
            "sequence_registry_sha256": sequence_hash,
            "baseline_feedback_records": 0,
        },
        "construct_validity": {
            "all_task_differentiation_gates_passed": True,
            "adapted_names_retained": True,
            "native_upstream_reproduction_claimed": False,
            "post_v4r1_inspection_exploratory": True,
        },
        "frozen_neurodiscovery": {
            "v4r1_final_audit": str(args.v4r1_audit.resolve()),
            "v4r1_final_audit_sha256": sha256_file(args.v4r1_audit),
            "regenerated": False,
        },
        "results_sha256_before_audit": sha256_file(args.output / "RESULTS.json"),
    }
    atomic_json(args.output / "FINAL_AUDIT.json", audit_result)
    return audit_result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4r1-evaluation", type=Path, default=DEFAULT_V4R1_EVALUATION)
    parser.add_argument("--v4r1-results", type=Path, default=DEFAULT_V4R1_RESULTS)
    parser.add_argument("--v4r1-audit", type=Path, default=DEFAULT_V4R1_AUDIT)
    parser.add_argument("--v4r2-discovery", type=Path, default=DEFAULT_V4R2_DISCOVERY)
    parser.add_argument("--v4r2-evaluation", type=Path, default=DEFAULT_V4R2_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = aggregate(args)
    final_audit = audit(args, result)
    primary = [
        row for row in result["method_summary"] if int(row["k"]) in {100, 1000}
    ]
    print(json.dumps({"audit": final_audit["status"], "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
