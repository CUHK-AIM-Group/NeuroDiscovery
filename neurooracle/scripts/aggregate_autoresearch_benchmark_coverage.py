"""Audit completion of the 17-task NeuroDiscovery/native-agent benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from neurooracle.src.autoresearch_benchmark_tasks import AUTORESEARCH_BENCHMARK_TASKS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CS3_ND = ROOT / "neurooracle/data/experiments/case3/case3_all_tasks_5year_phase3_seed10_20260722"
DEFAULT_CS3_NATIVE = ROOT / "neurooracle/data/experiments/case3/native_baselines_gpt56sol_high_seed10_20260722"
DEFAULT_CS1_ND = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260707_fullv2_kg_rerun\method_comparison_gpt55_trials10"
)
DEFAULT_CS1_NATIVE = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260707_fullv2_kg_rerun\native_baselines_gpt56sol_high"
)
DEFAULT_OUT = ROOT / "neurooracle/data/experiments/autoresearch_benchmark_17task_20260722"
METHODS = ("neurodiscovery", "brainpilot_native", "biomni_native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cs3-neurodiscovery-root", type=Path, default=DEFAULT_CS3_ND)
    parser.add_argument("--cs3-native-root", type=Path, default=DEFAULT_CS3_NATIVE)
    parser.add_argument("--cs1-neurodiscovery-root", type=Path, default=DEFAULT_CS1_ND)
    parser.add_argument("--cs1-native-root", type=Path, default=DEFAULT_CS1_NATIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-native-ranks", type=int, default=1000)
    return parser.parse_args()


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_csv(path, usecols=[0]))


def ordinary_status(args: argparse.Namespace, task: str, method: str) -> dict[str, Any]:
    if method == "neurodiscovery":
        task_root = args.cs3_neurodiscovery_root / task
        hypotheses = len(list(task_root.glob("kg*_to_*/hypotheses_general_raw.json")))
        metrics = len(list(task_root.glob("kg*_to_*/general_hindcasting/metrics.json")))
        status = "completed" if metrics == 5 else "partial" if hypotheses or metrics else "missing"
        return {"status": status, "generated_slots": None, "completed_windows": metrics, "path": str(task_root)}

    source = args.cs3_native_root / method / task / "seed_10" / "native_hypotheses.csv"
    generated = csv_rows(source)
    schema_valid = 0
    if generated:
        native = pd.read_csv(source, usecols=["native_schema_valid"])
        schema_valid = int(
            native["native_schema_valid"].astype(str).str.casefold().eq("true").sum()
        )
    scored_path = args.cs3_native_root / "hindcasting" / "native_baseline_metrics_by_k.csv"
    scored_windows = 0
    if scored_path.exists():
        metrics = pd.read_csv(scored_path)
        scored_windows = int(
            metrics[(metrics["method"] == method) & (metrics["task"] == task)]["freeze_year"].nunique()
        )
    status = (
        "completed" if generated == args.expected_native_ranks and schema_valid > 0 and scored_windows == 5
        else "generated" if generated == args.expected_native_ranks and schema_valid > 0
        else "failed_generation" if generated == args.expected_native_ranks
        else "partial" if generated
        else "missing"
    )
    return {
        "status": status,
        "generated_slots": generated,
        "schema_valid_slots": schema_valid,
        "completed_windows": scored_windows,
        "path": str(source),
    }


def case1_status(args: argparse.Namespace, method: str) -> dict[str, Any]:
    if method == "neurodiscovery":
        path = args.cs1_neurodiscovery_root / "case1_method_summary.csv"
        if not path.exists():
            return {"status": "missing", "generated_slots": None, "completed_windows": None, "path": str(path)}
        frame = pd.read_csv(path)
        present = "neurodiscovery" in set(frame.get("method", []))
        return {
            "status": "completed" if present else "missing",
            "generated_slots": None,
            "completed_windows": None,
            "path": str(path),
        }
    paths = sorted((args.cs1_native_root / method).glob("seed_*/mapped_hypotheses.csv"))
    rows_by_seed = [csv_rows(path) for path in paths]
    rows = sum(rows_by_seed)
    completed_runs = sum(count > 0 for count in rows_by_seed)
    return {
        "status": "completed" if completed_runs == 10 else "partial" if rows else "missing",
        "generated_slots": rows,
        "completed_windows": completed_runs,
        "path": str(args.cs1_native_root / method / "seed_*" / "mapped_hypotheses.csv"),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for task in AUTORESEARCH_BENCHMARK_TASKS:
        for method in METHODS:
            base = {
                "task": task.name,
                "task_label": task.label,
                "family": task.family,
                "protocol": task.protocol,
                "signature": task.signature,
                "method": method,
            }
            if task.default_deferred:
                detail = {
                    "status": "deferred",
                    "generated_slots": None,
                    "completed_windows": None,
                    "path": "",
                    "note": task.deferred_reason,
                }
            elif task.name == "case1_transdiagnostic":
                detail = {**case1_status(args, method), "note": ""}
            else:
                detail = {**ordinary_status(args, task.name, method), "note": ""}
            rows.append({**base, **detail})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(args.output_dir / "benchmark_coverage.csv", index=False)
    manifest = {
        "registered_tasks": 17,
        "ordinary_tasks": 15,
        "case_studies": 2,
        "scheduled_tasks": 16,
        "deferred_tasks": ["case2_pathway_mediation"],
        "methods": list(METHODS),
        "coverage_status_counts": coverage["status"].value_counts().to_dict(),
        "roots": {
            "cs3_neurodiscovery": str(args.cs3_neurodiscovery_root),
            "cs3_native": str(args.cs3_native_root),
            "cs1_neurodiscovery": str(args.cs1_neurodiscovery_root),
            "cs1_native": str(args.cs1_native_root),
        },
    }
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
