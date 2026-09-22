"""Seal a completed case-study table, benchmark, and model-robustness run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def expected_model_jobs(manifest: dict[str, Any]) -> int:
    return (
        len(manifest.get("atlases") or [])
        * len(manifest.get("diseases") or [])
        * len(manifest.get("models") or [])
        * len(manifest.get("seeds") or [])
        * int(manifest.get("folds") or 0)
    )


def seal_model_robustness(
    *, task_root: Path, task: str, model_run: Path
) -> dict[str, Any]:
    source_manifest_path = model_run / "manifest.json"
    source = load_json(source_manifest_path)
    if source.get("case_study_id") != task:
        raise ValueError("model run belongs to a different case study")
    expected = expected_model_jobs(source)
    performance_path = model_run / "performance_folds.csv"
    performance = pd.read_csv(performance_path)
    completed = len(
        performance.drop_duplicates(["atlas", "disease", "model", "seed", "fold"])
    )
    failure_path = model_run / "failed_model_folds.csv"
    failed = len(pd.read_csv(failure_path)) if failure_path.is_file() else 0
    if expected <= 0 or completed != expected or failed != 0:
        raise RuntimeError(
            f"incomplete model run: expected={expected} completed={completed} failed={failed}"
        )

    artifact_names = (
        "manifest.json",
        "performance_folds.csv",
        "performance_by_seed.csv",
        "performance_summary.csv",
        "best_model_by_atlas_disease.csv",
        "best_atlas_model_by_disease.csv",
        "fold_assignments.csv",
        "heldout_attribution_by_seed.csv",
        "heldout_attribution_summary.csv",
    )
    artifacts = {
        name: descriptor(model_run / name)
        for name in artifact_names
        if (model_run / name).is_file()
    }
    required = set(artifact_names) - {
        "heldout_attribution_by_seed.csv",
        "heldout_attribution_summary.csv",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise FileNotFoundError(f"missing model robustness artifacts: {missing}")

    robustness = {
        "schema_version": "case-study-model-robustness.v1",
        "created_at": utc_now(),
        "status": "complete",
        "task": task,
        "models": list(source["models"]),
        "atlases": list(source["atlases"]),
        "diseases": list(source["diseases"]),
        "seeds": list(source["seeds"]),
        "folds": int(source["folds"]),
        "expected_jobs": expected,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "sources": [
            {
                "name": str(source.get("run_name") or model_run.name),
                "status": "complete",
                "expected_jobs": expected,
                "completed_jobs": completed,
                "failed_jobs": failed,
                "artifacts": artifacts,
            }
        ],
    }
    target = task_root / "model_robustness" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(robustness, indent=2) + "\n", encoding="utf-8")
    return robustness


def seal_closure(
    *,
    task_root: Path,
    task: str,
    comparison_manifest_path: Path,
    robustness: dict[str, Any],
    holdout_manifest_path: Path | None = None,
) -> dict[str, Any]:
    table_path = task_root / "tables" / "table_manifest.json"
    table = load_json(table_path)
    comparison = load_json(comparison_manifest_path)
    benchmark = comparison.get("benchmark_manifest")
    if table.get("task") != task or comparison.get("task") != task:
        raise ValueError("table or comparison manifest belongs to a different task")
    if not isinstance(benchmark, dict) or benchmark.get("task") != task:
        raise ValueError("comparison manifest has no complete task benchmark")
    holdout = load_json(holdout_manifest_path) if holdout_manifest_path else None
    if holdout is not None and holdout.get("task") != task:
        raise ValueError("holdout manifest belongs to a different task")
    statuses = (
        table.get("status"),
        comparison.get("status"),
        benchmark.get("status"),
        robustness.get("status"),
        holdout.get("status") if holdout is not None else "complete",
    )
    if any(status != "complete" for status in statuses):
        raise RuntimeError(f"cannot seal incomplete inputs: {statuses}")

    closure = {
        "schema_version": "case-study-closure.v3",
        "created_at": utc_now(),
        "status": "complete",
        "task": task,
        "table_manifest": table,
        "benchmark_manifest": benchmark,
        "group_holdout_manifest": holdout,
        "model_robustness_manifest": robustness,
        "sources": {
            "table_manifest": descriptor(table_path),
            "comparison_manifest": descriptor(comparison_manifest_path),
            "model_robustness_manifest": descriptor(
                task_root / "model_robustness" / "manifest.json"
            ),
            **(
                {"group_holdout_manifest": descriptor(holdout_manifest_path)}
                if holdout_manifest_path is not None
                else {}
            ),
        },
    }
    target = task_root / "closure_manifest.json"
    target.write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    return closure


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    robustness = seal_model_robustness(
        task_root=args.task_root,
        task=args.task,
        model_run=args.model_run,
    )
    closure = seal_closure(
        task_root=args.task_root,
        task=args.task,
        comparison_manifest_path=args.comparison_manifest,
        robustness=robustness,
        holdout_manifest_path=args.holdout_manifest,
    )
    print(task_root := args.task_root / "closure_manifest.json")
    print(f"status={closure['status']} task={closure['task']} path={task_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
