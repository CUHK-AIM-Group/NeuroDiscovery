from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.scripts.finalize_case_study_closed_loop import (
    seal_closure,
    seal_model_robustness,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_seals_complete_model_and_closure_manifests(tmp_path: Path) -> None:
    task = "biomarker_discovery"
    task_root = tmp_path / task
    model_run = task_root / "model_robustness" / "run"
    model_run.mkdir(parents=True)
    source = {
        "case_study_id": task,
        "run_name": "formal",
        "atlases": ["atlas"],
        "diseases": ["disease"],
        "models": ["model"],
        "seeds": [1],
        "folds": 2,
    }
    write_json(model_run / "manifest.json", source)
    rows = [
        {"atlas": "atlas", "disease": "disease", "model": "model", "seed": 1, "fold": fold}
        for fold in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(model_run / "performance_folds.csv", index=False)
    for name in (
        "performance_by_seed.csv",
        "performance_summary.csv",
        "best_model_by_atlas_disease.csv",
        "best_atlas_model_by_disease.csv",
        "fold_assignments.csv",
        "heldout_attribution_by_seed.csv",
    ):
        pd.DataFrame({"value": [1]}).to_csv(model_run / name, index=False)

    robustness = seal_model_robustness(
        task_root=task_root, task=task, model_run=model_run
    )
    assert robustness["status"] == "complete"
    assert robustness["completed_jobs"] == 2
    assert "heldout_attribution_by_seed.csv" in robustness["sources"][0]["artifacts"]

    write_json(
        task_root / "tables" / "table_manifest.json",
        {"task": task, "status": "complete", "candidate_count": 2},
    )
    comparison_path = task_root / "comparison" / "manifest.json"
    write_json(
        comparison_path,
        {
            "task": task,
            "status": "complete",
            "benchmark_manifest": {"task": task, "status": "complete"},
        },
    )
    holdout_path = task_root / "comparison" / "holdout_manifest.json"
    write_json(
        holdout_path,
        {
            "task": task,
            "status": "complete",
            "holdout_gt_total": 1,
        },
    )
    closure = seal_closure(
        task_root=task_root,
        task=task,
        comparison_manifest_path=comparison_path,
        robustness=robustness,
        holdout_manifest_path=holdout_path,
    )
    assert closure["status"] == "complete"
    assert closure["group_holdout_manifest"]["holdout_gt_total"] == 1
    assert "group_holdout_manifest" in closure["sources"]
    assert (task_root / "closure_manifest.json").is_file()
