from __future__ import annotations

import csv
import json
from pathlib import Path

from core.scripts.attach_case_study_model_robustness import TASKS, attach_manifests


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_attach_manifests_records_generic_and_deep_sources(tmp_path: Path) -> None:
    generic = tmp_path / "generic"
    _write_json(
        generic / "model_sweep_audit.json",
        {"n_tasks": len(TASKS), "n_completed": len(TASKS), "n_failed": 0},
    )
    with (generic / "model_sweep_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_study", "model"])
        writer.writeheader()
        for task in TASKS:
            writer.writerow({"case_study": task, "model": "ridge"})
            config = generic / task / "ridge" / "seed_1" / "config.json"
            _write_json(config, {"task": task})
            _write_json(config.parent / "run_manifest.json", {"status": "complete"})

    deep = tmp_path / "deep"
    _write_json(
        deep / "manifest.json",
        {
            "models": ["bnt"],
            "atlases": ["schaefer_400"],
            "seeds": [1],
            "n_jobs_complete": 1,
            "n_jobs_failed": 0,
        },
    )
    (deep / "model_summary.csv").write_text("model\nbnt\n", encoding="utf-8")
    (deep / "model_results.csv").write_text(
        "model,status\nbnt,complete\n", encoding="utf-8"
    )

    closure = tmp_path / "closure"
    written = attach_manifests(
        closure,
        generic,
        connectome_deep_root=deep,
        brain_age_deep_root=deep,
    )
    assert len(written) == len(TASKS)
    connectome = json.loads(
        (closure / "connectome_behavior" / "model_robustness" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert connectome["status"] == "complete"
    assert connectome["models"] == ["bnt", "ridge"]
    assert len(connectome["sources"]) == 2
