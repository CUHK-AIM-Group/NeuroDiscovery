"""Attach lightweight, auditable model-robustness provenance to case studies.

The model sweeps are intentionally kept separate from the primary closed-loop
tables.  This command records their manifests and summary hashes without
copying checkpoints or treating pilot-model metrics as discovery outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = (
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "connectome_behavior",
    "brain_age",
    "progression_prediction",
    "prognosis",
    "imaging_genetics",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def generic_sources(root: Path) -> dict[str, dict[str, Any]]:
    audit_path = root / "model_sweep_audit.json"
    summary_path = root / "model_sweep_summary.csv"
    audit = load_json(audit_path)
    if int(audit.get("n_failed", -1)) != 0:
        raise ValueError(f"generic model sweep has failures: {audit_path}")
    if int(audit.get("n_completed", 0)) != int(audit.get("n_tasks", -1)):
        raise ValueError(f"generic model sweep is incomplete: {audit_path}")

    grouped: dict[str, list[str]] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["case_study"]), []).append(str(row["model"]))

    result: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        models = sorted(set(grouped.get(task, [])))
        if not models:
            raise ValueError(f"generic model sweep has no models for {task}")
        seeds = sorted(
            {
                int(path.name.removeprefix("seed_"))
                for path in (root / task).glob("*/seed_*")
                if path.is_dir() and path.name.removeprefix("seed_").isdigit()
            }
        )
        configs = sorted((root / task).glob("**/config.json"))
        run_manifests = sorted((root / task).glob("**/run_manifest.json"))
        if not configs or not run_manifests:
            raise ValueError(f"generic model sweep is missing run artifacts for {task}")
        result[task] = {
            "name": "task-native model sweep",
            "status": "complete",
            "role": "independent model-family robustness; not used for hypothesis ranking",
            "root": str(root),
            "models": models,
            "seeds": seeds,
            "completed_jobs": len(run_manifests),
            "failed_jobs": int(audit["n_failed"]),
            "artifacts": {
                "audit": file_descriptor(audit_path),
                "summary": file_descriptor(summary_path),
                "representative_config": file_descriptor(configs[0]),
            },
        }
    return result


def deep_source(
    root: Path,
    *,
    name: str,
    role: str,
    failed_key: str,
    complete_key: str,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    failed = int(manifest.get(failed_key, -1))
    completed = int(manifest.get(complete_key, 0))
    if failed != 0 or completed <= 0:
        raise ValueError(f"incomplete deep-model sweep: {manifest_path}")
    summary_candidates = [
        root / "model_summary.csv",
        root / "performance_summary.csv",
    ]
    summary_path = next((path for path in summary_candidates if path.is_file()), None)
    artifacts = {"manifest": file_descriptor(manifest_path)}
    if summary_path is not None:
        artifacts["summary"] = file_descriptor(summary_path)
    result_candidates = [root / "model_results.csv", root / "performance_folds.csv"]
    result_path = next((path for path in result_candidates if path.is_file()), None)
    if result_path is None:
        raise ValueError(f"deep-model sweep has no result table: {root}")
    with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    if len(result_rows) != completed:
        raise ValueError(
            f"deep-model result count differs from manifest: {len(result_rows)} != {completed}"
        )
    if result_rows and "status" in result_rows[0]:
        failed_rows = sum(str(row["status"]).lower() != "complete" for row in result_rows)
        if failed_rows:
            raise ValueError(f"deep-model result table contains {failed_rows} failed rows")
    artifacts["results"] = file_descriptor(result_path)
    return {
        "name": name,
        "status": "complete",
        "role": role,
        "root": str(root),
        "models": sorted(str(item) for item in manifest.get("models") or []),
        "atlases": sorted(str(item) for item in manifest.get("atlases") or []),
        "seeds": [int(item) for item in manifest.get("seeds") or []],
        "completed_jobs": completed,
        "failed_jobs": failed,
        "large_models_excluded": list(manifest.get("large_models_excluded") or []),
        "artifacts": artifacts,
    }


def attach_manifests(
    closure_root: Path,
    generic_root: Path,
    *,
    case1_deep_root: Path | None = None,
    connectome_deep_root: Path | None = None,
    brain_age_deep_root: Path | None = None,
) -> list[Path]:
    by_task = {task: [source] for task, source in generic_sources(generic_root).items()}
    if case1_deep_root is not None:
        by_task["biomarker_discovery"].append(
            deep_source(
                case1_deep_root,
                name="TCP multi-atlas NeuroRuntime sweep",
                role="19-atlas disease-vs-control model and attribution robustness",
                failed_key="n_failed_model_folds",
                complete_key="n_model_folds",
            )
        )
    if connectome_deep_root is not None:
        by_task["connectome_behavior"].append(
            deep_source(
                connectome_deep_root,
                name="HCP cognition NeuroRuntime sweep",
                role="19-atlas deep connectome-regression robustness",
                failed_key="n_jobs_failed",
                complete_key="n_jobs_complete",
            )
        )
    if brain_age_deep_root is not None:
        by_task["brain_age"].append(
            deep_source(
                brain_age_deep_root,
                name="lifespan NeuroRuntime sweep",
                role="19-atlas deep brain-age regression robustness",
                failed_key="n_jobs_failed",
                complete_key="n_jobs_complete",
            )
        )

    written: list[Path] = []
    for task in TASKS:
        sources = by_task[task]
        models = sorted({model for source in sources for model in source["models"]})
        payload = {
            "schema_version": "case-study-model-robustness.v1",
            "created_at": utc_now(),
            "task": task,
            "status": "complete",
            "interpretation": (
                "Independent model-family robustness only. These outputs neither expose "
                "validation labels to generators nor replace the current-KG primary closure."
            ),
            "models": models,
            "model_count": len(models),
            "sources": sources,
        }
        path = closure_root / task / "model_robustness" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", required=True, type=Path)
    parser.add_argument("--generic-root", required=True, type=Path)
    parser.add_argument("--case1-deep-root", type=Path)
    parser.add_argument("--connectome-deep-root", type=Path)
    parser.add_argument("--brain-age-deep-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    written = attach_manifests(
        args.closure_root,
        args.generic_root,
        case1_deep_root=args.case1_deep_root,
        connectome_deep_root=args.connectome_deep_root,
        brain_age_deep_root=args.brain_age_deep_root,
    )
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
