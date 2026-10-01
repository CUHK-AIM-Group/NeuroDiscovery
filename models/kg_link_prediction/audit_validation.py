"""Versioned withdrawal of invalid development claims, without retraining."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path) -> dict:
    names = ("ROOT_ADJUDICATION.json", "CANDIDATES.json", "CANDIDATES_PATHS.json",
             "FROZEN_LABEL_EVAL.json", "RESULT.json", "RESULT_MATCHED.json",
             "_smoke_run.json", "_diag_tuned_directional.json", "_diag_tuned_distmult.json")
    bindings = {name: sha256(directory / name) for name in names if (directory / name).is_file()}
    reference = json.loads((directory / "ROOT_ADJUDICATION.json").read_text(encoding="utf-8"))
    corrections = []
    for row in reference["items"]:
        corrections.append({
            "item_id": row["item_id"], "original_row_sha256": hashlib.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "source_work_key": row.get("source_work_key"),
            "source_sha256": row.get("source_sha256"),
            "old_verdict": row["verdict"],
            "status": "HELD_REVIEW_REQUIRED", "new_scientific_label": None,
            "reverse_label": "unresolved", "eligible_for_training_or_acceptance": False,
            "reason": "Historical sentence-only development judgment; exact entity, measurement, scope, role and source-family binding need review. Forward support does not establish reverse contradiction.",
        })
    paths = json.loads((directory / "CANDIDATES_PATHS.json").read_text(encoding="utf-8"))
    loops = []
    for item in paths["path_positive"]["items"]:
        hops = item["hops"]
        vertices = [hops[0]["source_id"], *[hop["target_id"] for hop in hops]] if hops else []
        if len(set(vertices)) != len(vertices):
            loops.append(item["item_id"])
    results = []
    for name in names:
        if name not in bindings:
            continue
        data = json.loads((directory / name).read_text(encoding="utf-8"))
        runs = data.get("runs", [])
        if name == "FROZEN_LABEL_EVAL.json":
            runs = [run for decoder in data["decoders"].values() for run in decoder["runs"]]
        if runs:
            results.append({"file": name, "retained_run_records": len(runs),
                            "peak_gpu_bytes": data.get("torch", {}).get("peak_gpu_bytes"),
                            "evidence_status": "development_only_not_validated_accuracy"})
    retained = sum(row["retained_run_records"] for row in results)
    return {
        "schema": "gnn-offline-correction-v1", "status": "HELD_NO_TRAINING",
        "input_sha256": bindings, "reading_scope": "existing scripts and development artifacts; no new source abstracts read",
        "independent_validation": False, "new_model_training_runs": 0,
        "new_external_calls": 0, "new_source_labels": 0,
        "old_verdict_counts_not_endorsed": dict(Counter(row["old_verdict"] for row in corrections)),
        "corrected_items": corrections, "repeated_vertex_path_ids": loops,
        "retained_training_records_lower_bound": retained,
        "total_training_runs": None,
        "count_limitation": "Overwritten evaluation attempts and unpersisted diagnostic runs are not recoverable from retained JSON; do not equate retained records with total executions.",
        "run_limit": 6, "training_slots_remaining": 0 if retained >= 6 else None,
        "results": results,
        "withdrawn_claims": [
            "24 source-verified hard negatives", "19 independent scientific facts",
            "0.841 or 0.889 validated accuracy", "64% of the whole graph unusable",
            "path validation infeasible within budget", "no model training occurred",
            "protected heldout exclusion verified by previous candidate builders",
        ],
        "reason_for_stop": "Run budget already exceeded; references and split are not qualified. No renewed training authorization is inferred from an offline repair request.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.input_dir)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    for name, expected in report["input_sha256"].items():
        if sha256(args.input_dir / name) != expected:
            raise RuntimeError(f"input changed during audit: {name}")
    print(json.dumps({key: report[key] for key in (
        "status", "new_model_training_runs", "retained_training_records_lower_bound",
        "training_slots_remaining", "repeated_vertex_path_ids",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
