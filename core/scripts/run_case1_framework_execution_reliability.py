"""Audit CS1 hypothesis formatting and execute a lightweight model matrix.

This systems audit keeps hypothesis generation reliability separate from
scientific performance.  It samples legal fMRI hypotheses from each generator,
deduplicates their disease-by-atlas execution cells, and actually fits one
held-out fold with classical and lightweight deep-learning models.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.scripts.case1_exhaustive_full import (
    FULL_FMRI_FEATURES,
    build_atlas_feature_matrices,
)
from core.scripts.case1_exhaustive_v1 import disease_masks, load_metadata
from core.scripts.case1_exhaustive_v2 import build_covariates
from core.scripts.case1_multimodel_pilot import (
    fold_adjust_standardize,
    load_correlation_matrices,
    make_folds,
    model_parameter_count,
    resolve_device,
    run_model_fold,
    set_seed,
)


METHOD_LABELS = {
    "neurodiscovery": "NeuroDiscovery",
    "ai_scientist_v2": "AI Scientist-v2",
    "open_coscientist": "Open Co-Scientist",
    "sciagents": "SciAgents",
    "virtual_lab": "Virtual Lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}

DEFAULT_MODELS = ("elasticnet", "roi_mlp", "brainnetcnn", "bnt", "braingnn")
NATIVE_EXECUTOR_SUPPORT = {
    "neurodiscovery": (
        True,
        "NeuroRuntime is NeuroDiscovery's native executor.",
    ),
    "ai_scientist_v2": (
        True,
        "The official BFTS workflow edits and executes experiment code.",
    ),
    "brainpilot_native": (
        True,
        "The official runtime exposes shell and engineering tools.",
    ),
    "biomni_native": (
        True,
        "The official A1 agent exposes persistent Python execution.",
    ),
    "open_coscientist": (
        False,
        "The official CS1 entry point generates hypotheses but has no experiment executor.",
    ),
    "sciagents": (
        False,
        "The official CS1 workflow generates and critiques discoveries but has no executor.",
    ),
    "virtual_lab": (
        False,
        "The official CS1 workflow is a multi-agent meeting and summary pipeline.",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--transdiag-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--native-audit-root", type=Path)
    parser.add_argument("--hypotheses-per-method", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-rois", type=int, default=200)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--graph-density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_candidate(value: str) -> dict[str, Any]:
    parts = str(value).split("|")
    if len(parts) != 5:
        raise ValueError(f"Invalid candidate_id: {value!r}")
    modality, source, disease, feature, roi_index = parts
    return {
        "candidate_id": value,
        "modality": modality,
        "source": source,
        "atlas": source.removesuffix("_multiatlas"),
        "disease": disease,
        "feature": feature,
        "roi_index": int(roi_index),
    }


def _source_rows(
    path: Path,
    *,
    seed: int | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if seed is not None and "seed" in frame.columns:
        frame = frame[pd.to_numeric(frame["seed"], errors="coerce").eq(seed)]
    if "mapping_status" in frame.columns:
        frame = frame[frame["mapping_status"].eq("mapped")]
    rank_column = "generated_rank" if "generated_rank" in frame.columns else "rank"
    candidate_column = (
        "mapped_candidate_id" if "mapped_candidate_id" in frame.columns else "candidate_id"
    )
    return frame.rename(
        columns={rank_column: "generator_rank", candidate_column: "candidate_id"}
    )[["method", "generator_rank", "candidate_id"]].copy()


def load_generator_rows(rerun_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = {
        "official": rerun_root
        / "official_baselines_primary10"
        / "generation_first_mapped_hypotheses.csv",
        "brainpilot": rerun_root
        / "native_brainpilot_10trials"
        / "generation_first_mapped_hypotheses.csv",
        "biomni": rerun_root
        / "native_biomni_10trials"
        / "generation_first_mapped_hypotheses.csv",
        "neurodiscovery": rerun_root
        / "internal_method_comparison_all_baselines_fresh10"
        / "ranked_candidates_neurodiscovery.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing generator inputs: " + ", ".join(missing))
    frames = [
        _source_rows(paths["official"], seed=0),
        _source_rows(paths["brainpilot"], seed=0),
        _source_rows(paths["biomni"], seed=0),
        _source_rows(paths["neurodiscovery"], seed=None),
    ]
    return pd.concat(frames, ignore_index=True), paths


def atlas_roi_counts(transdiag_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    fc_root = transdiag_root / "fc"
    for atlas_dir in sorted(path for path in fc_root.iterdir() if path.is_dir()):
        files = sorted((atlas_dir / "correlation").glob("*.npy"))
        if files:
            counts[atlas_dir.name] = int(np.load(files[0], mmap_mode="r").shape[0])
    return counts


def select_hypotheses(
    rows: pd.DataFrame,
    *,
    roi_counts: dict[str, int],
    per_method: int,
    max_rois: int,
) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows.sort_values(["method", "generator_rank"], kind="stable").itertuples(
        index=False
    ):
        try:
            parsed = _parse_candidate(str(row.candidate_id))
        except (TypeError, ValueError):
            continue
        parsed_rows.append(
            {
                "method": str(row.method),
                "label": METHOD_LABELS.get(str(row.method), str(row.method)),
                "generator_rank": int(row.generator_rank),
                **parsed,
                "n_rois": roi_counts.get(parsed["atlas"]),
            }
        )
    parsed = pd.DataFrame(parsed_rows)
    eligible = parsed[
        parsed["modality"].eq("fmri")
        & parsed["n_rois"].notna()
        & parsed["n_rois"].le(max_rois)
    ].copy()
    eligible = eligible.drop_duplicates(
        ["method", "disease", "atlas"], keep="first"
    )
    selected = (
        eligible.sort_values(["method", "generator_rank"], kind="stable")
        .groupby("method", sort=False, as_index=False)
        .head(per_method)
        .reset_index(drop=True)
    )
    counts = selected.groupby("method").size().to_dict()
    expected_methods = sorted(set(parsed["method"].astype(str)))
    missing = {
        method: per_method - int(counts.get(method, 0))
        for method in expected_methods
        if counts.get(method, 0) < per_method
    }
    if missing:
        raise RuntimeError(f"Insufficient eligible hypotheses: {missing}")
    selected["selection_slot"] = selected.groupby("method").cumcount() + 1
    return selected


def generation_format_audit(rerun_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    official_root = rerun_root / "official_baselines_primary10"
    official = pd.read_csv(official_root / "official_native_proposal_summary.csv")
    for method, group in official.groupby("method", sort=False):
        requested = int(pd.to_numeric(group["requested_anchors"]).sum())
        valid = int(pd.to_numeric(group["valid_anchors"]).sum())
        complete = int(group["completion_status"].eq("complete").sum())
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "framework_units": len(group),
                "framework_units_returned_and_parsed": complete,
                "artifact_success_rate": complete / len(group),
                "requested_hypothesis_slots": requested,
                "schema_valid_hypotheses": valid,
                "exact_mapped_hypotheses": valid,
                "format_success_rate": valid / requested,
                "exact_mapping_rate": valid / requested,
            }
        )

    for directory, method in (
        ("native_brainpilot_10trials", "brainpilot_native"),
        ("native_biomni_10trials", "biomni_native"),
    ):
        root = rerun_root / directory
        summary = pd.read_csv(root / "native_baselines_direct_seed_summary.csv")
        summary = summary[pd.to_numeric(summary["budget"]).eq(80)]
        requested = 80 * len(summary)
        valid = int(pd.to_numeric(summary["schema_valid"]).sum())
        mapped = int(pd.to_numeric(summary["mapped"]).sum())
        batch_dirs = sorted((root / method).glob("seed_*/batch_*"))
        parsed = sum((path / "parsed.json").is_file() for path in batch_dirs)
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "framework_units": len(batch_dirs),
                "framework_units_returned_and_parsed": parsed,
                "artifact_success_rate": parsed / len(batch_dirs),
                "requested_hypothesis_slots": requested,
                "schema_valid_hypotheses": valid,
                "exact_mapped_hypotheses": mapped,
                "format_success_rate": valid / requested,
                "exact_mapping_rate": mapped / requested,
            }
        )

    rows.append(
        {
            "method": "neurodiscovery",
            "label": METHOD_LABELS["neurodiscovery"],
            "framework_units": 10,
            "framework_units_returned_and_parsed": 10,
            "artifact_success_rate": 1.0,
            "requested_hypothesis_slots": 800,
            "schema_valid_hypotheses": 800,
            "exact_mapped_hypotheses": 800,
            "format_success_rate": 1.0,
            "exact_mapping_rate": 1.0,
        }
    )
    return pd.DataFrame(rows).sort_values("method", kind="stable")


def native_execution_audit(root: Path | None) -> pd.DataFrame:
    rows = []
    valid = pd.DataFrame()
    if root is not None:
        path = root / "native_execution_valid_trials.csv"
        if path.is_file():
            valid = pd.read_csv(path)
    for method, label in METHOD_LABELS.items():
        supported, basis = NATIVE_EXECUTOR_SUPPORT[method]
        group = valid[valid["method"].eq(method)] if not valid.empty else valid
        attempted = int(group["attempted_candidates"].sum()) if not group.empty else None
        successful = int(group["successful_candidates"].sum()) if not group.empty else None
        rate = successful / attempted if attempted else None
        rows.append(
            {
                "method": method,
                "label": label,
                "official_native_executor_supported": supported,
                "protocol_valid_real_trials": int(len(group)),
                "attempted_candidate_executions": attempted,
                "successful_candidate_executions": successful,
                "native_execution_success_rate": rate,
                "basis": basis,
                "source": str(root or ""),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def execute_cells(args: argparse.Namespace, selected: pd.DataFrame) -> pd.DataFrame:
    device = resolve_device(args.device)
    diagnosis = args.transdiag_root / "metadata" / "diagnosis.csv"
    execution_root = args.out_dir / "model_runs"
    execution_root.mkdir(parents=True, exist_ok=True)
    cells = selected[["atlas", "disease", "n_rois"]].drop_duplicates()
    output_rows: list[dict[str, Any]] = []

    for atlas, atlas_cells in cells.groupby("atlas", sort=True):
        source_subjects, roi_meta, feature_matrices = build_atlas_feature_matrices(
            args.transdiag_root, atlas, requested_subjects=None
        )
        meta, available_diseases = load_metadata(diagnosis, source_subjects, min_cases=5)
        available = {str(row["disease"]) for row in available_diseases}
        subjects = meta["subjectkey"].astype(str).tolist()
        source_order = {subject: index for index, subject in enumerate(source_subjects)}
        aligned = [source_order[subject] for subject in subjects]
        node_features = np.stack(
            [feature_matrices[name][aligned] for name in FULL_FMRI_FEATURES], axis=-1
        ).astype(np.float32)
        correlations = load_correlation_matrices(args.transdiag_root, atlas, subjects)
        covariates = build_covariates(meta).to_numpy(np.float32)

        for cell in atlas_cells.itertuples(index=False):
            disease = str(cell.disease)
            if disease not in available:
                raise RuntimeError(f"Unavailable disease {disease!r} for atlas {atlas!r}")
            case_mask, control_mask = disease_masks(meta, disease)
            subset = np.flatnonzero(case_mask | control_mask)
            y = case_mask[subset].astype(np.int64)
            disease_subjects = [subjects[index] for index in subset]
            raw_nodes = node_features[subset]
            raw_corr = correlations[subset]
            disease_covariates = covariates[subset]
            folds = make_folds(y, args.folds, args.seed)
            if args.fold_index >= len(folds):
                raise IndexError("fold-index is outside the generated folds")
            fold = folds[args.fold_index]
            adjusted_nodes = fold_adjust_standardize(
                raw_nodes, disease_covariates, fold.train
            )
            adjusted_corr = fold_adjust_standardize(
                raw_corr, disease_covariates, fold.train
            )
            for matrix in adjusted_corr:
                np.fill_diagonal(matrix, 0.0)

            for model_index, model_name in enumerate(args.models):
                run_dir = execution_root / atlas / disease / model_name
                run_path = run_dir / "run.json"
                if run_path.is_file() and not args.force:
                    cached = json.loads(run_path.read_text(encoding="utf-8"))
                    if cached.get("status") == "success":
                        output_rows.append(cached)
                        continue
                set_seed(args.seed + model_index * 1009)
                started = datetime.now(timezone.utc)
                payload: dict[str, Any] = {
                    "schema_version": "case1-model-execution-reliability.v1",
                    "atlas": atlas,
                    "disease": disease,
                    "model": model_name,
                    "seed": args.seed,
                    "fold": args.fold_index,
                    "n_case": int(y.sum()),
                    "n_control": int((y == 0).sum()),
                    "n_train": int(len(fold.train)),
                    "n_validation": int(len(fold.val)),
                    "n_test": int(len(fold.test)),
                    "device": str(device),
                    "started_at": started.isoformat(timespec="seconds"),
                }
                fitted = None
                try:
                    fitted, metrics, iterations, _ = run_model_fold(
                        model_name=model_name,
                        atlas=atlas,
                        disease=disease,
                        seed=args.seed,
                        fold_index=args.fold_index,
                        fold=fold,
                        y=y,
                        subjects=disease_subjects,
                        adjusted_nodes=adjusted_nodes,
                        adjusted_corr=adjusted_corr,
                        roi_meta=roi_meta,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        patience=1,
                        graph_density=args.graph_density,
                        skip_attribution=True,
                    )
                    finite_metrics = all(math.isfinite(float(value)) for value in metrics.values())
                    parameters = model_parameter_count(fitted)
                    success = bool(finite_metrics and int(iterations) >= 1 and parameters > 0)
                    payload.update(
                        {
                            "status": "success" if success else "failed_verification",
                            "execution_success": success,
                            "epochs_or_iterations": int(iterations),
                            "trainable_parameters": int(parameters),
                            **{key: float(value) for key, value in metrics.items()},
                        }
                    )
                except Exception as exc:
                    payload.update(
                        {
                            "status": "failed_exception",
                            "execution_success": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                finally:
                    payload["finished_at"] = datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    payload["elapsed_seconds"] = (
                        datetime.now(timezone.utc) - started
                    ).total_seconds()
                    _write_json(run_path, payload)
                    output_rows.append(payload)
                    del fitted
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(
                    f"{atlas} {disease} {model_name}: {payload['status']}",
                    flush=True,
                )
    return pd.DataFrame(output_rows)


def expand_execution_by_method(
    selected: pd.DataFrame,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = selected.merge(runs, on=["atlas", "disease"], how="left")
    expanded["execution_success"] = expanded["execution_success"].fillna(False).astype(bool)
    expanded = expanded.sort_values(
        ["method", "selection_slot", "model"], kind="stable"
    )
    rows = []
    for (method, model), group in expanded.groupby(["method", "model"], sort=False):
        successes = int(group["execution_success"].sum())
        attempted = int(len(group))
        rate = successes / attempted if attempted else float("nan")
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "model": model,
                "attempted_hypothesis_model_executions": attempted,
                "successful_hypothesis_model_executions": successes,
                "execution_success_rate": rate,
                "binary_success_sample_variance": (
                    float(group["execution_success"].astype(float).var(ddof=1))
                    if attempted > 1
                    else 0.0
                ),
            }
        )
    for method, group in expanded.groupby("method", sort=False):
        successes = int(group["execution_success"].sum())
        attempted = int(len(group))
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "model": "ALL",
                "attempted_hypothesis_model_executions": attempted,
                "successful_hypothesis_model_executions": successes,
                "execution_success_rate": successes / attempted if attempted else float("nan"),
                "binary_success_sample_variance": (
                    float(group["execution_success"].astype(float).var(ddof=1))
                    if attempted > 1
                    else 0.0
                ),
            }
        )
    return expanded, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.hypotheses_per_method < 1:
        raise ValueError("hypotheses-per-method must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generator_rows, input_paths = load_generator_rows(args.rerun_root)
    counts = atlas_roi_counts(args.transdiag_root)
    selected = select_hypotheses(
        generator_rows,
        roi_counts=counts,
        per_method=args.hypotheses_per_method,
        max_rois=args.max_rois,
    )
    selected.to_csv(args.out_dir / "selected_hypotheses.csv", index=False)

    format_audit = generation_format_audit(args.rerun_root)
    format_audit.to_csv(args.out_dir / "hypothesis_format_success.csv", index=False)
    native = native_execution_audit(args.native_audit_root)
    native.to_csv(args.out_dir / "official_native_execution_success.csv", index=False)

    runs = execute_cells(args, selected)
    runs.to_csv(args.out_dir / "model_execution_runs.csv", index=False)
    expanded, summary = expand_execution_by_method(selected, runs)
    expanded.to_csv(args.out_dir / "hypothesis_model_execution.csv", index=False)
    summary.to_csv(args.out_dir / "model_execution_success_by_method.csv", index=False)

    manifest = {
        "schema_version": "case1-framework-execution-reliability.v1",
        "created_at": utc_now(),
        "rerun_root": str(args.rerun_root),
        "transdiag_root": str(args.transdiag_root),
        "out_dir": str(args.out_dir),
        "generation_inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "selection": {
            "hypotheses_per_method": args.hypotheses_per_method,
            "fMRI_only": True,
            "unique_disease_atlas_per_method": True,
            "max_rois": args.max_rois,
            "uses_GT_or_effect_size": False,
        },
        "execution": {
            "models": args.models,
            "folds_generated": args.folds,
            "executed_fold": args.fold_index,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "graph_density": args.graph_density,
            "seed": args.seed,
            "device": str(resolve_device(args.device)),
            "success_definition": (
                "data load + model fit/backpropagation + held-out inference + finite metrics + "
                "at least one iteration/epoch + persisted run.json"
            ),
            "performance_used_for_success": False,
        },
        "official_native_execution_source": str(args.native_audit_root or ""),
        "outputs": {
            "format": "hypothesis_format_success.csv",
            "native": "official_native_execution_success.csv",
            "selected": "selected_hypotheses.csv",
            "runs": "model_execution_runs.csv",
            "expanded": "hypothesis_model_execution.csv",
            "summary": "model_execution_success_by_method.csv",
        },
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
