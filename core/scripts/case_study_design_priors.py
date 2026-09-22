"""Build outcome-blind design priors for registered case-study candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.scripts.case_study_neurodiscovery_policy import DESIGN_PRIOR_COLUMN


SCHEMA = "case-study-design-prior.v1"
ATLAS_RESOLUTION_COLUMN = "score_atlas_resolution_prior"
MODEL_CLASS_COLUMN = "score_model_class_prior"
ATLAS_ROI_COUNT_COLUMN = "design_atlas_n_rois"

# These values encode model class, not observed task performance. Regularized
# linear estimators receive full prior support; a nonlinear kernel estimator
# remains eligible with half support.
REGULARIZED_MODEL_CLASS_PRIOR = {
    "ridge": 1.0,
    "elastic_net": 1.0,
    "svm": 0.5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atlas_roi_counts(table_manifest: Mapping[str, Any]) -> dict[str, int]:
    provenance = table_manifest.get("provenance") or {}
    audits = provenance.get("feature_cache_audits") or {}
    protocol = provenance.get("protocol") or {}
    discovery_datasets = tuple(protocol.get("discovery_datasets") or ())
    counts: dict[str, int] = {}
    for atlas, dataset_records in audits.items():
        if not isinstance(dataset_records, Mapping):
            continue
        ordered_names = [
            *[name for name in discovery_datasets if name in dataset_records],
            *[name for name in dataset_records if name not in discovery_datasets],
        ]
        for dataset_name in ordered_names:
            record = dataset_records.get(dataset_name)
            if not isinstance(record, Mapping):
                continue
            raw_count = record.get("expected_n_rois")
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts[str(atlas)] = count
                break
    return counts


def attach_connectome_design_prior(
    public: pd.DataFrame,
    table_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach a weak public-metadata prior without reading outcome tables."""

    required = {"candidate_id", "atlas", "model"}
    missing = sorted(required - set(public.columns))
    if missing:
        raise ValueError(f"connectome candidate table lacks columns: {missing}")
    atlas_counts = _atlas_roi_counts(table_manifest)
    candidate_atlases = public["atlas"].fillna("").astype(str)
    missing_atlases = sorted(set(candidate_atlases) - set(atlas_counts))
    if missing_atlases:
        raise ValueError(f"ROI counts unavailable for atlases: {missing_atlases}")

    candidate_models = public["model"].fillna("").astype(str)
    missing_models = sorted(set(candidate_models) - set(REGULARIZED_MODEL_CLASS_PRIOR))
    if missing_models:
        raise ValueError(f"model classes unavailable for models: {missing_models}")

    roi_counts = candidate_atlases.map(atlas_counts).to_numpy(dtype=float)
    log_counts = np.log(roi_counts)
    low = float(log_counts.min())
    high = float(log_counts.max())
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("atlas ROI counts must span at least two positive values")
    atlas_resolution = (log_counts - low) / (high - low)
    model_class = candidate_models.map(REGULARIZED_MODEL_CLASS_PRIOR).to_numpy(
        dtype=float
    )
    design_prior = 0.5 * atlas_resolution + 0.5 * model_class

    out = public.copy()
    out[ATLAS_ROI_COUNT_COLUMN] = roi_counts.astype(int)
    out[ATLAS_RESOLUTION_COLUMN] = atlas_resolution
    out[MODEL_CLASS_COLUMN] = model_class
    out[DESIGN_PRIOR_COLUMN] = design_prior
    audit = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "task": "connectome_behavior",
        "score_column": DESIGN_PRIOR_COLUMN,
        "formula": ("0.5 * minmax(log(atlas_n_rois)) + 0.5 * model_class_prior"),
        "component_weights": {
            ATLAS_RESOLUTION_COLUMN: 0.5,
            MODEL_CLASS_COLUMN: 0.5,
        },
        "model_class_prior": dict(REGULARIZED_MODEL_CLASS_PRIOR),
        "atlas_roi_counts": dict(sorted(atlas_counts.items())),
        "uses_internal_outcomes": False,
        "uses_external_outcomes": False,
        "uses_generator_identity": False,
        "uses_kg_score": False,
        "candidate_count": int(len(out)),
        "score_min": float(design_prior.min()),
        "score_max": float(design_prior.max()),
    }
    return out, audit


def attach_brain_age_design_prior(
    public: pd.DataFrame,
    table_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Favor regularized estimators using public model metadata only."""

    del table_manifest
    required = {"candidate_id", "model"}
    missing = sorted(required - set(public.columns))
    if missing:
        raise ValueError(f"brain-age candidate table lacks columns: {missing}")
    candidate_models = public["model"].fillna("").astype(str)
    missing_models = sorted(set(candidate_models) - set(REGULARIZED_MODEL_CLASS_PRIOR))
    if missing_models:
        raise ValueError(f"model classes unavailable for models: {missing_models}")
    model_class = candidate_models.map(REGULARIZED_MODEL_CLASS_PRIOR).to_numpy(
        dtype=float
    )
    out = public.copy()
    out[MODEL_CLASS_COLUMN] = model_class
    out[DESIGN_PRIOR_COLUMN] = model_class
    audit = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "task": "brain_age",
        "score_column": DESIGN_PRIOR_COLUMN,
        "formula": "model_class_prior",
        "component_weights": {MODEL_CLASS_COLUMN: 1.0},
        "model_class_prior": dict(REGULARIZED_MODEL_CLASS_PRIOR),
        "uses_internal_outcomes": False,
        "uses_external_outcomes": False,
        "uses_generator_identity": False,
        "uses_kg_score": False,
        "candidate_count": int(len(out)),
        "score_min": float(model_class.min()),
        "score_max": float(model_class.max()),
    }
    return out, audit


def augment_task_bundle(
    *,
    source_root: Path,
    output_root: Path,
    task: str,
) -> dict[str, Any]:
    builders = {
        "brain_age": attach_brain_age_design_prior,
        "connectome_behavior": attach_connectome_design_prior,
    }
    if task not in builders:
        raise ValueError(f"no registered design prior for task: {task}")
    source_tables = source_root / task / "tables"
    output_tables = output_root / task / "tables"
    manifest_path = source_tables / "table_manifest.json"
    public_path = source_tables / "public_candidates.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public = pd.read_csv(public_path, low_memory=False)
    augmented, audit = builders[task](public, manifest)

    output_tables.mkdir(parents=True, exist_ok=True)
    for path in source_tables.iterdir():
        if path.is_file() and path.name not in {
            "public_candidates.csv",
            "table_manifest.json",
        }:
            shutil.copy2(path, output_tables / path.name)
    output_public = output_tables / "public_candidates.csv"
    augmented.to_csv(output_public, index=False)

    derived = deepcopy(manifest)
    derived["created_at"] = utc_now()
    derived["design_prior"] = audit
    derived["design_prior"]["source_bundle"] = {
        "table_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "public_candidates": {
            "path": str(public_path.resolve()),
            "sha256": sha256_file(public_path),
        },
    }
    files = derived.setdefault("files", {})
    files["public_candidates"] = {
        "path": str(output_public.resolve()),
        "sha256": sha256_file(output_public),
        "rows": int(len(augmented)),
        "columns": list(augmented.columns),
    }
    for key, filename in (
        ("internal_outcomes", "internal_outcomes.csv"),
        ("external_outcomes", "external_outcomes.csv"),
    ):
        copied = output_tables / filename
        if copied.is_file():
            descriptor = dict(files.get(key) or {})
            descriptor.update(
                {
                    "path": str(copied.resolve()),
                    "sha256": sha256_file(copied),
                }
            )
            files[key] = descriptor
    output_manifest = output_tables / "table_manifest.json"
    output_manifest.write_text(
        json.dumps(derived, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "public_candidates": str(output_public.resolve()),
        "public_candidates_sha256": sha256_file(output_public),
        "design_prior": audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["connectome_behavior", "brain_age"]
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = [
        augment_task_bundle(
            source_root=args.source_root,
            output_root=args.output_root,
            task=task,
        )
        for task in args.tasks
    ]
    manifest = {
        "schema_version": "case-study-design-prior-bundle.v1",
        "created_at": utc_now(),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "uses_internal_outcomes": False,
        "uses_external_outcomes": False,
        "tasks": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "design_prior_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
