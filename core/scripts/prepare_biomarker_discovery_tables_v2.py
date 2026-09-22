"""Build the independent Case Study 3 biomarker-discovery experiment tables.

The candidate registry is generated from TCP derivatives only.  Experimental
outcomes are kept in a physically separate table and never enter the KG score.
Unlike the retired migration script, this runner does not reuse Case Study 1
candidate IDs, rankings, labels, or external-validation artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from case1_exhaustive_full import (  # noqa: E402
    DEFAULT_ATLAS_ROOT,
    FULL_FMRI_FEATURES,
    atlas_roi_meta,
    build_atlas_feature_matrices,
    common_subjects_for_atlases,
    infer_target_shape,
    list_atlases,
)
from case1_exhaustive_v1 import (  # noqa: E402
    benjamini_hochberg,
    disease_masks,
    load_metadata,
    load_smri_source,
    rank_auc,
)
from case1_exhaustive_v2 import build_covariates, evaluate_matrix_v2  # noqa: E402
from case_study_candidate_tables import (  # noqa: E402
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
)
from case_study_closed_loop import sha256_file  # noqa: E402


TASK = "biomarker_discovery"
DEFAULT_TRANSDIAG_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\transdiag_preprocessed"
)
DEFAULT_DIAGNOSIS = DEFAULT_TRANSDIAG_ROOT / "metadata" / "diagnosis.csv"
DEFAULT_SMRI_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\tcp_fastsurfer_segonly\features"
)
DEFAULT_OUTPUT = Path(
    r"\\192.168.3.61\data\Public Dataset"
    r"\case_study_closed_loop_v8_biomarker_independent"
    r"\biomarker_discovery\tables"
)
DEFAULT_KG = (
    Path(__file__).resolve().parents[2]
    / "neurooracle"
    / "data"
    / "full_v2"
    / "knowledge_graph.json"
)
DEFAULT_CLAIMS = DEFAULT_KG.with_name("extracted_claims.jsonl")
DEFAULT_STATE = DEFAULT_KG.with_name("CURRENT_STATE.json")
DEFAULT_CV_SEEDS = (20260816, 20260817, 20260818, 20260819, 20260820)
IDENTITY_FIELDS = (
    "disease",
    "atlas",
    "roi_index",
    "anatomy",
    "feature_family",
    "feature",
)
SEMANTIC_FIELDS = (
    "disease",
    "atlas",
    "anatomy",
    "feature_family",
    "feature",
)


def feature_family(feature: object) -> str:
    name = str(feature)
    if name in {"roi_alff_proxy", "roi_falff_proxy"}:
        return "amplitude"
    if name.startswith("roi_temporal_"):
        return "temporal"
    if name.startswith("corr_"):
        return "correlation_fc"
    if name.startswith("partial_"):
        return "partial_fc"
    if name == "normalized_volume_fraction":
        return "structure"
    raise ValueError(f"unregistered biomarker feature: {name}")


def parse_seeds(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
        seeds = tuple(int(item) for item in values)
    else:
        seeds = tuple(int(item) for item in raw)
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds:
        raise ValueError("at least one CV seed is required")
    return seeds


def _site_strata(
    labels: np.ndarray,
    sites: pd.Series,
    *,
    n_splits: int,
) -> np.ndarray:
    site_values = sites.fillna("unknown").astype(str).to_numpy()
    composite = np.asarray(
        [f"{int(label)}|{site}" for label, site in zip(labels, site_values, strict=True)]
    )
    counts = pd.Series(composite).value_counts()
    if not counts.empty and int(counts.min()) >= n_splits:
        return composite
    return labels.astype(str)


def repeated_directional_validation(
    *,
    matrix: np.ndarray,
    meta: pd.DataFrame,
    covariates: pd.DataFrame,
    disease: str,
    seeds: Sequence[int],
    n_splits: int,
) -> dict[str, np.ndarray | int | str]:
    """Measure held-out direction and discrimination without test-fold tuning."""

    case_mask, control_mask = disease_masks(meta, disease)
    subset_mask = case_mask | control_mask
    labels = case_mask[subset_mask].astype(int)
    values = np.asarray(matrix[subset_mask], dtype=float)
    nuisance = covariates.loc[subset_mask].to_numpy(dtype=float)
    sites = meta.loc[subset_mask, "Site"] if "Site" in meta else pd.Series(
        "unknown", index=np.flatnonzero(subset_mask)
    )
    n_roi = values.shape[1]
    if int(labels.sum()) < n_splits or int((labels == 0).sum()) < n_splits:
        nan = np.full(n_roi, np.nan)
        return {
            "cv_auc_mean": nan,
            "cv_auc_sd": nan,
            "cv_auc_p10": nan,
            "cv_direction_stability": nan,
            "cv_splits": 0,
            "cv_stratification": "insufficient_cases",
        }

    strata = _site_strata(labels, sites, n_splits=n_splits)
    stratification = "diagnosis_x_site" if "|" in str(strata[0]) else "diagnosis"
    fold_auc: list[np.ndarray] = []
    fold_concordance: list[np.ndarray] = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for train_index, test_index in splitter.split(values, strata):
            train_values = values[train_index].copy()
            test_values = values[test_index].copy()
            train_median = np.nanmedian(train_values, axis=0)
            train_median[~np.isfinite(train_median)] = 0.0
            train_values = np.where(
                np.isfinite(train_values), train_values, train_median
            )
            test_values = np.where(
                np.isfinite(test_values), test_values, train_median
            )

            x_train = np.column_stack(
                [np.ones(len(train_index)), nuisance[train_index]]
            )
            x_test = np.column_stack(
                [np.ones(len(test_index)), nuisance[test_index]]
            )
            nuisance_beta = np.linalg.pinv(x_train) @ train_values
            train_residual = train_values - x_train @ nuisance_beta
            test_residual = test_values - x_test @ nuisance_beta

            y_train = labels[train_index]
            y_test = labels[test_index]
            train_delta = (
                np.mean(train_residual[y_train == 1], axis=0)
                - np.mean(train_residual[y_train == 0], axis=0)
            )
            train_direction = np.where(train_delta < 0.0, -1.0, 1.0)
            auc = rank_auc(
                test_residual[y_test == 1],
                test_residual[y_test == 0],
            )
            oriented_auc = np.where(train_direction > 0.0, auc, 1.0 - auc)
            test_delta = (
                np.mean(test_residual[y_test == 1], axis=0)
                - np.mean(test_residual[y_test == 0], axis=0)
            )
            fold_auc.append(oriented_auc)
            fold_concordance.append((train_direction * test_delta) > 0.0)

    auc_matrix = np.vstack(fold_auc)
    concordance_matrix = np.vstack(fold_concordance).astype(float)
    return {
        "cv_auc_mean": np.nanmean(auc_matrix, axis=0),
        "cv_auc_sd": np.nanstd(auc_matrix, axis=0, ddof=1),
        "cv_auc_p10": np.nanpercentile(auc_matrix, 10.0, axis=0),
        "cv_direction_stability": np.nanmean(concordance_matrix, axis=0),
        "cv_splits": int(auc_matrix.shape[0]),
        "cv_stratification": stratification,
    }


def _evaluate_feature_matrix(
    *,
    modality: str,
    source: str,
    atlas: str,
    feature: str,
    meta: pd.DataFrame,
    diseases: list[dict[str, object]],
    roi_meta: pd.DataFrame,
    matrix: np.ndarray,
    covariates: pd.DataFrame,
    cv_seeds: Sequence[int],
    cv_folds: int,
    seed: int,
) -> pd.DataFrame:
    rows = evaluate_matrix_v2(
        modality=modality,
        source=source,
        feature_name=feature,
        meta=meta,
        diseases=diseases,
        roi_meta=roi_meta,
        matrix=matrix,
        covariates=covariates,
        n_boot=0,
        seed=seed,
    )
    frame = pd.DataFrame(rows)
    frame["atlas"] = atlas
    frame["feature_family"] = feature_family(feature)
    frame["anatomy"] = (
        frame["anatomy_full"]
        .fillna("")
        .astype(str)
        .where(frame["anatomy_full"].fillna("").astype(str).str.strip().ne(""), frame["roi_name"])
    )
    for disease_row in diseases:
        disease = str(disease_row["disease"])
        metrics = repeated_directional_validation(
            matrix=matrix,
            meta=meta,
            covariates=covariates,
            disease=disease,
            seeds=cv_seeds,
            n_splits=cv_folds,
        )
        selected = frame["disease"].astype(str).eq(disease)
        if int(selected.sum()) != matrix.shape[1]:
            raise RuntimeError(
                f"unexpected ROI row count for {disease}/{atlas}/{feature}"
            )
        for column in (
            "cv_auc_mean",
            "cv_auc_sd",
            "cv_auc_p10",
            "cv_direction_stability",
        ):
            frame.loc[selected, column] = np.asarray(metrics[column], dtype=float)
        frame.loc[selected, "cv_splits"] = int(metrics["cv_splits"])
        frame.loc[selected, "cv_stratification"] = str(
            metrics["cv_stratification"]
        )
    return frame


def _add_group_fdr(
    frame: pd.DataFrame,
    *,
    group_fields: Sequence[str],
    output_column: str,
) -> None:
    q_values = np.full(len(frame), np.nan, dtype=float)
    for indices in frame.groupby(list(group_fields), sort=False).indices.values():
        compact = np.asarray(indices, dtype=int)
        p_values = pd.to_numeric(
            frame.iloc[compact]["p_value"], errors="coerce"
        ).to_numpy(dtype=float)
        q_values[compact] = benjamini_hochberg(p_values)
    frame[output_column] = q_values


def apply_validation_contract(
    frame: pd.DataFrame,
    *,
    family_q: float,
    effect_threshold: float,
    auc_threshold: float,
    stability_threshold: float,
) -> pd.DataFrame:
    out = frame.copy()
    _add_group_fdr(
        out,
        group_fields=("disease", "atlas", "feature_family"),
        output_column="q_fdr_registered_family",
    )
    _add_group_fdr(
        out,
        group_fields=("disease",),
        output_column="q_fdr_disease",
    )
    out["q_fdr_global"] = benjamini_hochberg(
        pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float)
    )
    effect = pd.to_numeric(out["adjusted_residual_d"], errors="coerce").abs()
    auc = pd.to_numeric(out["cv_auc_mean"], errors="coerce")
    stability = pd.to_numeric(out["cv_direction_stability"], errors="coerce")
    family_q_values = pd.to_numeric(
        out["q_fdr_registered_family"], errors="coerce"
    )
    execution_succeeded = (
        np.isfinite(effect.to_numpy(dtype=float))
        & np.isfinite(auc.to_numpy(dtype=float))
        & np.isfinite(stability.to_numpy(dtype=float))
        & np.isfinite(family_q_values.to_numpy(dtype=float))
    )
    statistical = (
        execution_succeeded
        & family_q_values.le(family_q)
        & effect.ge(effect_threshold)
    )
    validated = (
        statistical
        & auc.ge(auc_threshold)
        & stability.ge(stability_threshold)
    )
    strict = validated & pd.to_numeric(
        out["q_fdr_global"], errors="coerce"
    ).le(family_q)
    out["execution_succeeded"] = execution_succeeded
    out["statistically_validated"] = statistical
    out["validated"] = validated
    out["strict_validated"] = strict
    out["feedback_status"] = np.where(validated, "supported", "inconclusive")
    out["feedback_available"] = execution_succeeded

    q_component = np.clip(
        -np.log10(np.clip(family_q_values.to_numpy(dtype=float), 1e-12, 1.0))
        / 5.0,
        0.0,
        1.0,
    )
    effect_component = np.clip(effect.to_numpy(dtype=float) / 0.80, 0.0, 1.0)
    auc_component = np.clip(
        (auc.to_numpy(dtype=float) - 0.50) / 0.20, 0.0, 1.0
    )
    stability_component = np.clip(
        stability.to_numpy(dtype=float), 0.0, 1.0
    )
    utility = (
        0.30 * q_component
        + 0.25 * effect_component
        + 0.25 * auc_component
        + 0.20 * stability_component
    )
    utility[~execution_succeeded] = np.nan
    out["feedback_utility"] = utility
    return out


def _data_inventory(transdiag_root: Path, atlases: Sequence[str]) -> dict[str, Any]:
    atlas_rows = []
    for atlas in atlases:
        atlas_rows.append(
            {
                "atlas": atlas,
                "roi_files": len(list((transdiag_root / "roi" / atlas).glob("*.npy"))),
                "correlation_files": len(
                    list(
                        (transdiag_root / "fc" / atlas / "correlation").glob(
                            "*.npy"
                        )
                    )
                ),
                "partial_correlation_files": len(
                    list(
                        (
                            transdiag_root
                            / "fc"
                            / atlas
                            / "partial_correlation"
                        ).glob("*.npy")
                    )
                ),
            }
        )
    return {"atlases": atlas_rows}


def build_tables(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    cv_seeds = parse_seeds(args.cv_seeds)
    atlases = list_atlases(args.transdiag_root, args.atlas_pattern)
    if args.max_atlases:
        atlases = atlases[: args.max_atlases]
    if not atlases:
        raise FileNotFoundError("no prepared TCP atlases matched the requested pattern")
    subjects = common_subjects_for_atlases(args.transdiag_root, atlases)
    target_shape = infer_target_shape(args.transdiag_root)
    metadata, diseases = load_metadata(args.diagnosis, subjects, args.min_cases)
    subjects = metadata["subjectkey"].astype(str).tolist()
    print(
        f"[biomarker] subjects={len(subjects)} diseases={len(diseases)} "
        f"atlases={len(atlases)} features={len(FULL_FMRI_FEATURES)}",
        flush=True,
    )

    parts: list[pd.DataFrame] = []
    for atlas_index, atlas in enumerate(atlases):
        print(f"[atlas {atlas_index + 1}/{len(atlases)}] {atlas}", flush=True)
        atlas_subjects, roi_meta, matrices = build_atlas_feature_matrices(
            args.transdiag_root,
            atlas,
            subjects,
            atlas_root=args.atlas_root,
            target_shape=target_shape,
        )
        atlas_meta, atlas_diseases = load_metadata(
            args.diagnosis, atlas_subjects, args.min_cases
        )
        atlas_covariates = build_covariates(atlas_meta)
        for feature_index, (feature, matrix) in enumerate(matrices.items()):
            parts.append(
                _evaluate_feature_matrix(
                    modality="fmri",
                    source=f"{atlas}_multiatlas",
                    atlas=atlas,
                    feature=feature,
                    meta=atlas_meta,
                    diseases=atlas_diseases,
                    roi_meta=roi_meta,
                    matrix=matrix,
                    covariates=atlas_covariates,
                    cv_seeds=cv_seeds,
                    cv_folds=args.cv_folds,
                    seed=(
                        args.seed
                        + atlas_index * 1_000_003
                        + feature_index * 100_003
                    ),
                )
            )

    if args.include_smri:
        covariates = build_covariates(metadata)
        for source_index, source in enumerate(("aparc_dkt_aseg", "aseg")):
            print(f"[sMRI] {source}", flush=True)
            roi_meta, matrix = load_smri_source(args.smri_root, source, subjects)
            parts.append(
                _evaluate_feature_matrix(
                    modality="smri",
                    source=source,
                    atlas=source,
                    feature="normalized_volume_fraction",
                    meta=metadata,
                    diseases=diseases,
                    roi_meta=roi_meta,
                    matrix=matrix,
                    covariates=covariates,
                    cv_seeds=cv_seeds,
                    cv_folds=args.cv_folds,
                    seed=args.seed + (source_index + 100) * 100_003,
                )
            )

    statistics = pd.concat(parts, ignore_index=True)
    statistics = apply_validation_contract(
        statistics,
        family_q=args.family_q,
        effect_threshold=args.effect_threshold,
        auc_threshold=args.auc_threshold,
        stability_threshold=args.stability_threshold,
    )
    public_columns = [
        "disease",
        "modality",
        "atlas",
        "source",
        "roi_index",
        "roi_id",
        "roi_name",
        "anatomy",
        "anatomy_full",
        "hemisphere",
        "network",
        "structure_class",
        "feature_family",
        "feature",
    ]
    public = attach_candidate_ids(
        statistics[public_columns],
        task=TASK,
        identity_fields=IDENTITY_FIELDS,
    )
    print(f"[KG] scoring {len(public):,} candidates", flush=True)
    public, kg_audit = score_public_candidates(
        public,
        semantic_fields=SEMANTIC_FIELDS,
        case_study_id=TASK,
        kg_path=args.kg,
        seed=args.kg_seed,
    )
    statistics.insert(0, "candidate_id", public["candidate_id"].to_numpy())
    hidden_columns = [
        "candidate_id",
        "validated",
        "strict_validated",
        "statistically_validated",
        "execution_succeeded",
        "feedback_status",
        "feedback_available",
        "feedback_utility",
        "n_case",
        "n_control",
        "adjusted_beta_case_minus_control",
        "adjusted_beta_se",
        "adjusted_t",
        "p_value",
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "q_fdr_registered_family",
        "q_fdr_disease",
        "q_fdr_global",
        "cv_auc_mean",
        "cv_auc_sd",
        "cv_auc_p10",
        "cv_direction_stability",
        "cv_splits",
        "cv_stratification",
        "direction",
    ]
    internal = statistics[hidden_columns].copy()

    current_state = json.loads(args.current_state.read_text(encoding="utf-8"))
    provenance = {
        "scientific_contract": "biomarker-discovery-independent-tcp.v2",
        "discovery_dataset": "TCP",
        "external_validation_required": False,
        "candidate_space": "disease x atlas ROI x imaging feature",
        "subjects": len(subjects),
        "diseases": diseases,
        "atlases": atlases,
        "fmri_features": list(FULL_FMRI_FEATURES),
        "smri_sources": ["aparc_dkt_aseg", "aseg"] if args.include_smri else [],
        "transdiag_root": str(args.transdiag_root.resolve()),
        "diagnosis": str(args.diagnosis.resolve()),
        "diagnosis_sha256": sha256_file(args.diagnosis.resolve()),
        "smri_root": str(args.smri_root.resolve()) if args.include_smri else None,
        "atlas_root": str(args.atlas_root.resolve()),
        "data_inventory": _data_inventory(args.transdiag_root, atlases),
        "covariates": ["age", "sex", "site"],
        "cv": {
            "folds": args.cv_folds,
            "seeds": list(cv_seeds),
            "split_count_per_candidate": args.cv_folds * len(cv_seeds),
            "stratification": "diagnosis_x_site when every stratum supports all folds; diagnosis otherwise",
            "direction_selected_on_training_fold_only": True,
        },
        "validation_rule": {
            "fdr_family": "disease x atlas x feature_family",
            "family_q_max": args.family_q,
            "absolute_adjusted_cohen_d_min": args.effect_threshold,
            "heldout_directional_auc_min": args.auc_threshold,
            "heldout_direction_concordance_min": args.stability_threshold,
            "strict_secondary": "same rule plus global BH-FDR q<=family_q_max",
        },
        "current_state": current_state,
        "current_state_path": str(args.current_state.resolve()),
        "current_state_sha256": sha256_file(args.current_state.resolve()),
        "claims_path": str(args.claims.resolve()),
        "claims_sha256": sha256_file(args.claims.resolve()),
        "outcomes_used_for_kg_scoring": False,
        "retired_inputs_reused": False,
        "retired_case1_candidate_ids_reused": False,
        "retired_top_fraction_gt_reused": False,
    }
    manifest = export_table_bundle(
        task=TASK,
        public=public,
        internal=internal,
        external=None,
        factor_fields=IDENTITY_FIELDS,
        output_dir=args.output_dir,
        provenance=provenance,
        kg_audit=kg_audit,
    )
    summary = (
        statistics.groupby(["modality", "atlas", "feature_family"], dropna=False)
        .agg(
            candidates=("candidate_id", "size"),
            statistical_hits=("statistically_validated", "sum"),
            validated_hits=("validated", "sum"),
            strict_hits=("strict_validated", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    completion = {
        "schema_version": "biomarker-discovery-table-build.v2",
        "status": "complete",
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table_manifest": manifest,
        "counts": {
            "candidates": len(statistics),
            "statistically_validated": int(
                statistics["statistically_validated"].sum()
            ),
            "validated": int(statistics["validated"].sum()),
            "strict_validated": int(statistics["strict_validated"].sum()),
        },
        "summary": str((args.output_dir / "validation_summary.csv").resolve()),
    }
    completion_path = args.output_dir / "build_manifest.json"
    completion_path.write_text(
        json.dumps(completion, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transdiag-root", type=Path, default=DEFAULT_TRANSDIAG_ROOT)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_DIAGNOSIS)
    parser.add_argument("--smri-root", type=Path, default=DEFAULT_SMRI_ROOT)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atlas-pattern", default="")
    parser.add_argument("--max-atlases", type=int, default=0)
    parser.add_argument("--min-cases", type=int, default=5)
    parser.add_argument("--include-smri", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--cv-seeds",
        default=",".join(str(seed) for seed in DEFAULT_CV_SEEDS),
    )
    parser.add_argument("--family-q", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=0.15)
    parser.add_argument("--auc-threshold", type=float, default=0.55)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--kg-seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if not 0.0 < args.family_q < 1.0:
        parser.error("--family-q must be in (0, 1)")
    if not 0.5 <= args.auc_threshold <= 1.0:
        parser.error("--auc-threshold must be in [0.5, 1]")
    if not 0.0 <= args.stability_threshold <= 1.0:
        parser.error("--stability-threshold must be in [0, 1]")
    return args


def main() -> int:
    manifest = build_tables(parse_args())
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
