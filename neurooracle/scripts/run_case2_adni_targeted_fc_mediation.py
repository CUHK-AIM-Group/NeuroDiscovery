"""Run exact ADNI FC mediation tests for KG-prioritised Case Study 2 claims.

This analysis extracts pairwise resting-state functional connectivity among
bilateral amygdala, hippocampus, and anterior cingulate cortex from the
Harvard-Oxford merged atlas. It then screens every prespecified pathway PRS
and available ADNI outcome, before attaching the outcome-blind KG ranking.
The models are cross-sectional association tests and have no causal claim.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from statsmodels.stats.multitest import multipletests

from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import (
    CASE2_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_PROXY_MAPPING_ROOT,
    DEFAULT_PATHWAY_DIR,
    evaluate_prioritisation,
    fixed_threshold_exposures,
)
from neurooracle.scripts.run_case2_adni_mediation_smoke import (
    CATEGORICAL_COVARIATES,
    DEFAULT_DATASET_ROOT,
    NUMERIC_COVARIATES,
    OUTCOME_HIGHER_IS,
    build_covariate_matrix,
    complete_case_mask,
    load_pathway_exposures,
    mediation_screen,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "case1_adni_validation"
    / "fmri_multiatlas_full"
    / "adni_fmri_first_testable_runs.csv"
)
DEFAULT_ATLAS = (
    REPO_ROOT.parent
    / "NeuroSTORM"
    / "datasets"
    / "atlas"
    / "harvard_oxford_merged"
    / "atlas.nii.gz"
)
DEFAULT_PATHWAY_PRS = (
    DEFAULT_PATHWAY_DIR / "case2_adni_pathway_prs_wide.parquet"
)
DEFAULT_KG_MAPPING = (
    DEFAULT_PROXY_MAPPING_ROOT / "kg_ranked_mediation_results.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_targeted_fc_mediation_v1"
)
DEFAULT_OUTCOMES = ("diagnosis_code", "ADAS13", "MMSE", "CDRSB")
TARGET_LABELS = {
    "left_hippocampus": 17,
    "left_amygdala": 18,
    "right_hippocampus": 53,
    "right_amygdala": 54,
    "right_anterior_cingulate": 2901,
    "left_anterior_cingulate": 2902,
}
TARGET_FEATURES = (
    "triad_mean_fc_fisher_z",
    "amygdala_hippocampus_fc_fisher_z",
    "amygdala_acc_fc_fisher_z",
    "hippocampus_acc_fc_fisher_z",
)
FEATURE_SPECIFICITY = {
    "triad_mean_fc_fisher_z": 1.00,
    "amygdala_hippocampus_fc_fisher_z": 0.85,
    "amygdala_acc_fc_fisher_z": 0.85,
    "hippocampus_acc_fc_fisher_z": 0.85,
}
FEATURE_DESCRIPTION = {
    "triad_mean_fc_fisher_z": (
        "Mean Fisher-z FC across all amygdala-hippocampus, amygdala-ACC, "
        "and hippocampus-ACC edges"
    ),
    "amygdala_hippocampus_fc_fisher_z": (
        "Mean Fisher-z FC across bilateral amygdala-hippocampus edges"
    ),
    "amygdala_acc_fc_fisher_z": (
        "Mean Fisher-z FC across bilateral amygdala-anterior cingulate edges"
    ),
    "hippocampus_acc_fc_fisher_z": (
        "Mean Fisher-z FC across bilateral hippocampus-anterior cingulate edges"
    ),
}
CONFOUND_CANDIDATES = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "white_matter",
    "csf",
    "global_signal",
)


def translate_nas_path(value: object) -> Path:
    """Translate the historical Z: mount to the currently available UNC root."""

    text = str(value)
    if text.lower().startswith("z:\\"):
        text = (
            "\\\\192.168.3.61\\data\\"
            + text[3:].lstrip("\\/")
        )
    return Path(text)


def build_target_atlas(source_path: Path, output_path: Path) -> dict[str, int]:
    """Create a compact six-label atlas while preserving source geometry."""

    image = nib.load(str(source_path))
    source = np.asanyarray(image.dataobj)
    target = np.zeros(source.shape, dtype=np.int16)
    voxel_counts: dict[str, int] = {}
    for new_label, (name, source_label) in enumerate(
        TARGET_LABELS.items(),
        start=1,
    ):
        mask = source == source_label
        count = int(mask.sum())
        if count == 0:
            raise ValueError(
                f"Atlas label {source_label} ({name}) contains no voxels"
            )
        target[mask] = new_label
        voxel_counts[name] = count
    header = image.header.copy()
    header.set_data_dtype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(target, image.affine, header),
        str(output_path),
    )
    return voxel_counts


def _mean_cross_group(
    fisher_z: np.ndarray,
    group_a: Iterable[int],
    group_b: Iterable[int],
) -> float:
    values = [fisher_z[a, b] for a in group_a for b in group_b]
    return float(np.mean(values))


def compute_target_fc_features(timeseries: np.ndarray) -> dict[str, float]:
    """Compute four prespecified signed pairwise-FC summaries."""

    timeseries = np.asarray(timeseries, dtype=float)
    if timeseries.ndim != 2 or timeseries.shape[1] != len(TARGET_LABELS):
        raise ValueError(
            "Expected time-by-six target ROI matrix, got "
            f"{timeseries.shape}"
        )
    if not np.isfinite(timeseries).all():
        raise ValueError("Target ROI time series contains nonfinite values")
    correlation = np.corrcoef(timeseries, rowvar=False)
    correlation = np.clip(correlation, -0.999999, 0.999999)
    fisher_z = np.arctanh(correlation)
    np.fill_diagonal(fisher_z, np.nan)

    hippocampus = (0, 2)
    amygdala = (1, 3)
    acc = (4, 5)
    amygdala_hippocampus = _mean_cross_group(
        fisher_z,
        amygdala,
        hippocampus,
    )
    amygdala_acc = _mean_cross_group(fisher_z, amygdala, acc)
    hippocampus_acc = _mean_cross_group(fisher_z, hippocampus, acc)
    return {
        "triad_mean_fc_fisher_z": float(
            np.mean(
                [
                    amygdala_hippocampus,
                    amygdala_acc,
                    hippocampus_acc,
                ]
            )
        ),
        "amygdala_hippocampus_fc_fisher_z": amygdala_hippocampus,
        "amygdala_acc_fc_fisher_z": amygdala_acc,
        "hippocampus_acc_fc_fisher_z": hippocampus_acc,
    }


def _load_confounds(path: Path, n_timepoints: int) -> tuple[np.ndarray | None, float]:
    if not path.is_file():
        return None, float("nan")
    confounds = pd.read_csv(path, sep="\t")
    framewise_displacement = pd.to_numeric(
        confounds.get("framewise_displacement"),
        errors="coerce",
    ).fillna(0.0)
    mean_fd = float(framewise_displacement.mean())
    columns = [
        column for column in CONFOUND_CANDIDATES if column in confounds
    ]
    if not columns:
        return None, mean_fd
    matrix = (
        confounds.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return matrix[:n_timepoints], mean_fd


def _extract_subject(
    row: dict[str, object],
    target_atlas: str,
) -> dict[str, object]:
    subject_id = str(row["PTID"])
    bold_path = translate_nas_path(row["bold_path"])
    confounds_path = translate_nas_path(row["confounds_path"])
    if not bold_path.is_file():
        raise FileNotFoundError(bold_path)

    image = nib.load(str(bold_path))
    n_timepoints = int(image.shape[-1])
    zooms = image.header.get_zooms()
    repetition_time = (
        float(zooms[3])
        if len(zooms) >= 4 and float(zooms[3]) > 0
        else 3.0
    )
    confounds, mean_fd = _load_confounds(confounds_path, n_timepoints)
    masker = NiftiLabelsMasker(
        labels_img=target_atlas,
        standardize=False,
        detrend=True,
        low_pass=0.08,
        high_pass=0.01,
        t_r=repetition_time,
        memory=None,
        verbose=0,
    )
    timeseries = masker.fit_transform(str(bold_path), confounds=confounds)
    if timeseries.shape[1] != len(TARGET_LABELS):
        raise ValueError(
            f"{subject_id}: extracted {timeseries.shape[1]} target ROIs"
        )
    return {
        "subject_id": subject_id,
        "mean_fd_recomputed": mean_fd,
        "n_timepoints": n_timepoints,
        "repetition_time": repetition_time,
        **compute_target_fc_features(timeseries),
    }


def extract_target_features(
    runs: pd.DataFrame,
    target_atlas: Path,
    *,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract target FC in parallel and return successful rows plus errors."""

    tasks = runs.to_dict("records")
    successful: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_subject, row, str(target_atlas)): row
            for row in tasks
        }
        for future in as_completed(futures):
            row = futures[future]
            completed += 1
            try:
                successful.append(future.result())
            except Exception as exc:  # pragma: no cover - real-data audit path
                errors.append(
                    {
                        "subject_id": str(row.get("PTID", "")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"[target FC] {completed}/{len(tasks)} complete; "
                    f"errors={len(errors)}",
                    flush=True,
                )
    features = pd.DataFrame(successful).sort_values(
        "subject_id",
        kind="stable",
    )
    return features.reset_index(drop=True), pd.DataFrame(errors)


def run_targeted_screen(
    subjects: pd.DataFrame,
    target_features: pd.DataFrame,
    pathway_prs_path: Path,
    score_manifest: pd.DataFrame,
    *,
    outcomes: Iterable[str],
    fd_max: float,
    min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete fixed-threshold pathway/outcome target-FC screen."""

    pathway_scores, exposure_metadata = load_pathway_exposures(
        pathway_prs_path
    )
    exposures = fixed_threshold_exposures(score_manifest)
    pathway_scores = pathway_scores.set_index("subject_id")
    subjects = subjects.set_index("subject_id", drop=False)
    subjects = subjects.join(
        pathway_scores.loc[:, list(exposures)],
        how="left",
        validate="one_to_one",
    )
    target_features = target_features.set_index("subject_id")
    common_ids = subjects.index.intersection(
        target_features.index,
        sort=False,
    )
    subjects = subjects.loc[common_ids].copy()
    target_features = target_features.loc[common_ids]

    all_results: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    feature_metadata = pd.DataFrame(
        {
            "marker_id": TARGET_FEATURES,
            "feature": TARGET_FEATURES,
            "feature_description": [
                FEATURE_DESCRIPTION[feature] for feature in TARGET_FEATURES
            ],
            "atlas": "harvard_oxford_merged",
            "regions": "amygdala;hippocampus;anterior cingulate cortex",
            "measure": "signed Pearson correlation transformed to Fisher z",
        }
    )

    for exposure in exposures:
        for outcome in outcomes:
            family_mask = complete_case_mask(
                subjects,
                exposure,
                outcome,
                fd_max=fd_max,
            )
            family_subjects = subjects.loc[family_mask]
            if len(family_subjects) < min_n:
                summaries.append(
                    {
                        "exposure": exposure,
                        "outcome": outcome,
                        "n_complete": int(len(family_subjects)),
                        "status": "skipped_below_min_n",
                    }
                )
                continue
            covariates, covariate_names = build_covariate_matrix(
                family_subjects
            )
            result, marker_mask = mediation_screen(
                pd.to_numeric(
                    family_subjects[exposure],
                    errors="coerce",
                ).to_numpy(float),
                pd.to_numeric(
                    family_subjects[outcome],
                    errors="coerce",
                ).to_numpy(float),
                target_features.loc[
                    family_subjects.index,
                    list(TARGET_FEATURES),
                ].to_numpy(float),
                covariates,
            )
            metadata = feature_metadata.loc[marker_mask].reset_index(drop=True)
            result = pd.concat([metadata, result], axis=1)
            result.insert(0, "outcome", outcome)
            result.insert(0, "exposure", exposure)
            result["outcome_higher_is"] = OUTCOME_HIGHER_IS[outcome]
            result["abs_indirect_effect_std"] = result[
                "indirect_effect_std"
            ].abs()
            result["fdr_significant"] = result["sobel_q_family"].lt(0.05)
            result["covariates"] = ";".join(covariate_names)
            for key, value in exposure_metadata.get(exposure, {}).items():
                result[f"exposure_{key}"] = value
            all_results.append(result)
            summaries.append(
                {
                    "exposure": exposure,
                    "outcome": outcome,
                    "n_complete": int(len(family_subjects)),
                    "markers_tested": int(len(result)),
                    "minimum_sobel_p": float(result["sobel_p"].min()),
                    "family_fdr_hits": int(
                        result["sobel_q_family"].lt(0.05).sum()
                    ),
                    "status": "completed",
                }
            )

    combined = pd.concat(all_results, ignore_index=True)
    combined["sobel_q_global"] = np.nan
    finite = combined["sobel_p"].notna()
    combined.loc[finite, "sobel_q_global"] = multipletests(
        combined.loc[finite, "sobel_p"],
        method="fdr_bh",
    )[1]
    return combined, pd.DataFrame(summaries)


def _stable_unique(values: Iterable[object]) -> str:
    return ";".join(dict.fromkeys(str(value) for value in values if str(value)))


def build_exact_kg_ranking(
    kg_mapping: pd.DataFrame,
    targeted_results: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact target-FC tests to the pre-result KG mapping."""

    family_mapping = (
        kg_mapping.sort_values(
            ["mapping_score", "kg_rank"],
            ascending=[False, True],
            kind="stable",
        )
        .groupby(["exposure", "outcome"], sort=False)
        .agg(
            base_mapping_score=("mapping_score", "max"),
            primary_hypothesis_id=("primary_hypothesis_id", "first"),
            pathway_id=("pathway_id", "first"),
            pathway_name=("pathway_name", "first"),
            supporting_hypothesis_ids=(
                "supporting_hypothesis_ids",
                _stable_unique,
            ),
            supporting_genes=("supporting_genes", _stable_unique),
        )
        .reset_index()
    )
    candidates = targeted_results.merge(
        family_mapping,
        on=["exposure", "outcome"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_kg"),
    )
    candidates["feature_specificity"] = candidates["feature"].map(
        FEATURE_SPECIFICITY
    )
    candidates["mapping_score"] = (
        0.90 * candidates["base_mapping_score"]
        + 0.10 * candidates["feature_specificity"]
    )
    candidates["feature_priority"] = candidates["feature"].map(
        {
            feature: index
            for index, feature in enumerate(TARGET_FEATURES)
        }
    )

    exposure_order = (
        candidates.groupby("exposure")["mapping_score"]
        .max()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    queues = {
        exposure: candidates[candidates["exposure"].eq(exposure)]
        .sort_values(
            ["feature_priority", "mapping_score"],
            ascending=[True, False],
            kind="stable",
        )
        .index.tolist()
        for exposure in exposure_order
    }
    ranked_indices: list[int] = []
    for feature_round in range(len(TARGET_FEATURES)):
        for exposure in exposure_order:
            queue = queues[exposure]
            if feature_round < len(queue):
                ranked_indices.append(queue[feature_round])
    ranked = candidates.loc[ranked_indices].reset_index(drop=True)
    ranked.insert(0, "kg_rank", np.arange(1, len(ranked) + 1))
    return ranked


def run(args: argparse.Namespace) -> dict[str, object]:
    required = (
        args.run_manifest,
        args.dataset_root / "case2_adni_subject_table.parquet",
        args.atlas,
        args.pathway_prs,
        args.pathway_dir / "score_manifest.csv",
        args.kg_mapping,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. Use --force."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    subject_path = args.dataset_root / "case2_adni_subject_table.parquet"
    subjects = pd.read_parquet(subject_path)
    runs = pd.read_csv(args.run_manifest)
    runs = runs[runs["PTID"].isin(subjects["subject_id"])].copy()
    runs = runs.drop_duplicates("PTID", keep="first")
    if len(runs) != len(subjects):
        missing = sorted(set(subjects["subject_id"]) - set(runs["PTID"]))
        raise ValueError(
            f"Run manifest covers {len(runs)}/{len(subjects)} subjects; "
            f"missing={missing[:10]}"
        )

    target_atlas = args.output_root / "harvard_oxford_target6.nii.gz"
    voxel_counts = build_target_atlas(args.atlas, target_atlas)
    feature_cache = args.output_root / "target_fc_features.parquet"
    error_path = args.output_root / "target_fc_extraction_errors.csv"
    if feature_cache.is_file() and not args.force_extract:
        target_features = pd.read_parquet(feature_cache)
        extraction_errors = (
            pd.read_csv(error_path)
            if error_path.is_file()
            else pd.DataFrame()
        )
    else:
        target_features, extraction_errors = extract_target_features(
            runs,
            target_atlas,
            workers=args.workers,
        )
        target_features.to_parquet(
            feature_cache,
            index=False,
            compression="zstd",
        )
        extraction_errors.to_csv(error_path, index=False)
    if len(target_features) < args.min_n:
        raise RuntimeError(
            f"Only {len(target_features)} subjects have target FC features"
        )

    score_manifest = pd.read_csv(args.pathway_dir / "score_manifest.csv")
    targeted_results, family_summary = run_targeted_screen(
        subjects,
        target_features,
        args.pathway_prs,
        score_manifest,
        outcomes=args.outcomes,
        fd_max=args.fd_max,
        min_n=args.min_n,
    )
    targeted_results.to_parquet(
        args.output_root / "all_targeted_mediation_results.parquet",
        index=False,
        compression="zstd",
    )
    targeted_results.to_csv(
        args.output_root / "all_targeted_mediation_results.csv",
        index=False,
    )
    family_summary.to_csv(
        args.output_root / "family_summary.csv",
        index=False,
    )

    kg_mapping = pd.read_parquet(args.kg_mapping)
    ranked = build_exact_kg_ranking(kg_mapping, targeted_results)
    ranked, metrics, random_distribution = evaluate_prioritisation(
        ranked,
        targeted_results,
        score_manifest,
        k_values=args.k,
        nominal_alpha=args.nominal_alpha,
        minimum_effect=args.minimum_effect,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    ranked.to_parquet(
        args.output_root / "kg_ranked_exact_fc_results.parquet",
        index=False,
        compression="zstd",
    )
    ranked.to_csv(
        args.output_root / "kg_ranked_exact_fc_results.csv",
        index=False,
    )
    metrics.to_csv(
        args.output_root / "prioritisation_by_k.csv",
        index=False,
    )
    random_distribution.to_parquet(
        args.output_root / "random_distribution.parquet",
        index=False,
        compression="zstd",
    )

    now_hkt = datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload: dict[str, Any] = {
        "created_at_hkt": now_hkt,
        "analysis_type": (
            "exact target-region resting-state FC association-based "
            "mediation screen"
        ),
        "causal_interpretation": False,
        "atlas": "Harvard-Oxford merged target labels",
        "source_atlas": str(args.atlas),
        "target_labels": TARGET_LABELS,
        "target_label_voxel_counts": voxel_counts,
        "subjects_requested": int(len(runs)),
        "subjects_extracted": int(len(target_features)),
        "extraction_errors": int(len(extraction_errors)),
        "fixed_pathway_exposures": int(
            len(fixed_threshold_exposures(score_manifest))
        ),
        "outcomes": list(args.outcomes),
        "target_fc_features": list(TARGET_FEATURES),
        "complete_targeted_tests": int(len(targeted_results)),
        "family_fdr_hits": int(
            targeted_results["sobel_q_family"].lt(0.05).sum()
        ),
        "global_fdr_hits": int(
            targeted_results["sobel_q_global"].lt(0.05).sum()
        ),
        "kg_candidate_tests": int(len(ranked)),
        "kg_source_genes": sorted(
            {
                gene
                for value in ranked["supporting_genes"]
                for gene in str(value).split(";")
                if gene
            }
        ),
        "outcome_blind_design": (
            "Target regions and four FC summaries were fixed from KG claim "
            "semantics before mediation results were computed. One PRS "
            "threshold per pathway was fixed independently of outcomes."
        ),
        "covariates": {
            "numeric": list(NUMERIC_COVARIATES),
            "categorical": list(CATEGORICAL_COVARIATES),
        },
        "exploratory_signal_definition": {
            "sobel_p_lt": args.nominal_alpha,
            "abs_indirect_effect_std_gte": args.minimum_effect,
        },
        "random_resamples": args.resamples,
        "random_seed": args.seed,
        "outputs": {
            "target_atlas": str(target_atlas),
            "target_fc_features": str(feature_cache),
            "extraction_errors": str(error_path),
            "all_targeted_results": str(
                args.output_root / "all_targeted_mediation_results.parquet"
            ),
            "kg_ranked_results": str(
                args.output_root / "kg_ranked_exact_fc_results.parquet"
            ),
            "prioritisation_by_k": str(
                args.output_root / "prioritisation_by_k.csv"
            ),
            "random_distribution": str(
                args.output_root / "random_distribution.parquet"
            ),
            "family_summary": str(
                args.output_root / "family_summary.csv"
            ),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=DEFAULT_RUN_MANIFEST,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument(
        "--pathway-prs",
        type=Path,
        default=DEFAULT_PATHWAY_PRS,
    )
    parser.add_argument(
        "--pathway-dir",
        type=Path,
        default=DEFAULT_PATHWAY_DIR,
    )
    parser.add_argument(
        "--kg-mapping",
        type=Path,
        default=DEFAULT_KG_MAPPING,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--outcome",
        dest="outcomes",
        action="append",
        default=None,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fd-max", type=float, default=0.5)
    parser.add_argument("--min-n", type=int, default=80)
    parser.add_argument("--nominal-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument("--k", type=int, action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    args.outcomes = tuple(args.outcomes or DEFAULT_OUTCOMES)
    args.k = tuple(sorted(set(args.k or (5, 10, 20, 28))))
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Last Updated At: 2026-07-31 03:35 HKT
