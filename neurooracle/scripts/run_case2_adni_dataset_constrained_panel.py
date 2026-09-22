"""Run a dataset-constrained ADNI validation panel for Case Study 2.

The candidate rules are fixed from Case Study 2 claims that can be represented
by the available ADNI genotype, resting-state fMRI, and cognitive tables. The
script first evaluates the complete direct-variant imaging universe and only
then attaches an outcome-blind KG ranking. Statistical results never enter the
candidate selection or ranking.

All models are cross-sectional association-based indirect-effect tests. They
do not establish causal mediation.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from neurooracle.scripts.build_case2_adni_experiment_table import (
    normalize_subject_id,
)
from neurooracle.scripts.run_case2_adni_mediation_smoke import (
    DEFAULT_DATASET_ROOT,
    MARKER_METADATA_COLUMNS,
    NUMERIC_COVARIATES,
    OUTCOME_HIGHER_IS,
    build_covariate_matrix,
    complete_case_mask,
    load_imaging_matrix,
    mediation_screen,
)


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_PGEN_PREFIX = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "merged"
    / "case2_adni_common_dr2_0p8_autosomes"
)
DEFAULT_PLINK2 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references\tools"
) / "plink2_20260504" / "plink2.exe"
DEFAULT_OUTPUT_ROOT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_dataset_constrained_panel_v1"
)
DEFAULT_OUTCOMES = ("diagnosis_code", "ADAS13", "MMSE", "CDRSB")
DEFAULT_K_VALUES = (10, 20, 50, 100, 200, 500, 1000)
CONNECTIVITY_FEATURES = (
    "corr_mean",
    "corr_mean_abs",
    "partial_mean",
    "partial_mean_abs",
)
TARGET_VARIANTS = {
    "rs3865444": "cd33_rs3865444_dosage",
    "rs4680": "comt_rs4680_dosage",
}
EXPOSURE_METADATA = {
    "apoe_e4_dosage": {
        "gene": "APOE",
        "source": "direct APOE epsilon4 dosage reconstructed in the subject table",
    },
    "cd33_rs3865444_dosage": {
        "gene": "CD33",
        "source": "post-imputation rs3865444 additive dosage",
    },
    "comt_rs4680_dosage": {
        "gene": "COMT",
        "source": "post-imputation rs4680 additive dosage",
    },
}


@dataclass(frozen=True)
class CandidateRule:
    """Pre-result mapping from one KG claim family to executable markers."""

    rule_id: str
    claim_ids: tuple[str, ...]
    exposure: str
    gene: str
    marker_kind: str
    outcomes: tuple[str, ...]
    claim_confidence: float
    chain_scope: str
    interpretation: str


CANDIDATE_RULES = (
    CandidateRule(
        rule_id="apoe_hippocampal_connectivity",
        claim_ids=("CLM:2c00385a4908334c",),
        exposure="apoe_e4_dosage",
        gene="APOE",
        marker_kind="hippocampal_connectivity",
        outcomes=("MMSE", "ADAS13"),
        claim_confidence=0.90,
        chain_scope="full_chain_proxy",
        interpretation=(
            "APOE allele status -> intrinsic hippocampal connectivity -> cognition"
        ),
    ),
    CandidateRule(
        rule_id="apoe_global_fcd_decline",
        claim_ids=(
            "CLM:005c57b796e8e72c",
            "CLM:20d367d9b7730656",
        ),
        exposure="apoe_e4_dosage",
        gene="APOE",
        marker_kind="global_fcd",
        outcomes=("CDRSB", "ADAS13", "MMSE", "diagnosis_code"),
        claim_confidence=0.90,
        chain_scope="full_chain_proxy",
        interpretation=(
            "APOE epsilon4 -> functional connectivity density -> cognitive decline"
        ),
    ),
    CandidateRule(
        rule_id="apoe_insula_connectivity",
        claim_ids=("CLM:3a8182c7dec36451",),
        exposure="apoe_e4_dosage",
        gene="APOE",
        marker_kind="insula_connectivity",
        outcomes=("MMSE", "ADAS13", "CDRSB"),
        claim_confidence=0.84,
        chain_scope="full_chain_proxy",
        interpretation="APOE genotype -> insula connectivity -> cognition",
    ),
    CandidateRule(
        rule_id="apoe_large_scale_networks",
        claim_ids=("CLM:f2665c6f3e783c50",),
        exposure="apoe_e4_dosage",
        gene="APOE",
        marker_kind="large_scale_network_connectivity",
        outcomes=("MMSE", "ADAS13"),
        claim_confidence=0.84,
        chain_scope="full_chain_proxy",
        interpretation=(
            "APOE epsilon4 -> DMN/ECN/salience connectivity -> cognition"
        ),
    ),
    CandidateRule(
        rule_id="cd33_global_fcd_cognition",
        claim_ids=("CLM:5241a2ca586e7610",),
        exposure="cd33_rs3865444_dosage",
        gene="CD33",
        marker_kind="global_fcd",
        outcomes=("MMSE", "ADAS13", "CDRSB", "diagnosis_code"),
        claim_confidence=0.84,
        chain_scope="full_chain_proxy",
        interpretation="CD33 rs3865444 -> global FCD -> cognition",
    ),
    CandidateRule(
        rule_id="comt_dmn_imaging_control",
        claim_ids=("CLM:53bb33de7ddd", "CLM:337183362368"),
        exposure="comt_rs4680_dosage",
        gene="COMT",
        marker_kind="dmn_connectivity",
        outcomes=("MMSE", "ADAS13"),
        claim_confidence=0.50,
        chain_scope="gene_imaging_positive_control",
        interpretation=(
            "COMT rs4680 -> DMN connectivity; cognition is an ADNI transfer outcome"
        ),
    ),
)


def _run_command(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def parse_variant_dosages(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a PLINK2 additive export and retain target dosage metadata."""

    raw = pd.read_csv(raw_path, sep=r"\s+")
    if "IID" not in raw:
        raise ValueError(f"PLINK export has no IID column: {raw_path}")
    result = pd.DataFrame(
        {"subject_id": raw["IID"].map(normalize_subject_id)}
    )
    metadata_rows: list[dict[str, object]] = []
    for variant_id, output_column in TARGET_VARIANTS.items():
        matches = [
            str(column)
            for column in raw.columns
            if str(column).startswith(f"{variant_id}_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one dosage column for {variant_id}; found {matches}"
            )
        source_column = matches[0]
        effect_allele = source_column.removeprefix(f"{variant_id}_")
        result[output_column] = pd.to_numeric(
            raw[source_column],
            errors="coerce",
        )
        metadata_rows.append(
            {
                "variant_id": variant_id,
                "exposure": output_column,
                "plink_source_column": source_column,
                "counted_allele": effect_allele,
                "coding": f"additive count of {effect_allele} allele",
            }
        )
    if result["subject_id"].eq("").any():
        raise ValueError("At least one PLINK IID could not be normalized")
    if result["subject_id"].duplicated().any():
        raise ValueError("PLINK export contains duplicate normalized subject IDs")
    return result, pd.DataFrame(metadata_rows)


def export_variant_dosages(
    *,
    plink2: Path,
    pgen_prefix: Path,
    output_root: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export the fixed CD33 and COMT variants from the post-imputation PGEN."""

    output_prefix = output_root / "target_variant_additive"
    raw_path = output_prefix.with_suffix(".raw")
    if force or not raw_path.is_file():
        _run_command(
            [
                str(plink2),
                "--pfile",
                str(pgen_prefix),
                "--snps",
                *TARGET_VARIANTS,
                "--export",
                "A",
                "--out",
                str(output_prefix),
            ],
            output_root / "target_variant_export.log",
        )
    dosages, metadata = parse_variant_dosages(raw_path)
    dosages.to_parquet(
        output_root / "target_variant_dosages.parquet",
        index=False,
        compression="zstd",
    )
    metadata.to_csv(output_root / "target_variant_metadata.csv", index=False)
    return dosages, metadata


def _fdr(values: pd.Series) -> np.ndarray:
    adjusted = np.full(len(values), np.nan)
    finite = np.isfinite(pd.to_numeric(values, errors="coerce"))
    if finite.any():
        adjusted[finite] = multipletests(
            pd.to_numeric(values[finite], errors="coerce"),
            method="fdr_bh",
        )[1]
    return adjusted


def run_direct_variant_universe(
    subjects: pd.DataFrame,
    imaging_dir: Path,
    *,
    outcomes: Iterable[str],
    fd_max: float,
    min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all direct-variant-by-marker tests across every atlas."""

    subjects = subjects.set_index("subject_id", drop=False)
    exposures = tuple(EXPOSURE_METADATA)
    all_results: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    atlas_paths = sorted(imaging_dir.glob("*.parquet"))
    if not atlas_paths:
        raise FileNotFoundError(f"No imaging feature tables under {imaging_dir}")

    for atlas_index, imaging_path in enumerate(atlas_paths, start=1):
        imaging_matrix, marker_metadata = load_imaging_matrix(imaging_path)
        common_ids = subjects.index.intersection(imaging_matrix.index, sort=False)
        atlas_subjects = subjects.loc[common_ids]
        imaging_matrix = imaging_matrix.loc[common_ids]
        print(
            f"[direct panel] atlas {atlas_index}/{len(atlas_paths)} "
            f"{imaging_path.stem}: {imaging_matrix.shape[1]} markers",
            flush=True,
        )
        for exposure in exposures:
            for outcome in outcomes:
                family_mask = complete_case_mask(
                    atlas_subjects,
                    exposure,
                    outcome,
                    fd_max=fd_max,
                )
                family_subjects = atlas_subjects.loc[family_mask]
                if len(family_subjects) < min_n:
                    summaries.append(
                        {
                            "atlas": imaging_path.stem,
                            "exposure": exposure,
                            "outcome": outcome,
                            "n_complete": int(len(family_subjects)),
                            "markers_available": int(imaging_matrix.shape[1]),
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
                    imaging_matrix.loc[family_subjects.index].to_numpy(float),
                    covariates,
                )
                metadata = marker_metadata.loc[marker_mask].reset_index(drop=True)
                result = pd.concat([metadata, result], axis=1)
                result.insert(0, "outcome", outcome)
                result.insert(0, "exposure", exposure)
                result["gene"] = EXPOSURE_METADATA[exposure]["gene"]
                result["outcome_higher_is"] = OUTCOME_HIGHER_IS[outcome]
                result["abs_a_path_std"] = result["a_path_std"].abs()
                result["abs_indirect_effect_std"] = result[
                    "indirect_effect_std"
                ].abs()
                result["a_path_q_family"] = _fdr(result["a_path_p"])
                result["covariates"] = ";".join(covariate_names)
                all_results.append(result)
                summaries.append(
                    {
                        "atlas": imaging_path.stem,
                        "exposure": exposure,
                        "outcome": outcome,
                        "n_complete": int(len(family_subjects)),
                        "markers_available": int(imaging_matrix.shape[1]),
                        "markers_tested": int(len(result)),
                        "minimum_a_path_p": float(result["a_path_p"].min()),
                        "minimum_sobel_p": float(result["sobel_p"].min()),
                        "status": "completed",
                    }
                )

    combined = pd.concat(all_results, ignore_index=True)
    combined["sobel_q_exposure_outcome_all_atlases"] = np.nan
    combined["a_path_q_exposure_outcome_all_atlases"] = np.nan
    for _, index in combined.groupby(
        ["exposure", "outcome"],
        sort=False,
    ).groups.items():
        combined.loc[index, "sobel_q_exposure_outcome_all_atlases"] = _fdr(
            combined.loc[index, "sobel_p"]
        )
        combined.loc[index, "a_path_q_exposure_outcome_all_atlases"] = _fdr(
            combined.loc[index, "a_path_p"]
        )
    combined["sobel_q_global"] = _fdr(combined["sobel_p"])
    combined["a_path_q_global"] = _fdr(combined["a_path_p"])
    return combined, pd.DataFrame(summaries)


def _rule_universe_mask(
    results: pd.DataFrame,
    rule: CandidateRule,
    *,
    require_target_region: bool,
) -> pd.Series:
    atlas = results["atlas"].fillna("").astype(str).str.lower()
    roi = (
        results["roi_name"].fillna("").astype(str)
        + " "
        + results["parcel_name"].fillna("").astype(str)
    ).str.lower()
    network = results["network"].fillna("").astype(str)
    feature = results["feature"].fillna("").astype(str)
    mask = results["exposure"].eq(rule.exposure)
    mask &= results["outcome"].isin(rule.outcomes)

    if rule.marker_kind == "global_fcd":
        mask &= atlas.isin(
            (
                "schaefer_100_7net",
                "schaefer_200_7net",
                "schaefer_400_7net",
            )
        )
        mask &= feature.eq("corr_node_degree_abs_top10")
    elif rule.marker_kind == "hippocampal_connectivity":
        mask &= atlas.isin(("harvard_oxford_merged", "eickhoff_zilles"))
        mask &= feature.isin(CONNECTIVITY_FEATURES)
        if require_target_region:
            mask &= roi.str.contains("hippocampus", regex=False)
            mask &= ~roi.str.contains("parahippocamp", regex=False)
    elif rule.marker_kind == "insula_connectivity":
        mask &= atlas.isin(
            (
                "harvard_oxford_merged",
                "eickhoff_zilles",
                "talairach_tournoux",
            )
        )
        mask &= feature.isin(CONNECTIVITY_FEATURES)
        if require_target_region:
            mask &= roi.str.contains(r"insula|insular", regex=True)
    elif rule.marker_kind == "large_scale_network_connectivity":
        mask &= atlas.isin(
            (
                "msdl_39",
                "schaefer_100_7net",
                "schaefer_200_7net",
                "schaefer_400_7net",
            )
        )
        mask &= feature.isin(CONNECTIVITY_FEATURES)
        if require_target_region:
            mask &= network.isin(
                ("DMN", "Default", "Cont", "SalVentAttn")
            )
    elif rule.marker_kind == "dmn_connectivity":
        mask &= atlas.isin(
            (
                "msdl_39",
                "schaefer_100_7net",
                "schaefer_200_7net",
                "schaefer_400_7net",
            )
        )
        mask &= feature.isin(CONNECTIVITY_FEATURES)
        if require_target_region:
            mask &= network.isin(("DMN", "Default"))
    else:  # pragma: no cover - guarded by fixed rule registry
        raise ValueError(f"Unknown marker kind: {rule.marker_kind}")
    return mask


def _candidate_prior(
    candidates: pd.DataFrame,
    rule: CandidateRule,
) -> pd.Series:
    atlas_prior = candidates["atlas"].map(
        {
            "harvard_oxford_merged": 1.00,
            "eickhoff_zilles": 0.95,
            "talairach_tournoux": 0.90,
            "msdl_39": 1.00,
            "schaefer_400_7net": 0.95,
            "schaefer_200_7net": 0.90,
            "schaefer_100_7net": 0.85,
        }
    ).fillna(0.70)
    feature_prior = candidates["feature"].map(
        {
            "corr_mean": 1.00,
            "corr_node_degree_abs_top10": 1.00,
            "partial_mean": 0.90,
            "corr_mean_abs": 0.80,
            "partial_mean_abs": 0.70,
        }
    ).fillna(0.50)
    outcome_prior = candidates["outcome"].map(
        {
            "MMSE": 1.00,
            "ADAS13": 1.00,
            "CDRSB": 0.90,
            "diagnosis_code": 0.75,
        }
    ).fillna(0.50)
    return (
        0.55 * rule.claim_confidence
        + 0.20 * atlas_prior
        + 0.15 * feature_prior
        + 0.10 * outcome_prior
    )


def build_kg_candidate_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Select and deterministically rank dataset-matched KG candidates."""

    queues: dict[str, list[dict[str, object]]] = {}
    for rule in CANDIDATE_RULES:
        selected = results.loc[
            _rule_universe_mask(
                results,
                rule,
                require_target_region=True,
            )
        ].copy()
        if selected.empty:
            queues[rule.rule_id] = []
            continue
        selected["rule_id"] = rule.rule_id
        selected["claim_ids"] = ";".join(rule.claim_ids)
        selected["chain_scope"] = rule.chain_scope
        selected["claim_interpretation"] = rule.interpretation
        selected["claim_confidence"] = rule.claim_confidence
        selected["prior_score"] = _candidate_prior(selected, rule)
        selected = selected.sort_values(
            ["prior_score", "outcome", "atlas", "marker_id"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        queues[rule.rule_id] = selected.to_dict("records")

    ranked_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    positions = {rule.rule_id: 0 for rule in CANDIDATE_RULES}
    while True:
        added = False
        for rule in CANDIDATE_RULES:
            queue = queues[rule.rule_id]
            position = positions[rule.rule_id]
            while position < len(queue):
                row = queue[position]
                position += 1
                key = (
                    str(row["exposure"]),
                    str(row["outcome"]),
                    str(row["marker_id"]),
                )
                if key not in seen:
                    seen.add(key)
                    ranked_rows.append(row)
                    added = True
                    break
            positions[rule.rule_id] = position
        if not added:
            break

    ranked = pd.DataFrame(ranked_rows)
    if ranked.empty:
        return ranked
    ranked.insert(0, "kg_rank", np.arange(1, len(ranked) + 1))
    ranked.insert(
        1,
        "candidate_id",
        [
            f"CS2DC:{index:06d}"
            for index in range(1, len(ranked) + 1)
        ],
    )
    return ranked


def _add_signal_flags(
    frame: pd.DataFrame,
    *,
    nominal_alpha: float,
    minimum_effect: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["mediation_nominal"] = frame["sobel_p"].lt(nominal_alpha)
    frame["mediation_exploratory"] = (
        frame["mediation_nominal"]
        & frame["abs_indirect_effect_std"].ge(minimum_effect)
    )
    frame["gene_imaging_nominal"] = frame["a_path_p"].lt(nominal_alpha)
    frame["gene_imaging_exploratory"] = (
        frame["gene_imaging_nominal"]
        & frame["abs_a_path_std"].ge(minimum_effect)
    )
    return frame


def evaluate_ranking(
    ranked: pd.DataFrame,
    results: pd.DataFrame,
    *,
    k_values: Iterable[int],
    nominal_alpha: float,
    minimum_effect: float,
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate KG ranking against rule- and family-matched random draws."""

    ranked = _add_signal_flags(
        ranked,
        nominal_alpha=nominal_alpha,
        minimum_effect=minimum_effect,
    )
    results = _add_signal_flags(
        results,
        nominal_alpha=nominal_alpha,
        minimum_effect=minimum_effect,
    )
    rules = {rule.rule_id: rule for rule in CANDIDATE_RULES}
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    signal_columns = (
        "mediation_nominal",
        "mediation_exploratory",
        "gene_imaging_nominal",
        "gene_imaging_exploratory",
    )

    for k in k_values:
        if k > len(ranked):
            continue
        top = ranked.head(k)
        for signal_column in signal_columns:
            random_totals = np.zeros(n_resamples, dtype=int)
            for (rule_id, outcome), group in top.groupby(
                ["rule_id", "outcome"],
                sort=False,
            ):
                rule = rules[str(rule_id)]
                pool_mask = _rule_universe_mask(
                    results,
                    rule,
                    require_target_region=False,
                )
                pool_mask &= results["outcome"].eq(outcome)
                pool = results.loc[pool_mask, signal_column].fillna(False)
                draws = len(group)
                if draws > len(pool):
                    raise ValueError(
                        f"Random pool too small for {rule_id}/{outcome}: "
                        f"{draws}>{len(pool)}"
                    )
                hits = int(pool.sum())
                random_totals += rng.hypergeometric(
                    hits,
                    len(pool) - hits,
                    draws,
                    size=n_resamples,
                )
            observed = int(top[signal_column].fillna(False).sum())
            random_mean = float(random_totals.mean())
            ci_low, ci_high = np.quantile(random_totals, [0.025, 0.975])
            empirical_p = float(
                (1 + np.count_nonzero(random_totals >= observed))
                / (n_resamples + 1)
            )
            metrics.append(
                {
                    "k": int(k),
                    "signal": signal_column,
                    "kg_hits": observed,
                    "kg_rate": observed / k,
                    "random_mean_hits": random_mean,
                    "random_ci95_low": float(ci_low),
                    "random_ci95_high": float(ci_high),
                    "fold_over_random": (
                        observed / random_mean
                        if random_mean > 0
                        else math.inf if observed > 0 else 1.0
                    ),
                    "empirical_p_random_ge_kg": empirical_p,
                    "resamples": n_resamples,
                }
            )
            distributions.extend(
                {
                    "k": int(k),
                    "signal": signal_column,
                    "resample": index,
                    "random_hits": int(value),
                }
                for index, value in enumerate(random_totals, start=1)
            )
    return ranked, pd.DataFrame(metrics), pd.DataFrame(distributions)


def run(args: argparse.Namespace) -> dict[str, object]:
    required = (
        args.plink2,
        args.pgen_prefix.with_suffix(".pgen"),
        args.pgen_prefix.with_suffix(".pvar"),
        args.pgen_prefix.with_suffix(".psam"),
        args.dataset_root / "case2_adni_subject_table.parquet",
        args.dataset_root / "imaging_features",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. Use --force."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    dosages, dosage_metadata = export_variant_dosages(
        plink2=args.plink2,
        pgen_prefix=args.pgen_prefix,
        output_root=args.output_root,
        force=args.force_variant_export,
    )
    subjects = pd.read_parquet(
        args.dataset_root / "case2_adni_subject_table.parquet"
    )
    subjects = subjects.merge(
        dosages,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    for exposure in TARGET_VARIANTS.values():
        available = int(subjects[exposure].notna().sum())
        if available < args.min_n:
            raise RuntimeError(
                f"Only {available} subjects have {exposure}; min_n={args.min_n}"
            )

    results, family_summary = run_direct_variant_universe(
        subjects,
        args.dataset_root / "imaging_features",
        outcomes=args.outcomes,
        fd_max=args.fd_max,
        min_n=args.min_n,
    )
    results.to_parquet(
        args.output_root / "all_direct_variant_results.parquet",
        index=False,
        compression="zstd",
    )
    family_summary.to_csv(args.output_root / "family_summary.csv", index=False)

    ranked = build_kg_candidate_ranking(results)
    ranked, metrics, random_distribution = evaluate_ranking(
        ranked,
        results,
        k_values=args.k,
        nominal_alpha=args.nominal_alpha,
        minimum_effect=args.minimum_effect,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    ranked.to_parquet(
        args.output_root / "kg_ranked_dataset_constrained_results.parquet",
        index=False,
        compression="zstd",
    )
    ranked.head(args.top_n).to_csv(
        args.output_root / "top_kg_ranked_results.csv",
        index=False,
    )
    metrics.to_csv(args.output_root / "prioritisation_by_k.csv", index=False)
    random_distribution.to_parquet(
        args.output_root / "random_distribution.parquet",
        index=False,
        compression="zstd",
    )
    (args.output_root / "candidate_rules.json").write_text(
        json.dumps(
            [asdict(rule) for rule in CANDIDATE_RULES],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    now_hkt = datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload: dict[str, object] = {
        "created_at_hkt": now_hkt,
        "analysis_type": (
            "dataset-constrained direct-variant resting-state fMRI "
            "association-based mediation screen"
        ),
        "causal_interpretation": False,
        "selection_policy": (
            "Candidate rules were fixed from KG claim semantics and available "
            "ADNI variables before statistical results were inspected."
        ),
        "dataset_root": str(args.dataset_root),
        "pgen_prefix": str(args.pgen_prefix),
        "subjects": int(len(subjects)),
        "atlases": int(
            len(list((args.dataset_root / "imaging_features").glob("*.parquet")))
        ),
        "exposures": list(EXPOSURE_METADATA),
        "outcomes": list(args.outcomes),
        "variant_metadata": dosage_metadata.to_dict("records"),
        "candidate_rules": len(CANDIDATE_RULES),
        "complete_direct_variant_tests": int(len(results)),
        "kg_ranked_candidates": int(len(ranked)),
        "direct_universe_family_fdr_mediation_hits": int(
            results["sobel_q_family"].lt(0.05).sum()
        ),
        "direct_universe_all_atlas_fdr_mediation_hits": int(
            results["sobel_q_exposure_outcome_all_atlases"].lt(0.05).sum()
        ),
        "direct_universe_all_atlas_fdr_gene_imaging_hits": int(
            results["a_path_q_exposure_outcome_all_atlases"].lt(0.05).sum()
        ),
        "random_resamples": args.resamples,
        "random_seed": args.seed,
        "limitations": [
            "Cross-sectional indirect effects are association-based.",
            "Current outcomes are concurrent diagnosis and cognitive scores, "
            "not longitudinal decline.",
            "Functional connectivity density and network summaries are atlas "
            "proxies for the claim-level imaging constructs.",
            "COMT is an imaging positive-control transfer from healthy adults, "
            "not a primary Alzheimer disease chain.",
        ],
        "outputs": {
            "all_results": str(
                args.output_root / "all_direct_variant_results.parquet"
            ),
            "ranked_results": str(
                args.output_root
                / "kg_ranked_dataset_constrained_results.parquet"
            ),
            "top_results": str(
                args.output_root / "top_kg_ranked_results.csv"
            ),
            "prioritisation": str(
                args.output_root / "prioritisation_by_k.csv"
            ),
            "family_summary": str(args.output_root / "family_summary.csv"),
            "candidate_rules": str(args.output_root / "candidate_rules.json"),
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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pgen-prefix", type=Path, default=DEFAULT_PGEN_PREFIX)
    parser.add_argument("--plink2", type=Path, default=DEFAULT_PLINK2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--outcome",
        dest="outcomes",
        action="append",
        default=None,
    )
    parser.add_argument("--fd-max", type=float, default=0.5)
    parser.add_argument("--min-n", type=int, default=80)
    parser.add_argument("--nominal-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-variant-export", action="store_true")
    args = parser.parse_args()
    args.outcomes = tuple(args.outcomes or DEFAULT_OUTCOMES)
    args.k = tuple(sorted(set(args.k or DEFAULT_K_VALUES)))
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Last Updated At: 2026-07-31 03:57 HKT
