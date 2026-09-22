"""Build blinded Case Study 1 external-validation outcome labels.

The TCP ranking is frozen before any external result is inspected.  This
script joins that ranking to independently computed external cohort results
and distinguishes three scientifically different outcomes:

* confirmed: at least one significant, direction-concordant external result
  and no significant direction-discordant result;
* falsified: at least one significant, direction-discordant external result
  and no significant direction-concordant result;
* not_confirmed: executable externally, but without a significant result.

Candidates with both concordant and discordant significant results are marked
``conflicting`` and must not be used as either side of an Expert Study pair.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_DISEASES_BY_DATASET = {
    "adhd200": {"ADHD"},
    "cobre": {"psychosis_SZ_SZA"},
    "hcpep": {"psychosis_SZ_SZA"},
    "ucla": {"ADHD", "bipolar", "psychosis_SZ_SZA"},
}

KEY_COLUMNS = ("modality", "source", "disease", "feature", "roi_index")
DEFAULT_RESULT_NAME = "case1_exhaustive_full_all_tests.csv"


def candidate_id(frame: pd.DataFrame) -> pd.Series:
    """Return the stable TCP candidate identifier used by frozen rankings."""

    roi_index = pd.to_numeric(frame["roi_index"], errors="coerce").astype("Int64")
    return (
        frame["modality"].astype(str)
        + "|"
        + frame["source"].astype(str)
        + "|"
        + frame["disease"].astype(str)
        + "|"
        + frame["feature"].astype(str)
        + "|"
        + roi_index.astype(str)
    )


def load_external_results(
    external_root: Path,
    run_name: str,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    selected = datasets or list(PRIMARY_DISEASES_BY_DATASET)
    unknown = sorted(set(selected) - set(PRIMARY_DISEASES_BY_DATASET))
    if unknown:
        raise ValueError(f"Unknown external datasets: {unknown}")
    for dataset in selected:
        allowed_diseases = PRIMARY_DISEASES_BY_DATASET[dataset]
        path = external_root / dataset / run_name / DEFAULT_RESULT_NAME
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path, low_memory=False)
        required = {
            *KEY_COLUMNS,
            "adjusted_residual_d",
            "q_fdr_disease",
            "n_case",
            "n_control",
        }
        absent = sorted(required - set(frame.columns))
        if absent:
            raise ValueError(f"{path} is missing required columns: {absent}")
        frame = frame[frame["disease"].isin(allowed_diseases)].copy()
        frame["dataset"] = dataset
        frame["candidate_id"] = candidate_id(frame)
        rows.append(frame)
    if missing:
        raise FileNotFoundError(
            "External validation is incomplete. Missing result files:\n"
            + "\n".join(missing)
        )
    return pd.concat(rows, ignore_index=True)


def classify_results(
    ranking: pd.DataFrame,
    external: pd.DataFrame,
    *,
    q_threshold: float,
    min_abs_d: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_ranking = {
        *KEY_COLUMNS,
        "candidate_id",
        "rank",
        "score_neurodiscovery",
        "adjusted_residual_d",
    }
    absent = sorted(required_ranking - set(ranking.columns))
    if absent:
        raise ValueError(f"Frozen ranking is missing required columns: {absent}")

    frozen = ranking.copy()
    frozen["candidate_id"] = frozen["candidate_id"].astype(str)
    frozen["tcp_adjusted_residual_d"] = pd.to_numeric(
        frozen["adjusted_residual_d"], errors="coerce"
    )
    frozen = frozen.drop(columns=["adjusted_residual_d"])

    external = external.rename(
        columns={
            "adjusted_residual_d": "external_adjusted_residual_d",
            "q_fdr_disease": "external_q_fdr_disease",
            "n_case": "external_n_case",
            "n_control": "external_n_control",
        }
    )
    external_columns = [
        "candidate_id",
        "dataset",
        "external_adjusted_residual_d",
        "external_q_fdr_disease",
        "external_n_case",
        "external_n_control",
        "p_value",
    ]
    long = frozen.merge(external[external_columns], on="candidate_id", how="inner")
    long["externally_executable"] = np.isfinite(long["external_adjusted_residual_d"])
    long["externally_significant"] = (
        long["externally_executable"]
        & (pd.to_numeric(long["external_q_fdr_disease"], errors="coerce") < q_threshold)
        & (
            pd.to_numeric(
                long["external_adjusted_residual_d"], errors="coerce"
            ).abs()
            > min_abs_d
        )
    )
    long["direction_concordant"] = (
        np.sign(long["tcp_adjusted_residual_d"])
        == np.sign(long["external_adjusted_residual_d"])
    )
    long["external_outcome"] = np.select(
        [
            long["externally_significant"] & long["direction_concordant"],
            long["externally_significant"] & ~long["direction_concordant"],
            long["externally_executable"],
        ],
        ["confirmed", "falsified", "not_confirmed"],
        default="not_executable",
    )

    aggregate_rows: list[dict[str, Any]] = []
    for candidate, group in long.groupby("candidate_id", sort=False):
        base = group.iloc[0]
        confirmed = sorted(
            group.loc[group["external_outcome"] == "confirmed", "dataset"].unique()
        )
        falsified = sorted(
            group.loc[group["external_outcome"] == "falsified", "dataset"].unique()
        )
        executable = sorted(
            group.loc[group["externally_executable"], "dataset"].unique()
        )
        if confirmed and falsified:
            label = "conflicting"
        elif confirmed:
            label = "confirmed"
        elif falsified:
            label = "falsified"
        elif executable:
            label = "not_confirmed"
        else:
            label = "not_executable"
        aggregate_rows.append(
            {
                "candidate_id": candidate,
                "rank": int(base["rank"]),
                "disease": str(base["disease"]),
                "modality": str(base["modality"]),
                "source": str(base["source"]),
                "roi_index": int(base["roi_index"]),
                "roi_id": base.get("roi_id"),
                "roi_name": str(base.get("roi_name") or ""),
                "anatomy_full": str(base.get("anatomy_full") or ""),
                "network": str(base.get("network") or ""),
                "feature": str(base["feature"]),
                "feature_family": str(base.get("feature_family") or ""),
                "score_neurodiscovery": float(base["score_neurodiscovery"]),
                "tcp_adjusted_residual_d": float(base["tcp_adjusted_residual_d"]),
                "external_label": label,
                "executable_datasets": json.dumps(executable),
                "confirmed_datasets": json.dumps(confirmed),
                "falsified_datasets": json.dumps(falsified),
                "n_external_datasets": len(executable),
                "n_confirmed_datasets": len(confirmed),
                "n_falsified_datasets": len(falsified),
            }
        )
    labels = pd.DataFrame(aggregate_rows)
    return long, labels


def build_summary(
    labels: pd.DataFrame,
    long: pd.DataFrame,
    *,
    ranking_path: Path,
    external_root: Path,
    run_name: str,
    q_threshold: float,
    min_abs_d: float,
) -> dict[str, Any]:
    label_counts = Counter(labels["external_label"])
    by_disease = {
        disease: dict(Counter(group["external_label"]))
        for disease, group in labels.groupby("disease")
    }
    by_dataset = {
        dataset: dict(Counter(group["external_outcome"]))
        for dataset, group in long.groupby("dataset")
    }
    datasets_used = sorted(long["dataset"].unique())
    return {
        "schema_version": "case1-external-labels-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_ranking": str(ranking_path),
        "external_results_root": str(external_root),
        "external_run_name": run_name,
        "primary_diseases_by_dataset": {
            key: sorted(PRIMARY_DISEASES_BY_DATASET[key])
            for key in datasets_used
        },
        "datasets_used": datasets_used,
        "thresholds": {
            "bh_fdr_q": q_threshold,
            "bh_fdr_scope": "within external dataset and disease",
            "minimum_absolute_cohens_d": min_abs_d,
            "direction_must_match_tcp_for_confirmation": True,
        },
        "definitions": {
            "confirmed": (
                "At least one significant direction-concordant external result "
                "and no significant direction-discordant result."
            ),
            "falsified": (
                "At least one significant direction-discordant external result "
                "and no significant direction-concordant result."
            ),
            "conflicting": (
                "Both significant concordant and significant discordant external "
                "results; excluded from Expert Study pairing."
            ),
            "not_confirmed": (
                "Externally executable but no significant external result; never "
                "presented as falsified."
            ),
        },
        "n_ranked_candidates_with_external_results": int(len(labels)),
        "label_counts": dict(label_counts),
        "label_counts_by_disease": by_disease,
        "external_outcomes_by_dataset": by_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join frozen TCP NeuroDiscovery ranking to external validation."
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="primary")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=tuple(PRIMARY_DISEASES_BY_DATASET),
        default=None,
        help="Optional subset for interim audits; final labels should use all datasets.",
    )
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--min-abs-d", type=float, default=0.15)
    args = parser.parse_args()

    ranking = pd.read_csv(args.ranking, low_memory=False)
    external = load_external_results(
        args.external_root,
        args.run_name,
        args.datasets,
    )
    long, labels = classify_results(
        ranking,
        external,
        q_threshold=args.q_threshold,
        min_abs_d=args.min_abs_d,
    )
    summary = build_summary(
        labels,
        long,
        ranking_path=args.ranking,
        external_root=args.external_root,
        run_name=args.run_name,
        q_threshold=args.q_threshold,
        min_abs_d=args.min_abs_d,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_dir / "case1_external_results_long.csv", index=False)
    labels.to_csv(args.output_dir / "case1_external_candidate_labels.csv", index=False)
    (args.output_dir / "case1_external_label_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
