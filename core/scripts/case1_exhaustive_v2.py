"""Run Case Study 1 exhaustive v2 with covariate adjustment and bootstrap CIs.

Compared with v1, this version keeps the same ``disease x ROI x feature`` search
space but reports age/sex/site-adjusted effects, OLS p-values, and bootstrap
confidence intervals for adjusted residual Cohen's d.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from case1_exhaustive_v1 import (
    DEFAULT_HP2000_MAPPING,
    DEFAULT_SMRI_FEATURE_ROOT,
    DEFAULT_TCP_ROOT,
    DEFAULT_TRANS_DIAGNOSIS,
    add_fdr_columns,
    build_fmri_features,
    cohen_d,
    disease_masks,
    load_metadata,
    load_smri_source,
)


DEFAULT_OUT_ROOT = Path(r"Z:\Public Dataset\case1_exhaustive_v2")


def build_covariates(meta: pd.DataFrame) -> pd.DataFrame:
    cov = pd.DataFrame(index=meta.index)
    age = pd.to_numeric(meta.get("Age"), errors="coerce")
    if age.isna().all():
        age = pd.to_numeric(meta.get("interview_age"), errors="coerce") / 12.0
    cov["age_z"] = (age - age.mean()) / age.std(ddof=0)

    sex = meta.get("sex", pd.Series("", index=meta.index)).astype(str).str.strip().str.upper()
    cov["sex_male"] = (sex == "M").astype(float)

    site = meta.get("Site", pd.Series("", index=meta.index)).astype(str)
    site_dummies = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
    cov = pd.concat([cov, site_dummies], axis=1)
    return cov.fillna(0.0)


def ols_case_effect(
    matrix: np.ndarray,
    case_mask: np.ndarray,
    control_mask: np.ndarray,
    covariates: pd.DataFrame,
) -> dict[str, np.ndarray | int]:
    subset_mask = case_mask | control_mask
    y = matrix[subset_mask].astype(float)
    case = case_mask[subset_mask].astype(float)
    cov = covariates.loc[subset_mask].to_numpy(float)

    valid_subject = np.isfinite(y).all(axis=1) & np.isfinite(cov).all(axis=1)
    y = y[valid_subject]
    case = case[valid_subject]
    cov = cov[valid_subject]
    n_case = int(case.sum())
    n_control = int((case == 0).sum())
    if n_case < 2 or n_control < 2:
        n_roi = matrix.shape[1]
        nan = np.full(n_roi, np.nan)
        return {
            "n_case": n_case,
            "n_control": n_control,
            "beta": nan,
            "se": nan,
            "t": nan,
            "p": nan,
            "residual_d": nan,
            "residual_mean_case": nan,
            "residual_mean_control": nan,
        }

    x_full = np.column_stack([np.ones(len(case)), case, cov])
    x_cov = np.column_stack([np.ones(len(case)), cov])

    pinv_full = np.linalg.pinv(x_full)
    beta_full = pinv_full @ y
    residual_full = y - x_full @ beta_full
    dof = max(x_full.shape[0] - x_full.shape[1], 1)
    sigma2 = np.nansum(residual_full * residual_full, axis=0) / dof
    xtx_inv = np.linalg.pinv(x_full.T @ x_full)
    case_var = float(xtx_inv[1, 1])
    se = np.sqrt(np.maximum(sigma2 * case_var, 0.0))
    beta = beta_full[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = beta / se
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), dof)

    beta_cov = np.linalg.pinv(x_cov) @ y
    residual_cov = y - x_cov @ beta_cov
    case_resid = residual_cov[case == 1]
    control_resid = residual_cov[case == 0]
    residual_d = cohen_d(case_resid, control_resid)

    return {
        "n_case": n_case,
        "n_control": n_control,
        "beta": beta,
        "se": se,
        "t": t_stat,
        "p": p_value,
        "residual_d": residual_d,
        "residual_mean_case": np.nanmean(case_resid, axis=0),
        "residual_mean_control": np.nanmean(control_resid, axis=0),
    }


def bootstrap_residual_d_ci(
    matrix: np.ndarray,
    case_mask: np.ndarray,
    control_mask: np.ndarray,
    covariates: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset_mask = case_mask | control_mask
    y = matrix[subset_mask].astype(float)
    case = case_mask[subset_mask].astype(bool)
    cov = covariates.loc[subset_mask].to_numpy(float)
    valid_subject = np.isfinite(y).all(axis=1) & np.isfinite(cov).all(axis=1)
    y = y[valid_subject]
    case = case[valid_subject]
    cov = cov[valid_subject]

    n_roi = matrix.shape[1]
    if n_boot <= 0 or case.sum() < 2 or (~case).sum() < 2:
        nan = np.full(n_roi, np.nan)
        return nan, nan

    x_cov = np.column_stack([np.ones(len(case)), cov])
    beta_cov = np.linalg.pinv(x_cov) @ y
    residual = y - x_cov @ beta_cov
    case_resid = residual[case]
    control_resid = residual[~case]

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_roi), dtype=np.float32)
    n_case = case_resid.shape[0]
    n_control = control_resid.shape[0]
    for i in range(n_boot):
        case_idx = rng.integers(0, n_case, size=n_case)
        control_idx = rng.integers(0, n_control, size=n_control)
        boot[i] = cohen_d(case_resid[case_idx], control_resid[control_idx])
    return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def evaluate_matrix_v2(
    *,
    modality: str,
    source: str,
    feature_name: str,
    meta: pd.DataFrame,
    diseases: list[dict[str, object]],
    roi_meta: pd.DataFrame,
    matrix: np.ndarray,
    covariates: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for disease_idx, disease_row in enumerate(diseases):
        disease = str(disease_row["disease"])
        case_mask, control_mask = disease_masks(meta, disease)
        stats_row = ols_case_effect(matrix, case_mask, control_mask, covariates)
        ci_low, ci_high = bootstrap_residual_d_ci(
            matrix,
            case_mask,
            control_mask,
            covariates,
            n_boot=n_boot,
            seed=seed + disease_idx * 1009,
        )

        beta = np.asarray(stats_row["beta"], dtype=float)
        se = np.asarray(stats_row["se"], dtype=float)
        t_stat = np.asarray(stats_row["t"], dtype=float)
        p_value = np.asarray(stats_row["p"], dtype=float)
        residual_d = np.asarray(stats_row["residual_d"], dtype=float)
        residual_mean_case = np.asarray(stats_row["residual_mean_case"], dtype=float)
        residual_mean_control = np.asarray(stats_row["residual_mean_control"], dtype=float)

        for idx in range(matrix.shape[1]):
            roi = roi_meta.iloc[idx].to_dict()
            d_val = float(residual_d[idx]) if np.isfinite(residual_d[idx]) else float("nan")
            beta_val = float(beta[idx]) if np.isfinite(beta[idx]) else float("nan")
            se_val = float(se[idx]) if np.isfinite(se[idx]) else float("nan")
            low = float(ci_low[idx]) if np.isfinite(ci_low[idx]) else float("nan")
            high = float(ci_high[idx]) if np.isfinite(ci_high[idx]) else float("nan")
            rows.append(
                {
                    "modality": modality,
                    "source": source,
                    "disease": disease,
                    "feature": feature_name,
                    "roi_index": int(roi.get("roi_index", roi.get("h5_index", idx))),
                    "roi_id": roi.get("roi_id", idx + 1),
                    "roi_name": roi.get("parcel_name") or roi.get("roi_name") or roi.get("label_name") or f"roi_{idx + 1}",
                    "anatomy_key": roi.get("anatomy_key", ""),
                    "anatomy_full": roi.get("anatomy_full", ""),
                    "hemisphere": roi.get("hemisphere", ""),
                    "network": roi.get("network", ""),
                    "structure_class": roi.get("structure_class", ""),
                    "n_case": int(stats_row["n_case"]),
                    "n_control": int(stats_row["n_control"]),
                    "adjusted_beta_case_minus_control": beta_val,
                    "adjusted_beta_se": se_val,
                    "adjusted_t": float(t_stat[idx]) if np.isfinite(t_stat[idx]) else float("nan"),
                    "p_value": float(p_value[idx]) if np.isfinite(p_value[idx]) else float("nan"),
                    "adjusted_residual_d": d_val,
                    "abs_adjusted_residual_d": abs(d_val) if math.isfinite(d_val) else float("nan"),
                    "residual_mean_case": float(residual_mean_case[idx]) if np.isfinite(residual_mean_case[idx]) else float("nan"),
                    "residual_mean_control": float(residual_mean_control[idx]) if np.isfinite(residual_mean_control[idx]) else float("nan"),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_ci_excludes_zero": bool((low > 0 and high > 0) or (low < 0 and high < 0)),
                    "direction": "case_higher" if d_val > 0 else "case_lower" if d_val < 0 else "flat",
                }
            )
    return rows


def add_v2_fdr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"adjusted_residual_d": "cohen_d_case_minus_control", "abs_adjusted_residual_d": "abs_d"})
    out = add_fdr_columns(out)
    return out.rename(columns={"cohen_d_case_minus_control": "adjusted_residual_d", "abs_d": "abs_adjusted_residual_d"})


def summarize_shared_v2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["modality", "source", "feature", "roi_name", "anatomy_full", "hemisphere", "network"]
    for keys, g in df.groupby(group_cols, dropna=False):
        effects = g["adjusted_residual_d"].to_numpy(float)
        finite = effects[np.isfinite(effects)]
        if finite.size == 0:
            continue
        pos = int(np.sum(finite > 0))
        neg = int(np.sum(finite < 0))
        dominant_sign = "case_higher" if pos >= neg else "case_lower"
        same_sign = max(pos, neg)
        sig_global = g[np.isfinite(g["q_fdr_global"]) & (g["q_fdr_global"] < 0.05)]
        sig_modality = g[np.isfinite(g["q_fdr_modality"]) & (g["q_fdr_modality"] < 0.05)]
        ci_nonzero = g[g["bootstrap_ci_excludes_zero"].astype(bool)]
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n_diseases_tested": int(g["disease"].nunique()),
                "dominant_direction": dominant_sign,
                "direction_consistency": float(same_sign / finite.size),
                "mean_adjusted_d": float(np.nanmean(finite)),
                "mean_abs_adjusted_d": float(np.nanmean(np.abs(finite))),
                "median_adjusted_d": float(np.nanmedian(finite)),
                "max_abs_adjusted_d": float(np.nanmax(np.abs(finite))),
                "n_q05_global": int(sig_global["disease"].nunique()),
                "n_q05_modality": int(sig_modality["disease"].nunique()),
                "n_bootstrap_ci_excludes_zero": int(ci_nonzero["disease"].nunique()),
                "best_q_global": float(np.nanmin(g["q_fdr_global"])),
                "best_q_modality": float(np.nanmin(g["q_fdr_modality"])),
                "best_p": float(np.nanmin(g["p_value"])),
                "diseases_q05_global": "|".join(sorted(sig_global["disease"].unique().tolist())),
                "diseases_q05_modality": "|".join(sorted(sig_modality["disease"].unique().tolist())),
                "diseases_ci_excludes_zero": "|".join(sorted(ci_nonzero["disease"].unique().tolist())),
            }
        )
        row["shared_strength_score"] = (
            row["mean_abs_adjusted_d"]
            * (0.40 + 0.60 * row["direction_consistency"])
            * (1.0 + 0.20 * row["n_q05_modality"] + 0.08 * row["n_bootstrap_ci_excludes_zero"])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["n_q05_global", "n_q05_modality", "shared_strength_score"],
        ascending=[False, False, False],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-root", type=Path, default=DEFAULT_TCP_ROOT)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_TRANS_DIAGNOSIS)
    parser.add_argument("--smri-feature-root", type=Path, default=DEFAULT_SMRI_FEATURE_ROOT)
    parser.add_argument("--hp2000-mapping", type=Path, default=DEFAULT_HP2000_MAPPING)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-cases", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    start = time.perf_counter()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building feature matrices...", flush=True)
    subjects, hp2000_roi_meta, fmri_arrays = build_fmri_features(args.tcp_root, args.hp2000_mapping)
    meta, diseases = load_metadata(args.diagnosis, subjects, args.min_cases)
    covariates = build_covariates(meta)
    print(
        f"subjects={len(subjects)} controls={int(meta['is_control'].sum())} "
        f"diseases={len(diseases)} covariates={list(covariates.columns)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for feature_idx, (feature_name, matrix) in enumerate(fmri_arrays.items()):
        print(f"fMRI {feature_name}...", flush=True)
        rows.extend(
            evaluate_matrix_v2(
                modality="fmri",
                source="hp2000_rest_correlation",
                feature_name=feature_name,
                meta=meta,
                diseases=diseases,
                roi_meta=hp2000_roi_meta,
                matrix=matrix,
                covariates=covariates,
                n_boot=args.n_boot,
                seed=args.seed + feature_idx * 100_003,
            )
        )

    for source_idx, source in enumerate(("aparc_dkt_aseg", "aseg")):
        print(f"sMRI {source}...", flush=True)
        roi_meta, matrix = load_smri_source(args.smri_feature_root, source, subjects)
        rows.extend(
            evaluate_matrix_v2(
                modality="smri",
                source=source,
                feature_name="normalized_volume_fraction",
                meta=meta,
                diseases=diseases,
                roi_meta=roi_meta,
                matrix=matrix,
                covariates=covariates,
                n_boot=args.n_boot,
                seed=args.seed + (source_idx + 10) * 100_003,
            )
        )

    all_df = add_v2_fdr_columns(pd.DataFrame(rows))
    all_df = all_df.sort_values(["q_fdr_global", "abs_adjusted_residual_d"], ascending=[True, False])
    all_path = out_dir / "case1_exhaustive_v2_all_tests.csv"
    all_df.to_csv(all_path, index=False)

    top_path = out_dir / "case1_exhaustive_v2_top_hits.csv"
    all_df.sort_values(["abs_adjusted_residual_d", "q_fdr_modality"], ascending=[False, True]).head(1000).to_csv(top_path, index=False)

    sig_global_path = out_dir / "case1_exhaustive_v2_significant_global_q05.csv"
    all_df[all_df["q_fdr_global"] < 0.05].to_csv(sig_global_path, index=False)

    sig_modality_path = out_dir / "case1_exhaustive_v2_significant_modality_q05.csv"
    all_df[all_df["q_fdr_modality"] < 0.05].to_csv(sig_modality_path, index=False)

    ci_path = out_dir / "case1_exhaustive_v2_bootstrap_ci_nonzero.csv"
    all_df[all_df["bootstrap_ci_excludes_zero"]].to_csv(ci_path, index=False)

    shared_df = summarize_shared_v2(all_df)
    shared_path = out_dir / "case1_exhaustive_v2_shared_roi_feature_summary.csv"
    shared_df.to_csv(shared_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "out_dir": str(out_dir),
        "tcp_root": str(args.tcp_root),
        "diagnosis": str(args.diagnosis),
        "smri_feature_root": str(args.smri_feature_root),
        "hp2000_mapping": str(args.hp2000_mapping),
        "subjects": len(subjects),
        "controls": int(meta["is_control"].sum()),
        "diseases": diseases,
        "covariates": list(covariates.columns),
        "n_boot": args.n_boot,
        "n_tests": int(all_df.shape[0]),
        "n_q05_global": int((all_df["q_fdr_global"] < 0.05).sum()),
        "n_q05_modality": int((all_df["q_fdr_modality"] < 0.05).sum()),
        "n_bootstrap_ci_excludes_zero": int(all_df["bootstrap_ci_excludes_zero"].sum()),
        "all_tests": str(all_path),
        "top_hits": str(top_path),
        "significant_global_q05": str(sig_global_path),
        "significant_modality_q05": str(sig_modality_path),
        "bootstrap_ci_nonzero": str(ci_path),
        "shared_summary": str(shared_path),
        "elapsed_sec": round(time.perf_counter() - start, 3),
    }
    manifest_path = out_dir / "case1_exhaustive_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
