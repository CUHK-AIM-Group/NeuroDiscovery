"""Run first-pass ADNI fMRI validation for Case Study 1 rankings.

This is a deliberately narrow external-validation path:
- use existing ADNI fMRIPrep MNI-space BOLD files,
- extract Schaefer-400 7-network ROI time series,
- compute the fMRI features used by Case Study 1 that are reproducible from
  resting-state ROI time series,
- test AD vs CN and MCI vs CN,
- evaluate whether each method's Case Study 1 region-feature priority recovers
  ADNI-validated region-feature effects earlier.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from scipy import stats
from statsmodels.stats.multitest import multipletests

from case1_exhaustive_full import atlas_roi_meta


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("outputs/case1_adni_validation/adni_validation_manifest.csv")
DEFAULT_METHOD_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot\method_comparison"
)
DEFAULT_ATLAS_ROOT = REPO_ROOT.parent / "NeuroSTORM" / "datasets" / "atlas"
DEFAULT_OUT_DIR = Path("outputs/case1_adni_validation/fmri_schaefer400")
TCP_FULL_ATLASES = (
    "aal3_166",
    "aal_116",
    "basc_122",
    "cc200",
    "cc400",
    "destrieux_148",
    "dk_112",
    "dosenbach_160",
    "eickhoff_zilles",
    "glasser_360",
    "harvard_oxford_cort",
    "harvard_oxford_merged",
    "harvard_oxford_sub",
    "msdl_39",
    "power_264",
    "schaefer_100_7net",
    "schaefer_200_7net",
    "schaefer_400_7net",
    "talairach_tournoux",
)
METHODS = (
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
    "exhaustive_gt",
    "neurodiscovery",
)
ROI_FEATURES = (
    "roi_temporal_mean",
    "roi_temporal_std",
    "roi_temporal_variance",
    "roi_temporal_mean_abs",
    "roi_alff_proxy",
    "roi_falff_proxy",
)
CORR_FEATURES = (
    "corr_mean",
    "corr_mean_abs",
    "corr_positive_mean",
    "corr_negative_mean",
    "corr_node_degree_abs_top10",
)
PARTIAL_FEATURES = (
    "partial_mean",
    "partial_mean_abs",
    "partial_positive_mean",
    "partial_negative_mean",
)
FEATURES = ROI_FEATURES + CORR_FEATURES + PARTIAL_FEATURES
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--method-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=["schaefer_400_7net"],
        help="Atlas names, or 'all' for the 19-atlas TCP full-exhaustive set.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-subjects", type=int, default=0, help="Pilot limit after first-run-per-subject filtering; 0=all.")
    parser.add_argument(
        "--top-n-per-method",
        type=int,
        default=5000,
        help=(
            "Number of executable unique region-feature candidates to keep per "
            "method. Non-executable rows are skipped before computing top-k metrics."
        ),
    )
    parser.add_argument("--fd-thresh", type=float, default=0.5, help="QC exclusion threshold for mean framewise displacement.")
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true", help="Only write per-atlas feature caches and stop before statistics.")
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="Reuse an existing validation stats CSV in out-dir and only refresh ranking evaluation.",
    )
    return parser.parse_args()


def resolve_atlases(raw: Iterable[str]) -> list[str]:
    values = [str(v) for v in raw]
    if any(v.lower() == "all" for v in values):
        return list(TCP_FULL_ATLASES)
    return values


def first_testable_runs(manifest: pd.DataFrame, max_subjects: int = 0) -> pd.DataFrame:
    df = manifest.copy()
    df = df[df["adni_testable"].astype(str).str.lower().isin(["true", "1"])]
    df = df[df["diagnosis"].isin(["CN", "MCI", "AD"])].copy()
    df["session_month"] = df["VISCODE"].map(_viscode_month)
    df = df.sort_values(["PTID", "session_month", "VISCODE"], kind="mergesort")
    df = df.drop_duplicates("PTID", keep="first")
    if max_subjects and max_subjects > 0:
        # Keep a balanced, deterministic pilot rather than the first N subjects.
        parts = []
        per_group = max(1, math.ceil(max_subjects / 3))
        for dx in ("CN", "MCI", "AD"):
            parts.append(df[df["diagnosis"] == dx].head(per_group))
        df = pd.concat(parts, ignore_index=True).head(max_subjects)
    return df.reset_index(drop=True)


def _viscode_month(value: object) -> int:
    text = str(value or "").lower()
    if text == "bl":
        return 0
    match = re.fullmatch(r"m(\d+)", text)
    if match:
        return int(match.group(1))
    return 9999


def load_confounds(path: Path, n_tp: int) -> tuple[np.ndarray | None, float]:
    if not path.exists():
        return None, float("nan")
    conf = pd.read_csv(path, sep="\t")
    mean_fd = float(pd.to_numeric(conf.get("framewise_displacement"), errors="coerce").fillna(0.0).mean())
    cols = [c for c in CONFOUND_CANDIDATES if c in conf.columns]
    if not cols:
        return None, mean_fd
    mat = conf[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if len(mat) != n_tp:
        mat = mat[:n_tp]
    return mat, mean_fd


def tr_from_image(path: Path) -> float:
    zooms = nib.load(str(path)).header.get_zooms()
    return float(zooms[3]) if len(zooms) >= 4 and zooms[3] > 0 else 3.0


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(values, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def compute_roi_features(ts: np.ndarray) -> dict[str, np.ndarray]:
    # ts shape: time x ROI. Keep the cleaned signal scale for temporal/amplitude
    # features, but z-score a copy before estimating correlation features.
    ts = np.asarray(ts, dtype=float)
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    ts_mean = np.mean(ts, axis=0, keepdims=True)
    ts_std = np.std(ts, axis=0, ddof=1, keepdims=True)
    ts_z = (ts - ts_mean) / np.where(ts_std > 0, ts_std, 1.0)
    corr = np.corrcoef(ts_z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    positive = np.where(corr > 0, corr, np.nan)
    negative = np.where(corr < 0, corr, np.nan)
    abs_corr = np.abs(corr)
    ridge = 1e-3
    precision = np.linalg.pinv(corr + np.eye(corr.shape[0]) * ridge)
    precision_diag = np.clip(np.abs(np.diag(precision)), 1e-12, None)
    denom = np.sqrt(np.outer(precision_diag, precision_diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = -precision / denom
    partial = np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(partial, 0.0)
    partial_positive = np.where(partial > 0, partial, np.nan)
    partial_negative = np.where(partial < 0, partial, np.nan)
    abs_partial = np.abs(partial)
    top_k = min(10, max(1, abs_corr.shape[1] - 1))
    top_abs = np.partition(abs_corr, -top_k, axis=1)[:, -top_k:]
    temporal_std = np.std(ts, axis=0, ddof=1)
    temporal_var = np.var(ts, axis=0, ddof=1)
    return {
        "corr_mean": np.mean(corr, axis=1),
        "corr_mean_abs": np.mean(abs_corr, axis=1),
        "corr_positive_mean": _row_nanmean(positive),
        "corr_negative_mean": _row_nanmean(negative),
        "corr_node_degree_abs_top10": np.mean(top_abs, axis=1),
        "roi_temporal_mean": np.mean(ts, axis=0),
        "roi_temporal_mean_abs": np.abs(np.mean(ts, axis=0)),
        "roi_temporal_std": temporal_std,
        "roi_temporal_variance": temporal_var,
        "roi_alff_proxy": temporal_std,
        "roi_falff_proxy": temporal_std / (np.mean(np.abs(ts - np.mean(ts, axis=0)), axis=0) + 1e-6),
        "partial_mean": np.mean(partial, axis=1),
        "partial_mean_abs": np.mean(abs_partial, axis=1),
        "partial_positive_mean": _row_nanmean(partial_positive),
        "partial_negative_mean": _row_nanmean(partial_negative),
    }


def atlas_n_rois(atlas_path: Path) -> int:
    img = nib.load(str(atlas_path))
    shape = img.shape
    if len(shape) == 4:
        return int(shape[-1])
    data = img.get_fdata(dtype=np.float32)
    return int(sum(1 for value in np.unique(data) if int(value) > 0))


def make_masker(atlas_path: Path, t_r: float):
    kwargs = dict(
        standardize=False,
        detrend=True,
        low_pass=0.08,
        high_pass=0.01,
        t_r=t_r,
        memory=None,
        verbose=0,
    )
    if len(nib.load(str(atlas_path)).shape) == 4:
        return NiftiMapsMasker(maps_img=str(atlas_path), **kwargs)
    return NiftiLabelsMasker(labels_img=str(atlas_path), **kwargs)


def extracted_roi_meta(masker, roi_meta: pd.DataFrame, n_cols: int) -> pd.DataFrame:
    """Return ROI metadata aligned to masker output columns."""
    region_ids = getattr(masker, "region_ids_", None)
    if isinstance(region_ids, dict) and all(i in region_ids for i in range(n_cols)):
        extracted_ids = [int(region_ids[i]) for i in range(n_cols)]
        by_id = {int(row["roi_id"]): row for row in roi_meta.to_dict("records")}
        rows = []
        for out_idx, label_id in enumerate(extracted_ids):
            row = dict(by_id.get(label_id, {}))
            if not row:
                row = {
                    "roi_index": out_idx,
                    "roi_id": label_id,
                    "roi_name": f"label_{label_id}",
                    "parcel_name": str(label_id),
                    "hemisphere": "",
                    "network": "",
                    "structure_class": "",
                }
            row["feature_col"] = out_idx
            rows.append(row)
        return pd.DataFrame(rows)
    out = roi_meta.iloc[:n_cols].copy().reset_index(drop=True)
    out["feature_col"] = np.arange(len(out))
    return out


def extract_atlas_features(
    runs: pd.DataFrame,
    atlas: str,
    atlas_root: Path,
    out_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = out_dir / "atlas_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"adni_{atlas}_run_features.csv"
    if feature_path.exists() and not force:
        return pd.read_csv(feature_path)
    atlas_path = atlas_root / atlas / "atlas.nii.gz"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing atlas image: {atlas_path}")
    n_roi = atlas_n_rois(atlas_path)
    roi_meta = atlas_roi_meta(atlas, n_roi, atlas_root=atlas_root)
    rows: list[dict[str, object]] = []
    for i, row in runs.iterrows():
        bold_path = Path(str(row["bold_path"]))
        conf_path = Path(str(row["confounds_path"]))
        n_tp = int(nib.load(str(bold_path)).shape[-1])
        confounds, mean_fd = load_confounds(conf_path, n_tp)
        if not math.isfinite(mean_fd) or mean_fd > 10:
            mean_fd = float("nan")
        run_masker = make_masker(atlas_path, tr_from_image(bold_path))
        ts = run_masker.fit_transform(str(bold_path), confounds=confounds)
        run_roi_meta = extracted_roi_meta(run_masker, roi_meta, ts.shape[1])
        features = compute_roi_features(ts)
        for roi in run_roi_meta.to_dict("records"):
            roi_index = int(roi["roi_index"])
            feature_col = int(roi["feature_col"])
            base = {
                "PTID": row["PTID"],
                "subject_bids": row["subject_bids"],
                "session_bids": row["session_bids"],
                "VISCODE": row["VISCODE"],
                "diagnosis": row["diagnosis"],
                "AGE": row.get("AGE", np.nan),
                "PTGENDER": row.get("PTGENDER", ""),
                "SITE": row.get("SITE", ""),
                "mean_fd": mean_fd,
                "n_timepoints": n_tp,
                "source": f"{atlas}_multiatlas",
                "atlas": atlas,
                "roi_index": roi_index,
                "roi_id": int(roi["roi_id"]),
                "roi_name": roi["roi_name"],
                "parcel_name": roi.get("parcel_name", ""),
                "hemisphere": roi.get("hemisphere", ""),
                "network": roi.get("network", ""),
                "structure_class": roi.get("structure_class", ""),
            }
            for feature in FEATURES:
                rows.append({**base, "feature": feature, "value": float(features[feature][feature_col])})
        print(f"[{atlas} {i + 1}/{len(runs)}] extracted {row['subject_bids']} {row['session_bids']} {row['diagnosis']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(feature_path, index=False)
    return out


def combined_feature_path(out_dir: Path, atlases: list[str]) -> Path:
    if atlases == ["schaefer_400_7net"]:
        return out_dir / "adni_schaefer400_run_features.csv"
    return out_dir / "adni_multiatlas_run_features.csv"


def extract_features(
    runs: pd.DataFrame,
    out_dir: Path,
    atlases: list[str],
    atlas_root: Path,
    force: bool = False,
) -> pd.DataFrame:
    feature_path = combined_feature_path(out_dir, atlases)
    if atlases == ["schaefer_400_7net"] and feature_path.exists() and not force:
        return pd.read_csv(feature_path)
    parts = [
        extract_atlas_features(runs, atlas, atlas_root, out_dir, force=force)
        for atlas in atlases
    ]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(feature_path, index=False)
    return out


def ols_contrast(y: np.ndarray, diagnosis: pd.Series, covars: pd.DataFrame, case_label: str) -> tuple[float, float, float, int, int]:
    keep = diagnosis.isin([case_label, "CN"]).to_numpy()
    y = y[keep]
    dx = diagnosis[keep].to_numpy()
    cov = covars.loc[keep].copy()
    finite = np.isfinite(y)
    for col in cov.columns:
        cov[col] = pd.to_numeric(cov[col], errors="coerce")
        finite &= np.isfinite(cov[col].to_numpy(float))
    y = y[finite]
    dx = dx[finite]
    cov = cov.loc[finite]
    n_case = int(np.sum(dx == case_label))
    n_cn = int(np.sum(dx == "CN"))
    if n_case < 5 or n_cn < 5:
        return float("nan"), float("nan"), float("nan"), n_case, n_cn
    x_cols = [np.ones(len(y)), (dx == case_label).astype(float)]
    for col in cov.columns:
        vals = cov[col].to_numpy(float)
        std = np.std(vals)
        x_cols.append((vals - np.mean(vals)) / std if std > 0 else np.zeros_like(vals))
    x = np.column_stack(x_cols)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    if dof <= 0:
        return float("nan"), float("nan"), float("nan"), n_case, n_cn
    sigma2 = float((resid @ resid) / dof)
    cov_beta = sigma2 * np.linalg.pinv(x.T @ x)
    se = math.sqrt(max(cov_beta[1, 1], 0.0))
    t = float(beta[1] / se) if se > 0 else float("nan")
    p = float(2.0 * stats.t.sf(abs(t), dof)) if math.isfinite(t) else float("nan")
    pooled = np.sqrt((np.var(y[dx == case_label], ddof=1) + np.var(y[dx == "CN"], ddof=1)) / 2.0)
    d = float((np.mean(y[dx == case_label]) - np.mean(y[dx == "CN"])) / pooled) if pooled > 0 else float("nan")
    return float(beta[1]), p, d, n_case, n_cn


def run_statistics(features: pd.DataFrame, fd_thresh: float, fdr_alpha: float) -> pd.DataFrame:
    df = features.copy()
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["sex_female"] = df["PTGENDER"].astype(str).str.lower().eq("female").astype(float)
    df["site_code"] = pd.factorize(df["SITE"].astype(str))[0].astype(float)
    df["mean_fd"] = pd.to_numeric(df["mean_fd"], errors="coerce").fillna(df["mean_fd"].median())
    df = df[df["mean_fd"] <= fd_thresh].copy()
    covars = df[["AGE", "sex_female", "site_code", "mean_fd"]]
    rows = []
    grouped = df.groupby(["source", "roi_index", "roi_id", "roi_name", "feature"], sort=False)
    for key, sub in grouped:
        source, roi_index, roi_id, roi_name, feature = key
        y = pd.to_numeric(sub["value"], errors="coerce").to_numpy(float)
        cov = covars.loc[sub.index]
        for contrast, case_label in (("AD_vs_CN", "AD"), ("MCI_vs_CN", "MCI"), ("AD_or_MCI_vs_CN", "CASE")):
            if case_label == "CASE":
                dx = sub["diagnosis"].replace({"AD": "CASE", "MCI": "CASE"})
                label = "CASE"
            else:
                dx = sub["diagnosis"]
                label = case_label
            beta, p, d, n_case, n_cn = ols_contrast(y, dx, cov, label)
            rows.append(
                {
                    "source": source,
                    "roi_index": int(roi_index),
                    "roi_id": int(roi_id),
                    "roi_name": roi_name,
                    "feature": feature,
                    "contrast": contrast,
                    "beta_case_minus_cn": beta,
                    "p_value": p,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d) if math.isfinite(d) else np.nan,
                    "n_case": n_case,
                    "n_cn": n_cn,
                }
            )
    stats_df = pd.DataFrame(rows)
    stats_df["q_fdr_global"] = np.nan
    finite = np.isfinite(stats_df["p_value"].to_numpy(float))
    if finite.any():
        stats_df.loc[finite, "q_fdr_global"] = multipletests(stats_df.loc[finite, "p_value"], method="fdr_bh")[1]
    stats_df["q_fdr_contrast_feature"] = np.nan
    for (_contrast, _feature), idx in stats_df.groupby(["contrast", "feature"]).groups.items():
        sub_p = stats_df.loc[idx, "p_value"].to_numpy(float)
        sub_finite = np.isfinite(sub_p)
        if sub_finite.any():
            q = np.full(len(sub_p), np.nan, dtype=float)
            q[sub_finite] = multipletests(sub_p[sub_finite], method="fdr_bh")[1]
            stats_df.loc[idx, "q_fdr_contrast_feature"] = q
    stats_df["validated_global_fdr"] = (stats_df["q_fdr_global"] < fdr_alpha) & (stats_df["abs_cohens_d"] > 0.15)
    stats_df["validated_feature_fdr"] = (stats_df["q_fdr_contrast_feature"] < fdr_alpha) & (stats_df["abs_cohens_d"] > 0.15)
    stats_df["validated_nominal_p01"] = (stats_df["p_value"] < 0.01) & (stats_df["abs_cohens_d"] > 0.15)
    stats_df["adni_support_score"] = -np.log10(stats_df["p_value"].clip(lower=1e-300)) * stats_df["abs_cohens_d"].fillna(0.0)
    stats_df["validated"] = stats_df["validated_feature_fdr"]
    return stats_df


def _ranking_score_column(df: pd.DataFrame, method: str) -> str:
    if method == "neurodiscovery" and "score_neurodiscovery" in df.columns:
        return "score_neurodiscovery"
    return "rank"


def _region_feature_key(row: pd.Series) -> tuple[str, int, str]:
    return str(row["source"]), int(row["roi_index"]), str(row["feature"])


def validation_region_feature_keys(stats_df: pd.DataFrame) -> set[tuple[str, int, str]]:
    keys = stats_df[["source", "roi_index", "feature"]].drop_duplicates().copy()
    keys["roi_index"] = pd.to_numeric(keys["roi_index"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["roi_index"])
    return {
        (str(row.source), int(row.roi_index), str(row.feature))
        for row in keys.itertuples(index=False)
    }


def load_method_region_feature_rankings(
    method_dir: Path,
    top_n: int,
    atlases: list[str],
    executable_keys: set[tuple[str, int, str]] | None = None,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    allowed_sources = {f"{atlas}_multiatlas" for atlas in atlases}
    for method in METHODS:
        path = method_dir / f"ranked_candidates_{method}.csv"
        if not path.exists():
            continue
        selected = []
        seen: set[tuple[str, int, str]] = set()
        n_rows_read = 0
        n_metadata_testable = 0
        n_executable_rows_seen = 0
        n_duplicate_rows_skipped = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            n_rows_read += int(len(chunk))
            chunk = chunk[(chunk["modality"] == "fmri") & (chunk["source"].isin(allowed_sources))]
            chunk = chunk[chunk["feature"].isin(FEATURES)].copy()
            chunk["roi_index"] = pd.to_numeric(chunk["roi_index"], errors="coerce").astype("Int64")
            chunk = chunk.dropna(subset=["roi_index"])
            n_metadata_testable += int(len(chunk))
            if executable_keys is not None:
                mask = [
                    (str(row.source), int(row.roi_index), str(row.feature)) in executable_keys
                    for row in chunk.itertuples(index=False)
                ]
                chunk = chunk.loc[mask].copy()
            n_executable_rows_seen += int(len(chunk))
            if chunk.empty:
                continue
            score_col = _ranking_score_column(chunk, method)
            for row in chunk.itertuples(index=False):
                key = (str(row.source), int(row.roi_index), str(row.feature))
                if key in seen:
                    n_duplicate_rows_skipped += 1
                    continue
                seen.add(key)
                selected.append(
                    {
                        "source": key[0],
                        "roi_index": key[1],
                        "feature": key[2],
                        "method_rank": int(getattr(row, "rank")),
                        "method_score": getattr(row, score_col),
                        "executable_rank": len(selected) + 1,
                        "method": method,
                    }
                )
                if len(selected) >= top_n:
                    break
            if len(selected) >= top_n:
                break
        if selected:
            rows.append(pd.DataFrame(selected))
        audit_rows.append(
            {
                "method": method,
                "raw_rows_read_until_target": n_rows_read,
                "metadata_testable_rows_seen": n_metadata_testable,
                "executable_rows_seen": n_executable_rows_seen,
                "duplicate_executable_rows_skipped": n_duplicate_rows_skipped,
                "unique_executable_candidates_kept": len(selected),
                "target_unique_executable_candidates": top_n,
            }
        )
    rankings = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    return rankings, audit


def evaluate_rankings(stats_df: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = (
        stats_df.groupby(["source", "roi_index", "feature"], as_index=False)
        .agg(
            validated_any=("validated", "max"),
            validated_global_any=("validated_global_fdr", "max"),
            validated_nominal_any=("validated_nominal_p01", "max"),
            best_support_score=("adni_support_score", "max"),
            best_abs_d=("abs_cohens_d", "max"),
            best_q_global=("q_fdr_global", "min"),
            best_q_contrast_feature=("q_fdr_contrast_feature", "min"),
            best_contrast=("contrast", lambda s: ",".join(sorted(set(map(str, s))))),
        )
    )
    merged = rankings.merge(validation, how="inner", on=["source", "roi_index", "feature"])
    rows = []
    for method, sub in merged.groupby("method", sort=False):
        rank_col = "executable_rank" if "executable_rank" in sub.columns else "method_rank"
        sub = sub.sort_values(rank_col, kind="mergesort")
        labels = sub["validated_any"].astype(bool).to_numpy()
        total_validated = int(labels.sum())
        row = {
            "method": method,
            "n_executable_topk_pool": int(len(sub)),
            "n_validated_in_executable_pool": total_validated,
        }
        for k in (50, 100, 250, 500, 1000):
            kk = min(k, len(sub))
            row[f"validated_hits_at_{k}"] = int(labels[:kk].sum()) if kk else 0
            row[f"replication_at_{k}"] = float(labels[:kk].mean()) if kk else np.nan
            row[f"mean_support_at_{k}"] = float(sub["best_support_score"].head(kk).mean()) if kk else np.nan
        rows.append(row)
    return merged, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atlases = resolve_atlases(args.atlases)
    manifest = pd.read_csv(args.manifest)
    runs = first_testable_runs(manifest, args.max_subjects)
    runs.to_csv(args.out_dir / "adni_fmri_first_testable_runs.csv", index=False)
    if args.extract_only:
        for atlas in atlases:
            extract_atlas_features(runs, atlas, args.atlas_root, args.out_dir, force=args.force_extract)
        payload = {
            "n_subject_runs": int(len(runs)),
            "diagnosis_counts": runs["diagnosis"].value_counts().to_dict(),
            "atlases": atlases,
            "n_atlases": len(atlases),
            "feature_cache_dir": str(args.out_dir / "atlas_features"),
            "extract_only": True,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    stats_path = args.out_dir / ("adni_schaefer400_validation_stats.csv" if atlases == ["schaefer_400_7net"] else "adni_multiatlas_validation_stats.csv")
    if args.reuse_stats and stats_path.exists():
        stats_df = pd.read_csv(stats_path)
    else:
        features = extract_features(runs, args.out_dir, atlases, args.atlas_root, force=args.force_extract)
        features.to_csv(combined_feature_path(args.out_dir, atlases), index=False)
        stats_df = run_statistics(features, args.fd_thresh, args.fdr_alpha)
        stats_df.to_csv(stats_path, index=False)
    executable_keys = validation_region_feature_keys(stats_df)
    rankings, ranking_audit = load_method_region_feature_rankings(
        args.method_dir,
        args.top_n_per_method,
        atlases,
        executable_keys=executable_keys,
    )
    rankings_path = args.out_dir / ("case1_region_feature_rankings_schaefer400.csv" if atlases == ["schaefer_400_7net"] else "case1_region_feature_rankings_multiatlas.csv")
    rankings.to_csv(rankings_path, index=False)
    ranking_audit.to_csv(args.out_dir / "adni_case1_executable_ranking_audit.csv", index=False)
    merged, summary = evaluate_rankings(stats_df, rankings)
    merged.to_csv(args.out_dir / "adni_case1_ranking_validation_table.csv", index=False)
    summary.to_csv(args.out_dir / "adni_case1_ranking_validation_summary.csv", index=False)
    payload = {
        "n_subject_runs": int(len(runs)),
        "diagnosis_counts": runs["diagnosis"].value_counts().to_dict(),
        "atlases": atlases,
        "n_atlases": len(atlases),
        "fd_thresh": args.fd_thresh,
        "n_feature_tests": int(len(stats_df)),
        "n_validated_region_features": int(
            stats_df.groupby(["source", "roi_index", "feature"])["validated"].max().sum()
        ),
        "validation_definition": (
            "validated uses BH-FDR q<alpha within each contrast x feature family "
            "across the selected atlas-specific ROI-feature tests; global FDR and "
            "nominal p<0.01 are also exported."
        ),
        "ranking_definition": (
            "Top-k metrics are computed after filtering to executable unique "
            "source x roi_index x feature candidates. Non-executable rows caused "
            "by modality, atlas, feature, ROI, or dataset-metadata mismatch are "
            "skipped before top-k evaluation."
        ),
        "outputs": {
            "runs": str(args.out_dir / "adni_fmri_first_testable_runs.csv"),
            "features": str(combined_feature_path(args.out_dir, atlases)),
            "stats": str(stats_path),
            "ranking_summary": str(args.out_dir / "adni_case1_ranking_validation_summary.csv"),
            "ranking_audit": str(args.out_dir / "adni_case1_executable_ranking_audit.csv"),
        },
    }
    (args.out_dir / "adni_fmri_validation_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
