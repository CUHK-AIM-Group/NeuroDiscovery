"""Run Case Study 1 exhaustive v1 on prepared TCP fMRI and sMRI features.

This v1 search is intentionally limited to data that are already available:

* fMRI: TCP hp2000_rest ROI time series and FC correlation matrices.
* sMRI: FastSurfer segmentation-only regional volume fractions.

The unit of testing is ``disease x ROI x feature``. Outputs include all tests,
top hits, and ROI-feature summaries across diseases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_TCP_ROOT = Path(r"Z:\Public Dataset\tcp_preprocessed")
DEFAULT_TRANS_DIAGNOSIS = Path(r"Z:\Public Dataset\transdiag_preprocessed\metadata\diagnosis.csv")
DEFAULT_SMRI_FEATURE_ROOT = Path(r"Z:\Public Dataset\tcp_fastsurfer_segonly\features")
DEFAULT_HP2000_MAPPING = Path(r"Z:\Public Dataset\tcp_h5_hp2000\metadata\hp2000_roi_mapping.csv")
DEFAULT_OUT_ROOT = Path(r"Z:\Public Dataset\case1_exhaustive_v1")


FMRI_FEATURES = (
    "roi_temporal_std",
    "roi_temporal_mean_abs",
    "fc_mean_abs",
    "fc_positive_mean",
    "fc_negative_mean",
)


def split_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {tok.strip() for tok in re.split(r"[|;,]", str(value)) if tok.strip()}


def subject_from_roi_path(path: Path) -> str:
    suffix = "_hp2000_rest.npy"
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def load_metadata(path: Path, subjects: Iterable[str], min_cases: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    meta = pd.read_csv(path)
    meta["subjectkey"] = meta["subjectkey"].astype(str)
    subjects = list(subjects)
    meta = meta[meta["subjectkey"].isin(subjects)].copy()
    meta = meta.set_index("subjectkey", drop=False).loc[subjects].reset_index(drop=True)
    meta["is_control"] = (
        meta.get("is_genpop", "").astype(str).isin({"1", "True", "true"})
        | (meta.get("Group", "").astype(str).str.casefold() == "genpop")
    )
    token_counts: dict[str, int] = {}
    for value in meta["diagnosis_broad_any"].fillna(""):
        for token in split_tokens(value):
            token_counts[token] = token_counts.get(token, 0) + 1
    diseases = [
        {"disease": token, "n_case": count}
        for token, count in sorted(token_counts.items())
        if count >= min_cases
    ]
    return meta, diseases


def load_hp2000_mapping(path: Path, n_roi: int) -> pd.DataFrame:
    if path.exists():
        mapping = pd.read_csv(path)
    else:
        mapping = pd.DataFrame({"h5_index": range(n_roi)})
    mapping["h5_index"] = mapping["h5_index"].astype(int)
    mapping = mapping.set_index("h5_index", drop=False)
    rows = []
    for idx in range(n_roi):
        if idx in mapping.index:
            row = mapping.loc[idx].to_dict()
        else:
            row = {"h5_index": idx}
        row.setdefault("roi_id", idx + 1)
        row.setdefault("hemisphere", "")
        row.setdefault("network", "")
        row.setdefault("anatomy_key", "")
        row.setdefault("anatomy_full", "")
        row.setdefault("parcel_name", f"hp2000_roi_{idx + 1}")
        rows.append(row)
    return pd.DataFrame(rows)


def build_fmri_features(tcp_root: Path, mapping_path: Path) -> tuple[list[str], pd.DataFrame, dict[str, np.ndarray]]:
    roi_dir = tcp_root / "roi" / "hp2000_rest"
    fc_dir = tcp_root / "fc" / "hp2000_rest" / "correlation"
    roi_files = sorted(roi_dir.glob("*_hp2000_rest.npy"))
    subjects = [subject_from_roi_path(path) for path in roi_files]
    if not subjects:
        raise FileNotFoundError(f"No hp2000_rest ROI files found under {roi_dir}")

    first = np.load(roi_files[0])
    n_roi = int(first.shape[0])
    roi_meta = load_hp2000_mapping(mapping_path, n_roi)

    arrays = {name: [] for name in FMRI_FEATURES}
    kept_subjects: list[str] = []
    for subject, roi_file in zip(subjects, roi_files, strict=True):
        fc_file = fc_dir / f"{subject}_hp2000_rest_correlation.npy"
        if not fc_file.exists():
            continue
        roi = np.load(roi_file)
        fc = np.load(fc_file).astype(float)
        np.fill_diagonal(fc, np.nan)
        arrays["roi_temporal_std"].append(np.nanstd(roi, axis=1))
        arrays["roi_temporal_mean_abs"].append(np.nanmean(np.abs(roi), axis=1))
        arrays["fc_mean_abs"].append(np.nanmean(np.abs(fc), axis=1))
        arrays["fc_positive_mean"].append(np.nanmean(np.where(fc > 0, fc, np.nan), axis=1))
        arrays["fc_negative_mean"].append(np.nanmean(np.where(fc < 0, fc, np.nan), axis=1))
        kept_subjects.append(subject)

    feature_arrays = {name: np.vstack(values).astype(float) for name, values in arrays.items()}
    return kept_subjects, roi_meta, feature_arrays


def parse_smri_feature_column(column: str) -> tuple[int, str]:
    match = re.match(r"vol_frac_total__(\d+)__(.+)$", column)
    if not match:
        return -1, column
    return int(match.group(1)), match.group(2)


def load_smri_source(
    feature_root: Path,
    source: str,
    subjects: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    wide_path = feature_root / f"fastsurfer_{source}_volume_normalized_wide.csv"
    catalog_path = feature_root / f"fastsurfer_{source}_label_catalog.csv"
    wide = pd.read_csv(wide_path)
    wide["subject"] = wide["subject"].astype(str)
    wide = wide.set_index("subject", drop=False).loc[subjects].reset_index(drop=True)
    catalog = pd.read_csv(catalog_path)
    catalog = catalog.set_index("label_id", drop=False)

    columns = [c for c in wide.columns if c != "subject"]
    roi_rows = []
    for idx, column in enumerate(columns):
        label_id, fallback = parse_smri_feature_column(column)
        if label_id in catalog.index:
            cat = catalog.loc[label_id].to_dict()
            roi_name = str(cat.get("label_name") or fallback)
            hemisphere = str(cat.get("hemisphere") or "")
            structure_class = str(cat.get("structure_class") or "")
        else:
            roi_name = fallback
            hemisphere = ""
            structure_class = ""
        roi_rows.append(
            {
                "roi_index": idx,
                "roi_id": label_id,
                "roi_name": roi_name,
                "hemisphere": hemisphere,
                "network": "",
                "anatomy_key": "",
                "anatomy_full": roi_name,
                "parcel_name": roi_name,
                "structure_class": structure_class,
            }
        )
    return pd.DataFrame(roi_rows), wide[columns].to_numpy(float)


def cohen_d(case: np.ndarray, control: np.ndarray) -> np.ndarray:
    n1, n0 = case.shape[0], control.shape[0]
    mean1 = np.nanmean(case, axis=0)
    mean0 = np.nanmean(control, axis=0)
    var1 = np.nanvar(case, axis=0, ddof=1)
    var0 = np.nanvar(control, axis=0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / max(n1 + n0 - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (mean1 - mean0) / pooled
    return out


def rank_auc(case: np.ndarray, control: np.ndarray) -> np.ndarray:
    values = np.vstack([case, control])
    n1 = case.shape[0]
    n0 = control.shape[0]
    ranks = stats.rankdata(values, axis=0, nan_policy="omit")
    rank_sum_case = np.nansum(ranks[:n1], axis=0)
    auc = (rank_sum_case - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return auc


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[order] = adjusted
    return q


def disease_masks(meta: pd.DataFrame, disease: str) -> tuple[np.ndarray, np.ndarray]:
    case = meta["diagnosis_broad_any"].fillna("").apply(lambda value: disease in split_tokens(value)).to_numpy(bool)
    control = meta["is_control"].to_numpy(bool)
    return case, control


def evaluate_matrix(
    *,
    modality: str,
    source: str,
    feature_name: str,
    subjects: list[str],
    meta: pd.DataFrame,
    diseases: list[dict[str, object]],
    roi_meta: pd.DataFrame,
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for disease_row in diseases:
        disease = str(disease_row["disease"])
        case_mask, control_mask = disease_masks(meta, disease)
        case = matrix[case_mask]
        control = matrix[control_mask]
        if case.shape[0] < 2 or control.shape[0] < 2:
            continue

        d = cohen_d(case, control)
        auc = rank_auc(case, control)
        t_stat, p_value = stats.ttest_ind(case, control, axis=0, equal_var=False, nan_policy="omit")
        mean_case = np.nanmean(case, axis=0)
        mean_control = np.nanmean(control, axis=0)
        sd_case = np.nanstd(case, axis=0, ddof=1)
        sd_control = np.nanstd(control, axis=0, ddof=1)

        for idx in range(matrix.shape[1]):
            roi = roi_meta.iloc[idx].to_dict()
            d_val = float(d[idx]) if np.isfinite(d[idx]) else float("nan")
            auc_val = float(auc[idx]) if np.isfinite(auc[idx]) else float("nan")
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
                    "n_case": int(case.shape[0]),
                    "n_control": int(control.shape[0]),
                    "mean_case": float(mean_case[idx]),
                    "mean_control": float(mean_control[idx]),
                    "sd_case": float(sd_case[idx]),
                    "sd_control": float(sd_control[idx]),
                    "cohen_d_case_minus_control": d_val,
                    "abs_d": abs(d_val) if math.isfinite(d_val) else float("nan"),
                    "auc_case_higher": auc_val,
                    "auc_separation": max(auc_val, 1.0 - auc_val) if math.isfinite(auc_val) else float("nan"),
                    "welch_t": float(t_stat[idx]) if np.isfinite(t_stat[idx]) else float("nan"),
                    "p_value": float(p_value[idx]) if np.isfinite(p_value[idx]) else float("nan"),
                    "direction": "case_higher" if d_val > 0 else "case_lower" if d_val < 0 else "flat",
                }
            )
    return rows


def add_fdr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q_fdr_global"] = benjamini_hochberg(out["p_value"].to_numpy(float))
    out["q_fdr_modality"] = np.nan
    out["q_fdr_disease"] = np.nan
    for _, idx in out.groupby("modality").groups.items():
        out.loc[idx, "q_fdr_modality"] = benjamini_hochberg(out.loc[idx, "p_value"].to_numpy(float))
    for _, idx in out.groupby("disease").groups.items():
        out.loc[idx, "q_fdr_disease"] = benjamini_hochberg(out.loc[idx, "p_value"].to_numpy(float))
    return out


def summarize_shared(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["modality", "source", "feature", "roi_name", "anatomy_full", "hemisphere", "network"], dropna=False):
        effects = g["cohen_d_case_minus_control"].to_numpy(float)
        finite = effects[np.isfinite(effects)]
        if finite.size == 0:
            continue
        pos = int(np.sum(finite > 0))
        neg = int(np.sum(finite < 0))
        dominant_sign = "case_higher" if pos >= neg else "case_lower"
        same_sign = max(pos, neg)
        sig_modality = g[np.isfinite(g["q_fdr_modality"]) & (g["q_fdr_modality"] < 0.05)]
        rows.append(
            {
                "modality": keys[0],
                "source": keys[1],
                "feature": keys[2],
                "roi_name": keys[3],
                "anatomy_full": keys[4],
                "hemisphere": keys[5],
                "network": keys[6],
                "n_diseases_tested": int(g["disease"].nunique()),
                "dominant_direction": dominant_sign,
                "direction_consistency": float(same_sign / finite.size),
                "mean_d": float(np.nanmean(finite)),
                "mean_abs_d": float(np.nanmean(np.abs(finite))),
                "max_abs_d": float(np.nanmax(np.abs(finite))),
                "n_q05_modality": int(sig_modality["disease"].nunique()),
                "best_q_modality": float(np.nanmin(g["q_fdr_modality"])),
                "best_p": float(np.nanmin(g["p_value"])),
                "diseases_q05_modality": "|".join(sorted(sig_modality["disease"].unique().tolist())),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_q05_modality", "mean_abs_d", "direction_consistency"],
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
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    start = time.perf_counter()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building fMRI feature matrices...", flush=True)
    subjects, hp2000_roi_meta, fmri_arrays = build_fmri_features(args.tcp_root, args.hp2000_mapping)
    meta, diseases = load_metadata(args.diagnosis, subjects, args.min_cases)
    print(f"subjects={len(subjects)} diseases={len(diseases)} controls={int(meta['is_control'].sum())}", flush=True)

    rows: list[dict[str, object]] = []
    for feature_name, matrix in fmri_arrays.items():
        rows.extend(
            evaluate_matrix(
                modality="fmri",
                source="hp2000_rest_correlation",
                feature_name=feature_name,
                subjects=subjects,
                meta=meta,
                diseases=diseases,
                roi_meta=hp2000_roi_meta,
                matrix=matrix,
            )
        )

    for source in ("aparc_dkt_aseg", "aseg"):
        roi_meta, matrix = load_smri_source(args.smri_feature_root, source, subjects)
        rows.extend(
            evaluate_matrix(
                modality="smri",
                source=source,
                feature_name="normalized_volume_fraction",
                subjects=subjects,
                meta=meta,
                diseases=diseases,
                roi_meta=roi_meta,
                matrix=matrix,
            )
        )

    all_df = add_fdr_columns(pd.DataFrame(rows))
    all_df = all_df.sort_values(["q_fdr_global", "abs_d"], ascending=[True, False])
    all_path = out_dir / "case1_exhaustive_v1_all_tests.csv"
    all_df.to_csv(all_path, index=False)

    top_df = all_df.sort_values(["abs_d", "q_fdr_modality"], ascending=[False, True]).head(500)
    top_path = out_dir / "case1_exhaustive_v1_top_hits.csv"
    top_df.to_csv(top_path, index=False)

    sig_global_path = out_dir / "case1_exhaustive_v1_significant_global_q05.csv"
    all_df[all_df["q_fdr_global"] < 0.05].to_csv(sig_global_path, index=False)

    sig_modality_path = out_dir / "case1_exhaustive_v1_significant_modality_q05.csv"
    all_df[all_df["q_fdr_modality"] < 0.05].to_csv(sig_modality_path, index=False)

    shared_df = summarize_shared(all_df)
    shared_path = out_dir / "case1_exhaustive_v1_shared_roi_feature_summary.csv"
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
        "fmri_features": list(fmri_arrays.keys()),
        "smri_sources": ["aparc_dkt_aseg", "aseg"],
        "n_tests": int(all_df.shape[0]),
        "n_q05_global": int((all_df["q_fdr_global"] < 0.05).sum()),
        "n_q05_modality": int((all_df["q_fdr_modality"] < 0.05).sum()),
        "all_tests": str(all_path),
        "top_hits": str(top_path),
        "significant_global_q05": str(sig_global_path),
        "significant_modality_q05": str(sig_modality_path),
        "shared_summary": str(shared_path),
        "elapsed_sec": round(time.perf_counter() - start, 3),
    }
    manifest_path = out_dir / "case1_exhaustive_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
