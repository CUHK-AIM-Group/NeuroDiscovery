"""Run Case Study 1 full executable exhaustive search across prepared atlases.

This runner expands the v2 search from hp2000-only fMRI to every atlas already
available under ``transdiag_preprocessed``. It reports age/sex/site-adjusted
case-control effects for disease x atlas-specific ROI x feature hypotheses.
Bootstrap is optional and should usually be reserved for follow-up/top-hit runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from case1_exhaustive_v1 import (
    DEFAULT_SMRI_FEATURE_ROOT,
    DEFAULT_TRANS_DIAGNOSIS,
    add_fdr_columns,
    load_metadata,
    load_smri_source,
)
from case1_exhaustive_v2 import (
    build_covariates,
    evaluate_matrix_v2,
    summarize_shared_v2,
)


DEFAULT_TRANSDIAG_ROOT = Path(r"Z:\Public Dataset\transdiag_preprocessed")
DEFAULT_OUT_ROOT = Path(r"Z:\Public Dataset\case1_exhaustive_full")
DEFAULT_ATLAS_ROOT = Path(r"C:\Users\45846\Documents\Code\NeuroSTORM\datasets\atlas")

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
FULL_FMRI_FEATURES = ROI_FEATURES + CORR_FEATURES + PARTIAL_FEATURES


def subject_from_atlas_roi_file(path: Path, atlas: str) -> str:
    suffix = f"_{atlas}.npy"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def infer_target_shape(transdiag_root: Path) -> tuple[int, int, int] | None:
    img_root = transdiag_root / "img"
    if not img_root.exists():
        return None
    try:
        import torch

        data_pt = next(img_root.glob("*/data.pt"))
        blob = torch.load(data_pt, map_location="cpu", weights_only=False)
        frames = blob["frames"]
        if len(frames.shape) == 4:
            return tuple(int(v) for v in frames.shape[1:4])
    except Exception:
        return None
    return None


def _parse_overlap_field(raw: object) -> tuple[str, float] | None:
    text = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
    matches = re.findall(r'\["([^"]+)":\s*([0-9.]+)\]', text)
    for label, weight in matches:
        if label.strip().lower() == "none":
            continue
        return label.strip(), float(weight)
    return None


def _infer_hemisphere(name: str) -> str:
    low = name.lower()
    if low.startswith(("left ", "l_", "lh_", "7networks_lh_")) or low.endswith(("_l", " left")):
        return "left"
    if low.startswith(("right ", "r_", "rh_", "7networks_rh_")) or low.endswith(("_r", " right")):
        return "right"
    return ""


def _infer_network(atlas: str, name: str) -> str:
    if atlas.startswith("schaefer_"):
        match = re.search(r"7Networks_[LR]H_([^_]+)_", name)
        return match.group(1) if match else ""
    if atlas == "msdl_39":
        for key in ("DMN", "Aud", "Vis", "Motor", "DLPFC", "IPS", "ACC", "Ins", "TPJ", "Cereb"):
            if key.lower() in name.lower():
                return key
    return ""


def _infer_structure_class(name: str) -> str:
    low = name.lower()
    if any(key in low for key in ("cerebell", "cereb")):
        return "cerebellum"
    if any(key in low for key in ("thalam", "caudate", "putamen", "pallid", "accumbens", "amygdala", "hippocampus")):
        return "subcortical"
    if any(key in low for key in ("white matter", "ventricle", "csf", "brain-stem", "brain stem")):
        return "non-cortical"
    return "cortical"


def read_atlas_label_lookup(atlas: str, atlas_root: Path) -> dict[int, dict[str, object]]:
    label_path = atlas_root / atlas / "labels.csv"
    if not label_path.exists():
        return {}
    first_data_line = ""
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_data_line = stripped
            break
    if not first_data_line:
        return {}

    lookup: dict[int, dict[str, object]] = {}
    if first_data_line.lower().startswith("roi number"):
        df = pd.read_csv(label_path)
        df.columns = [str(col).strip() for col in df.columns]
        overlap_sources = ["Harvard-Oxford", "AAL", "Eickhoff-Zilles", "Talairach-Tournoux", "Dosenbach"]
        for _, row in df.iterrows():
            label_id = int(row["ROI number"])
            best_label = None
            best_source = ""
            best_weight = np.nan
            for source in overlap_sources:
                parsed = _parse_overlap_field(row.get(source))
                if parsed is not None:
                    best_label, best_weight = parsed
                    best_source = source
                    break
            label_name = best_label or f"{atlas}_label_{label_id}"
            lookup[label_id] = {
                "label_name": label_name,
                "anatomy_full": label_name,
                "label_source": best_source,
                "label_weight": best_weight,
                "center_of_mass": str(row.get("center of mass", "")).strip(),
            }
        return lookup

    rows: list[tuple[int, str]] = []
    with label_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        header = next(reader, None)
        if header is None:
            return {}
        header_l = [cell.strip().lower() for cell in header]
        if len(header_l) >= 2 and header_l[0] == "index" and header_l[1] == "name":
            data_iter = reader
        else:
            data_iter = [header, *reader]
        for row in data_iter:
            if len(row) < 2:
                continue
            try:
                rows.append((int(float(row[0])), str(row[1]).strip()))
            except ValueError:
                continue
    return {
        label_id: {
            "label_name": name,
            "anatomy_full": name,
            "label_source": "labels.csv",
            "label_weight": np.nan,
            "center_of_mass": "",
        }
        for label_id, name in rows
    }


def actual_atlas_label_ids(
    atlas: str,
    n_roi: int,
    atlas_root: Path,
    target_shape: tuple[int, int, int] | None,
) -> list[int]:
    atlas_path = atlas_root / atlas / "atlas.nii.gz"
    if not atlas_path.exists():
        return list(range(1, n_roi + 1))
    try:
        import nibabel as nib
        from scipy.ndimage import zoom

        data = nib.load(str(atlas_path)).get_fdata()
        if data.ndim == 4:
            max_prob = data.max(axis=3)
            labels = np.argmax(data, axis=3).astype(np.int32) + 1
            labels[max_prob <= 0] = 0
        else:
            labels = data.astype(np.int32)
        if target_shape is not None and labels.shape != target_shape:
            factors = [t / s for t, s in zip(target_shape, labels.shape, strict=True)]
            labels = zoom(labels, factors, order=0).astype(np.int32)
        ids = [int(v) for v in np.unique(labels) if int(v) > 0]
        if len(ids) == n_roi:
            return ids
    except Exception:
        pass
    return list(range(1, n_roi + 1))


def _resolve_label(
    label_id: int,
    label_lookup: dict[int, dict[str, object]],
    *,
    atlas: str,
    n_roi: int,
) -> dict[str, object]:
    shifted = label_id + 1
    first = label_lookup.get(1, {})
    first_name = str(first.get("label_name", "")).lower()
    if "background" in first_name and shifted in label_lookup:
        return label_lookup[shifted]
    if label_id in label_lookup:
        return label_lookup[label_id]
    return {
        "label_name": f"{atlas}_roi_{label_id}",
        "anatomy_full": f"{atlas} ROI {label_id}",
        "label_source": "",
        "label_weight": np.nan,
        "center_of_mass": "",
    }


def atlas_roi_meta(
    atlas: str,
    n_roi: int,
    atlas_root: Path = DEFAULT_ATLAS_ROOT,
    target_shape: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    label_lookup = read_atlas_label_lookup(atlas, atlas_root)
    label_ids = actual_atlas_label_ids(atlas, n_roi, atlas_root, target_shape)
    rows = []
    for idx, label_id in enumerate(label_ids):
        label = _resolve_label(label_id, label_lookup, atlas=atlas, n_roi=n_roi)
        label_name = str(label["label_name"])
        hemisphere = _infer_hemisphere(label_name)
        network = _infer_network(atlas, label_name)
        rows.append(
            {
                "roi_index": idx,
                "roi_id": label_id,
                "roi_name": f"{atlas}_{label_name}",
                "parcel_name": label_name,
                "anatomy_key": label_name,
                "anatomy_full": label.get("anatomy_full", label_name),
                "hemisphere": hemisphere,
                "network": network,
                "structure_class": _infer_structure_class(label_name),
                "atlas": atlas,
                "atlas_label_source": label.get("label_source", ""),
                "atlas_label_weight": label.get("label_weight", np.nan),
                "center_of_mass": label.get("center_of_mass", ""),
            }
        )
    return pd.DataFrame(rows)


def safe_row_nanmean(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def low_frequency_features(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coarse ALFF/fALFF-like ROI features from ROI time series.

    The TCP multi-atlas derivatives do not preserve full acquisition metadata in
    the ROI files, so this is an executable proxy over normalized frequency bins
    rather than a clinical ALFF/fALFF estimate.
    """
    centered = roi - np.nanmean(roi, axis=1, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    if spectrum.shape[1] <= 2:
        nan = np.full(roi.shape[0], np.nan)
        return nan, nan
    freqs = np.fft.rfftfreq(centered.shape[1])
    band = (freqs > 0.01) & (freqs <= 0.10)
    if not np.any(band):
        band = (freqs > 0) & (freqs <= np.quantile(freqs[freqs > 0], 0.20))
    alff = np.nanmean(spectrum[:, band], axis=1)
    total = np.nanmean(spectrum[:, freqs > 0], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = alff / total
    return alff, falff


def fc_node_degree_abs_top10(fc: np.ndarray) -> np.ndarray:
    abs_fc = np.abs(fc)
    finite = np.isfinite(abs_fc)
    if not finite.any():
        return np.full(fc.shape[0], np.nan)
    threshold = np.nanpercentile(abs_fc[finite], 90.0)
    return np.nansum(abs_fc >= threshold, axis=1).astype(float)


def fc_features(fc: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    fc = fc.astype(float, copy=True)
    np.fill_diagonal(fc, np.nan)
    out = {
        f"{prefix}_mean": safe_row_nanmean(fc),
        f"{prefix}_mean_abs": safe_row_nanmean(np.abs(fc)),
        f"{prefix}_positive_mean": safe_row_nanmean(np.where(fc > 0, fc, np.nan)),
        f"{prefix}_negative_mean": safe_row_nanmean(np.where(fc < 0, fc, np.nan)),
    }
    if prefix == "corr":
        out["corr_node_degree_abs_top10"] = fc_node_degree_abs_top10(fc)
    return out


def build_atlas_feature_matrices(
    transdiag_root: Path,
    atlas: str,
    requested_subjects: list[str] | None,
    atlas_root: Path = DEFAULT_ATLAS_ROOT,
    target_shape: tuple[int, int, int] | None = None,
) -> tuple[list[str], pd.DataFrame, dict[str, np.ndarray]]:
    roi_dir = transdiag_root / "roi" / atlas
    corr_dir = transdiag_root / "fc" / atlas / "correlation"
    pcorr_dir = transdiag_root / "fc" / atlas / "partial_correlation"
    roi_files = sorted(roi_dir.glob(f"*_{atlas}.npy"))
    subjects = [subject_from_atlas_roi_file(path, atlas) for path in roi_files]
    if requested_subjects is not None:
        keep = set(requested_subjects)
        pairs = [(subject, path) for subject, path in zip(subjects, roi_files, strict=True) if subject in keep]
        subjects = [subject for subject, _ in pairs]
        roi_files = [path for _, path in pairs]
    if not subjects:
        raise FileNotFoundError(f"No ROI files found for atlas={atlas} under {roi_dir}")

    first = np.load(roi_files[0], mmap_mode="r")
    n_roi = int(first.shape[0])
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in FULL_FMRI_FEATURES}
    kept_subjects: list[str] = []
    for subject, roi_file in zip(subjects, roi_files, strict=True):
        corr_file = corr_dir / f"{subject}_{atlas}_correlation.npy"
        pcorr_file = pcorr_dir / f"{subject}_{atlas}_partial_correlation.npy"
        if not corr_file.exists() or not pcorr_file.exists():
            continue
        roi = np.load(roi_file).astype(float)
        corr = np.load(corr_file).astype(float)
        pcorr = np.load(pcorr_file).astype(float)
        if roi.shape[0] != n_roi or corr.shape != (n_roi, n_roi) or pcorr.shape != (n_roi, n_roi):
            continue

        alff, falff = low_frequency_features(roi)
        roi_feature_values = {
            "roi_temporal_mean": np.nanmean(roi, axis=1),
            "roi_temporal_std": np.nanstd(roi, axis=1),
            "roi_temporal_variance": np.nanvar(roi, axis=1),
            "roi_temporal_mean_abs": np.nanmean(np.abs(roi), axis=1),
            "roi_alff_proxy": alff,
            "roi_falff_proxy": falff,
        }
        for name, values in roi_feature_values.items():
            arrays[name].append(values)
        for name, values in fc_features(corr, "corr").items():
            arrays[name].append(values)
        for name, values in fc_features(pcorr, "partial").items():
            arrays[name].append(values)
        kept_subjects.append(subject)

    return (
        kept_subjects,
        atlas_roi_meta(atlas, n_roi, atlas_root=atlas_root, target_shape=target_shape),
        {name: np.vstack(values).astype(float) for name, values in arrays.items()},
    )


def list_atlases(transdiag_root: Path, pattern: str) -> list[str]:
    atlases = [path.name for path in sorted((transdiag_root / "roi").iterdir()) if path.is_dir()]
    if pattern:
        regex = re.compile(pattern)
        atlases = [atlas for atlas in atlases if regex.search(atlas)]
    return atlases


def common_subjects_for_atlases(transdiag_root: Path, atlases: list[str]) -> list[str]:
    subject_sets = []
    for atlas in atlases:
        roi_dir = transdiag_root / "roi" / atlas
        subjects = {subject_from_atlas_roi_file(path, atlas) for path in roi_dir.glob(f"*_{atlas}.npy")}
        subject_sets.append(subjects)
    common = set.intersection(*subject_sets) if subject_sets else set()
    return sorted(common)


def add_full_fdr_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"adjusted_residual_d": "cohen_d_case_minus_control", "abs_adjusted_residual_d": "abs_d"})
    out = add_fdr_columns(renamed)
    return out.rename(columns={"cohen_d_case_minus_control": "adjusted_residual_d", "abs_d": "abs_adjusted_residual_d"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transdiag-root", type=Path, default=DEFAULT_TRANSDIAG_ROOT)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_TRANS_DIAGNOSIS)
    parser.add_argument("--smri-feature-root", type=Path, default=DEFAULT_SMRI_FEATURE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--min-cases", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--atlas-pattern", default="")
    parser.add_argument("--include-smri", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlases = list_atlases(args.transdiag_root, args.atlas_pattern)
    subjects = common_subjects_for_atlases(args.transdiag_root, atlases)
    target_shape = infer_target_shape(args.transdiag_root)
    meta, diseases = load_metadata(args.diagnosis, subjects, args.min_cases)
    covariates = build_covariates(meta)
    print(
        f"full_exhaustive subjects={len(subjects)} controls={int(meta['is_control'].sum())} "
        f"diseases={len(diseases)} atlases={len(atlases)} features={len(FULL_FMRI_FEATURES)} n_boot={args.n_boot}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    atlas_summary = []
    for atlas_idx, atlas in enumerate(atlases):
        print(f"[{atlas_idx + 1}/{len(atlases)}] atlas={atlas}", flush=True)
        atlas_subjects, roi_meta, feature_matrices = build_atlas_feature_matrices(
            args.transdiag_root,
            atlas,
            subjects,
            atlas_root=args.atlas_root,
            target_shape=target_shape,
        )
        atlas_summary.append({"atlas": atlas, "subjects": len(atlas_subjects), "n_roi": int(roi_meta.shape[0])})
        for feature_idx, (feature_name, matrix) in enumerate(feature_matrices.items()):
            rows.extend(
                evaluate_matrix_v2(
                    modality="fmri",
                    source=f"{atlas}_multiatlas",
                    feature_name=feature_name,
                    meta=meta,
                    diseases=diseases,
                    roi_meta=roi_meta,
                    matrix=matrix,
                    covariates=covariates,
                    n_boot=args.n_boot,
                    seed=args.seed + atlas_idx * 1_000_003 + feature_idx * 100_003,
                )
            )

    if args.include_smri:
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
                    seed=args.seed + (source_idx + 100) * 100_003,
                )
            )

    all_df = add_full_fdr_columns(pd.DataFrame(rows))
    all_df = all_df.sort_values(["q_fdr_global", "abs_adjusted_residual_d"], ascending=[True, False])

    stem = "case1_exhaustive_full"
    all_path = out_dir / f"{stem}_all_tests.csv"
    top_path = out_dir / f"{stem}_top_hits.csv"
    sig_global_path = out_dir / f"{stem}_significant_global_q05.csv"
    sig_modality_path = out_dir / f"{stem}_significant_modality_q05.csv"
    ci_path = out_dir / f"{stem}_bootstrap_ci_nonzero.csv"
    shared_path = out_dir / f"{stem}_shared_roi_feature_summary.csv"
    atlas_summary_path = out_dir / f"{stem}_atlas_summary.csv"

    all_df.to_csv(all_path, index=False)
    all_df.sort_values(["abs_adjusted_residual_d", "q_fdr_modality"], ascending=[False, True]).head(5000).to_csv(top_path, index=False)
    all_df[all_df["q_fdr_global"] < 0.05].to_csv(sig_global_path, index=False)
    all_df[all_df["q_fdr_modality"] < 0.05].to_csv(sig_modality_path, index=False)
    all_df[all_df["bootstrap_ci_excludes_zero"]].to_csv(ci_path, index=False)
    summarize_shared_v2(all_df).to_csv(shared_path, index=False)
    pd.DataFrame(atlas_summary).to_csv(atlas_summary_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "out_dir": str(out_dir),
        "transdiag_root": str(args.transdiag_root),
        "atlas_root": str(args.atlas_root),
        "target_shape": list(target_shape) if target_shape is not None else None,
        "diagnosis": str(args.diagnosis),
        "subjects": len(subjects),
        "controls": int(meta["is_control"].sum()),
        "diseases": diseases,
        "covariates": list(covariates.columns),
        "atlases": atlases,
        "n_atlases": len(atlases),
        "n_fmri_roi_total": int(sum(row["n_roi"] for row in atlas_summary)),
        "fmri_features": list(FULL_FMRI_FEATURES),
        "n_fmri_features": len(FULL_FMRI_FEATURES),
        "include_smri": bool(args.include_smri),
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
        "atlas_summary": str(atlas_summary_path),
        "elapsed_sec": round(time.perf_counter() - start, 3),
    }
    manifest_path = out_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
