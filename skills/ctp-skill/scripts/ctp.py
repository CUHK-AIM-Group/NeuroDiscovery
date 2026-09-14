"""Research CT perfusion: 4D HU series, cSVD maps, and supplied-map QA.

No clinical thresholds, automatic AIF selection, or scanner-specific DICOM
decoding. Read ../references/input-contract.md before using patient data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import circulant

from models.common.research_outputs import OutputBundle, read_json, write_json

UNITS = {"CBF": "mL/100g/min", "CBV": "mL/100g", "MTT": "s", "Tmax": "s"}


def load_image(path, ndim=None):
    img = nib.load(path)
    if ndim is not None and len(img.shape) != ndim:
        raise ValueError(f"Expected {ndim}D NIfTI: {path}")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-10:
        raise ValueError("Invalid NIfTI geometry")
    if img.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Spatial NIfTI units must explicitly be mm")
    return img, img.get_fdata(dtype=np.float32)


def same_grid(img, reference):
    if img.shape != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-5):
        raise ValueError("Shape/affine mismatch; register inputs before analysis")


def load_mask(path, reference, labels=False):
    img, data = load_image(path, 3)
    same_grid(img, reference)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Masks/atlas must contain finite nonnegative labels")
    if not np.equal(data, np.floor(data)).all():
        raise ValueError("Masks/atlas must contain integer labels")
    if not labels and not np.isin(data, [0, 1]).all():
        raise ValueError("Use a binary mask (0/1)")
    if not np.any(data > 0):
        raise ValueError("Empty mask/atlas")
    return data.astype(np.int32) if labels else data.astype(bool)


def save_image(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header.set_intent("none")
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    img.set_sform(reference.affine, code=int(reference.header["sform_code"]) or 1)
    nib.save(img, path)


def deconvolve(tissue, aif, dt, cutoff=0.2, density=1.04, hematocrit_factor=1.0,
               chunk_size=4096):
    """Return maps for delta-HU curves [voxel, time] on a uniform seconds grid.

    Rectangle-rule convolution with a 2T block-circulant matrix and relative
    truncated SVD. CBV uses measured finite-window area ratios, not clipped k.
    """
    tissue, aif = np.asarray(tissue, float), np.asarray(aif, float)
    if (tissue.ndim != 2 or aif.ndim != 1 or tissue.shape[1] != len(aif)
            or len(aif) < 6 or tissue.shape[0] == 0):
        raise ValueError("Expected nonempty [voxel,time] curves and matching AIF (>=6 frames)")
    controls = [dt, cutoff, density, hematocrit_factor]
    if not np.isfinite(controls).all() or not (dt > 0 and 0 < cutoff < 1 and density > 0
                                               and hematocrit_factor > 0 and chunk_size > 0):
        raise ValueError("Invalid deconvolution controls")
    if not np.isfinite(tissue).all() or not np.isfinite(aif).all():
        raise ValueError("Nonfinite perfusion curves")
    aif_area = float(trapezoid(aif, dx=dt))
    if aif.max() <= 0 or aif_area <= 0:
        raise ValueError("AIF must show positive enhancement and area")
    n = len(aif)
    matrix = circulant(np.r_[aif, np.zeros(n)]) * dt
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    retained = singular >= cutoff * singular[0]
    inverse = (vh[retained].T / singular[retained]) @ u[:, retained].T
    values = {key: np.full(len(tissue), np.nan) for key in UNITS}
    values["relative_residual"] = np.full(len(tissue), np.nan)
    scale = 100.0 * hematocrit_factor / density
    for start in range(0, len(tissue), chunk_size):
        stop = min(start + chunk_size, len(tissue))
        curves = tissue[start:stop]
        k = np.pad(curves, ((0, 0), (0, n))) @ inverse.T
        peak_index = k.argmax(axis=1)
        flow = k[np.arange(len(k)), peak_index]
        volume = trapezoid(curves, dx=dt, axis=1) / aif_area
        valid = (flow > 1e-10) & (volume > 0)
        cbf = scale * 60 * flow
        cbv = scale * volume
        # The second half represents negative lags, not a many-minute Tmax.
        lag = np.where(peak_index < n, peak_index, peak_index - 2 * n) * dt
        for key, data in (("CBF", cbf), ("CBV", cbv), ("Tmax", lag),
                          ("MTT", np.divide(60 * cbv, cbf, out=np.full_like(cbf, np.nan), where=valid))):
            values[key][start:stop] = np.where(valid, data, np.nan)
        reconstructed = (k @ matrix.T)[:, :n]
        values["relative_residual"][start:stop] = np.linalg.norm(reconstructed - curves, axis=1) / np.maximum(
            np.linalg.norm(curves, axis=1), 1e-12)
    return values, {"retained_singular_values": int(retained.sum()), "matrix_size": 2 * n}


def write_maps(bundle, maps, reference, mask, atlas=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    regions = [("brain", mask)]
    if atlas is not None:
        regions.extend((str(label), mask & (atlas == label)) for label in np.unique(atlas[mask]) if label > 0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    for ax, (name, unit) in zip(axes, UNITS.items()):
        data = maps[name].copy()
        data[~mask] = np.nan
        if not np.isfinite(data[mask]).any() or np.isinf(data[mask]).any():
            raise ValueError(f"No valid finite voxels, or infinite values, in {name}")
        save_image(bundle.root / f"{name}.nii.gz", data, reference)
        bundle.add(f"{name}.nii.gz", name, "nifti", shape=list(mask.shape),
                   affine=reference.affine.tolist(), finite=False, units=unit)
        for label, roi in regions:
            sample = data[roi]
            valid = sample[np.isfinite(sample)]
            rows.append({"region": label, "map": name, "units": unit,
                         "n_voxels": int(roi.sum()), "n_valid": len(valid),
                         "mean": float(valid.mean()) if len(valid) else None,
                         "median": float(np.median(valid)) if len(valid) else None,
                         "p05": float(np.percentile(valid, 5)) if len(valid) else None,
                         "p95": float(np.percentile(valid, 95)) if len(valid) else None})
        handle = ax.imshow(data[:, :, z].T, origin="lower", cmap="viridis")
        ax.set_title(f"{name} ({unit})")
        ax.set_xlabel("voxel i"); ax.set_ylabel("voxel j")
        fig.colorbar(handle, ax=ax, shrink=0.65)
    fig.suptitle(f"Research CTP maps — voxel slice k={z} (not radiological orientation)")
    fig.tight_layout()
    fig.savefig(bundle.root / "maps_qc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(bundle.root / "roi_summary.csv", index=False)
    bundle.add("roi_summary.csv", "roi_summary", "csv", columns=["region", "map", "units", "n_valid"])
    bundle.add("maps_qc.png", "map_figure")
    save_image(bundle.root / "analysis_mask.nii.gz", mask.astype(float), reference)
    bundle.add("analysis_mask.nii.gz", "analysis_mask", "nifti", shape=list(mask.shape), affine=reference.affine.tolist())


def compute(args):
    reference, hu = load_image(args.series, 4)
    metadata = read_json(args.metadata)
    if metadata.get("signal_units") != "HU" or metadata.get("motion_corrected") is not True:
        raise ValueError("Metadata must confirm HU and reviewed motion-corrected frames")
    times = np.asarray(metadata.get("frame_times_s", []), dtype=float)
    if (times.shape != (hu.shape[3],) or len(times) < 6 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("frame_times_s must be finite, strictly increasing and match the 4D series")
    baseline = metadata.get("baseline_frames")
    if (not isinstance(baseline, list) or len(baseline) < 2
            or any(type(i) is not int for i in baseline)
            or baseline != list(range(len(baseline))) or len(baseline) > len(times) - 3):
        raise ValueError("baseline_frames must be >=2 consecutive precontrast frames starting at 0")
    mask = load_mask(args.mask, reference)
    aif_mask = load_mask(args.aif_mask, reference)
    vof_mask = load_mask(args.vof_mask, reference) if args.vof_mask else None
    if vof_mask is not None and np.any(aif_mask & vof_mask):
        raise ValueError("AIF and VOF masks must not overlap")
    atlas = load_mask(args.atlas, reference, labels=True) if args.atlas else None
    all_used = mask | aif_mask | (vof_mask if vof_mask is not None else False)
    if not np.isfinite(hu[all_used]).all():
        raise ValueError("Nonfinite HU in analysis/AIF/VOF mask; repair acquisition inputs explicitly")
    tissue = hu[mask].astype(float)
    tissue -= tissue[:, baseline].mean(axis=1, keepdims=True)
    aif = hu[aif_mask].mean(axis=0).astype(float)
    aif -= aif[baseline].mean()
    vof = None
    if vof_mask is not None:
        vof = hu[vof_mask].mean(axis=0).astype(float)
        vof -= vof[baseline].mean()
    # Never extrapolate a bolus or silently replace acquisition timing.
    original_times = times.copy()
    if args.resample_dt is not None:
        if not np.isfinite(args.resample_dt) or args.resample_dt <= 0:
            raise ValueError("resample-dt must be positive seconds")
        times = np.arange(int(np.floor((times[-1] - times[0]) / args.resample_dt)) + 1) * args.resample_dt + times[0]
        tissue = interp1d(original_times, tissue, axis=1)(times)
        aif = np.interp(times, original_times, aif)
        if vof is not None:
            vof = np.interp(times, original_times, vof)
    if len(times) < 6 or not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-4, atol=1e-6):
        raise ValueError("Nonuniform timing: explicitly choose --resample-dt in seconds")
    aif_scale = 1.0
    if vof is not None:
        aa, va = trapezoid(aif, x=times), trapezoid(vof, x=times)
        if min(aa, va) <= 0:
            raise ValueError("AIF/VOF area must be positive")
        aif_scale = float(va / aa)
        aif *= aif_scale
    values, qc = deconvolve(tissue, aif, float(times[1] - times[0]), args.svd_cutoff,
                            args.density, args.hematocrit_factor, args.chunk_size)
    maps = {}
    for key in UNITS:
        maps[key] = np.full(mask.shape, np.nan, dtype=np.float32)
        maps[key][mask] = values[key]
    valid = np.isfinite(values["CBF"])
    qc.update({"analysis_voxels": int(mask.sum()), "valid_voxels": int(valid.sum()),
               "invalid_voxels": int((~valid).sum()), "aif_voxels": int(aif_mask.sum()),
               "aif_vof_scale": aif_scale, "mean_relative_residual": float(values["relative_residual"].mean()),
               "negative_tmax_voxels": int(np.sum(values["Tmax"] < 0)),
               "aif_tail_to_peak": float(aif[-1] / aif.max()),
               "tissue_tail_to_peak_median": float(np.median(tissue[:, -1] / np.maximum(tissue.max(axis=1), 1e-8))),
               "review_required": True, "warnings": [
                   "Review motion, masks, AIF/VOF, bolus coverage, residuals and negative lags before interpretation.",
                   "Finite-window CBV is biased by truncated bolus curves; SVD cutoff affects CBF and Tmax.",
                   "No leakage correction, clinical segmentation thresholds or diagnostic validation."]})
    sources = [args.series, args.metadata, args.mask, args.aif_mask]
    sources += [p for p in (args.vof_mask, args.atlas) if p]
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config["metadata"] = metadata
    bundle = OutputBundle(args.output_dir, "ctp-csvd", config, sources)
    write_maps(bundle, maps, reference, mask, atlas)
    curves = pd.DataFrame({"time_s": times, "aif_delta_hu": aif, "mean_tissue_delta_hu": tissue.mean(axis=0)})
    if vof is not None:
        curves["vof_delta_hu"] = vof
    curves.to_csv(bundle.root / "curves.csv", index=False)
    import matplotlib.pyplot as plt
    ax = curves.plot(x="time_s", ylabel="delta HU", title="Research CTP: review AIF/VOF and bolus coverage")
    ax.figure.savefig(bundle.root / "curves_qc.png", dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    write_json(bundle.root / "qc.json", qc)
    bundle.add("curves.csv", "curves", "csv", columns=["time_s", "aif_delta_hu"])
    bundle.add("curves_qc.png", "curve_figure")
    bundle.add("qc.json", "qc", "json")
    return bundle.finish([*UNITS, "analysis_mask", "roi_summary", "map_figure", "curves", "curve_figure", "qc"])


def summarize(args):
    metadata = read_json(args.metadata)
    if metadata.get("units") != UNITS:
        raise ValueError(f"Provide explicit canonical map units: {UNITS}; no implicit vendor scaling")
    paths = {key: getattr(args, key.lower()) for key in UNITS}
    reference, first = load_image(paths["CBF"], 3)
    maps = {"CBF": first}
    for key in ("CBV", "MTT", "Tmax"):
        img, maps[key] = load_image(paths[key], 3)
        same_grid(img, reference)
    mask = load_mask(args.mask, reference)
    atlas = load_mask(args.atlas, reference, labels=True) if args.atlas else None
    for name in ("CBF", "CBV", "MTT"):
        if np.any(maps[name][mask] < 0):
            raise ValueError(f"Negative {name}; resolve vendor missing-value conventions first")
    qc = {"review_required": True, "source": "supplied maps; not recomputed",
          "negative_tmax_voxels": int(np.sum(maps["Tmax"][mask] < 0)),
          "valid_voxels": {k: int(np.isfinite(v[mask]).sum()) for k, v in maps.items()}}
    config = {"metadata": metadata, "method": "supplied-map-summary"}
    bundle = OutputBundle(args.output_dir, "ctp-supplied-maps", config,
                          [*paths.values(), args.metadata, args.mask] + ([args.atlas] if args.atlas else []))
    write_maps(bundle, maps, reference, mask, atlas)
    write_json(bundle.root / "qc.json", qc)
    bundle.add("qc.json", "qc", "json")
    return bundle.finish([*UNITS, "analysis_mask", "roi_summary", "map_figure", "qc"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    raw = modes.add_parser("compute", help="Compute cSVD maps from motion-corrected 4D HU")
    raw.add_argument("--series", required=True, type=Path)
    raw.add_argument("--aif-mask", required=True, type=Path)
    raw.add_argument("--vof-mask", type=Path)
    raw.add_argument("--resample-dt", type=float)
    raw.add_argument("--svd-cutoff", type=float, default=0.2)
    raw.add_argument("--density", type=float, default=1.04, help="g/mL")
    raw.add_argument("--hematocrit-factor", type=float, default=1.0, help="Explicit common tissue scaling; 1 = none")
    raw.add_argument("--chunk-size", type=int, default=4096)
    raw.set_defaults(func=compute)
    supplied = modes.add_parser("summarize", help="Validate and summarize supplied CBF/CBV/MTT/Tmax maps")
    for name in UNITS:
        supplied.add_argument(f"--{name.lower()}", required=True, type=Path)
    supplied.set_defaults(func=summarize)
    for command in (raw, supplied):
        command.add_argument("--metadata", required=True, type=Path)
        command.add_argument("--mask", required=True, type=Path)
        command.add_argument("--atlas", type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(args.func(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
