"""Rigidly align 4D CT frames in physical space; human motion QA is still required."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ctp import OutputBundle, load_image, read_json, save_image, write_json


def sitk_image(data, affine):
    import SimpleITK as sitk
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    lps = np.diag([-1., -1., 1.])
    direction = lps @ affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sheared affine: explicitly resample onto an orthogonal grid first")
    img = sitk.GetImageFromArray(np.asarray(data, np.float32).transpose(2, 1, 0))
    img.SetSpacing(tuple(spacing))
    img.SetDirection(tuple(direction.ravel()))
    img.SetOrigin(tuple(lps @ affine[:3, 3]))
    return img


def main(argv=None):
    import SimpleITK as sitk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-displacement-mm", type=float, default=5.0,
                        help="Research QC flag, not a validated clinical exclusion threshold")
    args = parser.parse_args(argv)
    reference, hu = load_image(args.series, 4)
    meta = read_json(args.metadata)
    baseline = meta.get("baseline_frames")
    if (meta.get("signal_units") != "HU" or not isinstance(baseline, list) or len(baseline) < 2
            or any(type(i) is not int for i in baseline) or baseline != list(range(len(baseline)))
            or len(baseline) > hu.shape[3] - 3):
        raise ValueError("Provide HU metadata and at least two initial precontrast baseline frames")
    times = np.asarray(meta.get("frame_times_s", []), dtype=float)
    if (times.shape != (hu.shape[3],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid frame_times_s")
    if not np.isfinite(hu).all() or min(hu.shape[:3]) < 4:
        raise ValueError("Registration requires finite CT and at least four voxels along each axis")
    if (args.iterations < 1 or not np.isfinite(args.max_displacement_mm) or args.max_displacement_mm <= 0):
        raise ValueError("Invalid registration controls")
    # Register to one actual precontrast frame, avoiding blur in an unaligned mean.
    fixed = sitk_image(hu[..., baseline[0]], reference.affine)
    corrected = np.empty_like(hu)
    rows, transforms = [], []
    corners = [fixed.TransformIndexToPhysicalPoint((i, j, k))
               for i in (0, hu.shape[0] - 1) for j in (0, hu.shape[1] - 1) for k in (0, hu.shape[2] - 1)]
    for frame in range(hu.shape[3]):
        moving = sitk_image(hu[..., frame], reference.affine)
        transform = sitk.Euler3DTransform()
        transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize()) - 1) / 2)))
        if frame == baseline[0]:
            metric, stop = 0.0, "reference frame (identity)"
        else:
            registration = sitk.ImageRegistrationMethod()
            registration.SetNumberOfThreads(1)
            registration.SetMetricAsMattesMutualInformation(32)
            registration.SetMetricSamplingStrategy(registration.NONE)
            registration.SetInterpolator(sitk.sitkLinear)
            registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, args.iterations)
            registration.SetOptimizerScalesFromPhysicalShift()
            factors = [2, 1] if min(hu.shape[:3]) >= 8 else [1]
            registration.SetShrinkFactorsPerLevel(factors)
            registration.SetSmoothingSigmasPerLevel([1., 0.] if len(factors) == 2 else [0.])
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration.SetInitialTransform(transform, inPlace=True)
            registration.Execute(fixed, moving)
            metric, stop = float(registration.GetMetricValue()), registration.GetOptimizerStopConditionDescription()
            if not np.isfinite(metric):
                raise ValueError(f"Nonfinite registration metric at frame {frame}")
        resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, float("nan"), sitk.sitkFloat32)
        corrected[..., frame] = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)
        max_shift = max(np.linalg.norm(np.asarray(transform.TransformPoint(p)) - p) for p in corners)
        rows.append({"frame": frame, "time_s": float(times[frame]), "metric": metric,
                     "max_corner_displacement_mm": float(max_shift),
                     "large_motion_flag": bool(max_shift > args.max_displacement_mm), "optimizer_stop": stop})
        transforms.append(transform)
    coverage = np.isfinite(corrected).all(axis=3)
    if not coverage.any():
        raise ValueError("No common finite field of view after registration")
    bundle = OutputBundle(args.output_dir, "ctp-rigid-motion", {
        "reference_frame": baseline[0], "iterations": args.iterations,
        "max_displacement_mm": args.max_displacement_mm, "coordinates": "LPS mm",
        "transform_direction": "fixed/reference to moving frame", "review_required": True,
    }, [args.series, args.metadata])
    save_image(bundle.root / "motion_corrected.nii.gz", corrected, reference)
    save_image(bundle.root / "coverage_mask.nii.gz", coverage.astype(float), reference)
    pd.DataFrame(rows).to_csv(bundle.root / "motion.csv", index=False)
    for frame, transform in enumerate(transforms):
        name = f"frame_{frame:04d}.tfm"
        sitk.WriteTransform(transform, str(bundle.root / name))
        bundle.add(name, "motion_transform")
    meta.update({"motion_correction_performed": True, "motion_corrected": False,
                 "motion_review_required": True,
                 "review_instruction": "Inspect registered frames/motion.csv; intersect brain/AIF/VOF masks with coverage. Set motion_corrected=true only in a separately saved reviewed metadata file."})
    write_json(bundle.root / "metadata_for_review.json", meta)
    bundle.add("motion_corrected.nii.gz", "motion_series", "nifti", shape=list(hu.shape),
               affine=reference.affine.tolist(), finite=False)
    bundle.add("coverage_mask.nii.gz", "coverage_mask", "nifti", shape=list(coverage.shape), affine=reference.affine.tolist())
    bundle.add("motion.csv", "motion_qc", "csv", rows=hu.shape[3], columns=["frame", "max_corner_displacement_mm"])
    bundle.add("metadata_for_review.json", "metadata", "json")
    print(bundle.finish(["motion_series", "coverage_mask", "motion_qc", "motion_transform", "metadata"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
