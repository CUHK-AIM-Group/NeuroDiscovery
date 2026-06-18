"""Extract FastSurfer segmentation volumes into modeling-ready tables.

Inputs are FastSurferVINN segmentation-only outputs:

    <segmentation-root>/<subject>/mri/aparc.DKTatlas+aseg.deep.mgz
    <segmentation-root>/<subject>/mri/aseg.auto_noCCseg.mgz
    <segmentation-root>/<subject>/mri/mask.mgz

Outputs are CSV feature tables under ``features/`` by default. The raw volume
tables are intentionally simple: one row per subject in wide format, plus one
row per subject-label in long format for analysis and plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_SEGMENTATION_ROOT = Path(r"Z:\Public Dataset\tcp_fastsurfer_segonly")
DEFAULT_LUT = Path(r"C:\Users\45846\Documents\Code\FastSurfer\FastSurferCNN\config\FastSurfer_ColorLUT.tsv")

SEGMENTATION_IMAGES = {
    "aparc_dkt_aseg": "aparc.DKTatlas+aseg.deep.mgz",
    "aseg": "aseg.auto_noCCseg.mgz",
}


@dataclass(frozen=True)
class LabelInfo:
    label_id: int
    label_name: str
    hemisphere: str
    structure_class: str


def safe_feature_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def infer_hemisphere(label_id: int, label_name: str) -> str:
    name = label_name.lower()
    if 1000 <= label_id < 2000 or name.startswith(("left-", "ctx-lh-", "lh-")):
        return "left"
    if 2000 <= label_id < 3000 or name.startswith(("right-", "ctx-rh-", "rh-")):
        return "right"
    return "midline_or_bilateral"


def infer_structure_class(label_id: int, label_name: str) -> str:
    name = label_name.lower()
    if label_id == 0 or name in {"background", "unknown"}:
        return "background"
    if 1000 <= label_id < 3000 or name.startswith(("ctx-lh-", "ctx-rh-")):
        return "cortical_dkt"
    if "white-matter" in name or name.endswith("-wm") or "wm-" in name:
        return "white_matter"
    if "ventricle" in name or name in {"csf", "3rd-ventricle", "4th-ventricle"}:
        return "csf_ventricle"
    if any(token in name for token in ("hippocampus", "amygdala", "thalamus", "caudate", "putamen", "pallidum", "accumbens")):
        return "subcortical_gray"
    if "cerebellum" in name:
        return "cerebellum"
    if "brain-stem" in name or "brainstem" in name:
        return "brainstem"
    return "other"


def load_lut(path: Path) -> dict[int, LabelInfo]:
    labels: dict[int, LabelInfo] = {}
    if not path.exists():
        return labels

    if path.suffix.lower() == ".tsv":
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    label_id = int(row["ID"])
                except (KeyError, TypeError, ValueError):
                    continue
                label_name = row.get("LabelName") or f"label_{label_id}"
                labels[label_id] = LabelInfo(
                    label_id=label_id,
                    label_name=label_name,
                    hemisphere=infer_hemisphere(label_id, label_name),
                    structure_class=infer_structure_class(label_id, label_name),
                )
        return labels

    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                label_id = int(parts[0])
            except ValueError:
                continue
            label_name = parts[1]
            labels[label_id] = LabelInfo(
                label_id=label_id,
                label_name=label_name,
                hemisphere=infer_hemisphere(label_id, label_name),
                structure_class=infer_structure_class(label_id, label_name),
            )
    return labels


def subject_dirs(root: Path, subject_prefix: str | None = None) -> list[Path]:
    excluded = {"features", "logs", "metadata", "__pycache__"}
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in excluded]
    if subject_prefix:
        dirs = [p for p in dirs if p.name.startswith(subject_prefix)]
    return sorted(dirs)


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(abs(zooms[0] * zooms[1] * zooms[2]))


def mask_volume_mm3(subject_dir: Path) -> float:
    path = subject_dir / "mri" / "mask.mgz"
    if not path.exists():
        return float("nan")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return float(np.count_nonzero(data) * voxel_volume_mm3(img))


def label_info(lut: dict[int, LabelInfo], label_id: int) -> LabelInfo:
    if label_id in lut:
        return lut[label_id]
    name = f"label_{label_id}"
    return LabelInfo(
        label_id=label_id,
        label_name=name,
        hemisphere=infer_hemisphere(label_id, name),
        structure_class=infer_structure_class(label_id, name),
    )


def extract_one_image(
    subject_dir: Path,
    image_key: str,
    image_name: str,
    lut: dict[int, LabelInfo],
    include_background: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    subject = subject_dir.name
    path = subject_dir / "mri" / image_name
    if not path.exists():
        raise FileNotFoundError(path)

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    labels, counts = np.unique(data, return_counts=True)
    vv = voxel_volume_mm3(img)
    label_count = dict(zip((int(x) for x in labels), (int(x) for x in counts), strict=True))
    nonzero_voxels = sum(count for label, count in label_count.items() if label != 0)
    nonzero_volume = float(nonzero_voxels * vv)
    brain_mask_volume = mask_volume_mm3(subject_dir)

    rows: list[dict[str, object]] = []
    for label_id, count in sorted(label_count.items()):
        if label_id == 0 and not include_background:
            continue
        info = label_info(lut, label_id)
        volume = float(count * vv)
        rows.append(
            {
                "subject": subject,
                "source_image": image_key,
                "label_id": label_id,
                "label_name": info.label_name,
                "hemisphere": info.hemisphere,
                "structure_class": info.structure_class,
                "voxel_count": count,
                "voxel_volume_mm3": vv,
                "volume_mm3": volume,
                "volume_ml": volume / 1000.0,
                "total_nonzero_volume_mm3": nonzero_volume,
                "brain_mask_volume_mm3": brain_mask_volume,
                "volume_frac_total_nonzero": volume / nonzero_volume if nonzero_volume > 0 else float("nan"),
                "volume_frac_brain_mask": volume / brain_mask_volume if brain_mask_volume > 0 else float("nan"),
            }
        )

    summary = {
        "subject": subject,
        "source_image": image_key,
        "shape": "x".join(str(int(x)) for x in data.shape),
        "voxel_volume_mm3": vv,
        "n_labels_including_background": len(label_count),
        "n_nonzero_labels": len([label for label in label_count if label != 0]),
        "total_nonzero_voxels": nonzero_voxels,
        "total_nonzero_volume_mm3": nonzero_volume,
        "brain_mask_volume_mm3": brain_mask_volume,
    }
    return rows, summary


def write_tables(rows: list[dict[str, object]], output_dir: Path, image_key: str) -> dict[str, object]:
    long_path = output_dir / f"fastsurfer_{image_key}_volume_long.csv"
    wide_path = output_dir / f"fastsurfer_{image_key}_volume_wide.csv"
    norm_wide_path = output_dir / f"fastsurfer_{image_key}_volume_normalized_wide.csv"
    label_catalog_path = output_dir / f"fastsurfer_{image_key}_label_catalog.csv"

    long_df = pd.DataFrame(rows)
    long_df.to_csv(long_path, index=False)
    (
        long_df[["source_image", "label_id", "label_name", "hemisphere", "structure_class"]]
        .drop_duplicates()
        .sort_values(["label_id", "label_name"])
        .to_csv(label_catalog_path, index=False)
    )

    labels = (
        long_df[["label_id", "label_name"]]
        .drop_duplicates()
        .sort_values("label_id")
        .itertuples(index=False)
    )
    feature_names = {
        int(label_id): f"vol_mm3__{int(label_id):04d}__{safe_feature_name(str(label_name))}"
        for label_id, label_name in labels
    }
    norm_feature_names = {
        label_id: name.replace("vol_mm3__", "vol_frac_total__")
        for label_id, name in feature_names.items()
    }

    wide = long_df.pivot_table(index="subject", columns="label_id", values="volume_mm3", fill_value=0.0, aggfunc="sum")
    wide = wide.rename(columns=feature_names).reset_index()
    wide.to_csv(wide_path, index=False)

    norm_wide = long_df.pivot_table(
        index="subject",
        columns="label_id",
        values="volume_frac_total_nonzero",
        fill_value=0.0,
        aggfunc="sum",
    )
    norm_wide = norm_wide.rename(columns=norm_feature_names).reset_index()
    norm_wide.to_csv(norm_wide_path, index=False)

    return {
        "image_key": image_key,
        "long_path": str(long_path),
        "wide_path": str(wide_path),
        "normalized_wide_path": str(norm_wide_path),
        "label_catalog_path": str(label_catalog_path),
        "subjects": int(wide.shape[0]),
        "labels": int(wide.shape[1] - 1),
        "long_rows": int(long_df.shape[0]),
    }


def selected_images(names: Iterable[str]) -> dict[str, str]:
    if not names:
        return dict(SEGMENTATION_IMAGES)
    selected: dict[str, str] = {}
    for name in names:
        if name not in SEGMENTATION_IMAGES:
            raise ValueError(f"Unknown image key {name!r}; choose from {sorted(SEGMENTATION_IMAGES)}")
        selected[name] = SEGMENTATION_IMAGES[name]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT)
    parser.add_argument("--images", nargs="*", default=list(SEGMENTATION_IMAGES), help="Image keys to extract")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument(
        "--subject-prefix",
        default="",
        help="Optional subject directory prefix filter, e.g. NDAR_INV or sub-.",
    )
    args = parser.parse_args()

    segmentation_root = Path(args.segmentation_root)
    output_dir = Path(args.output_dir) if args.output_dir else segmentation_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    lut = load_lut(args.lut)
    subjects = subject_dirs(segmentation_root, args.subject_prefix or None)
    images = selected_images(args.images)
    if not subjects:
        prefix_note = f" with prefix {args.subject_prefix!r}" if args.subject_prefix else ""
        raise SystemExit(f"No subject directories{prefix_note} found under {segmentation_root}")

    all_summaries: list[dict[str, object]] = []
    table_reports: list[dict[str, object]] = []
    for image_key, image_name in images.items():
        rows: list[dict[str, object]] = []
        for subject_dir in subjects:
            subject_rows, summary = extract_one_image(
                subject_dir=subject_dir,
                image_key=image_key,
                image_name=image_name,
                lut=lut,
                include_background=args.include_background,
            )
            rows.extend(subject_rows)
            all_summaries.append(summary)
        table_reports.append(write_tables(rows, output_dir, image_key))

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_dir / "fastsurfer_subject_volume_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segmentation_root": str(segmentation_root),
        "output_dir": str(output_dir),
        "lut": str(args.lut),
        "subjects": len(subjects),
        "subject_prefix": args.subject_prefix,
        "images": images,
        "include_background": bool(args.include_background),
        "summary_path": str(summary_path),
        "tables": table_reports,
    }
    manifest_path = output_dir / "fastsurfer_volume_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
