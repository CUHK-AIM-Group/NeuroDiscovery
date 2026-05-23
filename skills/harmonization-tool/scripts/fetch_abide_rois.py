"""Fetch ABIDE I CPAC ROI time series for all 7 parcellations.

Downloads to data/abide/ via nilearn.datasets.fetch_abide_pcp. Idempotent:
nilearn caches per file, so re-running picks up where it left off.

CPAC pipeline + quality_checked=True keeps the subjects that passed every
rater's QC (~870 / 1112). band_pass_filtering=True and
global_signal_regression=False match the most-cited ABIDE preprocessing
choice in the literature; flip the flags here if you need a different recipe.
"""
from __future__ import annotations

from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data" / "abide"

ROI_DERIVATIVES = [
    "rois_aal",
    "rois_cc200",
    "rois_cc400",
    "rois_dosenbach160",
    "rois_ez",
    "rois_ho",
    "rois_tt",
]


def main() -> int:
    from nilearn.datasets import fetch_abide_pcp

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_abide] target dir: {DATA_DIR}")
    print(f"[fetch_abide] derivatives: {ROI_DERIVATIVES}")
    t0 = time.time()
    bunch = fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=ROI_DERIVATIVES,
        quality_checked=True,
        verbose=1,
    )
    dt = time.time() - t0
    print(f"\n[fetch_abide] done in {dt/60:.1f} min")
    pheno = bunch["phenotypic"]
    print(f"[fetch_abide] N subjects: {len(pheno)}")
    for d in ROI_DERIVATIVES:
        files = bunch.get(d)
        if files is None:
            print(f"  {d}: not in bunch")
        else:
            print(f"  {d}: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
