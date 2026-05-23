"""Robust ABIDE I CPAC ROI downloader.

nilearn.datasets.fetch_abide_pcp aborts on the first SSL hiccup. ABIDE has
~1100 subjects x 7 atlases, so even a 1% transient failure rate fails the
whole batch. This script uses the phenotype CSV's FILE_ID column to build
the S3 URLs directly and downloads with concurrent retry.

Files land at:
    data/abide/ABIDE_pcp/cpac/filt_noglobal/<atlas>/<FILE_ID>_<atlas>.1D

Idempotent: skips files that already exist with non-zero size.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data" / "abide" / "ABIDE_pcp"
PHENO = DATA_DIR / "Phenotypic_V1_0b_preprocessed1.csv"

S3_BASE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Outputs/cpac/filt_noglobal"
)

ATLASES = [
    "rois_aal",
    "rois_cc200",
    "rois_cc400",
    "rois_dosenbach160",
    "rois_ez",
    "rois_ho",
    "rois_tt",
]


def fetch_one(file_id: str, atlas: str, out_dir: Path,
              retries: int = 5, backoff: float = 2.0) -> tuple[str, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{file_id}_{atlas}.1D"
    dst = out_dir / fn
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", fn, "exists")
    url = f"{S3_BASE}/{atlas}/{fn}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 404:
                return ("404", fn, "not on S3")
            r.raise_for_status()
            tmp = dst.with_suffix(".1D.part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dst)
            return ("ok", fn, "")
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
    return ("fail", fn, f"{type(last_err).__name__}: {last_err}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+", default=ATLASES)
    p.add_argument("--quality-checked", action="store_true", default=True)
    p.add_argument(
        "--no-quality-checked", dest="quality_checked", action="store_false"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=5)
    args = p.parse_args()

    if not PHENO.exists():
        print(f"[fetch_abide] Missing phenotype CSV at {PHENO}; "
              "run nilearn fetch once to seed it.", file=sys.stderr)
        return 1

    pheno = pd.read_csv(PHENO)
    if args.quality_checked:
        for c in ("qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2",
                  "qc_anat_rater_3", "qc_func_rater_3"):
            if c in pheno.columns:
                pheno = pheno[pheno[c].fillna("OK").astype(str).str.upper() != "FAIL"]
    pheno = pheno[pheno["FILE_ID"].notna() & (pheno["FILE_ID"] != "no_filename")].copy()

    file_ids = pheno["FILE_ID"].tolist()
    print(f"[fetch_abide] N FILE_IDs: {len(file_ids)} | atlases: {args.atlases}")

    out_root = DATA_DIR / "cpac" / "filt_noglobal"
    tasks = [
        (fid, atlas, out_root / atlas)
        for atlas in args.atlases
        for fid in file_ids
    ]
    print(f"[fetch_abide] total files to verify/download: {len(tasks)}")

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_one, fid, atlas, dst, args.retries): (fid, atlas)
            for fid, atlas, dst in tasks
        }
        for i, fut in enumerate(as_completed(futs), 1):
            status, fn, msg = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "fail":
                failures.append(f"{fn}: {msg}")
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                rate = i / max(el, 1e-3)
                eta = (len(tasks) - i) / max(rate, 1e-3)
                print(f"[fetch_abide] {i}/{len(tasks)}  ok={counts['ok']}  "
                      f"skip={counts['skip']}  404={counts['404']}  "
                      f"fail={counts['fail']}  rate={rate:.1f}/s  "
                      f"eta={eta/60:.1f}min")

    dt = time.time() - t0
    print(f"\n[fetch_abide] done in {dt/60:.1f} min  | counts: {counts}")
    if failures:
        log = REPO / "runs" / "fetch_abide_failures.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(failures), encoding="utf-8")
        print(f"[fetch_abide] {len(failures)} failures logged to {log}")
        print("[fetch_abide] re-run to retry; downloads are idempotent.")
        return 1 if counts["ok"] == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
