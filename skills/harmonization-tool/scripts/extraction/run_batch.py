"""Build subject task list for ABIDE I and ABIDE II + run a streaming batch.

ABIDE I: takes FILE_IDs from
  data/abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv
URL pattern (CPAC filt_noglobal func_preproc):
  https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/
    Outputs/cpac/filt_noglobal/func_preproc/<FILE_ID>_func_preproc.nii.gz

ABIDE II: enumerates fmriprep subjects from S3 prefix
  data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/sub-XXXXX/
We pick the BOLD MNI 4D (preproc_bold in MNI152NLin2009cAsym, *.nii.gz, not the
gifti surface). Subjects without a usable BOLD MNI are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from worker import (SubjectTask, WorkerConfig, make_masker_cache,
                    process_subject)

REPO = Path(__file__).resolve().parents[4]
ABIDE1_PHENO = REPO / "data" / "abide" / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"

S3_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects"
ABIDE1_FUNC_PREFIX = (
    f"{S3_BASE}/ABIDE_Initiative/Outputs/cpac/filt_noglobal/func_preproc"
)
ABIDE2_S3_LIST = (
    "https://fcp-indi.s3.amazonaws.com/?prefix="
    "data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/&max-keys=1000"
)


def build_abide1_tasks(quality_checked: bool = True) -> list[SubjectTask]:
    pheno = pd.read_csv(ABIDE1_PHENO)
    if quality_checked:
        for c in ("qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2",
                  "qc_anat_rater_3", "qc_func_rater_3"):
            if c in pheno.columns:
                pheno = pheno[
                    pheno[c].fillna("OK").astype(str).str.upper() != "FAIL"
                ]
    pheno = pheno[pheno["FILE_ID"].notna() & (pheno["FILE_ID"] != "no_filename")]
    tasks = []
    for _, r in pheno.iterrows():
        fid = str(r["FILE_ID"])
        url = f"{ABIDE1_FUNC_PREFIX}/{fid}_func_preproc.nii.gz"
        tasks.append(SubjectTask(
            subject_id=f"abide1-{int(r['SUB_ID'])}",
            raw_url_or_path=url,
            dataset_tag="abide1",
        ))
    return tasks


def _s3_list(prefix: str, max_keys: int = 1000) -> list[str]:
    """List all keys under prefix using continuation."""
    keys = []
    marker = ""
    while True:
        url = (f"https://fcp-indi.s3.amazonaws.com/?prefix={prefix}"
               f"&max-keys={max_keys}")
        if marker:
            url += f"&marker={marker}"
        with urllib.request.urlopen(url, timeout=60) as r:
            xml = r.read().decode("utf-8")
        page_keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        keys.extend(page_keys)
        truncated = "<IsTruncated>true</IsTruncated>" in xml
        if not truncated or not page_keys:
            break
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        if m:
            marker = m.group(1)
        else:
            marker = page_keys[-1]
    return keys


def build_abide2_tasks() -> list[SubjectTask]:
    """Find ABIDE II BOLD MNI .nii.gz under fmriprep output."""
    keys = _s3_list("data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/")
    tasks = []
    seen = set()
    for k in keys:
        if "MNI152NLin2009cAsym" not in k:
            continue
        if not k.endswith("preproc_bold.nii.gz"):
            continue
        m = re.search(r"sub-(\d+)", k)
        if not m:
            continue
        sid = f"abide2-{m.group(1)}"
        if sid in seen:
            continue
        seen.add(sid)
        url = f"https://fcp-indi.s3.amazonaws.com/{k}"
        tasks.append(SubjectTask(sid, url, "abide2"))
    return tasks


def _run_one(task_dict: dict, cfg_dict: dict) -> dict:
    cfg = WorkerConfig(**{k: (Path(v) if k in ("out_root", "raw_tmp") and v else v)
                          for k, v in cfg_dict.items()})
    cache = make_masker_cache()
    task = SubjectTask(**task_dict)
    return process_subject(task, cfg, cache)


def run_batch(tasks: list[SubjectTask], cfg: WorkerConfig,
              num_workers: int = 4, limit: int = 0,
              progress_every: int = 5) -> dict:
    if limit > 0:
        tasks = tasks[:limit]
    out = {"ok": 0, "skip": 0, "fail": 0, "elapsed_s": 0.0}
    t0 = time.time()
    cfg_dict = {
        "out_root": str(cfg.out_root),
        "target_voxel_size": cfg.target_voxel_size,
        "quantize": cfg.quantize, "save_q8": cfg.save_q8,
        "save_roi": cfg.save_roi, "save_fc": cfg.save_fc,
        "fill_zeroback": cfg.fill_zeroback, "overwrite": cfg.overwrite,
        "raw_tmp": str(cfg.raw_tmp) if cfg.raw_tmp else None,
    }

    if num_workers <= 1:
        for i, task in enumerate(tasks, 1):
            r = _run_one(asdict(task), cfg_dict)
            status = r.get("status", "fail")
            out[status] = out.get(status, 0) + 1
            if i % progress_every == 0 or i == len(tasks):
                rate = i / max(1e-9, time.time() - t0)
                print(f"[batch] {i}/{len(tasks)}  ok={out['ok']} skip={out['skip']} "
                      f"fail={out['fail']}  rate={rate:.2f}/s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_run_one, asdict(t), cfg_dict): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"subject_id": futures[fut].subject_id,
                         "status": "fail", "reason": f"{type(e).__name__}: {e}"}
                status = r.get("status", "fail")
                out[status] = out.get(status, 0) + 1
                if i % progress_every == 0 or i == len(tasks):
                    rate = i / max(1e-9, time.time() - t0)
                    print(f"[batch] {i}/{len(tasks)}  ok={out['ok']} skip={out['skip']} "
                          f"fail={out['fail']}  rate={rate:.2f}/s", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["abide1", "abide2"], required=True)
    ap.add_argument("--out", required=True, help="output root, e.g. Z:\\Public Dataset\\abide")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all subjects; otherwise process first N")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-q8", action="store_true")
    ap.add_argument("--no-roi", action="store_true")
    ap.add_argument("--no-fc", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list-only", action="store_true",
                    help="just print first 10 task URLs and exit")
    args = ap.parse_args()

    if args.dataset == "abide1":
        tasks = build_abide1_tasks()
    else:
        tasks = build_abide2_tasks()

    print(f"[batch] dataset={args.dataset}  N_tasks={len(tasks)}", flush=True)
    if args.list_only:
        for t in tasks[:10]:
            print(f"  {t.subject_id}  {t.raw_url_or_path}")
        return 0

    cfg = WorkerConfig(
        out_root=Path(args.out),
        save_q8=not args.no_q8, save_roi=not args.no_roi,
        save_fc=not args.no_fc, overwrite=args.overwrite,
    )
    summary = run_batch(tasks, cfg, num_workers=args.workers, limit=args.limit)
    print(f"[batch] done: {summary}", flush=True)

    log_path = cfg.out_root / "batch_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0 if summary.get("fail", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
