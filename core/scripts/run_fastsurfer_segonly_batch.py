"""Run FastSurferVINN segmentation-only for a directory of T1w images.

This is a generic version of the TCP batch runner. It discovers T1w NIfTI files,
uses the closest ``sub-*`` path component as subject id, and writes FastSurfer
outputs under ``<output-root>/<subject>/mri``. The run is resumable.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FASTSURFER_ROOT = Path(r"C:\Users\45846\Documents\Code\FastSurfer")
DEFAULT_FS_LICENSE = Path(r"Z:\Public Dataset\tcp_preprocessed\metadata\licenses\freesurfer_license.txt")

EXPECTED_OUTPUTS = (
    Path("mri/aparc.DKTatlas+aseg.deep.mgz"),
    Path("mri/aseg.auto_noCCseg.mgz"),
    Path("mri/mask.mgz"),
)


def subject_from_t1(path: Path) -> str:
    for part in reversed(path.parent.parts):
        if part.startswith("sub-"):
            return part
    name = path.name
    for suffix in ("_T1w.nii.gz", "_T1w.nii", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_t1s(t1_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for path in sorted(t1_root.rglob("*_T1w.nii.gz")):
        rows.append((subject_from_t1(path), path))
    for path in sorted(t1_root.rglob("*_T1w.nii")):
        rows.append((subject_from_t1(path), path))
    return rows


def subject_complete(output_root: Path, subject: str) -> bool:
    subject_dir = output_root / subject
    return all((subject_dir / rel).is_file() for rel in EXPECTED_OUTPUTS)


def run_subject(args: argparse.Namespace, subject: str, t1_path: Path) -> dict[str, object]:
    output_root = Path(args.output_root)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if subject_complete(output_root, subject) and not args.overwrite:
        return {
            "subject": subject,
            "t1_path": str(t1_path),
            "status": "skip_exists",
            "returncode": 0,
            "elapsed_sec": 0.0,
        }

    run_prediction = Path(args.fastsurfer_root) / "FastSurferCNN" / "run_prediction.py"
    log_path = log_dir / f"{subject}_seg.log"
    cmd = [
        sys.executable,
        str(run_prediction),
        "--t1",
        str(t1_path),
        "--sid",
        subject,
        "--sd",
        str(output_root),
        "--seg_log",
        str(log_path),
        "--device",
        args.device,
        "--viewagg_device",
        args.viewagg_device,
        "--batch_size",
        str(args.batch_size),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.fastsurfer_root)
    env["FS_LICENSE"] = str(args.fs_license)

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(args.fastsurfer_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    status = "ok" if proc.returncode == 0 and subject_complete(output_root, subject) else "error"
    tail_path = log_dir / f"{subject}_console_tail.txt"
    tail_path.write_text(proc.stdout[-12000:], encoding="utf-8", errors="replace")
    return {
        "subject": subject,
        "t1_path": str(t1_path),
        "status": status,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "log_path": str(log_path),
        "console_tail": str(tail_path),
    }


def append_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "subject",
        "status",
        "returncode",
        "elapsed_sec",
        "t1_path",
        "log_path",
        "console_tail",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fastsurfer-root", type=Path, default=DEFAULT_FASTSURFER_ROOT)
    parser.add_argument("--fs-license", type=Path, default=DEFAULT_FS_LICENSE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewagg-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = discover_t1s(args.t1_root)
    if args.limit > 0:
        subjects = subjects[: args.limit]
    if not subjects:
        raise SystemExit(f"No T1w files found under {args.t1_root}")
    if not args.fs_license.exists():
        raise SystemExit(f"Missing FreeSurfer license: {args.fs_license}")

    manifest = Path(args.output_root) / "metadata" / "segmentation_manifest.csv"
    total = len(subjects)
    done = errors = skipped = 0
    print(f"FastSurfer segmentation-only subjects={total} output={args.output_root}", flush=True)
    for idx, (subject, t1_path) in enumerate(subjects, 1):
        row = run_subject(args, subject, t1_path)
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_manifest(manifest, row)
        status = row["status"]
        if status == "ok":
            done += 1
        elif status == "skip_exists":
            skipped += 1
        else:
            errors += 1
        print(
            f"{idx}/{total} {subject} {status} elapsed={row.get('elapsed_sec')}s "
            f"done={done} skipped={skipped} errors={errors}",
            flush=True,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
