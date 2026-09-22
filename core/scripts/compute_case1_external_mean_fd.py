"""Compute mean FD for Case Study 1 external cohorts.

The local external-validation images are already spatially preprocessed, so
motion must not be estimated from those derivatives. For UCLA, the runner
first reads the official fMRIPrep framewise-displacement confounds distributed
with OpenNeuro ds000030. If those are unavailable, it downloads one raw BOLD
run at a time, estimates six rigid-body parameters with AFNI ``3dvolreg`` under
WSL, records motion QC, and removes the temporary image.

Supported public sources:

* ADHD-200 raw BIDS data in the public FCP-INDI S3 bucket.
* UCLA CNP raw BIDS data (OpenNeuro ds000030).

COBRE and HCP-EP require authenticated source-data access and are intentionally
not approximated from their already motion-corrected local derivatives.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FCP_BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
FCP_ADHD_BIDS_PREFIX = "data/Projects/ADHD200/RawDataBIDS"
ADHD_SITES = (
    "Brown",
    "KKI",
    "NYU",
    "NeuroIMAGE",
    "OHSU",
    "Peking_1",
    "Peking_2",
    "Peking_3",
    "Pittsburgh",
    "WashU",
)
UCLA_OPENNEURO_PREFIX = (
    "https://s3.amazonaws.com/openneuro.org/ds000030"
)
UCLA_DERIVATIVE_ZIP_PREFIX = (
    "http://s3.amazonaws.com/openneuro/ds000030/"
    "ds000030_R1.0.4/compressed"
)
UCLA_DERIVATIVE_ZIP_RANGES = (
    "10159-10299",
    "10304-10388",
    "10428-10575",
    "10624-10697",
    "10704-10893",
    "10912-10998",
    "11019-11098",
    "11104-11156",
    "50004-50008",
    "50010-50038",
    "50043-50059",
    "50060-50085",
    "60001-60008",
    "60010-60038",
    "60042-60068",
    "60070-60089",
    "70001-70008",
    "70010-70049",
    "70051-70069",
    "70070-70086",
)
AFNI_DEFAULT = "$HOME/.local/afni/linux_ubuntu_24_64/3dvolreg"
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset"
    r"\case1_external_validation_v1\motion_qc"
)
DEFAULT_METADATA_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset"
    r"\case1_external_validation_v1\metadata"
)
DEFAULT_INPUT_ROOTS = {
    "adhd200": Path(r"\\192.168.3.61\data\Dataset\fMRI\adhd200"),
    "ucla": Path(r"\\192.168.3.61\data\Dataset\fMRI\UCLA"),
}
DEFAULT_METADATA = {
    "adhd200": DEFAULT_METADATA_ROOT / "adhd200_diagnosis.csv",
    "ucla": DEFAULT_METADATA_ROOT / "ucla_diagnosis.csv",
}


@dataclass(frozen=True)
class RawRun:
    dataset: str
    subjectkey: str
    session: str
    run: str
    source_site: str
    url: str

    @property
    def run_id(self) -> str:
        return (
            f"{self.dataset}:{self.subjectkey}:"
            f"ses-{self.session}:run-{self.run}"
        )


def power_fd(
    motion: np.ndarray,
    *,
    head_radius_mm: float = 50.0,
) -> np.ndarray:
    """Return Power FD from AFNI roll/pitch/yaw and dS/dL/dP parameters."""

    values = np.asarray(motion, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected [T, 6] motion parameters, got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("At least two frames are required to compute FD")
    delta = np.diff(values, axis=0)
    rotation_mm = (
        np.abs(np.deg2rad(delta[:, :3])).sum(axis=1) * head_radius_mm
    )
    translation_mm = np.abs(delta[:, 3:]).sum(axis=1)
    return rotation_mm + translation_mm


def summarize_motion(
    motion: np.ndarray,
    *,
    head_radius_mm: float,
) -> dict[str, float | int]:
    fd = power_fd(motion, head_radius_mm=head_radius_mm)
    return {
        "n_frames": int(np.asarray(motion).shape[0]),
        "n_fd_intervals": int(fd.size),
        "mean_fd": float(np.mean(fd)),
        "median_fd": float(np.median(fd)),
        "max_fd": float(np.max(fd)),
        "pct_fd_gt_0_2": float(np.mean(fd > 0.2) * 100.0),
        "pct_fd_gt_0_5": float(np.mean(fd > 0.5) * 100.0),
    }


def summarize_framewise_displacement(
    values: pd.Series,
    *,
    n_frames: int,
) -> dict[str, float | int]:
    """Summarize an existing FD column while excluding undefined first frames."""

    fd = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if fd.size == 0:
        raise ValueError("FramewiseDisplacement contains no finite values")
    return {
        "n_frames": int(n_frames),
        "n_fd_intervals": int(fd.size),
        "mean_fd": float(np.mean(fd)),
        "median_fd": float(np.median(fd)),
        "max_fd": float(np.max(fd)),
        "pct_fd_gt_0_2": float(np.mean(fd > 0.2) * 100.0),
        "pct_fd_gt_0_5": float(np.mean(fd > 0.5) * 100.0),
    }


def fetch_ucla_published_confounds(
    allowed_run_ids: set[str],
    *,
    retries: int = 6,
) -> list[dict[str, object]]:
    """Read official UCLA fMRIPrep FD columns using remote ZIP byte ranges."""

    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            "UCLA published-confound extraction requires `pip install remotezip`"
        ) from exc

    rows: list[dict[str, object]] = []
    pattern = re.compile(r"/sub-(\d+)_task-rest_bold_confounds\.tsv$")
    for zip_index, subject_range in enumerate(
        UCLA_DERIVATIVE_ZIP_RANGES,
        start=1,
    ):
        url = (
            f"{UCLA_DERIVATIVE_ZIP_PREFIX}/"
            f"ds000030_R1.0.4_derivatives_sub{subject_range}.zip"
        )
        error: Exception | None = None
        archive_rows: list[dict[str, object]] = []
        for attempt in range(retries):
            try:
                with RemoteZip(url) as archive:
                    names = [
                        name
                        for name in archive.namelist()
                        if pattern.search(name)
                    ]
                    for name in names:
                        subject_match = pattern.search(name)
                        if subject_match is None:
                            continue
                        subject = f"sub-{subject_match.group(1)}"
                        run_id = f"ucla:{subject}:ses-1:run-1"
                        if run_id not in allowed_run_ids:
                            continue
                        confounds = pd.read_csv(
                            io.BytesIO(archive.read(name)),
                            sep="\t",
                        )
                        if "FramewiseDisplacement" not in confounds:
                            raise ValueError(
                                f"{name} lacks FramewiseDisplacement"
                            )
                        archive_rows.append(
                            {
                                "dataset": "ucla",
                                "subjectkey": subject,
                                "session": "1",
                                "run": "1",
                                "source_site": (
                                    "OpenNeuro-ds000030-fMRIPrep-0.4.4"
                                ),
                                "url": f"{url}#{name}",
                                "run_id": run_id,
                                "status": "ok",
                                **summarize_framewise_displacement(
                                    confounds["FramewiseDisplacement"],
                                    n_frames=len(confounds),
                                ),
                                "motion_source": (
                                    "Official fMRIPrep 0.4.4 confounds; "
                                    "FramewiseDisplacement"
                                ),
                                "elapsed_sec": 0.0,
                                "error": "",
                            }
                        )
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = exc
                archive_rows = []
                time.sleep(min(2**attempt, 20))
        if error is not None:
            raise RuntimeError(f"Failed to read {url}: {error}") from error
        rows.extend(archive_rows)
        print(
            f"[published {zip_index}/{len(UCLA_DERIVATIVE_ZIP_RANGES)}] "
            f"sub{subject_range}: matched={len(archive_rows)}",
            flush=True,
        )
    return rows


def _fetch_bytes(url: str, *, retries: int = 6) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "NeuroClaw-case1-motion-qc/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Failed to fetch {url}: {error}") from error


def _s3_subject_site_index() -> dict[str, str]:
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    index: dict[str, str] = {}
    for site in ADHD_SITES:
        prefix = f"{FCP_ADHD_BIDS_PREFIX}/{site}/"
        query = urllib.parse.urlencode(
            {"prefix": prefix, "delimiter": "/", "max-keys": 1000}
        )
        root = ET.fromstring(_fetch_bytes(f"{FCP_BUCKET}/?{query}"))
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", namespace):
            match = re.search(r"/sub-(\d+)/$", str(node.text or ""))
            if match:
                subject = match.group(1)
                previous = index.get(subject)
                if previous and previous != site:
                    raise RuntimeError(
                        f"ADHD subject {subject} occurs in {previous} and {site}"
                    )
                index[subject] = site
    return index


def _metadata_subjects(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype={"subjectkey": str})
    if "subjectkey" not in frame:
        raise ValueError(f"{path} does not contain subjectkey")
    return {
        str(value).strip()
        for value in frame["subjectkey"]
        if str(value).strip()
    }


def discover_adhd_runs(
    input_root: Path,
    metadata_path: Path,
) -> tuple[list[RawRun], list[str]]:
    subjects = _metadata_subjects(metadata_path)
    site_index = _s3_subject_site_index()
    pattern = re.compile(
        r"fmri_X_(?P<subject>\d+)_session_(?P<session>\d+)_"
        r"run(?P<run>\d+)\.nii\.gz$"
    )
    runs: list[RawRun] = []
    missing: list[str] = []
    seen_subjects: set[str] = set()
    for path in sorted(input_root.glob("fmri_X_*_session_*_run*.nii.gz")):
        match = pattern.match(path.name)
        if not match:
            continue
        subject = match.group("subject")
        if subject not in subjects:
            continue
        seen_subjects.add(subject)
        site = site_index.get(subject)
        if not site:
            missing.append(subject)
            continue
        session = match.group("session")
        run = match.group("run")
        key = (
            f"{FCP_ADHD_BIDS_PREFIX}/{site}/sub-{subject}/ses-{session}/"
            f"func/sub-{subject}_ses-{session}_task-rest_run-{run}_bold.nii.gz"
        )
        runs.append(
            RawRun(
                dataset="adhd200",
                subjectkey=subject,
                session=session,
                run=run,
                source_site=site,
                url=f"{FCP_BUCKET}/{urllib.parse.quote(key, safe='/')}",
            )
        )
    missing.extend(sorted(subjects - seen_subjects))
    return runs, sorted(set(missing))


def discover_ucla_runs(
    input_root: Path,
    metadata_path: Path,
) -> tuple[list[RawRun], list[str]]:
    subjects = _metadata_subjects(metadata_path)
    available = {
        f"sub-{match.group(1)}"
        for path in input_root.glob("sub-*_task-rest_bold_*preproc.nii.gz")
        if (
            match := re.match(
                r"sub-(\d+)_task-rest_bold_.*preproc\.nii\.gz$",
                path.name,
            )
        )
    }
    runs: list[RawRun] = []
    for subject in sorted(subjects & available):
        numeric_subject = subject.removeprefix("sub-")
        runs.append(
            RawRun(
                dataset="ucla",
                subjectkey=subject,
                session="1",
                run="1",
                source_site="OpenNeuro-ds000030",
                url=(
                    f"{UCLA_OPENNEURO_PREFIX}/sub-{numeric_subject}/func/"
                    f"sub-{numeric_subject}_task-rest_bold.nii.gz"
                ),
            )
        )
    return runs, sorted(subjects - available)


def discover_runs(
    dataset: str,
    input_root: Path,
    metadata_path: Path,
) -> tuple[list[RawRun], list[str]]:
    if dataset == "adhd200":
        return discover_adhd_runs(input_root, metadata_path)
    if dataset == "ucla":
        return discover_ucla_runs(input_root, metadata_path)
    raise ValueError(f"Unsupported public raw-data source: {dataset}")


def _wsl_motion_command(
    raw_run: RawRun,
    *,
    afni_3dvolreg: str,
) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run.run_id)
    quoted_url = shlex.quote(raw_run.url)
    return (
        "set -euo pipefail; "
        f'work="$HOME/.cache/neuroclaw_mean_fd/{safe_id}"; '
        'mkdir -p "$work"; cd "$work"; '
        "trap 'rm -rf \"$work\"' EXIT; "
        "rm -f raw.nii.gz raw.nii.gz.aria2 motion.1D volreg.stderr; "
        "if command -v aria2c >/dev/null 2>&1; then "
        "aria2c -q -x 8 -s 8 -k 1M --file-allocation=none "
        "--max-tries=8 --retry-wait=2 --timeout=60 --connect-timeout=30 "
        f"-o raw.nii.gz {quoted_url}; "
        "else "
        f"curl -sS -L --retry 8 --retry-all-errors --retry-delay 2 "
        f"--max-time 900 -o raw.nii.gz {quoted_url}; "
        "fi; "
        f"if ! {afni_3dvolreg} -prefix NULL -Fourier -twopass -zpad 4 "
        "-base 0 -1Dfile motion.1D raw.nii.gz >/dev/null 2>volreg.stderr; "
        "then cat volreg.stderr >&2; exit 1; fi; "
        "cat motion.1D"
    )


def estimate_run(
    raw_run: RawRun,
    *,
    afni_3dvolreg: str,
    head_radius_mm: float,
    timeout_sec: int,
) -> dict[str, object]:
    started = time.perf_counter()
    command = _wsl_motion_command(
        raw_run,
        afni_3dvolreg=afni_3dvolreg,
    )
    try:
        result = subprocess.run(
            ["wsl.exe", "-e", "bash", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        motion = np.loadtxt(io.StringIO(result.stdout), dtype=float)
        if motion.ndim == 1:
            motion = motion.reshape(1, -1)
        summary = summarize_motion(
            motion,
            head_radius_mm=head_radius_mm,
        )
        return {
            **asdict(raw_run),
            "run_id": raw_run.run_id,
            "status": "ok",
            **summary,
            "motion_source": (
                "AFNI 3dvolreg on public raw BOLD; "
                f"Power FD, {head_radius_mm:g} mm radius"
            ),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            error = f"{error}\n{exc.stderr.strip()}"
        return {
            **asdict(raw_run),
            "run_id": raw_run.run_id,
            "status": "failed",
            "n_frames": 0,
            "n_fd_intervals": 0,
            "mean_fd": math.nan,
            "median_fd": math.nan,
            "max_fd": math.nan,
            "pct_fd_gt_0_2": math.nan,
            "pct_fd_gt_0_5": math.nan,
            "motion_source": (
                "AFNI 3dvolreg on public raw BOLD; "
                f"Power FD, {head_radius_mm:g} mm radius"
            ),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": error[:4000],
        }


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def subject_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    successful = run_frame[run_frame["status"].eq("ok")].copy()
    rows: list[dict[str, object]] = []
    for (dataset, subject), group in successful.groupby(
        ["dataset", "subjectkey"],
        sort=True,
    ):
        weights = pd.to_numeric(
            group["n_fd_intervals"],
            errors="coerce",
        ).to_numpy(float)
        valid = np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue
        weights = weights[valid]
        group = group.iloc[np.flatnonzero(valid)]

        def weighted(column: str) -> float:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).to_numpy(float)
            return float(np.average(values, weights=weights))

        sources = sorted(
            {
                str(value)
                for value in group.get(
                    "motion_source",
                    pd.Series(dtype=str),
                ).dropna()
                if str(value).strip()
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "subjectkey": subject,
                "mean_fd": weighted("mean_fd"),
                "median_fd_run_weighted": weighted("median_fd"),
                "max_fd": float(
                    pd.to_numeric(group["max_fd"], errors="coerce").max()
                ),
                "pct_fd_gt_0_2": weighted("pct_fd_gt_0_2"),
                "pct_fd_gt_0_5": weighted("pct_fd_gt_0_5"),
                "n_runs": int(len(group)),
                "n_frames": int(
                    pd.to_numeric(group["n_frames"], errors="coerce").sum()
                ),
                "n_fd_intervals": int(weights.sum()),
                "motion_source": "; ".join(sources) or "unspecified",
            }
        )
    return pd.DataFrame(rows)


def merge_metadata(
    metadata_path: Path,
    summary: pd.DataFrame,
    output_path: Path,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_path, dtype={"subjectkey": str})
    fd = summary[["subjectkey", "mean_fd"]].copy()
    fd["subjectkey"] = fd["subjectkey"].astype(str)
    merged = metadata.drop(columns=["mean_fd"], errors="ignore").merge(
        fd,
        on="subjectkey",
        how="left",
        validate="one_to_one",
    )
    _write_csv_atomic(merged, output_path)
    available = pd.to_numeric(merged["mean_fd"], errors="coerce").notna()
    return {
        "path": str(output_path),
        "subjects": int(len(merged)),
        "mean_fd_available": int(available.sum()),
        "coverage": float(available.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("adhd200", "ucla"),
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--head-radius-mm", type=float, default=50.0)
    parser.add_argument("--afni-3dvolreg", default=AFNI_DEFAULT)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Skip UCLA published confounds and estimate FD from raw BOLD.",
    )
    parser.add_argument(
        "--no-metadata-merge",
        action="store_true",
        help="Compute QC tables without writing diagnosis_with_mean_fd.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    input_root = args.input_root or DEFAULT_INPUT_ROOTS[args.dataset]
    metadata_path = args.metadata or DEFAULT_METADATA[args.dataset]
    output_root = args.output_root / args.dataset
    run_path = output_root / f"{args.dataset}_motion_runs.csv"
    subject_path = output_root / f"{args.dataset}_mean_fd.csv"
    manifest_path = output_root / f"{args.dataset}_mean_fd_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    runs, missing_subjects = discover_runs(
        args.dataset,
        input_root,
        metadata_path,
    )
    if args.limit > 0:
        runs = runs[: args.limit]

    existing = (
        pd.read_csv(run_path, dtype={"subjectkey": str})
        if run_path.is_file()
        else pd.DataFrame()
    )
    if not runs:
        raise RuntimeError(
            f"No raw runs matched {metadata_path} under {input_root}"
        )
    results_by_id = {
        str(row["run_id"]): row
        for row in (
            existing.to_dict(orient="records")
            if not existing.empty
            else []
        )
    }
    published_rows: list[dict[str, object]] = []
    if args.dataset == "ucla" and not args.raw_only:
        published_rows = fetch_ucla_published_confounds(
            {run.run_id for run in runs}
        )
        results_by_id.update(
            {str(row["run_id"]): row for row in published_rows}
        )
        _write_csv_atomic(
            pd.DataFrame(results_by_id.values()).sort_values("run_id"),
            run_path,
        )
    completed = {
        run_id
        for run_id, row in results_by_id.items()
        if str(row.get("status")) == "ok"
    }
    pending = [run for run in runs if run.run_id not in completed]
    results = list(results_by_id.values())
    print(
        f"dataset={args.dataset} discovered={len(runs)} "
        f"completed={len(completed)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                estimate_run,
                raw_run,
                afni_3dvolreg=args.afni_3dvolreg,
                head_radius_mm=args.head_radius_mm,
                timeout_sec=args.timeout_sec,
            ): raw_run
            for raw_run in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results = [
                old for old in results if old.get("run_id") != row["run_id"]
            ]
            results.append(row)
            frame = pd.DataFrame(results).sort_values("run_id")
            _write_csv_atomic(frame, run_path)
            print(
                f"[{index}/{len(pending)}] {row['run_id']} "
                f"{row['status']} mean_fd={row['mean_fd']} "
                f"elapsed={row['elapsed_sec']}s",
                flush=True,
            )

    run_frame = pd.DataFrame(results).sort_values("run_id")
    summary = subject_summary(run_frame)
    _write_csv_atomic(summary, subject_path)
    metadata_summary: dict[str, object] | None = None
    if not args.no_metadata_merge:
        merged_path = (
            metadata_path.parent
            / f"{args.dataset}_diagnosis_with_mean_fd.csv"
        )
        metadata_summary = merge_metadata(
            metadata_path,
            summary,
            merged_path,
        )

    status_counts = {
        str(key): int(value)
        for key, value in run_frame["status"].value_counts().items()
    }
    manifest = {
        "schema_version": "case1-external-mean-fd-v2",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "input_root": str(input_root),
        "metadata": str(metadata_path),
        "public_raw_sources": sorted({run.source_site for run in runs}),
        "motion_estimation": {
            "preferred_source": (
                "Official fMRIPrep 0.4.4 FramewiseDisplacement"
                if args.dataset == "ucla" and not args.raw_only
                else "AFNI 3dvolreg on public raw BOLD"
            ),
            "published_confounds_rows": len(published_rows),
            "fallback_tool": "AFNI 3dvolreg",
            "command_options": [
                "-prefix",
                "NULL",
                "-Fourier",
                "-twopass",
                "-zpad",
                "4",
                "-base",
                "0",
            ],
            "fd_definition": "Power FD",
            "rotation_units_from_afni": "degrees",
            "head_radius_mm": args.head_radius_mm,
            "mean_denominator": "T-1 frame-to-frame intervals",
        },
        "runs_discovered": len(runs),
        "subjects_discovered": len({run.subjectkey for run in runs}),
        "missing_subjects": missing_subjects,
        "status_counts": status_counts,
        "subject_mean_fd_rows": int(len(summary)),
        "run_output": str(run_path),
        "subject_output": str(subject_path),
        "metadata_output": metadata_summary,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if status_counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
