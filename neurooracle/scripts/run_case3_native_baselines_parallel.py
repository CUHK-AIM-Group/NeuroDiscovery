"""Parallel task-level launcher for Case Study 3 native-agent baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd

from neurooracle.src.case3_subtasks import select_case3_subtasks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT / "neurooracle/data/experiments/case3/native_baselines_gpt56sol_high_seed10_20260722"
)
METHODS = ("brainpilot_native", "biomni_native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[10])
    parser.add_argument("--n-hypotheses", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--brainpilot-urls", nargs="+", default=["http://127.0.0.1:9460/api"])
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=20.0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def run_job(
    args: argparse.Namespace,
    method: str,
    task: str,
    brainpilot_url: str | None,
) -> dict[str, Any]:
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{method}__{task}.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_case3_native_baselines.py")),
        "--out-dir", str(args.out_dir),
        "--tasks", task,
        "--methods", method,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--n-hypotheses", str(args.n_hypotheses),
        "--batch-size", str(args.batch_size),
        "--model", args.model,
        "--base-url", args.base_url,
        "--reasoning-effort", args.reasoning_effort,
        "--brainpilot-urls", brainpilot_url or args.brainpilot_urls[0],
        "--timeout-seconds", str(args.timeout_seconds),
        "--max-events", str(args.max_events),
        "--max-retries", str(args.max_retries),
        "--retry-wait-seconds", str(args.retry_wait_seconds),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    return {
        "method": method,
        "task": task,
        "returncode": result.returncode,
        "duration_seconds": time.time() - started,
        "log": str(log_path),
        "command": command,
    }


def rebuild_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(out_dir.glob("*/*/seed_*/native_hypotheses.csv"))
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_dir / "native_hypotheses_all.csv", index=False)
    summaries: list[dict[str, Any]] = []
    if not combined.empty:
        valid = combined["native_schema_valid"].astype(str).str.casefold().eq("true")
        combined = combined.assign(_valid=valid)
        for (method, task, seed), sub in combined.groupby(["method", "task", "seed"], sort=True):
            summaries.append(
                {
                    "method": method,
                    "task": task,
                    "seed": int(seed),
                    "requested": len(sub),
                    "schema_valid": int(sub["_valid"].sum()),
                    "schema_valid_rate": float(sub["_valid"].mean()),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "native_generation_summary.csv", index=False)
    return combined, summary


def main() -> None:
    args = parse_args()
    if not os.environ.get("AUTORESEARCH_LOCAL_API_KEY"):
        raise RuntimeError("AUTORESEARCH_LOCAL_API_KEY is required")
    if args.max_workers < 1:
        raise ValueError("max_workers must be positive")
    selected = select_case3_subtasks(args.tasks)
    jobs: list[tuple[str, str, str | None]] = []
    brainpilot_index = 0
    for task in selected:
        for method in args.methods:
            brainpilot_url = None
            if method == "brainpilot_native":
                brainpilot_url = args.brainpilot_urls[brainpilot_index % len(args.brainpilot_urls)]
                brainpilot_index += 1
            jobs.append((method, task.name, brainpilot_url))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_job, args, method, task, brainpilot_url): (method, task)
            for method, task, brainpilot_url in jobs
        }
        for future in as_completed(futures):
            method, task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "method": method,
                    "task": task,
                    "returncode": -1,
                    "duration_seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            run_rows.append(row)
            print(
                f"[{method}] {task}: returncode={row['returncode']} "
                f"duration={row['duration_seconds']:.1f}s",
                flush=True,
            )
            if args.fail_fast and row["returncode"] != 0:
                for pending in futures:
                    pending.cancel()
                break

    combined, summary = rebuild_tables(args.out_dir)
    manifest = {
        "tasks": [task.name for task in selected],
        "methods": args.methods,
        "seeds": args.seeds,
        "n_hypotheses_per_task_seed": args.n_hypotheses,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "max_retries": args.max_retries,
        "brainpilot_urls": args.brainpilot_urls,
        "jobs": run_rows,
        "completed_jobs": sum(row["returncode"] == 0 for row in run_rows),
        "failed_jobs": [
            f"{row['method']}:{row['task']}" for row in run_rows if row["returncode"] != 0
        ],
        "combined_rows": len(combined),
        "summary_rows": len(summary),
        "credential_policy": "The API key is inherited through process environment and is never serialized.",
    }
    (args.out_dir / "native_parallel_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if manifest["failed_jobs"]:
        raise SystemExit(1)
    print(args.out_dir)


if __name__ == "__main__":
    main()
