"""Reproduce legacy prompt-emulation CS1 baselines across repeated runs.

Primary experiments use ``case1_official_baseline_experiment.py``. This legacy
launcher is disabled unless the caller explicitly acknowledges prompt emulation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).with_name("case1_generation_baseline_experiment.py")
DEFAULT_ALL_TESTS = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\case1_exhaustive_full_all_tests_labeled.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-hypotheses", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--api-timeout-s", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--allow-prompt-emulation", action="store_true")
    return parser.parse_args()


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    seed_dir = args.out_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = seed_dir / "launcher.stdout.log"
    stderr_path = seed_dir / "launcher.stderr.log"
    command = [
        sys.executable,
        str(CHILD),
        "--all-tests",
        str(args.all_tests),
        "--out-dir",
        str(seed_dir),
        "--seed",
        str(seed),
        "--n-hypotheses",
        str(args.n_hypotheses),
        "--batch-size",
        str(args.batch_size),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--base-url",
        args.base_url,
        "--api",
        args.api,
        "--api-timeout-s",
        str(args.api_timeout_s),
    ]
    if args.methods:
        command.extend(["--methods", *args.methods])
    if args.force_api:
        command.append("--force-api")
    command.append("--allow-prompt-emulation")

    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    return {
        "seed": seed,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_dir": str(seed_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def combine_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for seed in args.seeds:
        seed_dir = args.out_dir / f"seed_{seed:02d}"
        mapped_path = seed_dir / "generation_first_mapped_hypotheses.csv"
        summary_path = seed_dir / "generation_first_summary.csv"
        if not mapped_path.is_file() or not summary_path.is_file():
            missing.append(str(seed_dir))
            continue
        mapped = pd.read_csv(mapped_path, low_memory=False)
        mapped["seed"] = int(seed)
        mapped_parts.append(mapped)
        summary = pd.read_csv(summary_path, low_memory=False)
        summary["seed"] = int(seed)
        summary_parts.append(summary)
    if missing:
        raise FileNotFoundError(
            "Cannot combine incomplete generation seeds:\n" + "\n".join(missing)
        )
    combined = pd.concat(mapped_parts, ignore_index=True)
    summaries = pd.concat(summary_parts, ignore_index=True)
    combined.to_csv(
        args.out_dir / "generation_first_mapped_hypotheses.csv", index=False
    )
    summaries.to_csv(
        args.out_dir / "generation_first_summary_by_seed.csv", index=False
    )
    return combined, summaries


def main() -> int:
    args = parse_args()
    if not args.allow_prompt_emulation:
        raise RuntimeError(
            "This launcher reproduces legacy role-prompt emulations only. "
            "Use case1_official_baseline_experiment.py for the primary comparison."
        )
    if not args.all_tests.is_file():
        raise FileNotFoundError(args.all_tests)
    if args.max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SUB2API_OPENAI_API_KEY")
    ):
        raise RuntimeError(
            "OPENAI_API_KEY or SUB2API_OPENAI_API_KEY must be set in the environment"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(args.seeds))
    ) as executor:
        futures = {
            executor.submit(run_seed, args, seed): seed for seed in args.seeds
        }
        for future in as_completed(futures):
            row = future.result()
            run_rows.append(row)
            print(
                f"seed={row['seed']} returncode={row['returncode']} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    failed = sorted(
        int(row["seed"]) for row in run_rows if int(row["returncode"]) != 0
    )
    combined_rows = 0
    summary_rows = 0
    if not failed:
        combined, summaries = combine_outputs(args)
        combined_rows = len(combined)
        summary_rows = len(summaries)

    manifest = {
        "schema_version": "case1-generation-seeds-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_tests": str(args.all_tests),
        "all_tests_sha256": sha256_file(args.all_tests),
        "output_dir": str(args.out_dir),
        "seeds": args.seeds,
        "n_hypotheses_per_method_seed": args.n_hypotheses,
        "batch_size": args.batch_size,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "base_url": args.base_url,
        "api": args.api,
        "max_workers": args.max_workers,
        "methods": args.methods or "case1_generation_baseline_experiment.DEFAULT",
        "runs": sorted(run_rows, key=lambda row: int(row["seed"])),
        "failed_seeds": failed,
        "combined_rows": combined_rows,
        "summary_rows": summary_rows,
        "credential_policy": (
            "API credentials are inherited through the process environment "
            "and are never serialized."
        ),
    }
    (args.out_dir / "generation_seed_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failed:
        print(f"failed seeds: {failed}", file=sys.stderr)
        return 1
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
