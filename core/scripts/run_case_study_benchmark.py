"""Outer entry point for primary experiments and hindcasting validation.

The launcher deliberately keeps case-specific scientific protocols separate:
Case 1 and Case 2 use their executable primary-experiment runners, while
hindcasting is a cross-case-study validation protocol. Arguments after the
benchmark mode are forwarded unchanged to the specialised runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "case1_primary": ROOT / "core/scripts/case1_official_baseline_experiment.py",
    "case2_primary": ROOT / "core/scripts/case2_official_baseline_experiment.py",
    "hindcasting": ROOT / "neurooracle/scripts/run_case_study_hindcasting_matrix.py",
}
ALIASES = {
    "1": "case1_primary",
    "cs1": "case1_primary",
    "case1": "case1_primary",
    "case1_transdiagnostic": "case1_primary",
    "2": "case2_primary",
    "cs2": "case2_primary",
    "case2": "case2_primary",
    "case2_pathway_mediation": "case2_primary",
    "hindcast": "hindcasting",
    "hindcasting": "hindcasting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(ALIASES))
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected case-specific runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = ALIASES[args.benchmark]
    runner = RUNNERS[benchmark]
    if not runner.is_file():
        raise FileNotFoundError(runner)
    forwarded = list(args.runner_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, str(runner), *forwarded]
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()


# Created At: 2026-08-01 17:36 HKT
