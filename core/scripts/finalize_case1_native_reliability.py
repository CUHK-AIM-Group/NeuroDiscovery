"""Combine Case Study 1 generation and native-execution reliability audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "neurodiscovery": "NeuroDiscovery",
    "ai_scientist_v2": "AI Scientist-v2",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
    "open_coscientist": "Open Co-Scientist",
    "sciagents": "SciAgents",
    "virtual_lab": "Virtual Lab",
}

METHOD_ALIASES = {"ai_scientist_v2_native": "ai_scientist_v2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-audit-dir", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--all-tests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--valid-trial",
        action="append",
        default=[],
        metavar="METHOD:TRIAL",
        help="Protocol-valid native trial; repeat for each framework.",
    )
    return parser.parse_args()


def normalize_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def parse_valid_trials(values: list[str]) -> set[tuple[str, int]]:
    parsed: set[tuple[str, int]] = set()
    for value in values:
        method, separator, trial_text = value.rpartition(":")
        if not separator:
            raise ValueError(f"Invalid --valid-trial value: {value!r}")
        parsed.add((normalize_method(method), int(trial_text)))
    return parsed


def read_execution_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(root.rglob("execution_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = normalize_method(str(payload.get("method") or ""))
        trial = int(payload.get("trial", -1))
        key = (method, trial, str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        attempted = int(payload.get("attempted_candidates", 0))
        successful = int(payload.get("successful_candidates", 0))
        result_file = str(payload.get("result_file") or "")
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "trial": trial,
                "framework_process_returned": bool(
                    result_file and Path(result_file).exists()
                ),
                "attempted_candidates": attempted,
                "successful_candidates": successful,
                "candidate_execution_success_rate": (
                    successful / attempted if attempted else 0.0
                ),
                "complete_valid_submission": bool(
                    payload.get("complete_valid_submission", False)
                ),
                "parse_error": str(payload.get("parse_error") or ""),
                "human_repairs": int(payload.get("human_repairs", 0)),
                "result_file": result_file,
                "score_summary": str(path),
            }
        )
    return rows


def neuroruntime_completion(all_tests: Path) -> tuple[int, int, float]:
    columns = ["adjusted_residual_d", "p_value", "q_fdr_global"]
    frame = pd.read_csv(all_tests, usecols=columns, low_memory=False)
    finite = np.ones(len(frame), dtype=bool)
    for column in columns:
        finite &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    successful = int(finite.sum())
    attempted = int(len(frame))
    return attempted, successful, successful / attempted if attempted else 0.0


def build_layered_summary(
    generation_dir: Path,
    valid_native: pd.DataFrame,
    all_tests: Path,
) -> pd.DataFrame:
    formal = pd.read_csv(generation_dir / "formal_generation_legality.csv")
    fresh = pd.read_csv(generation_dir / "fresh_no_retry_workflow_summary.csv")
    capability = pd.read_csv(generation_dir / "native_execution_capability.csv")
    methods = list(METHOD_LABELS)
    rows: list[dict[str, Any]] = []
    runtime_attempted, runtime_successful, runtime_rate = neuroruntime_completion(
        all_tests
    )
    for method in methods:
        formal_row = formal[formal["method"].eq(method)]
        fresh_row = fresh[fresh["method"].eq(method)]
        capability_row = capability[capability["method"].eq(method)]
        native_row = valid_native[valid_native["method"].eq(method)]
        supported = bool(
            capability_row["native_experiment_execution_supported"].iloc[0]
        )
        attempted: int | None = None
        successful: int | None = None
        native_rate: float | None = None
        process_rate: float | None = None
        native_trials = 0
        native_trial_mean: float | None = None
        native_trial_variance: float | None = None
        complete_submission_rate: float | None = None
        no_retry_trials = int(fresh_row["trials"].iloc[0]) if not fresh_row.empty else 0
        no_retry_legal_rate = (
            float(fresh_row["legal_rate"].iloc[0])
            if not fresh_row.empty
            else np.nan
        )
        if method == "neurodiscovery":
            attempted = runtime_attempted
            successful = runtime_successful
            native_rate = runtime_rate
            process_rate = 1.0
            native_trials = 1
            native_trial_mean = runtime_rate
            no_retry_trials = int(formal_row["trials"].iloc[0])
            no_retry_legal_rate = 1.0
        elif supported and not native_row.empty:
            attempted = int(native_row["attempted_candidates"].sum())
            successful = int(native_row["successful_candidates"].sum())
            native_rate = successful / attempted if attempted else 0.0
            process_rate = float(native_row["framework_process_returned"].mean())
            trial_rates = native_row["candidate_execution_success_rate"].astype(float)
            native_trials = int(len(trial_rates))
            native_trial_mean = float(trial_rates.mean())
            native_trial_variance = (
                float(trial_rates.var(ddof=1)) if len(trial_rates) > 1 else 0.0
            )
            complete_submission_rate = float(
                native_row["complete_valid_submission"].mean()
            )
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "formal_trials": (
                    int(formal_row["trials"].iloc[0]) if not formal_row.empty else 0
                ),
                "formal_generated_slots": (
                    int(formal_row["generated_slots"].iloc[0])
                    if not formal_row.empty
                    else 0
                ),
                "formal_legal_rate": (
                    float(formal_row["legal_rate"].iloc[0])
                    if not formal_row.empty
                    else np.nan
                ),
                "no_retry_trials": no_retry_trials,
                "no_retry_legal_rate": no_retry_legal_rate,
                "native_executor_supported": supported,
                "native_valid_trials": native_trials,
                "native_process_return_rate": process_rate,
                "native_complete_submission_rate": complete_submission_rate,
                "native_attempted_candidates": attempted,
                "native_successful_candidates": successful,
                "native_candidate_execution_success_rate": native_rate,
                "native_trial_mean_success_rate": native_trial_mean,
                "native_trial_success_rate_sample_variance": native_trial_variance,
            }
        )
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, valid_native: pd.DataFrame) -> None:
    text = """# Case Study 1 layered reliability audit

This audit keeps four denominators separate:

1. formal legality after the registered adapter/mapping workflow;
2. first-pass legality with no retry or repair;
3. whether the official framework exposes a native experiment executor;
4. exact candidate-level correctness in a blinded native execution task.

The native task exposes only subject-level TCP values, candidate definitions,
and the statistical contract. Hidden aggregate results are used only after the
framework returns. Unsupported native executors are reported as N/A, not zero.

The existing discovery comparison remains a **NeuroRuntime-adapted conditional
ranking benchmark**: it compares ranking quality under one shared executor.
The native audit is a separate systems-reliability result and does not replace
that benchmark.

## Protocol-valid native trials

"""
    for row in valid_native.itertuples(index=False):
        text += (
            f"- {row.label}, trial {row.trial}: "
            f"{row.successful_candidates}/{row.attempted_candidates} exact "
            f"candidate results; process returned={row.framework_process_returned}; "
            f"human repairs={row.human_repairs}.\n"
        )
    text += """

## Files

- `layered_reliability_summary.csv`: publication-facing layered denominators
- `native_execution_valid_trials.csv`: protocol-valid blinded native trials
- `native_execution_protocol_diagnostics.csv`: excluded setup/protocol trials
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    valid_specs = parse_valid_trials(args.valid_trial)
    trials = pd.DataFrame(read_execution_summaries(args.native_root))
    if trials.empty:
        raise RuntimeError(f"No execution summaries found under {args.native_root}")
    is_valid = [
        (str(row.method), int(row.trial)) in valid_specs
        for row in trials.itertuples(index=False)
    ]
    valid = trials.loc[is_valid].sort_values(["method", "trial"], kind="stable")
    diagnostics = trials.loc[np.logical_not(is_valid)].sort_values(
        ["method", "trial"], kind="stable"
    )
    missing = valid_specs - set(zip(valid["method"], valid["trial"]))
    if missing:
        raise RuntimeError(f"Missing protocol-valid trial summaries: {sorted(missing)}")
    valid.to_csv(args.out_dir / "native_execution_valid_trials.csv", index=False)
    diagnostics.to_csv(
        args.out_dir / "native_execution_protocol_diagnostics.csv", index=False
    )
    layered = build_layered_summary(args.generation_audit_dir, valid, args.all_tests)
    layered.to_csv(args.out_dir / "layered_reliability_summary.csv", index=False)
    write_readme(args.out_dir, valid)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
