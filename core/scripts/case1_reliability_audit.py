"""Audit Case Study 1 generation legality and execution reliability.

The audit deliberately separates three events that were previously easy to
conflate:

1. the native framework process returned an artifact;
2. the artifact contained a legal, unique registered hypothesis;
3. the statistical experiment produced a finite result.

It never repairs outputs and never reads GT labels when computing legality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "neurodiscovery": "NeuroDiscovery",
    "ai_scientist_v2": "AI Scientist-v2",
    "open_coscientist": "Open Co-Scientist",
    "sciagents": "SciAgents",
    "virtual_lab": "Virtual Lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}

NATIVE_EXECUTION_CAPABILITY = {
    "neurodiscovery": (
        True,
        "NeuroRuntime is the native NeuroDiscovery experiment executor.",
    ),
    "ai_scientist_v2": (
        True,
        "Official BFTS experimentation pipeline can edit and execute code.",
    ),
    "brainpilot_native": (
        True,
        "Official runtime exposes engineer/system shell tools.",
    ),
    "biomni_native": (
        True,
        "Official A1 agent exposes persistent Python code execution.",
    ),
    "open_coscientist": (
        False,
        "Official CS1 entry point is a hypothesis generator, not an experiment executor.",
    ),
    "sciagents": (
        False,
        "Official CS1 entry point generates and critiques discoveries but has no native CS1 executor.",
    ),
    "virtual_lab": (
        False,
        "Official CS1 entry point is a multi-agent meeting and summary workflow.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--all-tests", type=Path, required=True)
    parser.add_argument("--fresh-official-root", type=Path)
    parser.add_argument("--fresh-native-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matched-slots", type=int, default=80)
    parser.add_argument("--matched-seeds", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def formal_generation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    official_path = (
        args.formal_root
        / "official_baselines"
        / "official_native_proposal_summary.csv"
    )
    official = pd.read_csv(official_path)
    for method, group in official.groupby("method", sort=False):
        requested = int(group["requested_anchors"].sum())
        legal = int(group["valid_anchors"].sum())
        returned = int(group["completion_status"].isin(["complete", "partial"]).sum())
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "trials": int(len(group)),
                "framework_artifacts_returned": returned,
                "framework_completion_rate": _rate(returned, len(group)),
                "generated_slots": requested,
                "legal_unique_hypotheses": legal,
                "illegal_duplicate_or_missing": requested - legal,
                "legal_rate": _rate(legal, requested),
                "source": str(official_path),
            }
        )

    native_path = (
        args.formal_root
        / "native_baselines"
        / "native_baselines_direct_seed_summary.csv"
    )
    native = pd.read_csv(native_path)
    native = native[native["budget"].eq(args.matched_slots)].copy()
    for method, group in native.groupby("method", sort=False):
        requested = int(group["budget"].sum())
        legal = int(group["mapped"].sum())
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "trials": int(group["seed"].nunique()),
                "framework_artifacts_returned": int(group["seed"].nunique()),
                "framework_completion_rate": 1.0,
                "generated_slots": requested,
                "legal_unique_hypotheses": legal,
                "illegal_duplicate_or_missing": requested - legal,
                "legal_rate": _rate(legal, requested),
                "source": str(native_path),
            }
        )

    nd_slots = args.matched_slots * args.matched_seeds
    rows.append(
        {
            "method": "neurodiscovery",
            "label": METHOD_LABELS["neurodiscovery"],
            "trials": args.matched_seeds,
            "framework_artifacts_returned": args.matched_seeds,
            "framework_completion_rate": 1.0,
            "generated_slots": nd_slots,
            "legal_unique_hypotheses": nd_slots,
            "illegal_duplicate_or_missing": 0,
            "legal_rate": 1.0,
            "source": "registered SearchPolicy construction",
        }
    )
    return rows


def fresh_official_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        method = method_dir.name
        if method not in METHOD_LABELS:
            continue
        for trial_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            task_path = trial_dir / "task.json"
            if not task_path.exists():
                continue
            task = json.loads(task_path.read_text(encoding="utf-8"))
            requested = int(task.get("n_anchors", 0))
            audit_path = trial_dir / "native_compile_audit.json"
            audit = (
                json.loads(audit_path.read_text(encoding="utf-8"))
                if audit_path.exists()
                else {}
            )
            legal = int(audit.get("valid_unique_anchors", 0))
            process_returned = (trial_dir / "native_result.json").exists()
            policy_returned = (trial_dir / "search_policy.json").exists()
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "trial": int(task.get("trial", -1)),
                    "attempts_allowed": 1,
                    "framework_process_returned": process_returned,
                    "policy_compiled": policy_returned,
                    "first_attempt_workflow_success": bool(
                        process_returned and policy_returned
                    ),
                    "requested_slots": requested,
                    "legal_unique_hypotheses": legal,
                    "illegal_duplicate_or_missing": requested - legal,
                    "legal_rate": _rate(legal, requested),
                    "duration_seconds": _read_float(
                        trial_dir / "adapter_attempt_01.duration.txt"
                    ),
                }
            )
    return rows


def _read_float(path: Path) -> float:
    if not path.exists():
        return float("nan")
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return float("nan")


def fresh_native_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        method = method_dir.name
        if method not in {"brainpilot_native", "biomni_native"}:
            continue
        for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            mapped_path = seed_dir / "mapped_hypotheses.csv"
            batch_dirs = sorted(seed_dir.glob("batch_*"))
            final_count = sum((batch / "final.txt").exists() for batch in batch_dirs)
            parsed_count = sum((batch / "parsed.json").exists() for batch in batch_dirs)
            mapped = pd.read_csv(mapped_path) if mapped_path.exists() else pd.DataFrame()
            requested = int(len(mapped))
            legal = (
                int(mapped["native_schema_valid"].fillna(False).astype(bool).sum())
                if not mapped.empty
                else 0
            )
            durations = [
                _read_float(batch / "launcher_duration_seconds.txt")
                for batch in batch_dirs
            ]
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "trial": int(seed_dir.name.rsplit("_", 1)[-1]),
                    "attempts_allowed": 1,
                    "framework_process_returned": bool(
                        batch_dirs and final_count == len(batch_dirs)
                    ),
                    "policy_compiled": mapped_path.exists(),
                    "first_attempt_workflow_success": bool(
                        batch_dirs
                        and final_count == len(batch_dirs)
                        and parsed_count == len(batch_dirs)
                        and mapped_path.exists()
                    ),
                    "requested_slots": requested,
                    "legal_unique_hypotheses": legal,
                    "illegal_duplicate_or_missing": requested - legal,
                    "legal_rate": _rate(legal, requested),
                    "duration_seconds": float(np.nansum(durations)),
                }
            )
    return rows


def neuroruntime_rows(all_tests: Path) -> list[dict[str, Any]]:
    columns = [
        "modality",
        "source",
        "feature",
        "adjusted_residual_d",
        "p_value",
        "q_fdr_global",
    ]
    frame = pd.read_csv(all_tests, usecols=columns, low_memory=False)
    finite = np.ones(len(frame), dtype=bool)
    for column in ("adjusted_residual_d", "p_value", "q_fdr_global"):
        finite &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    frame["experiment_completed"] = finite
    rows = []
    for keys, group in frame.groupby(["modality", "source", "feature"], sort=False):
        completed = int(group["experiment_completed"].sum())
        rows.append(
            {
                "modality": keys[0],
                "source": keys[1],
                "feature": keys[2],
                "attempted_hypotheses": int(len(group)),
                "completed_hypotheses": completed,
                "failed_or_nonfinite": int(len(group) - completed),
                "completion_rate": _rate(completed, len(group)),
            }
        )
    total_completed = int(frame["experiment_completed"].sum())
    rows.append(
        {
            "modality": "ALL",
            "source": "ALL",
            "feature": "ALL",
            "attempted_hypotheses": int(len(frame)),
            "completed_hypotheses": total_completed,
            "failed_or_nonfinite": int(len(frame) - total_completed),
            "completion_rate": _rate(total_completed, len(frame)),
        }
    )
    return rows


def write_readme(args: argparse.Namespace, outputs: dict[str, Path]) -> None:
    text = f"""# Case Study 1 reliability audit

This directory separates three denominators:

1. **Framework completion**: the native framework returned its requested artifact.
2. **Hypothesis legality**: an exact, unique candidate from the frozen public registry was returned.
3. **Experiment completion**: the statistical execution produced finite effect, p-value, and FDR values.

Invalid, duplicate, missing, non-finite, and failed outputs are never repaired by this audit.

## Inputs

- Formal comparison root: `{args.formal_root}`
- Exhaustive result table: `{args.all_tests}`
- Exhaustive SHA256: `{sha256_file(args.all_tests)}`
- Fresh official no-retry root: `{args.fresh_official_root or ''}`
- Fresh BrainPilot/Biomni no-retry root: `{args.fresh_native_root or ''}`

## Outputs

"""
    for label, path in outputs.items():
        text += f"- `{path.name}`: {label}\n"
    text += """

## Interpretation boundary

The formal CS1 discovery curves are a **NeuroRuntime-adapted conditional ranking
benchmark**. They remain useful because every generator is evaluated with the
same executor. Native end-to-end execution is a separate systems benchmark and
is only applicable to systems whose official source exposes an experiment or
code-execution path.
"""
    (args.out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    formal = pd.DataFrame(formal_generation_rows(args)).sort_values(
        "method", kind="stable"
    )
    formal_path = args.out_dir / "formal_generation_legality.csv"
    formal.to_csv(formal_path, index=False)

    fresh_rows: list[dict[str, Any]] = []
    if args.fresh_official_root is not None:
        fresh_rows.extend(fresh_official_rows(args.fresh_official_root))
    if args.fresh_native_root is not None:
        fresh_rows.extend(fresh_native_rows(args.fresh_native_root))
    fresh = pd.DataFrame(fresh_rows)
    fresh_path = args.out_dir / "fresh_no_retry_workflow_trials.csv"
    fresh.to_csv(fresh_path, index=False)
    fresh_summary_path = args.out_dir / "fresh_no_retry_workflow_summary.csv"
    if fresh.empty:
        pd.DataFrame().to_csv(fresh_summary_path, index=False)
    else:
        summary = (
            fresh.groupby(["method", "label"], as_index=False)
            .agg(
                trials=("trial", "size"),
                first_attempt_successes=("first_attempt_workflow_success", "sum"),
                generated_slots=("requested_slots", "sum"),
                legal_unique_hypotheses=("legal_unique_hypotheses", "sum"),
                total_duration_seconds=("duration_seconds", "sum"),
            )
        )
        summary["first_attempt_workflow_success_rate"] = (
            summary["first_attempt_successes"] / summary["trials"]
        )
        summary["legal_rate"] = (
            summary["legal_unique_hypotheses"] / summary["generated_slots"]
        )
        summary.to_csv(fresh_summary_path, index=False)

    runtime = pd.DataFrame(neuroruntime_rows(args.all_tests))
    runtime_path = args.out_dir / "neuroruntime_hypothesis_execution.csv"
    runtime.to_csv(runtime_path, index=False)

    capability = pd.DataFrame(
        [
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "native_experiment_execution_supported": supported,
                "basis": basis,
            }
            for method, (supported, basis) in NATIVE_EXECUTION_CAPABILITY.items()
        ]
    )
    capability_path = args.out_dir / "native_execution_capability.csv"
    capability.to_csv(capability_path, index=False)

    outputs = {
        "formal legality using all 10 seeds and 80 slots": formal_path,
        "fresh per-trial no-retry audit": fresh_path,
        "fresh no-retry aggregate": fresh_summary_path,
        "hypothesis-level NeuroRuntime completion": runtime_path,
        "official native execution capability boundary": capability_path,
    }
    write_readme(args, outputs)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
