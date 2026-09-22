"""Run a small 2x2 CS1 generator-by-executor reliability experiment.

The four cells are adapted/raw framework hypothesis generation crossed with
NeuroRuntime/native framework execution. Generation validity and conditional
execution success are reported separately so unsupported native executors and
failed generators are never silently converted into ordinary execution errors.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from core.scripts.case1_native_execution_task import (
        candidate_id_series,
        execute_neuroruntime,
        prepare_bundle,
        score_results,
    )
    from core.scripts.run_case1_unadapted_format_audit import nested_values
except ModuleNotFoundError:
    from case1_native_execution_task import (
        candidate_id_series,
        execute_neuroruntime,
        prepare_bundle,
        score_results,
    )
    from run_case1_unadapted_format_audit import nested_values


METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)
REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}
NATIVE_EXECUTOR_SUPPORT = {
    "ai_scientist_v2": True,
    "open_coscientist": False,
    "sciagents": False,
    "virtual_lab": False,
    "brainpilot_native": True,
    "biomni_native": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted-mapped", type=Path, required=True)
    parser.add_argument("--raw-audit-root", type=Path, required=True)
    parser.add_argument("--all-tests", type=Path, required=True)
    parser.add_argument("--transdiag-root", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapted-requested", type=int, default=80)
    parser.add_argument("--raw-trial", type=int, default=0)
    parser.add_argument("--raw-requested", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--brainpilot-url")
    parser.add_argument("--node", default="node")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_payload_candidates(
    artifact: Any,
    *,
    known_ids: set[str],
    requested: int,
) -> list[str]:
    best: list[str] = []
    for value in nested_values(artifact):
        if not isinstance(value, dict) or not isinstance(value.get("hypotheses"), list):
            continue
        ranks: dict[int, list[dict[str, Any]]] = {}
        for item in value["hypotheses"]:
            if not isinstance(item, dict):
                continue
            try:
                rank = int(item.get("rank"))
            except (TypeError, ValueError):
                continue
            ranks.setdefault(rank, []).append(item)
        ids: list[str] = []
        used: set[str] = set()
        for rank in range(1, requested + 1):
            items = ranks.get(rank, [])
            if len(items) != 1:
                break
            item = items[0]
            cid = str(item.get("candidate_id") or "").strip()
            rationale = item.get("rationale")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                confidence = float("nan")
            if (
                cid not in known_ids
                or cid in used
                or not isinstance(rationale, str)
                or not rationale.strip()
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                break
            ids.append(cid)
            used.add(cid)
        if len(ids) > len(best):
            best = ids
    return best


def _adapted_generators(
    path: Path,
    *,
    methods: list[str],
    seed: int,
    requested_slots: int,
    known_ids: set[str],
) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    frame = frame.loc[pd.to_numeric(frame["seed"], errors="coerce").eq(seed)].copy()
    rows: dict[str, dict[str, Any]] = {}
    for method in methods:
        subset = frame.loc[frame["method"].astype(str).eq(method)].copy()
        subset["rank"] = pd.to_numeric(subset["generated_rank"], errors="coerce")
        subset = subset.sort_values("rank", kind="stable")
        requested = max(len(subset), requested_slots)
        valid = subset.loc[
            subset["mapping_status"].astype(str).eq("mapped")
            & subset["mapped_candidate_id"].astype(str).isin(known_ids)
        ].copy()
        unique_ids = valid["mapped_candidate_id"].astype(str).drop_duplicates().tolist()
        executable = [value for value in unique_ids if value.startswith("fmri|")]
        rows[method] = {
            "generator_condition": "adapted",
            "generator_requested": requested,
            "generator_valid": len(unique_ids),
            "generator_success_rate": len(unique_ids) / requested if requested else 0.0,
            "generator_complete": bool(requested and len(unique_ids) == requested),
            "selected_candidate_id": executable[0] if executable else "",
            "selected_rank": int(
                valid.loc[
                    valid["mapped_candidate_id"].astype(str).eq(executable[0]), "rank"
                ].iloc[0]
            )
            if executable
            else None,
            "generator_artifact": str(path),
            "generator_error": "" if executable else "no_executable_fmri_candidate",
        }
    return rows


def _raw_trial_dir(root: Path, method: str, trial: int) -> Path:
    candidates = (
        root / method / method / f"trial_{trial:02d}",
        root / method / f"trial_{trial:02d}",
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _raw_generators(
    root: Path,
    *,
    methods: list[str],
    trial: int,
    requested: int,
    known_ids: set[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method in methods:
        trial_dir = _raw_trial_dir(root, method, trial)
        artifact_path = trial_dir / "native_result.json"
        audit_path = trial_dir / "audit.json"
        audit = (
            json.loads(audit_path.read_text(encoding="utf-8"))
            if audit_path.is_file()
            else {}
        )
        ids: list[str] = []
        artifact_error = ""
        if artifact_path.is_file():
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                ids = strict_payload_candidates(
                    artifact,
                    known_ids=known_ids,
                    requested=requested,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"
        executable = [value for value in ids if value.startswith("fmri|")]
        process_status = str(audit.get("process_status") or "missing_audit")
        error = artifact_error or str(audit.get("error") or "")
        if len(ids) != requested and not error:
            error = str(audit.get("schema_errors") or process_status)
        rows[method] = {
            "generator_condition": "raw_upstream",
            "generator_requested": requested,
            "generator_valid": len(ids),
            "generator_success_rate": len(ids) / requested,
            "generator_complete": len(ids) == requested,
            "selected_candidate_id": executable[0] if len(ids) == requested and executable else "",
            "selected_rank": ids.index(executable[0]) + 1
            if len(ids) == requested and executable
            else None,
            "generator_artifact": str(artifact_path),
            "generator_error": error[:2000],
        }
    return rows


def prepare_exact_bundle(
    args: argparse.Namespace,
    *,
    bundle_dir: Path,
    candidate: str,
) -> None:
    candidate_file = bundle_dir / "selected_candidate.json"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(
        json.dumps({"candidate_ids": [candidate]}, indent=2), encoding="utf-8"
    )
    prepare_bundle(
        Namespace(
            all_tests=args.all_tests,
            transdiag_root=args.transdiag_root,
            diagnosis=args.diagnosis,
            out_dir=bundle_dir,
            n_candidates=1,
            seed=args.seed,
            sources=[],
            candidate_file=candidate_file,
        )
    )


def score_output(
    *,
    bundle_dir: Path,
    result_file: Path,
    method: str,
    out_dir: Path,
) -> dict[str, Any]:
    score_results(
        Namespace(
            bundle_dir=bundle_dir,
            result_file=result_file,
            method=method,
            trial=0,
            out_dir=out_dir,
            rtol=1e-4,
            atol=1e-7,
        )
    )
    return json.loads((out_dir / "execution_summary.json").read_text(encoding="utf-8"))


def run_neuroruntime(
    *,
    bundle_dir: Path,
    out_dir: Path,
    method: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result_file = out_dir / "result.json"
    execute_neuroruntime(
        Namespace(bundle_public=bundle_dir / "public", out=result_file)
    )
    summary = score_output(
        bundle_dir=bundle_dir,
        result_file=result_file,
        method=method,
        out_dir=out_dir / "score",
    )
    summary["duration_seconds"] = time.perf_counter() - started
    return summary


def brainpilot_prompt(task_dir: Path) -> str:
    return f"""Execute the blinded Case Study 1 task in {task_dir}.

Read TASK.md and task_manifest.json, then use your native shell or Python tools
to calculate every registered result from tcp_subject_level_features.csv. Do
not inspect hidden, reference, exhaustive, or precomputed aggregate results.
Return exactly one JSON object with outer key results and no prose. Do not omit
null or negative findings.
"""


def native_command(
    args: argparse.Namespace,
    *,
    method: str,
    bundle_dir: Path,
    out_dir: Path,
) -> tuple[list[str], Path, Path]:
    repo = args.clean_root / REPOSITORIES[method]
    if method == "ai_scientist_v2":
        python = args.runtime_root / REPOSITORIES[method] / ".venv" / "Scripts" / "python.exe"
        command = [
            str(python),
            str(SCRIPT_DIR / "case1_ai_scientist_native_execution.py"),
            "--repo",
            str(repo),
            "--bundle-public",
            str(bundle_dir / "public"),
            "--out-dir",
            str(out_dir),
            "--model",
            args.model,
            "--max-steps",
            "3",
        ]
        return command, repo, out_dir / "final.txt"
    if method == "biomni_native":
        python = args.runtime_root / REPOSITORIES[method] / ".venv" / "Scripts" / "python.exe"
        command = [
            str(python),
            str(SCRIPT_DIR / "case1_biomni_native_execution.py"),
            "--repo",
            str(repo),
            "--bundle-public",
            str(bundle_dir / "public"),
            "--out-dir",
            str(out_dir),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
        ]
        return command, repo, out_dir / "final.txt"
    if method == "brainpilot_native":
        if not args.brainpilot_url:
            raise RuntimeError("--brainpilot-url is required for native BrainPilot")
        task_dir = out_dir / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        for name in ("TASK.md", "task_manifest.json", "tcp_subject_level_features.csv"):
            shutil.copy2(bundle_dir / "public" / name, task_dir / name)
        prompt_path = out_dir / "prompt.txt"
        prompt_path.write_text(brainpilot_prompt(task_dir.resolve()), encoding="utf-8")
        command = [
            args.node,
            str(SCRIPT_DIR / "run_brainpilot_unadapted_client.mjs"),
            str(repo),
            args.brainpilot_url,
            str(prompt_path),
        ]
        return command, repo, out_dir / "final.txt"
    raise KeyError(method)


def run_native(
    args: argparse.Namespace,
    *,
    method: str,
    bundle_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    command, cwd, result_file = native_command(
        args,
        method=method,
        bundle_dir=bundle_dir,
        out_dir=out_dir,
    )
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = args.base_url
    env["OPENAI_API_BASE"] = args.base_url
    env["OPENAI_REASONING_EFFORT"] = "high"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env.get("OPENAI_API_KEY"):
        env["BIOMNI_API_KEY"] = env["OPENAI_API_KEY"]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout_seconds,
        check=False,
    )
    (out_dir / "process.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "process.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if method == "brainpilot_native":
        result_file.write_text(completed.stdout, encoding="utf-8")
    if not result_file.is_file():
        result_file.write_text(completed.stdout, encoding="utf-8")
    summary = score_output(
        bundle_dir=bundle_dir,
        result_file=result_file,
        method=method,
        out_dir=out_dir / "score",
    )
    summary.update(
        {
            "process_returncode": completed.returncode,
            "duration_seconds": time.perf_counter() - started,
            "command_completed": completed.returncode == 0,
            "failure_category": ""
            if summary["complete_valid_submission"]
            else (
                "native_process_failed"
                if completed.returncode != 0
                else "invalid_or_incomplete_result"
            ),
        }
    )
    return summary


def blocked_row(
    *,
    method: str,
    generator: dict[str, Any],
    executor: str,
    status: str,
    executor_supported: bool,
) -> dict[str, Any]:
    return {
        "method": method,
        **generator,
        "executor_condition": executor,
        "executor_supported": executor_supported,
        "execution_attempted": False,
        "execution_success": None,
        "complete_valid_submission": None,
        "end_to_end_success": False,
        "cell_status": status,
        "failure_category": status,
        "duration_seconds": 0.0,
    }


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (generator, executor), group in cells.groupby(
        ["generator_condition", "executor_condition"], sort=False
    ):
        attempted = group["execution_attempted"].fillna(False).astype(bool)
        execution = group.loc[attempted, "execution_success"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        rows.append(
            {
                "generator_condition": generator,
                "executor_condition": executor,
                "methods_total": len(group),
                "generator_complete_methods": int(group["generator_complete"].sum()),
                "mean_generator_slot_success_rate": float(
                    group["generator_success_rate"].mean()
                ),
                "executor_supported_methods": int(group["executor_supported"].sum()),
                "execution_attempts": int(attempted.sum()),
                "execution_successes": int(execution.sum()),
                "conditional_execution_success_rate": float(execution.mean())
                if len(execution)
                else None,
                "end_to_end_successes": int(group["end_to_end_success"].sum()),
                "end_to_end_success_rate_all_methods": float(
                    group["end_to_end_success"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_tests = pd.read_csv(args.all_tests, low_memory=False)
    all_tests["candidate_id"] = candidate_id_series(all_tests)
    known_ids = set(all_tests["candidate_id"].astype(str))
    methods = list(args.methods)
    generators = {
        "adapted": _adapted_generators(
            args.adapted_mapped,
            methods=methods,
            seed=args.seed,
            requested_slots=args.adapted_requested,
            known_ids=known_ids,
        ),
        "raw_upstream": _raw_generators(
            args.raw_audit_root,
            methods=methods,
            trial=args.raw_trial,
            requested=args.raw_requested,
            known_ids=known_ids,
        ),
    }

    cells: list[dict[str, Any]] = []
    for generator_name, by_method in generators.items():
        for method in methods:
            generator = by_method[method]
            if not generator["selected_candidate_id"]:
                for executor in ("neuroruntime", "native_upstream"):
                    supported = executor == "neuroruntime" or NATIVE_EXECUTOR_SUPPORT[method]
                    cells.append(
                        blocked_row(
                            method=method,
                            generator=generator,
                            executor=executor,
                            status="blocked_by_generator",
                            executor_supported=supported,
                        )
                    )
                continue

            bundle_dir = args.out_dir / "bundles" / generator_name / method
            if args.force and bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            if not (bundle_dir / "public" / "task_manifest.json").is_file():
                prepare_exact_bundle(
                    args,
                    bundle_dir=bundle_dir,
                    candidate=str(generator["selected_candidate_id"]),
                )

            neuro_dir = args.out_dir / "executions" / generator_name / method / "neuroruntime"
            neuro_summary = run_neuroruntime(
                bundle_dir=bundle_dir,
                out_dir=neuro_dir,
                method=method,
            )
            cells.append(
                {
                    "method": method,
                    **generator,
                    "executor_condition": "neuroruntime",
                    "executor_supported": True,
                    "execution_attempted": True,
                    "execution_success": bool(neuro_summary["complete_valid_submission"]),
                    "complete_valid_submission": bool(
                        neuro_summary["complete_valid_submission"]
                    ),
                    "end_to_end_success": bool(
                        neuro_summary["complete_valid_submission"]
                    ),
                    "cell_status": "success"
                    if neuro_summary["complete_valid_submission"]
                    else "execution_failed",
                    "failure_category": ""
                    if neuro_summary["complete_valid_submission"]
                    else "invalid_or_incomplete_result",
                    "execution_result_file": neuro_summary["result_file"],
                    "duration_seconds": neuro_summary["duration_seconds"],
                }
            )

            if not NATIVE_EXECUTOR_SUPPORT[method]:
                cells.append(
                    blocked_row(
                        method=method,
                        generator=generator,
                        executor="native_upstream",
                        status="native_executor_not_available",
                        executor_supported=False,
                    )
                )
                continue
            if args.skip_native:
                cells.append(
                    blocked_row(
                        method=method,
                        generator=generator,
                        executor="native_upstream",
                        status="native_execution_skipped",
                        executor_supported=True,
                    )
                )
                continue
            native_dir = args.out_dir / "executions" / generator_name / method / "native"
            try:
                native_summary = run_native(
                    args,
                    method=method,
                    bundle_dir=bundle_dir,
                    out_dir=native_dir,
                )
                success = bool(native_summary["complete_valid_submission"])
                cells.append(
                    {
                        "method": method,
                        **generator,
                        "executor_condition": "native_upstream",
                        "executor_supported": True,
                        "execution_attempted": True,
                        "execution_success": success,
                        "complete_valid_submission": success,
                        "end_to_end_success": success,
                        "cell_status": "success" if success else "execution_failed",
                        "failure_category": native_summary["failure_category"],
                        "execution_result_file": native_summary["result_file"],
                        "duration_seconds": native_summary["duration_seconds"],
                    }
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                cells.append(
                    {
                        "method": method,
                        **generator,
                        "executor_condition": "native_upstream",
                        "executor_supported": True,
                        "execution_attempted": True,
                        "execution_success": False,
                        "complete_valid_submission": False,
                        "end_to_end_success": False,
                        "cell_status": f"runner_error:{type(exc).__name__}",
                        "failure_category": "runner_exception",
                        "duration_seconds": 0.0,
                        "execution_error": str(exc)[:2000],
                    }
                )

    cell_frame = pd.DataFrame(cells)
    cell_frame.to_csv(args.out_dir / "factorial_cells.csv", index=False)
    summary = summarize(cell_frame)
    summary.to_csv(args.out_dir / "factorial_summary.csv", index=False)
    provenance = {
        "schema_version": "case1-generator-executor-factorial.v1",
        "created_at": utc_now(),
        "design": {
            "generator_conditions": ["adapted", "raw_upstream"],
            "executor_conditions": ["neuroruntime", "native_upstream"],
            "candidate_count_per_available_cell": 1,
            "generation_and_execution_denominators_separated": True,
            "unsupported_native_executor_is_na": True,
        },
        "inputs": {
            "adapted_mapped": str(args.adapted_mapped),
            "adapted_mapped_sha256": sha256_file(args.adapted_mapped),
            "raw_audit_root": str(args.raw_audit_root),
            "all_tests": str(args.all_tests),
            "all_tests_sha256": sha256_file(args.all_tests),
            "transdiag_root": str(args.transdiag_root),
            "clean_root": str(args.clean_root),
        },
        "model": args.model,
        "base_url": args.base_url,
        "api_key_recorded": False,
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
