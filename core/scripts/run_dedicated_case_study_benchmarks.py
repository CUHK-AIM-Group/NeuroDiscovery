"""Rerun the dedicated CS1 and CS2 benchmarks on a frozen KG release.

The expensive neuroimaging outcomes and outcome-blind baseline policies are
immutable inputs.  Every KG-derived hypothesis, score, ranking, feedback trace,
and external-validation table is rebuilt in the new run directory.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
    write_release_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "dedicated-case-study-rerun.v1"
DEFAULT_KG = ROOT / "neurooracle/data/full_v2/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
DEFAULT_STATE = ROOT / "neurooracle/data/full_v2/CURRENT_STATE.json"
DEFAULT_CASE1_SOURCE = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_rerun"
    r"\20260810_kg737C884E_seeded10"
)
DEFAULT_CASE1_CONFIG = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_tuning"
    r"\20260810_kgdd0d4037_auxiliary_cv_v5\best_config.json"
)
DEFAULT_CASE1_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_CASE1_EXTERNAL_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_external_validation_v1\full"
)
DEFAULT_CASE2_SOURCE = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
    r"\case2_adni_genetics_v1\experiments\case2_adni_kg_guided_longitudinal_v1"
    r"\20260812_kgA8C354B5_rerun"
)
DEFAULT_CASE2_POLICIES = (
    DEFAULT_CASE2_SOURCE
    / "official_baselines_reused"
    / "case2_search_policies.jsonl"
)
DEFAULT_CASE2_SOURCE_MANIFEST = DEFAULT_CASE2_SOURCE / "manifest.json"
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\dedicated_case_study_benchmarks"
    r"\20260814_kg89e40d_cross_release_v1"
)

CASE1_OFFICIAL_FILES = (
    "case1_search_policies.jsonl",
    "generation_first_mapped_hypotheses.csv",
    "official_native_proposal_summary.csv",
    "official_adapter_manifest.json",
    "case1_policy_independence_audit.json",
    "case1_policy_independence_audit.csv",
)
CASE1_NATIVE_FILES = (
    "native_baselines_direct_seed_summary.csv",
    "native_baselines_full_curve_seed.csv",
    "native_baselines_recall_cost_seed.csv",
    "native_baselines_manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_memory_gb() -> float:
    if os.name != "nt":
        return float("inf")
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return float(status.ullAvailPhys) / (1024.0**3)


def suite_is_complete(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "suite manifest is not available"
    payload = load_json(path)
    jobs = list(payload.get("jobs") or [])
    failed = [job for job in jobs if job.get("status") == "failed"]
    if failed:
        names = [f"{job.get('track')}:{job.get('task')}" for job in failed]
        raise RuntimeError(f"dependency suite failed: {names}")
    complete = bool(jobs) and all(job.get("status") == "complete" for job in jobs)
    counts = {
        state: sum(job.get("status") == state for job in jobs)
        for state in ("complete", "running", "planned", "failed")
    }
    return complete, json.dumps(counts, sort_keys=True)


def wait_for_resources(
    *,
    suite_manifest: Path | None,
    minimum_free_memory_gb: float,
    poll_seconds: int,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    while True:
        suite_ready = True
        suite_note = "not requested"
        if suite_manifest is not None:
            suite_ready, suite_note = suite_is_complete(suite_manifest)
        free_gb = free_memory_gb()
        memory_ready = free_gb >= minimum_free_memory_gb
        manifest["resource_wait"] = {
            "checked_at": utc_now(),
            "suite_manifest": str(suite_manifest) if suite_manifest else None,
            "suite_ready": suite_ready,
            "suite_note": suite_note,
            "minimum_free_memory_gb": minimum_free_memory_gb,
            "observed_free_memory_gb": round(free_gb, 3),
            "memory_ready": memory_ready,
        }
        manifest["status"] = "ready" if suite_ready and memory_ready else "waiting"
        write_json(manifest_path, manifest)
        if suite_ready and memory_ready:
            return
        time.sleep(max(5, poll_seconds))


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source_info = descriptor(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_info = descriptor(destination)
        if destination_info["sha256"] != source_info["sha256"]:
            raise ValueError(f"existing copied input has changed: {destination}")
    else:
        shutil.copy2(source, destination)
        destination_info = descriptor(destination)
    return {"source": source_info, "copy": destination_info}


def prepare_inputs(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    case1_root = args.output_root / "case1_transdiagnostic"
    case2_root = args.output_root / "case2_pathway_mediation"
    reused: dict[str, Any] = {}

    for name in CASE1_OFFICIAL_FILES:
        reused[f"case1_official/{name}"] = copy_verified(
            args.case1_source / "official_baselines" / name,
            case1_root / "official_baselines" / name,
        )
    for name in CASE1_NATIVE_FILES:
        reused[f"case1_native/{name}"] = copy_verified(
            args.case1_source / "native_baselines" / name,
            case1_root / "native_baselines" / name,
        )
    reused["case1_neurodiscovery_config"] = copy_verified(
        args.case1_config,
        case1_root / "frozen_inputs" / "neurodiscovery_config.json",
    )
    reused["case2_baseline_policies"] = copy_verified(
        args.case2_policies,
        case2_root / "official_baselines_reused" / "case2_search_policies.jsonl",
    )

    source_case1_manifest = load_json(
        args.case1_source
        / "internal_method_comparison"
        / "case1_method_comparison_manifest.json"
    )
    if sha256_file(args.case1_all_tests) != str(
        source_case1_manifest["all_tests_sha256"]
    ).upper():
        raise ValueError("CS1 exhaustive candidate table differs from the frozen source run")

    source_case2_manifest = load_json(args.case2_source_manifest)
    smoke_results = Path(str(source_case2_manifest["smoke_results_input"]))
    if not smoke_results.is_file():
        raise FileNotFoundError(smoke_results)

    manifest["frozen_reuse"] = {
        "policy": (
            "Reuse only outcome-blind baseline policies and immutable experimental "
            "readouts. Rebuild every KG-derived hypothesis, support score, ranking, "
            "feedback trace, and validation summary on the application release."
        ),
        "selection_release_precedes_application_release": True,
        "target_release_outcomes_used_for_policy_selection": False,
        "case1_candidate_table": descriptor(args.case1_all_tests),
        "case1_external_root": str(args.case1_external_root),
        "case2_smoke_results_path": str(smoke_results),
        "case2_source_mapping_manifest": descriptor(args.case2_source_manifest),
        "copied_inputs": reused,
    }
    manifest["resolved_inputs"] = {
        "case2_smoke_results": str(smoke_results),
    }


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    expected: tuple[Path, ...]


def completed_stage_is_valid(record: dict[str, Any], expected: Iterable[Path]) -> bool:
    if record.get("status") != "complete":
        return False
    recorded = record.get("outputs") or {}
    for path in expected:
        item = recorded.get(str(path.resolve()))
        if not path.is_file() or not item:
            return False
        if sha256_file(path) != str(item.get("sha256", "")).upper():
            return False
    return True


def run_stage(
    stage: Stage,
    *,
    output_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    stages = manifest.setdefault("stages", {})
    existing = stages.get(stage.name) or {}
    if completed_stage_is_valid(existing, stage.expected):
        return

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{stage.name}.stdout.log"
    stderr_path = log_dir / f"{stage.name}.stderr.log"
    record = {
        "status": "running",
        "started_at": utc_now(),
        "command": list(stage.command),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    stages[stage.name] = record
    manifest["status"] = "running"
    write_json(manifest_path, manifest)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            stage.command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    record["finished_at"] = utc_now()
    record["returncode"] = result.returncode
    if result.returncode != 0:
        record["status"] = "failed"
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(result.returncode, stage.command)

    missing = [str(path) for path in stage.expected if not path.is_file()]
    if missing:
        record["status"] = "failed"
        record["missing_outputs"] = missing
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise FileNotFoundError(f"{stage.name} did not create {missing}")
    record["outputs"] = {
        str(path.resolve()): descriptor(path) for path in stage.expected
    }
    record["status"] = "complete"
    write_json(manifest_path, manifest)


def build_stages(args: argparse.Namespace, manifest: dict[str, Any]) -> list[Stage]:
    python = str(args.python.resolve())
    case1_root = args.output_root / "case1_transdiagnostic"
    case2_root = args.output_root / "case2_pathway_mediation"
    case1_internal = case1_root / "internal_method_comparison"
    case1_external = case1_root / "external_method_comparison"
    case1_config = case1_root / "frozen_inputs" / "neurodiscovery_config.json"
    case2_generation = case2_root / "hypothesis_generation"
    case2_mapping = case2_root / "mapped_experiment"
    case2_comparison = case2_root / "comparison"
    audit_root = args.output_root / "audit"
    case2_smoke = Path(manifest["resolved_inputs"]["case2_smoke_results"])

    canonical_release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id="case2_pathway_mediation",
    )
    case2_generation.mkdir(parents=True, exist_ok=True)
    write_release_manifest(
        case2_generation / "canonical_kg_release.json", canonical_release
    )

    return [
        Stage(
            "case2_generate",
            (
                python,
                "-m",
                "neurooracle.src.hypothesis_cli",
                "--graph",
                str(args.kg),
                "--generation-seed",
                str(args.seed),
                "case-study",
                "case2_pathway_mediation",
                "--output-dir",
                str(case2_generation),
                "--stages",
                "batch",
            ),
            (case2_generation / "hypotheses_raw.json",),
        ),
        Stage(
            "case2_map",
            (
                python,
                str(
                    ROOT
                    / "neurooracle/scripts/map_case2_kg_hypotheses_to_adni_longitudinal.py"
                ),
                "--hypotheses",
                str(case2_generation / "hypotheses_raw.json"),
                "--canonical-release",
                str(case2_generation / "canonical_kg_release.json"),
                "--smoke-results",
                str(case2_smoke),
                "--output-root",
                str(case2_mapping),
                "--seed",
                str(args.seed),
            ),
            (
                case2_mapping / "manifest.json",
                case2_mapping / "kg_ranked_longitudinal_results.parquet",
            ),
        ),
        Stage(
            "case2_compare",
            (
                python,
                str(ROOT / "core/scripts/case2_method_comparison.py"),
                "--results",
                str(case2_mapping / "kg_ranked_longitudinal_results.parquet"),
                "--policies",
                str(
                    case2_root
                    / "official_baselines_reused"
                    / "case2_search_policies.jsonl"
                ),
                "--out-dir",
                str(case2_comparison),
                "--kg",
                str(args.kg),
                "--claims",
                str(args.claims),
                "--current-state",
                str(args.current_state),
                "--nd-trials",
                str(args.trials),
                "--nd-batch-size",
                "5",
                "--nd-prior",
                "evidence_consensus_v2",
                "--seed",
                str(args.seed),
                "--fail-on-collapse",
            ),
            (
                case2_comparison / "manifest.json",
                case2_comparison / "case2_method_metrics_aggregate.csv",
                case2_comparison / "case2_paired_comparisons.csv",
                case2_comparison / "case2_policy_independence_audit.json",
            ),
        ),
        Stage(
            "case1_internal",
            (
                python,
                str(ROOT / "core/scripts/case1_method_comparison.py"),
                "--all-tests",
                str(args.case1_all_tests),
                "--kg",
                str(args.kg),
                "--claims",
                str(args.claims),
                "--current-state",
                str(args.current_state),
                "--out-dir",
                str(case1_internal),
                "--neurodiscovery-config",
                str(case1_config),
                "--generation-first-dir",
                str(case1_root / "official_baselines"),
                "--seed",
                str(args.seed),
                "--trials",
                str(args.trials),
                "--skip-negative-feedback-ablation",
            ),
            (
                case1_internal / "case1_method_comparison_manifest.json",
                case1_internal / "case1_discovery_curves_by_trial.csv",
                case1_internal / "case1_method_summary_by_trial.csv",
            ),
        ),
        Stage(
            "case1_external",
            (
                python,
                str(ROOT / "core/scripts/case1_external_method_comparison.py"),
                "--all-tests",
                str(args.case1_all_tests),
                "--kg",
                str(args.kg),
                "--claims",
                str(args.claims),
                "--current-state",
                str(args.current_state),
                "--generation-first-dir",
                str(case1_root / "official_baselines"),
                "--neurodiscovery-config",
                str(case1_config),
                "--external-root",
                str(args.case1_external_root),
                "--external-run-name",
                args.case1_external_run_name,
                "--out-dir",
                str(case1_external),
                "--neurodiscovery-overlay-dir",
                str(case1_internal / "experimental_overlays"),
                "--trials",
                str(args.trials),
                "--seed",
                str(args.seed),
            ),
            (
                case1_external / "case1_external_method_comparison_manifest.json",
                case1_external / "case1_external_metrics_by_trial.csv",
                case1_external / "case1_external_recall_cost_by_trial.csv",
            ),
        ),
        Stage(
            "case1_finalize",
            (
                python,
                str(ROOT / "core/scripts/finalize_case1_rerun.py"),
                "--run-root",
                str(case1_root),
            ),
            (
                case1_root / "final_summary" / "case1_final_manifest.json",
                case1_root
                / "final_summary"
                / "internal_same_experiments_headline.csv",
                case1_root / "final_summary" / "internal_primary_p_values.csv",
                case1_root
                / "final_summary"
                / "external_pooled_same_experiments_headline.csv",
                case1_root / "final_summary" / "external_primary_p_values.csv",
            ),
        ),
        Stage(
            "dedicated_audit",
            (
                python,
                str(
                    ROOT
                    / "core/scripts/audit_dedicated_case_study_benchmarks.py"
                ),
                "--suite-root",
                str(args.output_root),
            ),
            (
                audit_root / "dedicated_audit.json",
                audit_root / "dedicated_audit_checks.csv",
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--case1-source", type=Path, default=DEFAULT_CASE1_SOURCE)
    parser.add_argument("--case1-config", type=Path, default=DEFAULT_CASE1_CONFIG)
    parser.add_argument("--case1-all-tests", type=Path, default=DEFAULT_CASE1_ALL_TESTS)
    parser.add_argument(
        "--case1-external-root", type=Path, default=DEFAULT_CASE1_EXTERNAL_ROOT
    )
    parser.add_argument("--case1-external-run-name", default="primary_mean_fd")
    parser.add_argument("--case2-policies", type=Path, default=DEFAULT_CASE2_POLICIES)
    parser.add_argument(
        "--case2-source-manifest", type=Path, default=DEFAULT_CASE2_SOURCE_MANIFEST
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--wait-suite-manifest", type=Path)
    parser.add_argument("--minimum-free-memory-gb", type=float, default=16.0)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "dedicated_suite_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "initializing",
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "trials": args.trials,
        "protocol": {
            "case_studies": [
                "case1_transdiagnostic",
                "case2_pathway_mediation",
            ],
            "case1_internal": "TCP exhaustive discovery recovery",
            "case1_external": "UCLA, COBRE, HCP-EP, and ADHD200",
            "case2_internal": "ADNI longitudinal pathway-imaging-outcome mediation",
            "case2_external": "not applicable: no independent matched cohort is registered",
            "formal_kg_mutation": False,
            "outcome_leakage_permitted": False,
        },
        "sources": {
            "runner": descriptor(Path(__file__).resolve()),
            "auditor": descriptor(
                ROOT / "core/scripts/audit_dedicated_case_study_benchmarks.py"
            ),
        },
    }
    manifest["sources"] = {
        "runner": descriptor(Path(__file__).resolve()),
        "auditor": descriptor(
            ROOT / "core/scripts/audit_dedicated_case_study_benchmarks.py"
        ),
        "case2_comparison": descriptor(
            ROOT / "core/scripts/case2_method_comparison.py"
        ),
    }

    canonical = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
    )
    actual = {
        name: str((canonical["files"][name])["sha256"]).upper()
        for name in CURRENT_CANONICAL_SHA256
    }
    if actual != {key: value.upper() for key, value in CURRENT_CANONICAL_SHA256.items()}:
        raise ValueError("canonical release does not match the registered application release")
    manifest["canonical_release"] = canonical
    write_json(manifest_path, manifest)

    if "frozen_reuse" not in manifest:
        prepare_inputs(args, manifest)
        write_json(manifest_path, manifest)

    wait_for_resources(
        suite_manifest=args.wait_suite_manifest,
        minimum_free_memory_gb=args.minimum_free_memory_gb,
        poll_seconds=args.poll_seconds,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    try:
        for stage in build_stages(args, manifest):
            run_stage(
                stage,
                output_root=args.output_root,
                manifest=manifest,
                manifest_path=manifest_path,
            )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at"] = utc_now()
    manifest.pop("error", None)
    write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
