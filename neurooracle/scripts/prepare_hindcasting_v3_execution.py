"""Create and verify the immutable execution plan for one locked v3 cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.scripts.manage_hindcasting_v3 import verify_cohort
from neurooracle.src.experiment_source_bundle import sha256_file


ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826"
)
DEFAULT_COHORT = (
    V3_ROOT
    / "cohorts/locked_r1_two_tasks_20260826/cohort_manifest.json"
)
DEFAULT_PROTOCOL = V3_ROOT / "protocol/formal_hindcasting_design_v3.json"
DEFAULT_ROUTING = V3_ROOT / "protocol/llm_api_routing_policy_v2.json"
SCHEMA = "neurodiscovery-hindcasting-v3-execution-plan.v1"
MATRIX_SCHEMA = "neurodiscovery-hindcasting-v3-run-matrix.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _locked_file_record(
    path: Path,
    *,
    lock_name: str,
    hash_key: str,
) -> dict[str, Any]:
    path = path.resolve()
    lock_path = path.with_name(lock_name)
    lock = _read_json(lock_path)
    digest = sha256_file(path)
    _require(
        digest == str(lock.get(hash_key) or "").upper(),
        f"file differs from lock: {path}",
    )
    return {
        "path": str(path),
        "sha256": digest,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
    }


def _release_artifact_path(
    cohort: Mapping[str, Any], *, artifact: str
) -> tuple[Path, str]:
    release_record = cohort.get("kg_release") or {}
    release_manifest_path = Path(str(release_record["manifest_path"])).resolve()
    release_manifest = _read_json(release_manifest_path)
    artifact_record = (release_manifest.get("artifacts") or {})[artifact]
    path = (
        release_manifest_path.parent / str(artifact_record["relative_path"])
    ).resolve()
    _require(path.is_file(), f"missing release artifact: {path}")
    expected = str(artifact_record["sha256"]).upper()
    _require(
        path.stat().st_size == int(artifact_record["bytes"]),
        f"release artifact size mismatch: {path}",
    )
    return path, expected


def _matrix_rows(
    *,
    cohort_id: str,
    task_ids: Sequence[str],
    methods: Sequence[str],
    windows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    budgets: Sequence[int],
    formal_output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for window in windows:
            freeze = int(window["freeze_year"])
            start = int(window["future_start_year"])
            end = int(window["future_end_year"])
            label = f"kg{freeze}_to_{start}_{end}"
            for method in methods:
                for seed in seeds:
                    if method == "neurodiscovery":
                        artifact = (
                            formal_output_root
                            / "neurodiscovery"
                            / f"seed_{seed:02d}"
                            / task_id
                            / label
                            / "run_manifest.json"
                        )
                    else:
                        artifact = (
                            formal_output_root
                            / "baseline_evaluation"
                            / method
                            / f"seed_{seed:02d}"
                            / task_id
                            / label
                            / "hindcasting/metrics.json"
                        )
                    rows.append(
                        {
                            "cell_id": (
                                f"{cohort_id}:{task_id}:KG_{freeze}:"
                                f"{method}:seed_{seed:02d}"
                            ),
                            "task_id": task_id,
                            "freeze_year": freeze,
                            "future_start_year": start,
                            "future_end_year": end,
                            "method": method,
                            "seed": int(seed),
                            "experiment_counts": list(budgets),
                            "expected_primary_artifact": str(artifact.resolve()),
                        }
                    )
    return rows


def _validate_protocol(
    protocol: Mapping[str, Any], cohort: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "neurodiscovery-hindcasting-formal-design.v3",
        "incompatible v3 protocol",
    )
    _require(
        protocol.get("status") == "core_protocol_locked_before_v3_method_execution",
        "v3 protocol is not locked before execution",
    )
    methods = list((protocol.get("fixed_methods") or {}).get("primary") or ())
    seeds = [int(value) for value in (protocol.get("fixed_replication") or {}).get("seeds") or ()]
    budgets = [
        int(value)
        for value in (protocol.get("fixed_replication") or {}).get(
            "experiment_counts"
        )
        or ()
    ]
    windows = list(protocol.get("fixed_temporal_windows") or ())
    _require(methods == ["neurodiscovery", "sciagents", "openscholar_rag"], "unexpected method set")
    _require(seeds == list(range(10)), "unexpected seed set")
    _require(budgets == [10, 20, 50, 100, 200, 500, 1000], "unexpected budget set")
    _require(
        [int(row["freeze_year"]) for row in windows]
        == [2016, 2017, 2018, 2019, 2020],
        "unexpected temporal windows",
    )
    expected_runs = len(cohort["task_ids"]) * len(windows) * len(methods) * len(seeds)
    _require(
        int((cohort.get("run_matrix") or {}).get("total_method_runs") or -1)
        == expected_runs,
        "cohort and protocol matrix sizes differ",
    )
    return {
        "methods": methods,
        "seeds": seeds,
        "budgets": budgets,
        "windows": windows,
        "neurodiscovery": dict(protocol["locked_neurodiscovery_config"]),
    }


def create_execution_plan(
    *,
    cohort_path: Path,
    protocol_path: Path,
    routing_path: Path,
    output_dir: Path,
    source_bundle_dir: Path,
    formal_output_root: Path | None = None,
    smoke_output_root: Path | None = None,
    plan_label: str = "ops_v1",
    supersedes_plan_path: Path | None = None,
) -> dict[str, Any]:
    cohort_path = cohort_path.resolve()
    output_dir = output_dir.resolve()
    plan_path = output_dir / "execution_plan.json"
    if plan_path.exists():
        return verify_execution_plan(plan_path)

    cohort_verified = verify_cohort(
        cohort_path, deep_execution_assets=False
    )
    _require(cohort_verified["runnable"] is True, "cohort is not runnable")
    cohort = _read_json(cohort_path)
    protocol_record = _locked_file_record(
        protocol_path,
        lock_name="formal_hindcasting_design_v3.lock.json",
        hash_key="protocol_sha256",
    )
    routing_record = _locked_file_record(
        routing_path,
        lock_name="llm_api_routing_policy_v2.lock.json",
        hash_key="policy_sha256",
    )
    protocol = _read_json(protocol_path)
    contract = _validate_protocol(protocol, cohort)

    execution_assets = cohort["execution_assets"]
    year_assets = execution_assets["kge"]["year_assets"]
    snapshot_roots = {
        Path(row["snapshot"]["knowledge_graph"]["path"]).resolve().parent.parent
        for row in year_assets
    }
    kge_roots = {
        Path(row["checkpoint"]["path"]).resolve().parent for row in year_assets
    }
    _require(len(snapshot_roots) == 1, "snapshot assets do not share one root")
    _require(len(kge_roots) == 1, "KGE assets do not share one root")
    snapshot_root = next(iter(snapshot_roots))
    kge_root = next(iter(kge_roots))
    future_claims, future_claims_hash = _release_artifact_path(
        cohort, artifact="extracted_claims"
    )

    formal_output_root = (
        formal_output_root.resolve()
        if formal_output_root is not None
        else (V3_ROOT / "results" / cohort_verified["cohort_id"]).resolve()
    )
    smoke_output_root = (
        smoke_output_root.resolve()
        if smoke_output_root is not None
        else (V3_ROOT / "smoke" / cohort_verified["cohort_id"]).resolve()
    )
    matrix_path = output_dir / "run_matrix.json"
    rows = _matrix_rows(
        cohort_id=cohort_verified["cohort_id"],
        task_ids=cohort["task_ids"],
        methods=contract["methods"],
        windows=contract["windows"],
        seeds=contract["seeds"],
        budgets=contract["budgets"],
        formal_output_root=formal_output_root,
    )
    _atomic_json(
        matrix_path,
        {
            "schema_version": MATRIX_SCHEMA,
            "cohort_id": cohort_verified["cohort_id"],
            "rows": rows,
        },
    )

    audits = (cohort.get("method_blind_selection_evidence") or {}).get(
        "formal_audits"
    ) or {}
    created_at = datetime.now(timezone.utc).isoformat()
    plan = {
        "schema_version": SCHEMA,
        "status": "locked_before_v3_method_execution",
        "created_at": created_at,
        "experiment_id": (
            f"hindcasting_v3::{cohort_verified['cohort_id']}::{plan_label}"
        ),
        "operational_plan_label": plan_label,
        "supersedes": (
            {
                "path": str(supersedes_plan_path.resolve()),
                "sha256": sha256_file(supersedes_plan_path.resolve()),
                "reason": "shorter Windows-safe execution and source-bundle roots before formal method execution",
                "formal_results_generated_by_superseded_plan": False,
            }
            if supersedes_plan_path is not None
            else None
        ),
        "protocol": protocol_record,
        "llm_api": {
            "required_by_current_implementations": False,
            "reason": (
                "All three locked implementations are deterministic local "
                "KG/literature algorithms; route only if a future in-scope step "
                "actually requires a large-model request."
            ),
            "routing_policy_if_required": routing_record,
        },
        "cohort": {
            "cohort_id": cohort_verified["cohort_id"],
            "path": str(cohort_path),
            "sha256": cohort_verified["cohort_manifest_sha256"],
            "lock_path": str(cohort_path.with_name("cohort.lock.json")),
            "lock_sha256": sha256_file(cohort_path.with_name("cohort.lock.json")),
            "task_ids": list(cohort["task_ids"]),
        },
        "matrix": {
            "methods": contract["methods"],
            "seeds": contract["seeds"],
            "experiment_counts": contract["budgets"],
            "windows": contract["windows"],
            "tasks": len(cohort["task_ids"]),
            "total_method_runs": len(rows),
            "path": str(matrix_path.resolve()),
            "sha256": sha256_file(matrix_path),
        },
        "inputs": {
            "snapshot_root": str(snapshot_root),
            "kge_root": str(kge_root),
            "kge_checkpoint_pattern": contract["neurodiscovery"][
                "kge_checkpoint_pattern"
            ],
            "future_claims": {
                "path": str(future_claims),
                "sha256": future_claims_hash,
                "bytes": future_claims.stat().st_size,
            },
            "dynamic_eligibility": audits["dynamic_eligibility"],
            "static_eligibility": audits["static_eligibility"],
            "eligibility_pipeline": execution_assets["eligibility_pipeline"],
            "kge_assets_manifest": execution_assets["kge"]["manifest"],
        },
        "implementations": {
            "neurodiscovery": "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py",
            "sciagents": "neurooracle/scripts/generate_case_study_frozen_baselines.py",
            "openscholar_rag": "neurooracle/scripts/generate_case_study_frozen_baselines.py",
            "baseline_evaluation": "neurooracle/scripts/evaluate_case_study_frozen_baselines.py",
            "formal_runner": "neurooracle/scripts/run_hindcasting_v3.py",
        },
        "configuration": {
            "neurodiscovery": contract["neurodiscovery"],
            "baselines": {
                "target_per_case_study": 1000,
                "feedback_enabled": False,
                "random_trials": 1000,
            },
            "pythonhashseed": 0,
        },
        "smoke": {
            "formal_result": False,
            "task_id": cohort["task_ids"][0],
            "window": contract["windows"][0],
            "seed": 0,
            "experiment_counts": [10, 20, 50, 100],
            "maximum_executions": 100,
            "output_root": str(smoke_output_root),
        },
        "outputs": {
            "formal_root": str(formal_output_root),
            "source_bundle_manifest": str(
                source_bundle_dir.resolve() / "bundle_manifest.json"
            ),
        },
        "resume_contract": {
            "required_identity": [
                "protocol_sha256",
                "cohort_sha256",
                "source_bundle_manifest_sha256",
                "snapshot_sha256",
                "kge_checkpoint_sha256",
            ],
            "mismatched_checkpoint_action": "refuse_reuse_and_require_explicit_force",
        },
    }
    _atomic_json(plan_path, plan)
    _atomic_json(
        output_dir / "execution_plan.lock.json",
        {
            "schema_version": "neurodiscovery-hindcasting-v3-execution-plan-lock.v1",
            "cohort_id": cohort_verified["cohort_id"],
            "locked_at": created_at,
            "plan": plan_path.name,
            "plan_sha256": sha256_file(plan_path),
            "immutable": True,
        },
    )
    return verify_execution_plan(plan_path)


def verify_execution_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = _read_json(plan_path)
    lock_path = plan_path.with_name("execution_plan.lock.json")
    lock = _read_json(lock_path)
    digest = sha256_file(plan_path)
    _require(plan.get("schema_version") == SCHEMA, "incompatible execution plan")
    _require(
        plan.get("status") == "locked_before_v3_method_execution",
        "execution plan is not locked",
    )
    _require(
        digest == str(lock.get("plan_sha256") or "").upper(),
        "execution plan differs from lock",
    )
    for label in ("protocol",):
        record = plan[label]
        _require(
            sha256_file(Path(record["path"])) == record["sha256"],
            f"execution plan {label} hash mismatch",
        )
    route = plan["llm_api"]["routing_policy_if_required"]
    _require(
        sha256_file(Path(route["path"])) == route["sha256"],
        "routing policy hash mismatch",
    )
    cohort_record = plan["cohort"]
    cohort_verified = verify_cohort(
        Path(cohort_record["path"]), deep_execution_assets=False
    )
    _require(cohort_verified["runnable"] is True, "plan cohort is not runnable")
    _require(
        cohort_verified["cohort_manifest_sha256"] == cohort_record["sha256"],
        "plan cohort hash mismatch",
    )
    matrix_record = plan["matrix"]
    matrix_path = Path(matrix_record["path"])
    _require(
        sha256_file(matrix_path) == matrix_record["sha256"],
        "run matrix hash mismatch",
    )
    matrix = _read_json(matrix_path)
    rows = list(matrix.get("rows") or ())
    _require(matrix.get("schema_version") == MATRIX_SCHEMA, "incompatible run matrix")
    _require(
        len(rows) == int(matrix_record["total_method_runs"]),
        "run matrix row count mismatch",
    )
    _require(len({row["cell_id"] for row in rows}) == len(rows), "duplicate cell IDs")
    return {
        "status": plan["status"],
        "experiment_id": plan["experiment_id"],
        "plan": str(plan_path),
        "plan_sha256": digest,
        "cohort_id": cohort_record["cohort_id"],
        "tasks": list(cohort_record["task_ids"]),
        "total_method_runs": len(rows),
        "source_bundle_manifest": plan["outputs"]["source_bundle_manifest"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--routing-policy", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-bundle-dir", type=Path)
    parser.add_argument("--formal-output-root", type=Path)
    parser.add_argument("--smoke-output-root", type=Path)
    parser.add_argument("--plan-label", default="ops_v1")
    parser.add_argument("--supersedes-plan", type=Path)
    parser.add_argument("--verify", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify is not None:
        result = verify_execution_plan(args.verify)
    else:
        cohort_id = _read_json(args.cohort)["cohort_id"]
        output_dir = args.output_dir or V3_ROOT / "execution" / cohort_id
        source_bundle_dir = (
            args.source_bundle_dir or V3_ROOT / "source_bundles" / cohort_id
        )
        result = create_execution_plan(
            cohort_path=args.cohort,
            protocol_path=args.protocol,
            routing_path=args.routing_policy,
            output_dir=output_dir,
            source_bundle_dir=source_bundle_dir,
            formal_output_root=args.formal_output_root,
            smoke_output_root=args.smoke_output_root,
            plan_label=args.plan_label,
            supersedes_plan_path=args.supersedes_plan,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
