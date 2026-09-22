"""Prepare and run the leakage-safe v4r2 adapted-baseline continuation.

``prepare`` locks a new exploratory protocol without opening the retrospective
publication corpus. ``discover`` loads only the five historical snapshots and
seals 200 baseline cells.  Retrospective evaluation must be invoked later in a
separate process with ``evaluate_hindcasting_v4.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Import the isolation adapter first.  It installs the minimal CLI-window shim
# before the unchanged legacy generator is imported.
from neurooracle.scripts.hindcasting_v4r2_adapted_baselines import (
    ADAPTER_VERSION,
    MAX_RANK,
    METHODS,
    POOL_SIZE,
    TASK_PROFILES,
    generate_adapted_hypotheses,
    sequence_pair_overlap,
)
from neurooracle.scripts.generate_case_study_frozen_baselines import (
    FrozenGraphIndex,
    load_semantic_claim_adjacencies,
)
from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.hindcasting_v4_feedback_boundary import (
    DISCOVERY_INPUT_SCHEMA,
    assert_discovery_payload_blind,
    build_discovery_seal,
    canonical_sha256,
    scan_discovery_source,
    sha256_file,
    validate_discovery_input_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
V4_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v4_computational_feedback_20260916"
)
V4R1_RELEASE = V4_ROOT / "execution/v4r1_two_tasks_teas5_20260916"
DEFAULT_RELEASE = V4_ROOT / "execution/v4r2r1_adapted_baselines_20260916"
DEFAULT_OUTPUT = ROOT / "neurooracle/data/hv4_runs/r2r1_adapted_baselines_20260916/discovery"
TASKS = ("case1_transdiagnostic", "biomarker_discovery")
SEEDS = tuple(range(10))
BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
TOP100_OVERLAP_LIMIT = 0.95
SCHEMA = "neurodiscovery-hindcasting-v4r2-adapted-baseline-discovery.v1"
CELL_SCHEMA = "neurodiscovery-hindcasting-v4r2-adapted-baseline-cell.v1"
FORBIDDEN_RUNTIME_MODULES = frozenset(
    {
        "neurooracle.scripts.case_study_hindcasting_eval",
        "neurooracle.scripts.evaluate_hindcasting_v4",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_record(
    path: Path,
    *,
    digest: str | None = None,
    deep_verify_at_start: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": str(digest or sha256_file(path)).casefold(),
        "deep_verify_at_start": bool(deep_verify_at_start),
    }


def write_locked(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    atomic_json(path, payload)
    digest = sha256_file(path)
    atomic_json(
        path.with_name(path.name + ".lock.json"),
        {
            "schema_version": "hindcasting-v4r2-file-lock.v1",
            "status": "locked",
            "locked_at": utc_now(),
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "immutable": True,
        },
    )
    return file_record(path, digest=digest, deep_verify_at_start=True)


def _v4r1_window_records(year: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    source = read_json(V4R1_RELEASE / "discovery_inputs" / f"kg_{year}.json")
    roles = {str(row["role"]): dict(row) for row in source["inputs"]}
    graph = roles["historical_kg"]
    claims = roles["historical_claims"]
    snapshot = dict(source["snapshot_manifest"])
    plan = read_json(V4R1_RELEASE / "execution_plan.json")
    window = next(
        item for item in plan["matrix"]["windows"] if int(item["freeze_year"]) == year
    )
    return graph, claims, snapshot, str(window["snapshot_root"])


def _source_bundle(release: Path) -> dict[str, Any]:
    explicit = [
        Path(__file__).resolve(),
        ROOT / "neurooracle/scripts/hindcasting_v4r2_adapted_baselines.py",
        ROOT / "neurooracle/scripts/generate_case_study_frozen_baselines.py",
        ROOT / "neurooracle/src/hindcasting_v4_feedback_boundary.py",
    ]
    for path in explicit[:4]:
        scan_discovery_source(path)
    forbidden_loaded = sorted(FORBIDDEN_RUNTIME_MODULES & set(sys.modules))
    if forbidden_loaded:
        raise RuntimeError(f"retrospective modules loaded during preparation: {forbidden_loaded}")
    module_files: dict[Path, list[str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        if not (module_name == "neurooracle" or module_name.startswith("neurooracle.")):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix != ".py":
            continue
        module_files.setdefault(path, []).append(module_name)
    files = sorted(set(explicit + list(module_files)))
    manifest = {
        "schema_version": "hindcasting-v4r2-discovery-source-bundle.v1",
        "status": "hash_locked_before_discovery",
        "created_at": utc_now(),
        "files": [file_record(path, deep_verify_at_start=True) for path in files],
        "source_scan": [scan_discovery_source(path) for path in explicit[:4]],
        "retrospective_evaluator_in_bundle": False,
        "retrospective_modules_loaded_during_preparation": forbidden_loaded,
        "runtime_dependency_modules": {
            str(path): sorted(names) for path, names in sorted(module_files.items())
        },
        "legacy_generator_cli_import_isolated_by_compatibility_shim": True,
    }
    path = release / "source_bundle_manifest.json"
    atomic_json(path, manifest)
    return file_record(path, deep_verify_at_start=True)


def prepare(release: Path, output: Path) -> dict[str, Any]:
    release = release.resolve()
    if release.exists():
        raise FileExistsError(f"v4r2 release already exists and is immutable: {release}")
    release.mkdir(parents=True, exist_ok=False)
    created_at = utc_now()

    task_cohort = write_locked(
        release / "task_cohort.json",
        {
            "schema_version": "hindcasting-v4r2-task-cohort.v1",
            "status": "inherited_from_locked_v4r1_without_later_literature",
            "created_at": created_at,
            "tasks": list(TASKS),
            "source_v4r1_task_cohort_sha256": sha256_file(V4R1_RELEASE / "task_cohort.json"),
            "later_literature_used_for_admission": False,
            "cohort_changed_from_v4r1": False,
        },
    )
    seed_schedule = write_locked(
        release / "seed_schedule.json",
        {
            "schema_version": "hindcasting-v4r2-seed-schedule.v1",
            "seeds": list(SEEDS),
            "paired_with_frozen_v4r1_neurodiscovery": True,
            "rng_rule": "method + task + seed + freeze_year hashed with SHA-256",
        },
    )
    method_config = write_locked(
        release / "method_config.json",
        {
            "schema_version": "hindcasting-v4r2-adapted-method-config.v1",
            "status": "locked_before_v4r2_discovery",
            "methods": list(METHODS),
            "display_names": {
                "sciagents_adapted": "SciAgents-adapted",
                "openscholar_rag_adapted": "OpenScholar-RAG-adapted",
            },
            "adapter_version": ADAPTER_VERSION,
            "maximum_rank": MAX_RANK,
            "candidate_pool_size": POOL_SIZE,
            "budgets": list(BUDGETS),
            "task_profiles": TASK_PROFILES,
            "task_conditioning_gate": {
                "rank": 100,
                "maximum_pair_jaccard": TOP100_OVERLAP_LIMIT,
                "identical_order_forbidden": True,
            },
            "baseline_feedback_enabled": False,
            "llm_api_enabled": False,
            "contemporary_model_forbidden_reason": (
                "model weights may encode publications after a historical cutoff"
            ),
        },
    )
    metric_contract = write_locked(
        release / "metric_contract.json",
        {
            "schema_version": "hindcasting-v4r2-teas5-comparability-contract.v1",
            "status": "inherited_unchanged_from_v4r1",
            "primary_metric": "TEAS-5",
            "definition": (
                "For each task x cutoff x seed cell and K, TEAS-5 = "
                "20 * min(unique exact primary discoveries among the first K, 5)."
            ),
            "budgets": list(BUDGETS),
            "frozen_v4r1_neurodiscovery_is_reused_without_regeneration": True,
            "source_v4r1_metric_sha256": sha256_file(V4R1_RELEASE / "metric_contract.json"),
        },
    )
    protocol = write_locked(
        release / "formal_hindcasting_design_v4r2.json",
        {
            "schema_version": "neurodiscovery-formal-hindcasting-design.v4r2",
            "protocol_id": "hindcasting_v4r2_adapted_baselines_20260916",
            "status": "exploratory_corrective_protocol_locked_before_rerun",
            "created_at": created_at,
            "reason": (
                "v4r1 inspection found a task-blind SciAgents proxy and an OpenScholar "
                "proxy without an explicit retrieval-feedback cycle"
            ),
            "post_v4r1_result_inspection": True,
            "technical_revision": (
                "v4r2r1 replaces the failed v4r2 source-bundle preflight; it "
                "removes an unused mutable KG source and blocks the legacy "
                "generator's transitive retrospective-evaluator import"
            ),
            "parent_v4r2_evaluation_corpus_opened": False,
            "confirmatory_claim_permitted": False,
            "methods_rerun": list(METHODS),
            "frozen_comparator": {
                "method": "neurodiscovery",
                "release": "v4r1_two_tasks_teas5_20260916",
                "regenerated": False,
            },
            "tasks": list(TASKS),
            "freeze_years": list(range(2016, 2021)),
            "evaluation_horizon_years": 5,
            "seeds": list(SEEDS),
            "maximum_rank": MAX_RANK,
            "method_labels_are_adaptations_not_native_reproductions": True,
            "sciagents_adaptation": (
                "task-scoped KG paths plus deterministic ontologist, scientist, "
                "critic, and ranker score decomposition"
            ),
            "openscholar_adaptation": (
                "task query, retrieval, deterministic pseudo-relevance feedback, "
                "re-retrieval, citation-aware synthesis, and a fixed hypothesis compiler"
            ),
            "upstream_method_references": {
                "sciagents": "https://doi.org/10.1002/adma.202413523",
                "openscholar": "https://doi.org/10.1038/s41586-025-10072-4",
            },
            "experimental_design_reference": (
                "Kassis et al. (2026), Scientific Critical Thinking, arXiv:2609.00065"
            ),
            "later_publications_available_during_discovery": False,
            "llm_api_calls_permitted": False,
            "discovery_and_evaluation_are_separate_processes": True,
            "retrospective_evaluator_module_imported_during_discovery": False,
            "v4r1_baseline_orders_reused": False,
            "future_labels_used_to_set_weights": False,
        },
    )
    source_bundle = _source_bundle(release)

    windows: list[dict[str, Any]] = []
    for year in range(2016, 2021):
        graph, claims, snapshot_manifest, snapshot_root = _v4r1_window_records(year)
        manifest = {
            "schema_version": DISCOVERY_INPUT_SCHEMA,
            "status": "locked_before_discovery",
            "freeze_year": year,
            "generator_mode": "deterministic_frozen",
            "llm_api_enabled": False,
            "evaluation_data_mounted": False,
            "evaluation_module_imported": False,
            "inputs": [
                graph,
                claims,
                {"role": "task_schema", **task_cohort},
                {
                    "role": "deterministic_generator",
                    **file_record(
                        ROOT / "neurooracle/scripts/hindcasting_v4r2_adapted_baselines.py"
                    ),
                },
                {"role": "method_config", **method_config},
                {"role": "seed_schedule", **seed_schedule},
            ],
            "snapshot_manifest": snapshot_manifest,
            "current_kg_candidate_score_columns_read": False,
        }
        validate_discovery_input_manifest(manifest)
        manifest_path = release / "discovery_inputs" / f"kg_{year}.json"
        atomic_json(manifest_path, manifest)
        windows.append(
            {
                "freeze_year": year,
                "future_start_year": year + 1,
                "future_end_year": year + 5,
                "snapshot_root": snapshot_root,
                "discovery_input_manifest": str(manifest_path.resolve()),
                "discovery_input_manifest_sha256": sha256_file(manifest_path),
            }
        )

    plan_path = release / "execution_plan.json"
    plan = {
        "schema_version": "neurodiscovery-hindcasting-v4r2-execution-plan.v1",
        "status": "locked_before_discovery",
        "created_at": created_at,
        "experiment_id": "hindcasting_v4r2r1_adapted_baselines_20260916",
        "protocol": protocol,
        "task_cohort": task_cohort,
        "metric_contract": metric_contract,
        "method_config": method_config,
        "discovery_source_bundle": source_bundle,
        "matrix": {
            "methods": list(METHODS),
            "tasks": list(TASKS),
            "seeds": list(SEEDS),
            "budgets": list(BUDGETS),
            "maximum_rank": MAX_RANK,
            "windows": windows,
            "total_cells": len(METHODS) * len(TASKS) * len(SEEDS) * len(windows),
        },
        "configuration": read_json(Path(method_config["path"])),
        "output_root": str(output.resolve()),
        "future_publication_corpus_recorded_in_discovery_plan": False,
        "llm_api_calls": 0,
        "frozen_v4r1_neurodiscovery_discovery_manifest": str(
            (ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/discovery/discovery_manifest.json").resolve()
        ),
    }
    atomic_json(plan_path, plan)
    lock = {
        "schema_version": "neurodiscovery-hindcasting-v4r2-execution-plan-lock.v1",
        "status": "locked",
        "locked_at": utc_now(),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "plan_bytes": plan_path.stat().st_size,
        "immutable": True,
        "later_publication_corpus_opened": False,
    }
    atomic_json(release / "execution_plan.lock.json", lock)
    report = {
        "status": "prepared_and_locked",
        "release": str(release),
        "plan": str(plan_path.resolve()),
        "plan_sha256": lock["plan_sha256"],
        "source_bundle_sha256": source_bundle["sha256"],
        "protocol_sha256": protocol["sha256"],
        "cells": plan["matrix"]["total_cells"],
        "later_publication_corpus_opened": False,
        "configuration_sha256": canonical_sha256(plan["configuration"]),
    }
    atomic_json(release / "PREEXECUTION_AUDIT.json", report)
    return report


def verify_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    plan = read_json(path)
    if plan.get("schema_version") != "neurodiscovery-hindcasting-v4r2-execution-plan.v1":
        raise ValueError("incompatible v4r2 plan")
    if plan.get("status") != "locked_before_discovery":
        raise ValueError("v4r2 plan is not locked")
    lock = read_json(path.with_name("execution_plan.lock.json"))
    if sha256_file(path) != str(lock.get("plan_sha256") or ""):
        raise ValueError("v4r2 plan differs from its lock")
    for key in ("protocol", "task_cohort", "metric_contract", "method_config"):
        record = plan[key]
        if sha256_file(Path(str(record["path"]))) != str(record["sha256"]):
            raise ValueError(f"locked {key} changed")
    source_record = plan["discovery_source_bundle"]
    source_path = Path(str(source_record["path"]))
    if sha256_file(source_path) != str(source_record["sha256"]):
        raise ValueError("source-bundle manifest changed")
    source_manifest = read_json(source_path)
    for record in source_manifest["files"]:
        member = Path(str(record["path"]))
        if member.stat().st_size != int(record["bytes"]):
            raise ValueError(f"discovery source size changed: {member}")
        if sha256_file(member) != str(record["sha256"]):
            raise ValueError(f"discovery source changed: {member}")
    return plan


def _prepare_hypotheses(
    payload: Mapping[str, Any],
    *,
    method: str,
    task_id: str,
    freeze_year: int,
    seed: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in payload.get("hypotheses") or ():
        source = str(raw.get("source_id") or "")
        target = str(raw.get("target_id") or "")
        pair = (source, target)
        if not source or not target or pair in seen:
            continue
        seen.add(pair)
        row = dict(raw)
        row["id"] = (
            f"V4R2:{method}:{task_id}:KG{freeze_year}:S{seed:02d}:"
            f"{len(prepared) + 1:04d}"
        )
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "case_study_id": task_id,
                "freeze_year": freeze_year,
                "discovery_method": method,
                "discovery_seed": seed,
                "publication_labels_available": False,
            }
        )
        row["metadata"] = metadata
        prepared.append(row)
        if len(prepared) == MAX_RANK:
            break
    if len(prepared) != MAX_RANK:
        raise RuntimeError(f"incomplete adapted sequence: {method}/{task_id}/{freeze_year}/{seed}")
    return prepared


def _complete_pointer(cell_root: Path) -> dict[str, Any] | None:
    pointer_path = cell_root / "COMPLETE.json"
    if not pointer_path.is_file():
        return None
    pointer = read_json(pointer_path)
    attempt = Path(str(pointer["attempt_dir"]))
    if sha256_file(attempt / "hypotheses_raw.json") != str(pointer["hypotheses_sha256"]):
        raise ValueError(f"completed hypotheses changed: {attempt}")
    seal = read_json(attempt / "discovery_seal.json")
    unsigned = dict(seal)
    stored = str(unsigned.pop("discovery_seal_sha256"))
    if canonical_sha256(unsigned) != stored:
        raise ValueError(f"completed seal changed: {attempt}")
    return pointer


def _next_attempt(cell_root: Path) -> Path:
    cell_root.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = cell_root / f"attempt_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"too many incomplete attempts: {cell_root}")


def discover(plan_path: Path) -> dict[str, Any]:
    forbidden_loaded = sorted(FORBIDDEN_RUNTIME_MODULES & set(sys.modules))
    if forbidden_loaded:
        raise RuntimeError(f"retrospective modules loaded before discovery: {forbidden_loaded}")
    window_module = sys.modules.get("neurooracle.scripts.run_case_study_hindcasting")
    if not getattr(window_module, "__hindcasting_discovery_shim__", False):
        raise RuntimeError("legacy hindcasting CLI was not isolated from discovery")
    plan = verify_plan(plan_path)
    output_root = Path(str(plan["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    differentiation: list[dict[str, Any]] = []
    total = int(plan["matrix"]["total_cells"])
    started_at = utc_now()
    source_sha = str(plan["discovery_source_bundle"]["sha256"])

    for window in plan["matrix"]["windows"]:
        freeze_year = int(window["freeze_year"])
        input_path = Path(str(window["discovery_input_manifest"]))
        input_manifest = read_json(input_path)
        validate_discovery_input_manifest(input_manifest)
        input_sha = sha256_file(input_path)
        snapshot = Path(str(window["snapshot_root"]))
        print(f"[load] adapted-baseline KG_{freeze_year}", flush=True)
        index = FrozenGraphIndex.load(snapshot / "knowledge_graph.json")
        _all_claims, literature, semantic_audit = load_semantic_claim_adjacencies(
            snapshot / "extracted_claims.jsonl", index, TASKS
        )

        for seed in SEEDS:
            for method in METHODS:
                generated_by_task: dict[str, dict[str, Any]] = {}
                reuse_by_task: dict[str, dict[str, Any]] = {}
                for task_id in TASKS:
                    cell_root = (
                        output_root
                        / method
                        / f"seed_{seed:02d}"
                        / task_id
                        / f"kg{freeze_year}_to_{freeze_year + 1}_{freeze_year + 5}"
                    )
                    pointer = _complete_pointer(cell_root)
                    if pointer is not None:
                        reuse_by_task[task_id] = pointer
                        payload = read_json(Path(str(pointer["row"]["hypotheses_path"])))
                        generated_by_task[task_id] = payload
                    else:
                        generated_by_task[task_id] = generate_adapted_hypotheses(
                            method=method,
                            case=case_study_by_name(task_id),
                            index=index,
                            graph_adjacency=index.adjacency,
                            scoped_claim_adjacency=literature.get(task_id, {}),
                            freeze_year=freeze_year,
                            seed=seed,
                            target_count=MAX_RANK,
                            pool_size=POOL_SIZE,
                        )

                overlap = sequence_pair_overlap(
                    generated_by_task[TASKS[0]], generated_by_task[TASKS[1]], k=100
                )
                gate_passed = (
                    float(overlap["jaccard"]) < TOP100_OVERLAP_LIMIT
                    and overlap["identical_order"] is False
                )
                differentiation_row = {
                    "method": method,
                    "freeze_year": freeze_year,
                    "seed": seed,
                    **overlap,
                    "maximum_jaccard": TOP100_OVERLAP_LIMIT,
                    "status": "passed" if gate_passed else "failed",
                }
                if not gate_passed:
                    raise RuntimeError(f"task-conditioning gate failed: {differentiation_row}")
                differentiation.append(differentiation_row)

                for task_id in TASKS:
                    if task_id in reuse_by_task:
                        rows.append(dict(reuse_by_task[task_id]["row"]))
                        print(
                            f"[reuse {len(rows)}/{total}] {method} {task_id} "
                            f"seed={seed} KG_{freeze_year}",
                            flush=True,
                        )
                        continue
                    cell_root = (
                        output_root
                        / method
                        / f"seed_{seed:02d}"
                        / task_id
                        / f"kg{freeze_year}_to_{freeze_year + 1}_{freeze_year + 5}"
                    )
                    attempt = _next_attempt(cell_root)
                    generated = generated_by_task[task_id]
                    hypotheses = _prepare_hypotheses(
                        generated,
                        method=method,
                        task_id=task_id,
                        freeze_year=freeze_year,
                        seed=seed,
                    )
                    payload = {
                        "metadata": {
                            "schema_version": CELL_SCHEMA,
                            "status": "discovery_complete",
                            "created_at": utc_now(),
                            "method": method,
                            "case_study_id": task_id,
                            "freeze_year": freeze_year,
                            "seed": seed,
                            "requested": MAX_RANK,
                            "valid": len(hypotheses),
                            "generator_mode": "deterministic_frozen_method_family_adaptation",
                            "llm_api_enabled": False,
                            "publication_labels_available": False,
                            "formal_kg_mutated": False,
                            "computational_feedback_used": False,
                            "adaptation_audit": generated["metadata"]["adaptation_audit"],
                            "task_differentiation_gate": differentiation_row,
                            "semantic_generation_audit": semantic_audit,
                            "discovery_input_manifest_sha256": input_sha,
                            "discovery_source_bundle_sha256": source_sha,
                        },
                        "hypotheses": hypotheses,
                    }
                    assert_discovery_payload_blind(payload, freeze_year=freeze_year)
                    hypotheses_path = attempt / "hypotheses_raw.json"
                    atomic_json(hypotheses_path, payload)
                    feedback_path = attempt / "computational_feedback.jsonl"
                    feedback_path.write_text("", encoding="utf-8")
                    feedback_sha = canonical_sha256(
                        {"method": method, "computational_feedback_used": False}
                    )
                    ordered_ids = [str(row["id"]) for row in hypotheses]
                    seal = build_discovery_seal(
                        freeze_year=freeze_year,
                        task_id=task_id,
                        method=method,
                        seed=seed,
                        ordered_hypothesis_ids=ordered_ids,
                        discovery_input_manifest_sha256=input_sha,
                        discovery_source_bundle_sha256=source_sha,
                        feedback_chain_sha256=feedback_sha,
                    )
                    seal_path = attempt / "discovery_seal.json"
                    atomic_json(seal_path, seal)
                    row = {
                        "method": method,
                        "seed": seed,
                        "case_study_id": task_id,
                        "freeze_year": freeze_year,
                        "future_start_year": freeze_year + 1,
                        "future_end_year": freeze_year + 5,
                        "hypotheses_path": str(hypotheses_path.resolve()),
                        "hypotheses_sha256": sha256_file(hypotheses_path),
                        "feedback_path": str(feedback_path.resolve()),
                        "feedback_sha256": sha256_file(feedback_path),
                        "discovery_seal_path": str(seal_path.resolve()),
                        "discovery_seal_sha256": str(seal["discovery_seal_sha256"]),
                        "ordered_hypothesis_ids_sha256": str(
                            seal["ordered_hypothesis_ids_sha256"]
                        ),
                        "feedback_activated": False,
                        "selection_changed_after_feedback": False,
                        "mapped_computational_hypotheses": 0,
                    }
                    pointer = {
                        "schema_version": "hindcasting-v4r2-complete-pointer.v1",
                        "status": "complete",
                        "attempt_dir": str(attempt.resolve()),
                        "hypotheses_sha256": row["hypotheses_sha256"],
                        "row": row,
                    }
                    atomic_json(cell_root / "COMPLETE.json", pointer)
                    rows.append(row)
                    atomic_json(
                        output_root / "discovery_state.json",
                        {
                            "schema_version": SCHEMA,
                            "status": "running",
                            "updated_at": utc_now(),
                            "completed_cells": len(rows),
                            "total_cells": total,
                            "percent": 100.0 * len(rows) / total,
                            "last_cell": row,
                        },
                    )
                    print(
                        f"[sealed {len(rows)}/{total}] {method} {task_id} "
                        f"seed={seed} KG_{freeze_year} overlap={overlap['jaccard']:.3f}",
                        flush=True,
                    )
        del index, literature, _all_claims

    if len(rows) != total:
        raise RuntimeError(f"discovery matrix incomplete: {len(rows)}/{total}")
    manifest = {
        "schema_version": SCHEMA,
        "status": "discovery_complete_all_cells_sealed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path.resolve()),
        "discovery_source_bundle": dict(plan["discovery_source_bundle"]),
        "methods": list(METHODS),
        "case_studies": list(TASKS),
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "windows": list(plan["matrix"]["windows"]),
        "runs": rows,
        "run_count": len(rows),
        "llm_api_calls": 0,
        "retrospective_evaluator_modules_loaded": sorted(
            FORBIDDEN_RUNTIME_MODULES & set(sys.modules)
        ),
        "publication_evaluation_data_loaded": False,
        "task_differentiation_audit": differentiation,
        "all_task_differentiation_gates_passed": all(
            row["status"] == "passed" for row in differentiation
        ),
        "frozen_v4r1_neurodiscovery_discovery_manifest": plan[
            "frozen_v4r1_neurodiscovery_discovery_manifest"
        ],
    }
    manifest_path = output_root / "discovery_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output_root / "discovery_state.json",
        {
            "schema_version": SCHEMA,
            "status": manifest["status"],
            "updated_at": utc_now(),
            "completed_cells": len(rows),
            "total_cells": total,
            "percent": 100.0,
            "discovery_manifest": str(manifest_path.resolve()),
            "discovery_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument(
        "--plan", type=Path, default=DEFAULT_RELEASE / "execution_plan.json"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(args.release, args.output)
    else:
        result = discover(args.plan)
    print(json.dumps({
        "status": result["status"],
        "cells": result.get("run_count", result.get("cells")),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
