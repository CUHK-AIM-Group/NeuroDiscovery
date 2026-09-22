"""Run the publication-blind discovery phase of Hindcasting v4r1.

The process receives only freeze-year KG snapshots, public TCP candidate
coordinates, and computational outcome vaults.  Each NeuroDiscovery batch is
persistently hash-committed before the matching computational outcomes are
revealed.  This file must remain import-independent from every retrospective
literature evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from neurooracle.scripts.generate_case_study_frozen_baselines import (
    FrozenGraphIndex,
    _merge_adjacencies,
    generate_case_hypotheses,
    load_semantic_claim_adjacencies,
)
from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.hindcasting_v4_computational_executor import (
    ComputationalMapping,
    ComputationalOutcomeVault,
    FeedbackRanker,
    PublicCandidateMapper,
    mapping_summary,
)
from neurooracle.src.hindcasting_v4_feedback_boundary import (
    COMPUTATIONAL_FEEDBACK_SCHEMA,
    assert_discovery_payload_blind,
    build_discovery_seal,
    canonical_sha256,
    sha256_file,
    validate_computational_feedback,
    validate_discovery_input_manifest,
)


SCHEMA = "neurodiscovery-hindcasting-v4r1-discovery.v1"
CELL_SCHEMA = "neurodiscovery-hindcasting-v4r1-discovery-cell.v1"


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


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"input byte-size mismatch: {path}")
    if bool(record.get("deep_verify_at_start", False)):
        if sha256_file(path) != str(record["sha256"]).casefold():
            raise ValueError(f"input SHA-256 mismatch: {path}")
    return path


def verify_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    plan = read_json(path)
    lock = read_json(path.with_name("execution_plan.lock.json"))
    if plan.get("schema_version") != "neurodiscovery-hindcasting-v4r1-execution-plan.v1":
        raise ValueError("incompatible v4r1 execution plan")
    if plan.get("status") != "locked_before_discovery":
        raise ValueError("v4r1 plan is not locked before discovery")
    if sha256_file(path) != str(lock.get("plan_sha256") or "").casefold():
        raise ValueError("execution plan differs from its lock")
    protocol = Path(str(plan["protocol"]["path"]))
    if sha256_file(protocol) != str(plan["protocol"]["sha256"]).casefold():
        raise ValueError("protocol differs from execution plan")
    source_manifest = Path(str(plan["discovery_source_bundle"]["path"]))
    if sha256_file(source_manifest) != str(
        plan["discovery_source_bundle"]["sha256"]
    ).casefold():
        raise ValueError("discovery source bundle manifest differs from plan")
    bundle = read_json(source_manifest)
    for record in bundle.get("files") or ():
        member = Path(str(record["path"]))
        if sha256_file(member) != str(record["sha256"]).casefold():
            raise ValueError(f"discovery source changed after locking: {member}")
    for record in plan["computational_resources"].values():
        _verify_record(record["public_candidates"])
        _verify_record(record["outcomes"])
    return plan


def _load_task_resources(plan: Mapping[str, Any]) -> dict[str, tuple[PublicCandidateMapper, ComputationalOutcomeVault]]:
    resources: dict[str, tuple[PublicCandidateMapper, ComputationalOutcomeVault]] = {}
    for task_id, record in plan["computational_resources"].items():
        public_record = record["public_candidates"]
        outcome_record = record["outcomes"]
        public = pd.read_csv(
            public_record["path"],
            usecols=list(public_record["columns_read"]),
            low_memory=False,
        )
        outcomes = pd.read_csv(
            outcome_record["path"],
            usecols=list(outcome_record["columns_read"]),
            low_memory=False,
        )
        resources[str(task_id)] = (
            PublicCandidateMapper(public),
            ComputationalOutcomeVault(outcomes, task_id=str(task_id)),
        )
    return resources


def _semantic_pair(hypothesis: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(hypothesis.get("source_id") or ""),
        str(hypothesis.get("target_id") or ""),
    )


def _prepare_hypotheses(
    payload: Mapping[str, Any],
    *,
    method: str,
    task_id: str,
    freeze_year: int,
    seed: int,
    budget: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for raw in payload.get("hypotheses") or ():
        metadata = dict(raw.get("metadata") or {})
        if raw.get("hypothesis_type") == "generation_failure" or metadata.get(
            "generation_failure"
        ):
            continue
        pair = _semantic_pair(raw)
        if not all(pair) or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        row = dict(raw)
        row["id"] = (
            f"V4R1:{method}:{task_id}:KG{freeze_year}:S{seed:02d}:"
            f"{len(prepared) + 1:04d}"
        )
        metadata.update(
            {
                "case_study_id": task_id,
                "freeze_year": int(freeze_year),
                "discovery_method": method,
                "discovery_seed": int(seed),
                "publication_labels_available": False,
            }
        )
        row["metadata"] = metadata
        prepared.append(row)
        if len(prepared) >= budget:
            break
    while len(prepared) < budget:
        rank = len(prepared) + 1
        prepared.append(
            {
                "id": f"V4R1:{method}:{task_id}:KG{freeze_year}:S{seed:02d}:FAIL:{rank:04d}",
                "hypothesis_type": "generation_failure",
                "source_id": "",
                "source_name": "",
                "target_id": "",
                "target_name": "",
                "path": [],
                "confidence_score": 0.0,
                "novelty_score": 0.0,
                "evidence_score": 0.0,
                "testability_score": 0.0,
                "composite_score": 0.0,
                "supporting_claims": [],
                "explanation": "No additional unique frozen-KG hypothesis was available.",
                "metadata": {
                    "case_study_id": task_id,
                    "freeze_year": int(freeze_year),
                    "discovery_method": method,
                    "discovery_seed": int(seed),
                    "generation_failure": True,
                    "publication_labels_available": False,
                },
            }
        )
    return prepared


def _normalized_base_scores(hypotheses: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [
            float(row.get("composite_score") or row.get("confidence_score") or 0.0)
            for row in hypotheses
        ],
        dtype=float,
    )
    values[~np.isfinite(values)] = 0.0
    if len(values) and float(np.ptp(values)) > 1e-12:
        values = (values - float(values.min())) / float(np.ptp(values))
    return values


def _feedback_record(
    *,
    hypothesis: Mapping[str, Any],
    mapping: ComputationalMapping,
    result: Mapping[str, Any],
    task_id: str,
    dataset_id: str,
    dataset_sha256: str,
    analysis_plan_sha256: str,
    pipeline_sha256: str,
    selection_commit_sha256: str,
    rank: int,
    previous_feedback_sha256: str,
) -> dict[str, Any]:
    experiment_id = mapping.candidate_id or f"UNMAPPED:{hypothesis['id']}"
    record: dict[str, Any] = {
        "schema_version": COMPUTATIONAL_FEEDBACK_SCHEMA,
        "source_kind": "computational_experiment",
        "hypothesis_id": str(hypothesis["id"]),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "executor": str(result.get("executor") or "registered_tcp_executor"),
        "analysis_plan_sha256": analysis_plan_sha256,
        "pipeline_sha256": pipeline_sha256,
        "selection_commit_sha256": selection_commit_sha256,
        "execution_status": str(result["execution_status"]),
        "feedback_status": str(result["feedback_status"]),
        "feedback_available": bool(result["feedback_available"]),
        "feedback_utility": result.get("feedback_utility"),
        "statistics": dict(result.get("statistics") or {}),
        "mapping_status": mapping.status,
        "mapping_reason": mapping.reason,
        "mapping_coordinate": {
            "candidate_id": mapping.candidate_id,
            "disease": mapping.disease,
            "modality": mapping.modality,
            "feature": mapping.feature,
            "feature_family": mapping.feature_family,
            "anatomy_tag": mapping.anatomy_tag,
        },
        "execution_rank": int(rank),
        "outcome_observed_after_selection": True,
        "publication_evaluation_fields_read": False,
        "previous_feedback_sha256": previous_feedback_sha256,
    }
    unsigned = dict(record)
    record["feedback_record_sha256"] = canonical_sha256(unsigned)
    return record


def _closed_loop_order(
    hypotheses: list[dict[str, Any]],
    mappings: list[ComputationalMapping],
    vault: ComputationalOutcomeVault,
    *,
    task_id: str,
    seed: int,
    freeze_year: int,
    batch_size: int,
    warmup_budget: int,
    feedback_weight: float,
    pair_feedback_weight: float,
    exploration_weight: float,
    dataset: Mapping[str, Any],
    analysis_plan_sha256: str,
    pipeline_sha256: str,
    attempt_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(hypotheses) != len(mappings):
        raise ValueError("hypotheses and computational mappings differ in length")
    base = _normalized_base_scores(hypotheses)
    rng = np.random.default_rng(20260916 + 1009 * seed + freeze_year)
    perturbation = rng.gumbel(0.0, 0.005, size=len(hypotheses))
    ranker = FeedbackRanker(
        mappings,
        feedback_weight=feedback_weight,
        pair_feedback_weight=pair_feedback_weight,
        exploration_weight=exploration_weight,
    )
    selected = np.zeros(len(hypotheses), dtype=bool)
    order: list[int] = []
    feedback_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    previous_commit = "0" * 64
    previous_feedback = "0" * 64
    batches_dir = attempt_dir / "batch_commitments"
    batches_dir.mkdir(parents=True, exist_ok=True)
    while len(order) < len(hypotheses):
        feedback_active = len(order) >= warmup_budget and ranker.informative_records > 0
        adjustments = (
            ranker.adjustments(total_observations=ranker.informative_records)
            if feedback_active
            else np.zeros(len(hypotheses), dtype=float)
        )
        scores = base + perturbation + adjustments
        scores[selected] = -np.inf
        remaining = np.flatnonzero(~selected)
        take = min(batch_size, len(remaining))
        chosen = remaining[np.lexsort((remaining, -scores[remaining]))][:take]
        counterfactual_scores = base + perturbation
        counterfactual_scores[selected] = -np.inf
        counterfactual = remaining[
            np.lexsort((remaining, -counterfactual_scores[remaining]))
        ][:take]
        selection_changed_by_feedback = bool(
            feedback_active and not np.array_equal(chosen, counterfactual)
        )
        candidate_ids = [str(hypotheses[int(index)]["id"]) for index in chosen]
        mapped_ids = [mappings[int(index)].candidate_id for index in chosen]
        commit = {
            "schema_version": "hindcasting-v4r1-selection-commitment.v1",
            "task_id": task_id,
            "seed": int(seed),
            "freeze_year": int(freeze_year),
            "batch": len(batch_rows),
            "start_rank": len(order) + 1,
            "end_rank": len(order) + len(chosen),
            "hypothesis_ids": candidate_ids,
            "mapped_computational_candidate_ids": mapped_ids,
            "feedback_active_for_selection": feedback_active,
            "informative_computational_records_before_selection": ranker.informative_records,
            "outcomes_read_before_commit": False,
            "previous_commit_sha256": previous_commit,
        }
        commit_sha = canonical_sha256(commit)
        committed = {**commit, "selection_commit_sha256": commit_sha}
        atomic_json(batches_dir / f"batch_{len(batch_rows):03d}.json", committed)
        previous_commit = commit_sha

        status_counts: Counter[str] = Counter()
        batch_feedback: list[str] = []
        for index in chosen:
            raw_index = int(index)
            hypothesis = hypotheses[raw_index]
            mapping = mappings[raw_index]
            result = vault.reveal(mapping)
            result = {**result, "executor": dataset["executor"]}
            rank = len(order) + 1
            record = _feedback_record(
                hypothesis=hypothesis,
                mapping=mapping,
                result=result,
                task_id=task_id,
                dataset_id=str(dataset["dataset_id"]),
                dataset_sha256=str(dataset["outcomes"]["sha256"]),
                analysis_plan_sha256=analysis_plan_sha256,
                pipeline_sha256=pipeline_sha256,
                selection_commit_sha256=commit_sha,
                rank=rank,
                previous_feedback_sha256=previous_feedback,
            )
            validate_computational_feedback(
                record,
                freeze_year=freeze_year,
                committed_candidate_ids=candidate_ids,
            )
            previous_feedback = str(record["feedback_record_sha256"])
            feedback_rows.append(record)
            batch_feedback.append(previous_feedback)
            status_counts[str(record["feedback_status"])] += 1
            ranker.update(mapping, result)
            selected[raw_index] = True
            order.append(raw_index)
        batch_rows.append(
            {
                "batch": len(batch_rows),
                "start_rank": len(order) - len(chosen) + 1,
                "end_rank": len(order),
                "selection_commit_sha256": commit_sha,
                "feedback_record_sha256s": batch_feedback,
                "feedback_active_for_selection": feedback_active,
                "status_counts": dict(sorted(status_counts.items())),
                "informative_records_after_batch": ranker.informative_records,
                "selection_adjustment_nonzero": bool(feedback_active and np.any(adjustments[chosen])),
                "selection_changed_by_feedback": selection_changed_by_feedback,
            }
        )
    ordered = [hypotheses[index] for index in order]
    return ordered, feedback_rows, {
        "batches": batch_rows,
        "batch_count": len(batch_rows),
        "warmup_budget": warmup_budget,
        "informative_records": ranker.informative_records,
        "feedback_activated": any(row["feedback_active_for_selection"] for row in batch_rows),
        "selection_changed_after_feedback": any(
            row["selection_changed_by_feedback"] for row in batch_rows
        ),
        "final_commit_chain_sha256": previous_commit,
        "final_feedback_chain_sha256": previous_feedback,
    }


def _write_feedback(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _next_attempt(cell_root: Path) -> Path:
    for index in range(1000):
        candidate = cell_root / f"attempt_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"too many incomplete attempts: {cell_root}")


def _complete_attempt(cell_root: Path) -> dict[str, Any] | None:
    pointer = cell_root / "COMPLETE.json"
    if not pointer.is_file():
        return None
    value = read_json(pointer)
    attempt = Path(str(value["attempt_dir"]))
    seal = read_json(attempt / "discovery_seal.json")
    hypotheses = attempt / "hypotheses_raw.json"
    if sha256_file(hypotheses) != str(value["hypotheses_sha256"]):
        raise ValueError(f"completed discovery hypotheses changed: {hypotheses}")
    unsigned = dict(seal)
    stored = str(unsigned.pop("discovery_seal_sha256"))
    if canonical_sha256(unsigned) != stored:
        raise ValueError(f"completed discovery seal changed: {attempt}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = verify_plan(args.plan)
    output_root = Path(str(plan["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_bundle_sha = str(plan["discovery_source_bundle"]["sha256"])
    analysis_plan_sha = str(plan["analysis_plan"]["sha256"])
    pipeline_sha = str(plan["pipeline"]["sha256"])
    resources = _load_task_resources(plan)
    methods = list(plan["matrix"]["methods"])
    tasks = list(plan["matrix"]["tasks"])
    seeds = [int(value) for value in plan["matrix"]["seeds"]]
    budget = int(plan["matrix"]["maximum_rank"])
    total_cells = len(methods) * len(tasks) * len(seeds) * len(plan["matrix"]["windows"])
    rows: list[dict[str, Any]] = []
    started_at = utc_now()

    for window in plan["matrix"]["windows"]:
        freeze_year = int(window["freeze_year"])
        input_manifest_path = Path(str(window["discovery_input_manifest"]))
        input_manifest = read_json(input_manifest_path)
        validate_discovery_input_manifest(input_manifest)
        input_manifest_sha = sha256_file(input_manifest_path)
        snapshot = Path(str(window["snapshot_root"]))
        graph_path = snapshot / "knowledge_graph.json"
        claims_path = snapshot / "extracted_claims.jsonl"
        print(f"[load] publication-blind KG_{freeze_year}", flush=True)
        index = FrozenGraphIndex.load(graph_path)
        claim_adjacency, literature, semantic_audit = load_semantic_claim_adjacencies(
            claims_path,
            index,
            tasks,
        )
        sciagents_adjacency = _merge_adjacencies(index.adjacency, claim_adjacency)
        for seed in seeds:
            for method in methods:
                for task_id in tasks:
                    cell_root = (
                        output_root
                        / method
                        / f"seed_{seed:02d}"
                        / task_id
                        / f"kg{freeze_year}_to_{freeze_year + 1}_{freeze_year + 5}"
                    )
                    complete = _complete_attempt(cell_root)
                    if complete is not None:
                        rows.append(dict(complete["row"]))
                        print(
                            f"[reuse {len(rows)}/{total_cells}] {method} {task_id} "
                            f"seed={seed} KG_{freeze_year}",
                            flush=True,
                        )
                        continue
                    attempt = _next_attempt(cell_root)
                    case = case_study_by_name(task_id)
                    adjacency = (
                        sciagents_adjacency
                        if method == "sciagents"
                        else literature.get(task_id, {})
                    )
                    generated = generate_case_hypotheses(
                        method=method,
                        case=case,
                        index=index,
                        adjacency=adjacency,
                        freeze_year=freeze_year,
                        target_count=budget,
                        seed=seed,
                        replicate_index=seed,
                        evidence_frontier_fraction=(0.35 if method == "neurodiscovery" else 0.0),
                        endpoint_canonical_quality_weight=(0.20 if method == "neurodiscovery" else 0.0),
                    )
                    hypotheses = _prepare_hypotheses(
                        generated,
                        method=method,
                        task_id=task_id,
                        freeze_year=freeze_year,
                        seed=seed,
                        budget=budget,
                    )
                    feedback_rows: list[dict[str, Any]] = []
                    loop_audit: dict[str, Any]
                    mapping_audit: dict[str, Any]
                    if method == "neurodiscovery":
                        mapper, vault = resources[task_id]
                        mappings = mapper.map_all(hypotheses)
                        mapping_audit = mapping_summary(mappings)
                        hypotheses, feedback_rows, loop_audit = _closed_loop_order(
                            hypotheses,
                            mappings,
                            vault,
                            task_id=task_id,
                            seed=seed,
                            freeze_year=freeze_year,
                            batch_size=int(plan["configuration"]["batch_size"]),
                            warmup_budget=int(plan["configuration"]["warmup_budget"]),
                            feedback_weight=float(plan["configuration"]["feedback_weight"]),
                            pair_feedback_weight=float(plan["configuration"]["pair_feedback_weight"]),
                            exploration_weight=float(plan["configuration"]["exploration_weight"]),
                            dataset=plan["computational_resources"][task_id],
                            analysis_plan_sha256=analysis_plan_sha,
                            pipeline_sha256=pipeline_sha,
                            attempt_dir=attempt,
                        )
                        feedback_chain_sha = str(loop_audit["final_feedback_chain_sha256"])
                    else:
                        mapping_audit = {
                            "mapper_version": None,
                            "total": 0,
                            "mapped": 0,
                            "unmapped": 0,
                            "reason_counts": {},
                        }
                        feedback_chain_sha = canonical_sha256(
                            {"method": method, "computational_feedback_used": False}
                        )
                        loop_audit = {
                            "batches": [],
                            "batch_count": 0,
                            "warmup_budget": None,
                            "informative_records": 0,
                            "feedback_activated": False,
                            "selection_changed_after_feedback": False,
                            "final_commit_chain_sha256": "0" * 64,
                            "final_feedback_chain_sha256": feedback_chain_sha,
                        }

                    payload = {
                        "metadata": {
                            "schema_version": CELL_SCHEMA,
                            "status": "discovery_complete",
                            "created_at": utc_now(),
                            "method": method,
                            "case_study_id": task_id,
                            "freeze_year": freeze_year,
                            "seed": seed,
                            "requested": budget,
                            "valid": sum(
                                row.get("hypothesis_type") != "generation_failure"
                                for row in hypotheses
                            ),
                            "generator_mode": "deterministic_frozen",
                            "llm_api_enabled": False,
                            "publication_labels_available": False,
                            "formal_kg_mutated": False,
                            "computational_feedback_used": method == "neurodiscovery",
                            "mapping_audit": mapping_audit,
                            "loop_audit": loop_audit,
                            "semantic_generation_audit": semantic_audit,
                            "discovery_input_manifest_sha256": input_manifest_sha,
                            "discovery_source_bundle_sha256": source_bundle_sha,
                        },
                        "hypotheses": hypotheses,
                    }
                    assert_discovery_payload_blind(payload, freeze_year=freeze_year)
                    hypotheses_path = attempt / "hypotheses_raw.json"
                    atomic_json(hypotheses_path, payload)
                    feedback_path = attempt / "computational_feedback.jsonl"
                    _write_feedback(feedback_path, feedback_rows)
                    ordered_ids = [str(row["id"]) for row in hypotheses]
                    seal = build_discovery_seal(
                        freeze_year=freeze_year,
                        task_id=task_id,
                        method=method,
                        seed=seed,
                        ordered_hypothesis_ids=ordered_ids,
                        discovery_input_manifest_sha256=input_manifest_sha,
                        discovery_source_bundle_sha256=source_bundle_sha,
                        feedback_chain_sha256=feedback_chain_sha,
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
                        "feedback_activated": bool(loop_audit["feedback_activated"]),
                        "selection_changed_after_feedback": bool(
                            loop_audit["selection_changed_after_feedback"]
                        ),
                        "mapped_computational_hypotheses": int(mapping_audit["mapped"]),
                    }
                    pointer = {
                        "schema_version": "hindcasting-v4r1-complete-pointer.v1",
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
                            "total_cells": total_cells,
                            "percent": 100.0 * len(rows) / total_cells,
                            "last_cell": row,
                        },
                    )
                    print(
                        f"[sealed {len(rows)}/{total_cells}] {method} {task_id} "
                        f"seed={seed} KG_{freeze_year} mapped={mapping_audit['mapped']}",
                        flush=True,
                    )
        del index, claim_adjacency, literature, sciagents_adjacency

    if len(rows) != total_cells:
        raise RuntimeError(f"discovery matrix incomplete: {len(rows)}/{total_cells}")
    manifest = {
        "schema_version": SCHEMA,
        "status": "discovery_complete_all_cells_sealed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan.resolve()),
        "discovery_source_bundle": dict(plan["discovery_source_bundle"]),
        "methods": methods,
        "case_studies": tasks,
        "seeds": seeds,
        "budgets": list(plan["matrix"]["budgets"]),
        "windows": list(plan["matrix"]["windows"]),
        "runs": rows,
        "run_count": len(rows),
        "llm_api_calls": 0,
        "publication_evaluation_data_loaded": False,
        "neurodiscovery_runs_with_feedback_activated": sum(
            row["method"] == "neurodiscovery" and row["feedback_activated"]
            for row in rows
        ),
        "neurodiscovery_runs_with_selection_change": sum(
            row["method"] == "neurodiscovery"
            and row["selection_changed_after_feedback"]
            for row in rows
        ),
    }
    manifest_path = output_root / "discovery_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output_root / "discovery_state.json",
        {
            "schema_version": SCHEMA,
            "status": "discovery_complete_all_cells_sealed",
            "updated_at": utc_now(),
            "completed_cells": len(rows),
            "total_cells": total_cells,
            "percent": 100.0,
            "discovery_manifest": str(manifest_path.resolve()),
            "discovery_manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps({"status": manifest["status"], "runs": len(rows)}, indent=2))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
