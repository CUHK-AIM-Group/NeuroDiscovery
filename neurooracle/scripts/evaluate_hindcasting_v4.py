"""Evaluate sealed Hindcasting v4r1 discoveries against later publications.

This command is intentionally separate from discovery.  It refuses to open
the later-publication corpus until every requested method/task/year/seed cell
has a valid discovery seal and immutable hypothesis hash.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.scripts.case_study_hindcasting_eval import (
    _future_indexes,
    _historical_pairs,
    evaluate,
    load_future_claim_records,
)
from neurooracle.scripts.temporal_hindcasting_core import load_kg_index
from neurooracle.src.hindcasting_v4_feedback_boundary import (
    EVALUATION_HANDOFF_SCHEMA,
    canonical_sha256,
    sha256_file,
    validate_evaluation_handoff,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISCOVERY = (
    ROOT
    / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/discovery"
    / "discovery_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/evaluation"
DEFAULT_FUTURE = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826/releases"
    / "kg_20260825_2c02732582da_705b0799/extracted_claims.jsonl"
)
AUXILIARY_RANDOM_TRIALS = 1000
INHERITED_EVALUATOR_SOURCE = ROOT / "neurooracle/scripts/case_study_hindcasting_eval.py"


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


def _validate_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    unsigned = dict(seal)
    stored = str(unsigned.pop("discovery_seal_sha256", ""))
    if not stored or canonical_sha256(unsigned) != stored:
        raise ValueError(f"invalid discovery seal: {path}")
    if seal.get("status") != "discovery_complete":
        raise ValueError(f"incomplete discovery seal: {path}")
    return seal


def _preflight(discovery_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    discovery_path = discovery_path.resolve()
    manifest = read_json(discovery_path)
    if manifest.get("status") != "discovery_complete_all_cells_sealed":
        raise ValueError("all discovery cells must be sealed before evaluation")

    plan_path = Path(str(manifest["plan_path"]))
    if sha256_file(plan_path) != str(manifest["plan_sha256"]):
        raise ValueError(f"locked discovery plan changed: {plan_path}")
    source_bundle = dict(manifest["discovery_source_bundle"])
    source_bundle_path = Path(str(source_bundle["path"]))
    if sha256_file(source_bundle_path) != str(source_bundle["sha256"]):
        raise ValueError(f"discovery source-bundle manifest changed: {source_bundle_path}")
    source_manifest = read_json(source_bundle_path)
    if source_manifest.get("status") != "hash_locked_before_discovery":
        raise ValueError("discovery source bundle was not locked before discovery")
    for source in source_manifest.get("files") or ():
        source_path = Path(str(source["path"]))
        if source_path.stat().st_size != int(source["bytes"]):
            raise ValueError(f"discovery source size changed: {source_path}")
        if sha256_file(source_path) != str(source["sha256"]):
            raise ValueError(f"discovery source changed: {source_path}")

    windows_by_year: dict[int, dict[str, Any]] = {}
    for raw_window in manifest["windows"]:
        window = dict(raw_window)
        freeze_year = int(window["freeze_year"])
        if freeze_year in windows_by_year:
            raise ValueError(f"duplicate discovery window: {freeze_year}")
        input_manifest_path = Path(str(window["discovery_input_manifest"]))
        if sha256_file(input_manifest_path) != str(window["discovery_input_manifest_sha256"]):
            raise ValueError(f"discovery-input manifest changed: {input_manifest_path}")
        windows_by_year[freeze_year] = window

    rows = [dict(row) for row in manifest.get("runs") or ()]
    expected = (
        len(manifest["methods"])
        * len(manifest["case_studies"])
        * len(manifest["seeds"])
        * len(manifest["windows"])
    )
    if len(rows) != expected or len(rows) != int(manifest.get("run_count", -1)):
        raise ValueError(f"discovery matrix is incomplete: {len(rows)}/{expected}")
    expected_identities = {
        (str(method), str(task), int(window["freeze_year"]), int(seed))
        for method in manifest["methods"]
        for task in manifest["case_studies"]
        for seed in manifest["seeds"]
        for window in manifest["windows"]
    }
    identities: set[tuple[str, str, int, int]] = set()
    sequence_registry: list[dict[str, Any]] = []
    for row in rows:
        identity = (
            str(row["method"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["seed"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate discovery identity: {identity}")
        identities.add(identity)
        hypotheses_path = Path(str(row["hypotheses_path"]))
        feedback_path = Path(str(row["feedback_path"]))
        seal_path = Path(str(row["discovery_seal_path"]))
        if sha256_file(hypotheses_path) != str(row["hypotheses_sha256"]):
            raise ValueError(f"discovery hypotheses changed: {hypotheses_path}")
        if sha256_file(feedback_path) != str(row["feedback_sha256"]):
            raise ValueError(f"computational-feedback record changed: {feedback_path}")
        seal = _validate_seal(seal_path)
        if str(seal["discovery_seal_sha256"]) != str(row["discovery_seal_sha256"]):
            raise ValueError(f"discovery seal identity mismatch: {seal_path}")
        expected_window = windows_by_year[int(row["freeze_year"])]
        if (
            str(seal["method"]) != str(row["method"])
            or str(seal["task_id"]) != str(row["case_study_id"])
            or int(seal["freeze_year"]) != int(row["freeze_year"])
            or int(seal["seed"]) != int(row["seed"])
            or str(seal["discovery_input_manifest_sha256"])
            != str(expected_window["discovery_input_manifest_sha256"])
            or str(seal["discovery_source_bundle_sha256"]) != str(source_bundle["sha256"])
            or seal.get("evaluation_data_loaded") is not False
        ):
            raise ValueError(f"discovery seal is not bound to the locked cell inputs: {seal_path}")
        payload = read_json(hypotheses_path)
        ordered_ids = [str(item["id"]) for item in payload.get("hypotheses") or ()]
        if len(ordered_ids) != max(int(value) for value in manifest["budgets"]):
            raise ValueError(f"unexpected sealed sequence length: {hypotheses_path}")
        if canonical_sha256(ordered_ids) != str(seal["ordered_hypothesis_ids_sha256"]):
            raise ValueError(f"hypothesis sequence differs from seal: {hypotheses_path}")
        if ordered_ids != list(seal["ordered_hypothesis_ids"]):
            raise ValueError(f"hypothesis order differs from seal: {hypotheses_path}")
        sequence_registry.append(
            {
                "identity": list(identity),
                "hypotheses_sha256": str(row["hypotheses_sha256"]),
                "seal_sha256": str(row["discovery_seal_sha256"]),
                "sequence_sha256": str(row["ordered_hypothesis_ids_sha256"]),
            }
        )
    if identities != expected_identities:
        missing = expected_identities - identities
        extra = identities - expected_identities
        raise ValueError(f"discovery identity matrix mismatch: missing={missing}, extra={extra}")
    sequence_registry.sort(key=lambda value: tuple(map(str, value["identity"])))
    return manifest, rows, canonical_sha256(sequence_registry)


def _labels_digest(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        item = (
            str(row.get("id") or ""),
            str(row.get("subject_id") or ""),
            str(row.get("object_id") or ""),
            str(row.get("predicate") or ""),
            str(row.get("year") or ""),
        )
        digest.update("\x1f".join(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _permuted_labels_digest(records: Sequence[Mapping[str, Any]]) -> str:
    order = list(range(len(records)))
    random.Random(20260916).shuffle(order)
    digest = hashlib.sha256()
    for index, source_index in enumerate(order):
        identity = str(records[index].get("id") or "")
        source = records[source_index]
        item = (
            identity,
            str(source.get("subject_id") or ""),
            str(source.get("object_id") or ""),
            str(source.get("predicate") or ""),
            str(source.get("year") or ""),
        )
        digest.update("\x1f".join(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, rows, pre_sequence_hash = _preflight(args.discovery_manifest)
    sealed_at = str(manifest["completed_at"])
    opened_at = utc_now()
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_root / "PRE_EVALUATION_GATE.json",
        {
            "schema_version": "hindcasting-v4r1-pre-evaluation-gate.v1",
            "status": "passed",
            "checked_at": opened_at,
            "discovery_completed_at": sealed_at,
            "discovery_manifest": str(args.discovery_manifest.resolve()),
            "discovery_manifest_sha256": sha256_file(args.discovery_manifest.resolve()),
            "sealed_cells": len(rows),
            "all_cells_sealed_before_evaluation": True,
            "sequence_registry_sha256": pre_sequence_hash,
            "locked_plan_sha256": str(manifest["plan_sha256"]),
            "deep_verified_discovery_source_bundle_sha256": str(
                manifest["discovery_source_bundle"]["sha256"]
            ),
            "computational_feedback_files_verified": len(rows),
            "auxiliary_random_trials": AUXILIARY_RANDOM_TRIALS,
            "auxiliary_random_trials_affect_locked_teas5": False,
            "evaluation_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "inherited_evaluator_source": {
                "path": str(INHERITED_EVALUATOR_SOURCE.resolve()),
                "sha256": sha256_file(INHERITED_EVALUATOR_SOURCE.resolve()),
            },
        },
    )

    # This is the first evaluation-stage access to the later-publication file.
    future_sha = sha256_file(args.future_claims)
    future_records = load_future_claim_records(
        args.future_claims,
        min_year=min(int(row["future_start_year"]) for row in rows),
        max_year=max(int(row["future_end_year"]) for row in rows),
    )
    original_label_sha = _labels_digest(future_records)
    permuted_label_sha = _permuted_labels_digest(future_records)
    if original_label_sha == permuted_label_sha:
        raise RuntimeError("deterministic label permutation did not change labels")

    evaluated: list[dict[str, Any]] = []
    windows = sorted({int(row["freeze_year"]) for row in rows})
    window_manifest_by_year = {
        int(window["freeze_year"]): dict(window) for window in manifest["windows"]
    }
    for freeze_year in windows:
        year_rows = [row for row in rows if int(row["freeze_year"]) == freeze_year]
        snapshot = Path(str(window_manifest_by_year[freeze_year]["snapshot_root"]))
        kg_path = snapshot / "knowledge_graph.json"
        historical_claims = snapshot / "extracted_claims.jsonl"
        print(f"[index] retrospective evaluation KG_{freeze_year}", flush=True)
        kg_index = load_kg_index(kg_path)
        concepts, edges, _ = kg_index
        historical_counter: Counter[str] = Counter()
        endpoint_atoms: dict[str, set[str]] = {}
        historical_pairs = _historical_pairs(
            edges,
            claims_path=historical_claims,
            concepts=concepts,
            stats=historical_counter,
            endpoint_atoms_out=endpoint_atoms,
        )
        by_task: dict[str, tuple[dict[str, Any], dict[str, int]]] = {}
        for task_id in sorted({str(row["case_study_id"]) for row in year_rows}):
            by_task[task_id] = _future_indexes(
                args.future_claims,
                concepts,
                historical_pairs,
                freeze_year + 1,
                freeze_year + 5,
                case_study_id=task_id,
                future_records=future_records,
                historical_endpoint_atoms=endpoint_atoms,
            )
        for cell_index, row in enumerate(year_rows, 1):
            method = str(row["method"])
            task_id = str(row["case_study_id"])
            seed = int(row["seed"])
            output = (
                args.output_root
                / method
                / f"seed_{seed:02d}"
                / task_id
                / f"kg{freeze_year}_to_{freeze_year + 1}_{freeze_year + 5}"
            )
            metrics_path = output / "metrics.json"
            if metrics_path.exists():
                raise FileExistsError(
                    f"evaluation result already exists; use a successor output: {metrics_path}"
                )
            future_index, future_stats = by_task[task_id]
            metrics = evaluate(
                kg_path=kg_path,
                hypotheses_path=Path(str(row["hypotheses_path"])),
                future_claims_path=args.future_claims,
                output_dir=output,
                freeze_year=freeze_year,
                future_start_year=freeze_year + 1,
                future_end_year=freeze_year + 5,
                top_ks=list(args.top_k),
                # The inherited evaluator requires at least one trial.  This
                # deterministic auxiliary baseline is not used by locked TEAS-5.
                random_trials=AUXILIARY_RANDOM_TRIALS,
                seed=seed,
                case_study_id=task_id,
                method=method,
                kg_index=kg_index,
                historical_pairs=historical_pairs,
                historical_claims_path=historical_claims,
                historical_stats=dict(historical_counter),
                historical_endpoint_atoms=endpoint_atoms,
                future_index=future_index,
                future_stats=future_stats,
            )
            seal = _validate_seal(Path(str(row["discovery_seal_path"])))
            handoff = {
                "schema_version": EVALUATION_HANDOFF_SCHEMA,
                "status": "evaluation_complete",
                "created_at": utc_now(),
                "discovery_seal_sha256": str(row["discovery_seal_sha256"]),
                "evaluation_loaded_after_discovery_seal": True,
                "discovery_completed_at": sealed_at,
                "evaluation_corpus_first_opened_at": opened_at,
                "evaluation_corpus_sha256": future_sha,
                "metrics_path": str(metrics_path.resolve()),
            }
            validate_evaluation_handoff(seal, handoff)
            metrics["v4r1_execution_identity"] = {
                "discovery_manifest_sha256": sha256_file(args.discovery_manifest.resolve()),
                "discovery_seal_sha256": str(row["discovery_seal_sha256"]),
                "hypotheses_sha256": str(row["hypotheses_sha256"]),
                "evaluation_corpus_sha256": future_sha,
                "evaluation_loaded_after_discovery_seal": True,
                "method": method,
                "task_id": task_id,
                "seed": seed,
                "freeze_year": freeze_year,
                "auxiliary_random_trials": AUXILIARY_RANDOM_TRIALS,
                "auxiliary_random_trials_affect_locked_teas5": False,
            }
            atomic_json(metrics_path, metrics)
            atomic_json(output / "evaluation_handoff.json", handoff)
            evaluated.append(
                {
                    **{key: row[key] for key in (
                        "method",
                        "seed",
                        "case_study_id",
                        "freeze_year",
                        "future_start_year",
                        "future_end_year",
                        "discovery_seal_sha256",
                        "ordered_hypothesis_ids_sha256",
                    )},
                    "metrics_path": str(metrics_path.resolve()),
                    "metrics_sha256": sha256_file(metrics_path),
                    "evaluation_handoff": str((output / "evaluation_handoff.json").resolve()),
                }
            )
            print(
                f"[evaluated {cell_index}/{len(year_rows)}] {method} {task_id} "
                f"seed={seed} KG_{freeze_year}",
                flush=True,
            )
        del kg_index, concepts, edges, historical_pairs, endpoint_atoms, by_task

    _, _, post_sequence_hash = _preflight(args.discovery_manifest)
    invariance = {
        "schema_version": "hindcasting-v4r1-label-permutation-invariance.v1",
        "status": "passed" if post_sequence_hash == pre_sequence_hash else "failed",
        "permutation_seed": 20260916,
        "original_label_sha256": original_label_sha,
        "permuted_label_sha256": permuted_label_sha,
        "labels_changed": original_label_sha != permuted_label_sha,
        "discovery_sequence_registry_before_sha256": pre_sequence_hash,
        "discovery_sequence_registry_after_sha256": post_sequence_hash,
        "all_discovery_sequences_unchanged": post_sequence_hash == pre_sequence_hash,
        "discovery_rerun_after_evaluation": False,
    }
    if invariance["status"] != "passed":
        raise RuntimeError("discovery sequences changed after evaluation access")
    atomic_json(args.output_root / "LABEL_PERMUTATION_INVARIANCE.json", invariance)
    result = {
        "schema_version": "neurodiscovery-hindcasting-v4r1-evaluation.v1",
        "status": "complete",
        "started_at": opened_at,
        "completed_at": utc_now(),
        "discovery_manifest": str(args.discovery_manifest.resolve()),
        "discovery_manifest_sha256": sha256_file(args.discovery_manifest.resolve()),
        "evaluation_corpus": {
            "path": str(args.future_claims.resolve()),
            "sha256": future_sha,
            "bytes": args.future_claims.stat().st_size,
            "records_in_2017_2025_view": len(future_records),
        },
        "top_k": list(args.top_k),
        "auxiliary_random_trials": AUXILIARY_RANDOM_TRIALS,
        "auxiliary_random_trials_affect_locked_teas5": False,
        "runs": evaluated,
        "run_count": len(evaluated),
        "label_permutation_invariance": invariance,
        "discovery_resumed_after_evaluation": False,
    }
    atomic_json(args.output_root / "evaluation_manifest.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-manifest", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--future-claims", type=Path, default=DEFAULT_FUTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200, 500, 1000],
    )
    args = parser.parse_args(argv)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("evaluation output root must be absent or empty")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "runs": result["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
