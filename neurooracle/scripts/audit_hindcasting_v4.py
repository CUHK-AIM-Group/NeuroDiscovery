"""Independent final integrity audit for the formal Hindcasting v4r1 run."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from neurooracle.scripts.evaluate_hindcasting_v4 import _preflight
from neurooracle.src.hindcasting_v4_feedback_boundary import (
    canonical_sha256,
    scan_discovery_source,
    sha256_file,
    validate_computational_feedback,
    validate_discovery_input_manifest,
    validate_evaluation_handoff,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916"
DEFAULT_DISCOVERY = RUN_ROOT / "discovery/discovery_manifest.json"
DEFAULT_EVALUATION = RUN_ROOT / "evaluation/evaluation_manifest.json"
DEFAULT_RESULTS = RUN_ROOT / "result_summary/RESULTS.json"
DEFAULT_OUTPUT = RUN_ROOT / "result_summary/FINAL_AUDIT.json"
DISCOVERY_SOURCES = (
    ROOT / "neurooracle/scripts/run_hindcasting_v4_discovery.py",
    ROOT / "neurooracle/src/hindcasting_v4_computational_executor.py",
    ROOT / "neurooracle/scripts/generate_case_study_frozen_baselines.py",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value}")
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> dict[str, Any]:
    discovery, discovery_rows, current_sequence_hash = _preflight(args.discovery)
    plan_path = Path(str(discovery["plan_path"]))
    plan = read_json(plan_path)
    if plan.get("status") != "locked_before_discovery":
        raise ValueError("discovery plan was not locked before execution")
    if int(plan.get("llm_api_calls", -1)) != 0:
        raise ValueError("discovery plan does not attest zero LLM API calls")
    if (plan.get("configuration") or {}).get("llm_api_enabled") is not False:
        raise ValueError("LLM-backed discovery was not disabled")
    if plan.get("future_publication_corpus_recorded_in_discovery_plan") is not False:
        raise ValueError("future-publication corpus leaked into the discovery plan")

    input_audits: list[dict[str, Any]] = []
    for window in discovery["windows"]:
        input_manifest = read_json(Path(str(window["discovery_input_manifest"])))
        input_audits.append(validate_discovery_input_manifest(input_manifest))

    source_audits = [scan_discovery_source(path) for path in DISCOVERY_SOURCES]
    if any(audit.get("status") != "passed" for audit in source_audits):
        raise ValueError("a discovery source imports or invokes the publication evaluator")

    discovery_by_identity: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    feedback_statuses: Counter[str] = Counter()
    feedback_records = 0
    neurodiscovery_cells = 0
    feedback_activated_cells = 0
    selection_changed_cells = 0
    for row in discovery_rows:
        identity = (
            str(row["method"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["seed"]),
        )
        discovery_by_identity[identity] = row
        records: list[dict[str, Any]] = []
        hypotheses = read_json(Path(str(row["hypotheses_path"])))
        committed_ids = [str(item["id"]) for item in hypotheses["hypotheses"]]
        feedback_path = Path(str(row["feedback_path"]))
        previous_feedback_sha256 = "0" * 64
        with feedback_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    validate_computational_feedback(
                        record,
                        freeze_year=int(row["freeze_year"]),
                        committed_candidate_ids=committed_ids,
                    )
                    if str(record["previous_feedback_sha256"]) != previous_feedback_sha256:
                        raise ValueError(f"broken feedback chain: {identity}")
                    unsigned = dict(record)
                    stored_record_sha256 = str(unsigned.pop("feedback_record_sha256"))
                    if canonical_sha256(unsigned) != stored_record_sha256:
                        raise ValueError(f"invalid feedback record hash: {identity}")
                    previous_feedback_sha256 = stored_record_sha256
                    records.append(record)
                    feedback_statuses[str(record["feedback_status"])] += 1
        feedback_records += len(records)
        mapped = int(row["mapped_computational_hypotheses"])
        if str(row["method"]) == "neurodiscovery":
            neurodiscovery_cells += 1
            mapped_records = sum(
                str(record.get("mapping_status")) == "mapped" for record in records
            )
            seal = read_json(Path(str(row["discovery_seal_path"])))
            if (
                len(records) != max(int(value) for value in discovery["budgets"])
                or mapped_records != mapped
                or previous_feedback_sha256 != str(seal["feedback_chain_sha256"])
            ):
                raise ValueError(f"NeuroDiscovery feedback count mismatch: {identity}")
            if row.get("feedback_activated") is True:
                feedback_activated_cells += 1
            if row.get("selection_changed_after_feedback") is True:
                selection_changed_cells += 1
        elif records or mapped != 0 or row.get("feedback_activated") is not False:
            raise ValueError(f"baseline unexpectedly consumed computational feedback: {identity}")

    evaluation = read_json(args.evaluation)
    gate = read_json(args.evaluation.parent / "PRE_EVALUATION_GATE.json")
    invariance = read_json(args.evaluation.parent / "LABEL_PERMUTATION_INVARIANCE.json")
    if evaluation.get("status") != "complete" or int(evaluation.get("run_count", -1)) != 300:
        raise ValueError("evaluation matrix is incomplete")
    if gate.get("status") != "passed" or int(gate.get("sealed_cells", -1)) != 300:
        raise ValueError("pre-evaluation gate did not pass for 300 cells")
    if sha256_file(args.discovery) != str(gate["discovery_manifest_sha256"]):
        raise ValueError("discovery manifest changed after the pre-evaluation gate")
    if sha256_file(args.discovery) != str(evaluation["discovery_manifest_sha256"]):
        raise ValueError("evaluation is not bound to the current discovery manifest")
    if not (
        parse_time(str(discovery["completed_at"]))
        < parse_time(str(gate["checked_at"]))
        <= parse_time(str(evaluation["started_at"]))
        <= parse_time(str(evaluation["completed_at"]))
    ):
        raise ValueError("discovery/evaluation timestamps violate the reveal boundary")
    if not (
        invariance.get("status") == "passed"
        and invariance.get("labels_changed") is True
        and invariance.get("all_discovery_sequences_unchanged") is True
        and invariance.get("discovery_rerun_after_evaluation") is False
        and str(invariance["discovery_sequence_registry_before_sha256"])
        == str(invariance["discovery_sequence_registry_after_sha256"])
        == current_sequence_hash
    ):
        raise ValueError("label-permutation discovery invariance did not pass")
    for source_key in ("evaluation_source", "inherited_evaluator_source"):
        source = dict(gate[source_key])
        if sha256_file(Path(str(source["path"]))) != str(source["sha256"]):
            raise ValueError(f"evaluation source changed after the gate: {source['path']}")

    evaluation_identities: set[tuple[str, str, int, int]] = set()
    handoffs_verified = 0
    for row in evaluation["runs"]:
        identity = (
            str(row["method"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["seed"]),
        )
        if identity in evaluation_identities or identity not in discovery_by_identity:
            raise ValueError(f"invalid evaluation identity: {identity}")
        evaluation_identities.add(identity)
        discovery_row = discovery_by_identity[identity]
        metrics_path = Path(str(row["metrics_path"]))
        if sha256_file(metrics_path) != str(row["metrics_sha256"]):
            raise ValueError(f"metrics changed after evaluation: {metrics_path}")
        metrics = read_json(metrics_path)
        bound = dict(metrics["v4r1_execution_identity"])
        if (
            str(bound["discovery_seal_sha256"])
            != str(discovery_row["discovery_seal_sha256"])
            or str(bound["hypotheses_sha256"]) != str(discovery_row["hypotheses_sha256"])
            or bound.get("evaluation_loaded_after_discovery_seal") is not True
        ):
            raise ValueError(f"metrics are not bound to discovery: {metrics_path}")
        seal = read_json(Path(str(discovery_row["discovery_seal_path"])))
        handoff = read_json(Path(str(row["evaluation_handoff"])))
        validate_evaluation_handoff(seal, handoff)
        if Path(str(handoff["metrics_path"])).resolve() != metrics_path.resolve():
            raise ValueError(f"handoff metrics path mismatch: {identity}")
        handoffs_verified += 1
    if evaluation_identities != set(discovery_by_identity):
        raise ValueError("evaluation and discovery identity matrices differ")

    results = read_json(args.results)
    if results.get("status") != "complete":
        raise ValueError("aggregate result is incomplete")
    if sha256_file(args.evaluation) != str(results["evaluation_manifest_sha256"]):
        raise ValueError("aggregate result is not bound to the evaluation manifest")
    cell_path = Path(str(results["artifacts"]["cell_metrics"]))
    summary_path = Path(str(results["artifacts"]["method_summary"]))
    paired_path = Path(str(results["artifacts"]["paired_comparisons"]))
    cells = csv_rows(cell_path)
    summaries = csv_rows(summary_path)
    paired = csv_rows(paired_path)
    cell_keys = {
        (
            row["method"],
            row["task_id"],
            int(row["freeze_year"]),
            int(row["seed"]),
            int(row["k"]),
        )
        for row in cells
    }
    if len(cells) != 2100 or len(cell_keys) != 2100 or len(summaries) != 21 or len(paired) != 14:
        raise ValueError("aggregate output matrix has the wrong shape")
    for row in cells:
        expected_teas = 20.0 * min(int(row["unique_exact_discoveries"]), 5)
        if not math.isclose(float(row["teas5"]), expected_teas):
            raise ValueError("a cell violates the locked TEAS-5 formula")
    for summary in summaries:
        subset = [
            row
            for row in cells
            if row["method"] == summary["method"] and int(row["k"]) == int(summary["k"])
        ]
        if len(subset) != 100:
            raise ValueError("method summary does not contain 100 locked cells")
        if not math.isclose(mean(float(row["teas5"]) for row in subset), float(summary["teas5_mean"])):
            raise ValueError("reported TEAS-5 mean does not reproduce from cell data")

    strict_historical_resources = all(
        bool(resource.get("strict_historical_resource"))
        for resource in plan["computational_resources"].values()
    )
    audit = {
        "schema_version": "neurodiscovery-hindcasting-v4r1-final-audit.v1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "formal v4r1 discovery, evaluation, aggregate, and leakage-boundary audit",
        "matrix": {
            "discovery_cells": len(discovery_rows),
            "evaluation_cells": len(evaluation_identities),
            "cell_budget_rows": len(cells),
            "methods": list(discovery["methods"]),
            "tasks": list(discovery["case_studies"]),
            "cutoffs": [int(window["freeze_year"]) for window in discovery["windows"]],
            "seeds": list(discovery["seeds"]),
            "budgets": list(discovery["budgets"]),
        },
        "no_leakage": {
            "all_discovery_inputs_validated": len(input_audits) == 5,
            "discovery_sources_isolated_from_publication_evaluator": True,
            "llm_api_calls": 0,
            "future_publication_corpus_absent_from_discovery_plan": True,
            "all_cells_sealed_before_first_evaluation_access": True,
            "discovery_completed_at": str(discovery["completed_at"]),
            "evaluation_gate_passed_at": str(gate["checked_at"]),
            "label_permutation_changed_labels": True,
            "discovery_sequences_unchanged_after_label_access": True,
            "discovery_sequence_registry_sha256": current_sequence_hash,
            "discovery_rerun_after_evaluation": False,
        },
        "computational_feedback": {
            "neurodiscovery_cells": neurodiscovery_cells,
            "feedback_activated_cells": feedback_activated_cells,
            "selection_changed_after_feedback_cells": selection_changed_cells,
            "validated_feedback_records": feedback_records,
            "feedback_status_counts": dict(sorted(feedback_statuses.items())),
            "publication_evaluation_fields_read": False,
            "baseline_feedback_records": 0,
        },
        "evaluation": {
            "handoffs_verified": handoffs_verified,
            "evaluation_manifest_sha256": sha256_file(args.evaluation),
            "evaluation_corpus_sha256": str(evaluation["evaluation_corpus"]["sha256"]),
        },
        "aggregate": {
            "teas5_formula_reproduced_for_all_rows": True,
            "all_method_means_reproduced_from_100_cells": True,
            "result_sha256_before_audit": sha256_file(args.results),
        },
        "interpretation_boundary": {
            "algorithmically_blinded_retrospective_hindcasting": True,
            "strict_historical_resource_availability": strict_historical_resources,
            "reason": (
                None
                if strict_historical_resources
                else "registered TCP computational datasets were released after the historical cutoffs"
            ),
        },
    }
    atomic_json(args.output, audit)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    audit = run(parse_args())
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
