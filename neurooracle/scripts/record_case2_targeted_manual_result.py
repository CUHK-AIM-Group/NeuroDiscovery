"""Compile a manually adjudicated Case-2 shard into a validated result file.

The judgment lives in a small, reviewer-authored specification.  This helper
only expands the explicitly acknowledged paper range, copies immutable task
hashes, and checks that every paper and currently labelled claim is covered.
It never reads or writes the formal knowledge graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "case2_targeted_manual_reaudit_result.v1"
SPEC_SCHEMA = "case2_targeted_manual_review_spec.v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_remove_rationale(paper: dict[str, Any]) -> str:
    spans = paper.get("deterministic_calibration", {}).get("component_spans", {})
    present = [name for name, values in spans.items() if values]
    present_text = ", ".join(present) if present else "none of the four components"
    return (
        "Manual paper-level review found no explicit, connected complete "
        "genetic/pathway -> brain imaging/physiology -> longitudinal clinical/"
        "cognitive outcome mediation or causal chain. The stored evidence only "
        f"supports: {present_text}; co-occurrence, ordinary prediction, or parallel "
        "intervention effects are insufficient under the frozen rubric."
    )


def compile_result(task: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError("unknown manual review specification schema")
    if spec.get("manual_review_complete") is not True:
        raise ValueError("manual_review_complete must be true")
    if int(spec.get("shard_index", -1)) != int(task.get("shard_index", -2)):
        raise ValueError("spec/task shard_index mismatch")

    papers = task.get("papers") or []
    indices = [int(paper["paper_index"]) for paper in papers]
    if not indices:
        raise ValueError("task has no papers")
    acknowledged = spec.get("reviewed_paper_index_range") or []
    if acknowledged != [indices[0], indices[-1]] or indices != list(
        range(indices[0], indices[-1] + 1)
    ):
        raise ValueError("reviewed range does not exactly cover the contiguous task")

    overrides_raw = spec.get("decision_overrides") or []
    overrides = {int(item["paper_index"]): item for item in overrides_raw}
    if len(overrides) != len(overrides_raw):
        raise ValueError("duplicate paper_index in decision_overrides")
    unknown = sorted(set(overrides) - set(indices))
    if unknown:
        raise ValueError(f"override paper indices are outside this task: {unknown}")

    decisions: list[dict[str, Any]] = []
    for paper in papers:
        paper_index = int(paper["paper_index"])
        claims = paper.get("currently_case2_labelled_claims") or []
        claim_ids = [str(claim["claim_id"]) for claim in claims]
        override = overrides.get(paper_index)
        if override is None:
            paper_decision = "remove"
            confidence = float(spec.get("default_remove_confidence", 0.99))
            evidence = {
                "genetic_or_pathway": [],
                "brain_imaging_or_physiology": [],
                "longitudinal_clinical_or_cognitive_outcome": [],
                "mediation_or_causal_chain": [],
            }
            rationale = _default_remove_rationale(paper)
            retained_claims: set[str] = set()
        else:
            paper_decision = str(override.get("paper_decision") or "")
            if paper_decision not in {"retain", "remove", "unresolved_source"}:
                raise ValueError(f"invalid paper decision for {paper_index}")
            confidence = float(override.get("confidence"))
            evidence = override.get("component_evidence") or {}
            required_components = {
                "genetic_or_pathway",
                "brain_imaging_or_physiology",
                "longitudinal_clinical_or_cognitive_outcome",
                "mediation_or_causal_chain",
            }
            if set(evidence) != required_components:
                raise ValueError(f"component evidence mismatch for {paper_index}")
            if any(not isinstance(value, list) for value in evidence.values()):
                raise ValueError(f"component evidence must be lists for {paper_index}")
            rationale = str(override.get("rationale") or "").strip()
            retained_claims = {
                str(value) for value in (override.get("retain_claim_ids") or [])
            }
            if not retained_claims <= set(claim_ids):
                raise ValueError(f"unknown retained claim for {paper_index}")
            if paper_decision == "retain":
                if any(not evidence[name] for name in required_components):
                    raise ValueError(f"retained paper {paper_index} lacks evidence")
                if not retained_claims:
                    raise ValueError(f"retained paper {paper_index} has no retained claim")
            elif retained_claims:
                raise ValueError(f"non-retained paper {paper_index} retains a claim")

        claim_decisions = [
            {
                "claim_id": claim_id,
                "decision": "retain" if claim_id in retained_claims else "remove",
                "reason": (
                    "This claim directly forms or states part of the manually "
                    "verified paper-level complete Case-2 chain."
                    if claim_id in retained_claims
                    else "The claim is not part of a verified complete paper-level Case-2 chain."
                ),
            }
            for claim_id in claim_ids
        ]
        decisions.append(
            {
                "paper_index": paper_index,
                "paper_key": paper["paper_key"],
                "audit_input_sha256": paper["audit_input_sha256"],
                "paper_decision": paper_decision,
                "confidence": confidence,
                "component_evidence": evidence,
                "rationale": rationale,
                "claim_decisions": claim_decisions,
            }
        )

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "shard_index": task["shard_index"],
        "task_sha256": task["task_sha256"],
        "reviewer_id": str(spec.get("reviewer_id") or "codex_root_manual"),
        "decisions": decisions,
    }
    canonical = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_result(_read_json(args.task), _read_json(args.spec))
    _write_json(args.output, result)
    retained = sum(row["paper_decision"] == "retain" for row in result["decisions"])
    unresolved = sum(
        row["paper_decision"] == "unresolved_source" for row in result["decisions"]
    )
    retained_claims = sum(
        claim["decision"] == "retain"
        for row in result["decisions"]
        for claim in row["claim_decisions"]
    )
    print(
        json.dumps(
            {
                "papers": len(result["decisions"]),
                "retained_papers": retained,
                "unresolved_papers": unresolved,
                "retained_claims": retained_claims,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
