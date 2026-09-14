"""Validate reference results using distinct scientific and provenance edge rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_kg_integrity import (
    DEFAULT_OUTPUT, JsonlWriter, atomic_json, compact, read_json, readonly_db,
    unchanged_inputs, utc_now,
)
from neurooracle.src.kg_quality_checks import edge_claim_agreement


def about_edge_checks(edge: dict, source_claim: dict | None) -> list[dict]:
    """An about edge runs from a claim node to one of that claim's endpoints."""
    md = edge.get("metadata") or {}
    issues = []
    if source_claim is None:
        return [{"code": "about_source_is_not_a_claim", "source_id": edge["source_id"]}]
    if md.get("claim_id") and md["claim_id"] != source_claim["id"]:
        issues.append({"code": "about_claim_pointer_differs_from_source", "claim_id": md["claim_id"]})
    if edge["target_id"] not in {source_claim["subject_id"], source_claim["object_id"]}:
        issues.append({"code": "about_target_not_claim_endpoint", "target_id": edge["target_id"]})
    role = md.get("anchor_role")
    if role in ("subject", "object") and edge["target_id"] != source_claim[role + "_id"]:
        issues.append({"code": "about_anchor_role_endpoint_disagreement", "role": role,
                       "target_id": edge["target_id"], "expected_id": source_claim[role + "_id"]})
    elif role and role not in ("subject", "object"):
        issues.append({"code": "about_unrecognized_anchor_role_review", "role": role})
    return issues


def reconcile(output: Path, original: dict | None = None) -> dict:
    db = readonly_db(output)
    counts, fields, roles, reference_shapes = Counter(), Counter(), Counter(), Counter()
    scientific = JsonlWriter(output / "EDGE_CLAIM_DISAGREEMENTS_CORRECTED.jsonl")
    provenance = JsonlWriter(output / "ABOUT_CLAIM_POINTER_ISSUES.jsonl")
    query = """SELECT e.ordinal,e.payload_json,c.summary_json FROM edges e
        LEFT JOIN claims c ON c.id=CASE WHEN e.r='about' THEN e.s ELSE e.claim_id END
        WHERE e.r='about' OR (e.claim_id IS NOT NULL AND e.claim_id!='')"""
    for ordinal, payload, summary in db.execute(query):
        edge = json.loads(payload)
        claim = json.loads(summary) if summary else None
        md = edge.get("metadata") or {}
        if md.get("claim_id"):
            counts["edges_with_claim_pointer_checked"] += 1
        if edge["relation_type"] == "about":
            counts["about_edges_checked"] += 1
            counts["about_edges_with_claim_pointer" if md.get("claim_id") else "about_edges_without_claim_pointer"] += 1
            roles[str(md.get("anchor_role") or "absent")] += 1
            issues = about_edge_checks(edge, claim)
            writer = provenance
        else:
            counts["scientific_edges_with_claim_pointer_checked"] += 1
            issues = edge_claim_agreement(edge, claim)
            writer = scientific
        if issues:
            writer.add({"edge_ordinal": ordinal, "edge": edge, "claim": claim, "issues": issues})
            counts.update({issue["code"] for issue in issues})
            fields.update(issue["field"] for issue in issues if issue.get("field"))
    db.close()
    ref_path = output / "REFERENCE_ISSUES.jsonl"
    if ref_path.exists():
        with ref_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["code"] != "claim_about_links_disagree_with_endpoints":
                    continue
                missing = not row["subject_links"] or not row["object_links"]
                extra = bool(row["extra_links"])
                reference_shapes["missing_and_extra" if missing and extra else "missing_only" if missing else "extra_only"] += 1
    original = original or {"counts": {}, "artifacts": {}}
    accepted_counts = {key: value for key, value in original["counts"].items()
                       if key not in {"edges_with_claim_pointer_checked", "edge_claim_payload_disagreement",
                                      "edge_claim_negation_disagreement", "edge_references_missing_claim"}}
    accepted_counts.update(counts)
    return {
        "status": "VALIDATED_WITH_DISTINCT_PROVENANCE_POLICY", "completed_at": utc_now(),
        "counts": accepted_counts, "scientific_disagreement_field_events": dict(fields),
        "about_anchor_role_inventory": dict(roles), "claim_about_disagreement_shapes": dict(reference_shapes),
        "artifacts": {"scientific_agreement": scientific.close(), "about_agreement": provenance.close(),
                      **{key: value for key, value in original["artifacts"].items() if key != "edge_claim_agreement"}},
        "supersedes": {
            "file": "STRUCTURE_COMPLETE.json",
            "excluded_artifact": "EDGE_CLAIM_DISAGREEMENTS.jsonl",
            "reason": "The provisional comparison applied subject-object rules to about provenance edges. This report separates the policies and replaces those counts; unrelated structural findings remain valid.",
        },
        "source_graph_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    target = output / "STRUCTURE_VALIDATED.json"
    if target.exists():
        raise ValueError("validated reference report already exists")
    inputs = read_json(output / "INPUTS.json")
    unchanged_inputs(inputs)
    result = reconcile(output, read_json(output / "STRUCTURE_COMPLETE.json"))
    unchanged_inputs(inputs)
    atomic_json(target, result)
    print(compact(result), flush=True)


if __name__ == "__main__":
    main()
