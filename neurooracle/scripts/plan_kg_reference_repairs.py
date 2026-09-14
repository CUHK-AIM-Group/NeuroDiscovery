"""Conservative, claim-local reference repair; never merge or rename entities."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from repair_kg_claim_compatibility import AUDIT, OUTPUT, check_inputs
from audit_kg_integrity import JsonlWriter, atomic_json, compact, readonly_db, utc_now
from analyze_kg_integrity import NodeLookup
from neurooracle.src.claim_semantics import semantic_claim_endpoint


def exact_label(value):
    # Keep signs/punctuation: normalization must not turn -1 into +1.
    return " ".join(str(value or "").casefold().split())


def propose_reference_patch(edge, claim, concepts):
    """Use explicit claim role + identical labels + actual canonical-role guard."""
    md = edge.get("metadata") or {}
    after = deepcopy(edge)
    if not claim:
        return None, "missing_claim"
    if edge["relation_type"] == "about":
        role = md.get("anchor_role")
        if edge["source_id"] != claim["id"] or md.get("claim_id") != claim["id"]:
            return None, "about_lacks_explicit_matching_claim_pointer"
        if role not in ("subject", "object"):
            return None, "about_lacks_explicit_endpoint_role"
        fields = [("target_id", role)]
    else:
        if md.get("claim_id") != claim["id"] or edge["relation_type"] != claim["predicate"]:
            return None, "claim_or_predicate_disagreement"
        if "negated" in md and md["negated"] != claim.get("negated", False):
            return None, "negation_disagreement"
        fields = [("source_id", "subject"), ("target_id", "object")]
    reasons = []
    for field, role in fields:
        previous, target = edge[field], claim[role + "_id"]
        if previous == target:
            continue
        old_node, new_node = concepts.get(previous), concepts.get(target)
        if not old_node or not new_node:
            return None, "missing_endpoint"
        labels = [exact_label(claim[role + "_name"]), exact_label(old_node["preferred_name"]), exact_label(new_node["preferred_name"])]
        if not labels[0] or len(set(labels)) != 1:
            return None, "claim_old_new_primary_labels_not_exactly_equal"
        if not previous.startswith("CLM_CONCEPT:") or target.startswith(("CLM_CONCEPT:", "CLM:", "CLM_ATOM:")):
            return None, "not_source_mention_to_existing_canonical_reference"
        endpoint = semantic_claim_endpoint(claim, role, concepts)
        if endpoint is None or not endpoint.uses_canonical_id or not endpoint.role_compatible:
            return None, "current_claim_endpoint_fails_canonical_guard"
        after[field] = target
        reasons.append({"field": field, "role": role, "before": previous, "after": target,
                        "exact_label": labels[0], "canonical_role_guard_passed": True})
    if not reasons:
        return None, "no_endpoint_change"
    if after["source_id"] == after["target_id"]:
        return None, "would_create_self_loop"
    return {"before": edge, "after": after, "changes": reasons,
            "claim_id": claim["id"], "policy": "explicit_claim_role_exact_primary_labels_and_canonical_guard",
            "node_merge": False, "semantic_truth_adjudicated": False}, "eligible"


def main():
    check_inputs(OUTPUT)
    db = readonly_db(AUDIT)
    lookup = NodeLookup(db)
    planned_pairs, counts = set(), Counter()
    plan = JsonlWriter(OUTPUT / "EDGE_REPAIR_PLAN.jsonl")
    deferred = JsonlWriter(OUTPUT / "EDGE_REVIEW_DEFERRED.jsonl")
    for name in ("EDGE_CLAIM_DISAGREEMENTS_CORRECTED.jsonl", "ABOUT_CLAIM_POINTER_ISSUES.jsonl"):
        with (AUDIT / name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                patch, reason = propose_reference_patch(row["edge"], row["claim"], lookup)
                if patch:
                    pair = (patch["after"]["source_id"], patch["after"]["target_id"])
                    collision = db.execute("SELECT 1 FROM edges WHERE s=? AND t=? AND ordinal!=? LIMIT 1", (*pair, row["edge_ordinal"])).fetchone()
                    if collision or pair in planned_pairs:
                        patch, reason = None, "would_add_parallel_edge_to_existing_or_planned_pair"
                counts[reason] += 1
                if patch:
                    planned_pairs.add(pair)
                    plan.add({"edge_ordinal": row["edge_ordinal"], **patch})
                    counts["eligible_about" if row["edge"]["relation_type"] == "about" else "eligible_scientific"] += 1
                else:
                    deferred.add({"edge_ordinal": row["edge_ordinal"], "claim_id": row["claim"]["id"],
                                  "relation_type": row["edge"]["relation_type"], "reason": reason,
                                  "source_issue_artifact": name, "source_issues": row["issues"]})
    lookup.get.cache_clear()
    db.close()
    result = {"status": "BOUNDED_REFERENCE_REPAIR_PLAN", "created_at": utc_now(), "counts": dict(counts),
              "plan": plan.close(), "deferred": deferred.close(), "node_merges_authorized": False,
              "clinical_identity_certified": False}
    check_inputs(OUTPUT)
    atomic_json(OUTPUT / "REFERENCE_PLAN.json", result)
    print(compact(result), flush=True)


if __name__ == "__main__":
    main()
