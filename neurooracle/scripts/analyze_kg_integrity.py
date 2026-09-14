"""Analyze frozen scan indexes for graph defects, preserving all original records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import networkx as nx

from audit_kg_integrity import (
    DEFAULT_OUTPUT, JsonlWriter, atomic_json, compact, progress, read_json, readonly_db,
    unchanged_inputs, utc_now,
)
from neurooracle.src.claim_semantics import audit_claim_endpoints, semantic_claim_endpoint
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_quality_checks import edge_claim_agreement
from neurooracle.src.schema import ConceptNode, Edge
from neurooracle.src.storage import load_graph


def graph_structure(output: Path) -> dict:
    db = readonly_db(output)
    progress(output, "GRAPH_REFERENTIAL_AND_RELATION_CHECKS")
    counters = Counter()
    refs = JsonlWriter(output / "REFERENCE_ISSUES.jsonl")
    queries = {
        "edge_missing_source": "SELECT e.ordinal,e.s FROM edges e LEFT JOIN nodes n ON n.id=e.s WHERE n.id IS NULL",
        "edge_missing_target": "SELECT e.ordinal,e.t FROM edges e LEFT JOIN nodes n ON n.id=e.t WHERE n.id IS NULL",
        "claim_missing_subject": "SELECT c.id,c.s FROM claims c LEFT JOIN nodes n ON n.id=c.s WHERE n.id IS NULL",
        "claim_missing_object": "SELECT c.id,c.t FROM claims c LEFT JOIN nodes n ON n.id=c.t WHERE n.id IS NULL",
        "atom_missing_source_mention": "SELECT n.id,n.parent_id FROM nodes n LEFT JOIN nodes p ON p.id=n.parent_id WHERE n.kind='umls_atom' AND p.id IS NULL",
        "atom_parent_not_source_mention": "SELECT n.id,n.parent_id FROM nodes n JOIN nodes p ON p.id=n.parent_id WHERE n.kind='umls_atom' AND p.kind!='source_mention'",
    }
    for code, query in queries.items():
        for record_id, endpoint in db.execute(query):
            refs.add({"code": code, "record_id": record_id, "endpoint": endpoint})
            counters[code] += 1
    for row in db.execute("""SELECT c.id,c.s,c.t,COUNT(e.ordinal),
            COALESCE(SUM(e.t=c.s),0),COALESCE(SUM(e.t=c.t),0),
            COALESCE(SUM(e.t!=c.s AND e.t!=c.t),0)
            FROM claims c LEFT JOIN edges e ON e.s=c.id AND e.r='about' GROUP BY c.id"""):
        claim_id, s, t, count, subject_links, object_links, extra = row
        if not subject_links or not object_links or extra:
            refs.add({"code": "claim_about_links_disagree_with_endpoints", "claim_id": claim_id,
                      "subject_id": s, "object_id": t, "about_count": count,
                      "subject_links": subject_links, "object_links": object_links, "extra_links": extra})
            counters["claim_about_links_disagree_with_endpoints"] += 1
    progress(output, "EDGE_TO_CLAIM_AGREEMENT")
    agreement = JsonlWriter(output / "EDGE_CLAIM_DISAGREEMENTS.jsonl")
    for ordinal, payload, summary in db.execute("""SELECT e.ordinal,e.payload_json,c.summary_json FROM edges e
            LEFT JOIN claims c ON c.id=e.claim_id WHERE e.claim_id IS NOT NULL AND e.claim_id!=''"""):
        edge = json.loads(payload)
        issues = edge_claim_agreement(edge, json.loads(summary) if summary else None)
        counters["edges_with_claim_pointer_checked"] += 1
        if issues:
            agreement.add({"edge_ordinal": ordinal, "claim_id": edge["metadata"]["claim_id"], "issues": issues})
            counters.update({row["code"] for row in issues})
    counters["claim_source_edges_without_claim_pointer"] = db.execute("SELECT COUNT(*) FROM edges WHERE source LIKE 'claim:%' AND (claim_id IS NULL OR claim_id='')").fetchone()[0]

    progress(output, "REPLAY_PARALLEL_RELATIONS_IN_ACTUAL_GRAPH_MANAGER")
    collisions = JsonlWriter(output / "PARALLEL_EDGE_GROUPS.jsonl")
    witness_edges, witness_ids = [], set()
    for s, t, n, relation_count, distinct_records in db.execute("SELECT s,t,COUNT(*),COUNT(DISTINCT r),COUNT(DISTINCT record_sha256) FROM edges WHERE s!=t GROUP BY s,t HAVING COUNT(*)>1"):
        group = [(ordinal, json.loads(payload)) for ordinal, payload in db.execute("SELECT ordinal,payload_json FROM edges WHERE s=? AND t=? ORDER BY ordinal", (s, t))]
        kg = KnowledgeGraph()
        for node_id in (s, t):
            kg.add_concept(ConceptNode(id=node_id, preferred_name=node_id))
        for ordinal, edge in group:
            kg.add_edge(Edge.from_dict(edge))
        expected_ordinal, expected = max(group, key=lambda item: (item[1].get("confidence", 1.0), -item[0]))
        if kg.G.number_of_edges() != 1 or kg.G.edges[s, t] != Edge.from_dict(expected).to_dict():
            raise ValueError("actual graph manager differs from predicted relation collapse")
        code = "different_relations_same_pair" if relation_count > 1 else ("exact_duplicate_edge_records" if distinct_records == 1 else "same_relation_different_records")
        counters["parallel_pair_groups"] += 1
        counters[code] += 1
        counters["edge_records_not_separately_retained_in_digraph"] += n - 1
        if relation_count > 1:
            counters["relation_identities_not_separately_retained_in_digraph"] += relation_count - 1
        collisions.add({"category": code, "source_id": s, "target_id": t, "records": n,
                        "distinct_relations": relation_count, "distinct_full_records": distinct_records,
                        "selected_ordinal_by_actual_reader": expected_ordinal,
                        "selected_relation": expected["relation_type"],
                        "edges": [{"ordinal": ordinal, "record": edge} for ordinal, edge in group],
                        "all_group_edges_replayed_using_actual_add_edge": True,
                        "original_evidence_not_deleted": True})
        # A bounded, real load_graph witness; full collision groups remain indexed.
        if len(witness_ids) < 120:
            witness_ids.update((s, t))
            witness_edges.extend(edge for _, edge in group)
    concepts = {node_id: json.loads(db.execute("SELECT brief_json FROM nodes WHERE id=?", (node_id,)).fetchone()[0]) for node_id in witness_ids}
    witness = output / "RUNTIME_PARALLEL_RELATION_WITNESS.json"
    atomic_json(witness, {"metadata": {"purpose": "exact source edges, abridged endpoint attributes, read-only loading witness"}, "concepts": concepts, "edges": witness_edges})
    loaded = load_graph(witness)
    unique_pairs = {(edge["source_id"], edge["target_id"]) for edge in witness_edges}
    if loaded.G.number_of_edges() != len(unique_pairs):
        raise ValueError("real storage.load_graph witness failed")
    counters["actual_storage_witness_input_edges"] = len(witness_edges)
    counters["actual_storage_witness_loaded_edges"] = loaded.G.number_of_edges()
    counters["serialized_unique_nonself_pairs"] = db.execute("SELECT COUNT(*) FROM (SELECT s,t FROM edges WHERE s!=t GROUP BY s,t)").fetchone()[0]
    counters["same_spo_mixed_explicit_negation_groups"] = db.execute("SELECT COUNT(*) FROM (SELECT s,t,r FROM edges WHERE negated IN ('true','false') GROUP BY s,t,r HAVING COUNT(DISTINCT negated)>1)").fetchone()[0]
    counters["same_paper_same_spo_claim_candidate_groups"] = db.execute("SELECT COUNT(*) FROM (SELECT paper_key,s,t,p,negated FROM claims WHERE paper_key!='' GROUP BY paper_key,s,t,p,negated HAVING COUNT(*)>1)").fetchone()[0]
    db.close()
    return {"status": "AUDITED", "counts": dict(counters), "artifacts": {"references": refs.close(), "edge_claim_agreement": agreement.close(), "parallel_relations": collisions.close()},
            "repeat_evidence_not_automatically_a_duplicate_fact": True}


def entity_identity(output: Path) -> dict:
    progress(output, "ENTITY_IDENTITY_AND_TAXONOMY_REVIEW")
    db = readonly_db(output)
    groups = JsonlWriter(output / "ENTITY_IDENTITY_REVIEW.jsonl")
    counts = Counter()
    for field in ("cui", "normalized_name"):
        query = f"SELECT {field},COUNT(*) FROM nodes WHERE kind IN ('existing_entity_or_infrastructure','new_umls_cui') AND {field}!='' GROUP BY {field} HAVING COUNT(*)>1"
        for value, count in db.execute(query):
            rows = [json.loads(row[0]) for row in db.execute(f"SELECT brief_json FROM nodes WHERE {field}=? AND kind IN ('existing_entity_or_infrastructure','new_umls_cui') ORDER BY id", (value,))]
            same_domain_pairs = sum(bool(set(left["domain_tags"]) & set(right["domain_tags"])) for i, left in enumerate(rows) for right in rows[i + 1:])
            code = "same_recorded_cui_multiple_entity_ids" if field == "cui" else "same_normalized_name_multiple_entity_ids"
            if field == "normalized_name" and not same_domain_pairs:
                continue
            counts[code] += 1
            groups.add({"category": code, "value": value, "node_count": count, "same_domain_pairs": same_domain_pairs,
                        "nodes": rows, "automatic_merge_authorized": False,
                        "interpretation": "recorded identifier or lexical overlap is a review candidate, not independent biomedical identity proof"})
    taxonomy = nx.DiGraph()
    taxonomy.add_edges_from(db.execute("SELECT s,t FROM edges WHERE r='is_a'"))
    cycles = [sorted(component) for component in nx.strongly_connected_components(taxonomy) if len(component) > 1]
    atomic_json(output / "TAXONOMY_CYCLE_REVIEW.json", {"is_a_edges": taxonomy.number_of_edges(), "strongly_connected_groups": cycles,
                "interpretation": "Only is_a hierarchy checked; cycles in general biological relations are not automatically defects."})
    counts["is_a_cycle_groups"] = len(cycles)
    isolated = JsonlWriter(output / "ISOLATED_NODES.jsonl")
    for node_id, category, brief in db.execute("""SELECT n.id,n.kind,n.brief_json FROM nodes n
            LEFT JOIN (SELECT s AS id FROM edges UNION SELECT t AS id FROM edges) active ON active.id=n.id
            WHERE active.id IS NULL"""):
        isolated.add({"node_id": node_id, "node_class": category, "node": json.loads(brief), "automatic_delete_authorized": False})
        counts["isolated_nodes"] += 1
    db.close()
    return {"status": "REVIEW_CANDIDATES_ONLY", "counts": dict(counts), "artifacts": {"identity": groups.close(), "isolated_nodes": isolated.close()}}


class NodeLookup:
    def __init__(self, db):
        self.db = db

    @lru_cache(maxsize=50000)
    def get(self, key, default=None):
        row = self.db.execute("SELECT brief_json FROM nodes WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default


def semantic_endpoints(output: Path) -> dict:
    db = readonly_db(output)
    lookup, counts, projections = NodeLookup(db), Counter(), Counter()
    writer = JsonlWriter(output / "SEMANTIC_ENDPOINT_REVIEW.jsonl")
    progress(output, "ACTUAL_CLAIM_ENDPOINT_RULE_REVIEW")
    for number, (payload,) in enumerate(db.execute("SELECT summary_json FROM claims ORDER BY id"), 1):
        claim = json.loads(payload)
        result = audit_claim_endpoints(claim, lookup)
        counts[result.reason] += 1
        if not result.valid:
            endpoints = {side: semantic_claim_endpoint(claim, side, lookup) for side in ("subject", "object")}
            for endpoint in endpoints.values():
                projections["suppressed" if endpoint is None else ("canonical" if endpoint.uses_canonical_id else "claim_local")] += 1
            writer.add({"claim": claim, "audit": asdict(result),
                        "runtime_projection": {side: asdict(endpoint) if endpoint else None for side, endpoint in endpoints.items()},
                        "interpretation": "existing lexical/type heuristic flag; not a final semantic error or permission to merge"})
        if number % 100000 == 0:
            progress(output, "CLAIM_ENDPOINT_REVIEW_PROGRESS", claims=number, flagged=number - counts["ok"])
    lookup.get.cache_clear()
    db.close()
    return {"status": "HEURISTIC_REVIEW_NOT_MEDICAL_ADJUDICATION", "claims_checked": sum(counts.values()),
            "reason_counts": dict(counts), "flagged_endpoint_projection_counts": dict(projections), "artifact": writer.close()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("structure", "identity", "semantic"), required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    inputs = read_json(output / "INPUTS.json")
    unchanged_inputs(inputs)
    if not (output / "SCAN_COMPLETE.json").exists():
        raise ValueError("full graph scan has not completed")
    target = output / f"{args.phase.upper()}_COMPLETE.json"
    if target.exists():
        raise ValueError("phase already completed")
    function = {"structure": graph_structure, "identity": entity_identity, "semantic": semantic_endpoints}[args.phase]
    result = function(output)
    unchanged_inputs(inputs)
    atomic_json(target, result)
    print(compact({"phase": args.phase, "status": result["status"], "counts": result.get("counts", result.get("reason_counts"))}), flush=True)


if __name__ == "__main__":
    main()
