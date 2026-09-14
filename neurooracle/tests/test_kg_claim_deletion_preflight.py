from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_kg_claim_deletion import FLOOR, exact_references, scan, select_targets, storage_check
from record_kg_source_findings import record_digest


def fixture(science=True):
    node = {"id": "CLM:x", "metadata": {"raw_text": "preserve evidence", "negated": False}}
    edges = {1: {"source_id": "CLM:x", "target_id": "a", "relation_type": "about", "metadata": {}},
             2: {"source_id": "CLM:x", "target_id": "b", "relation_type": "about", "metadata": {}}}
    if science:
        edges[3] = {"source_id": "a", "target_id": "b", "relation_type": "is_associated_with", "metadata": {"claim_id": "CLM:x", "negated": False}}
    queue = [{"claim_id": "CLM:x", "route": "source_mismatch", "source_round": 5, "classification": "confirmed",
              "original_issue_category": "unsupported_by_bound_input_abstract", "current_node_sha256": record_digest(node),
              "related_edge_ordinals": list(edges), "owned_science_ordinals": [3] if science else [], "about_ordinals": [1, 2], "actual_graph_mutations": []}]
    bundles = [{"claim_id": "CLM:x", "before_node": node, "before_node_sha256": record_digest(node),
                "before_edges": [{"candidate_edge_ordinal": k, "edge": v} for k, v in edges.items()],
                "before_edge_sha256": {str(k): record_digest(v) for k, v in edges.items()}, "actual_applied_mutations": []}]
    return queue, bundles


def graph(science=True, refs=False):
    queue, bundles = fixture(science)
    nodes, edges = select_targets(queue, bundles)
    records = [("metadata", "", {"stats": {}}), ("node", "CLM:x", deepcopy(nodes["CLM:x"]))]
    for key in ("a", "b", "CLM:keep"):
        records.append(("node", key, {"id": key, "metadata": {"history": ["CLM:x"]} if refs and key == "a" else {}}))
    records.extend(("edge", str(k), deepcopy(v)) for k, v in edges.items())
    records.append(("edge", str(len(edges) + 1), {"source_id": "a", "target_id": "CLM:keep", "relation_type": "is_a", "metadata": {}}))
    return records, nodes, edges


@pytest.mark.parametrize("science", [True, False])
def test_exact_scope_including_zero_science(science):
    q, b = fixture(science)
    snapshot = deepcopy((q, b))
    nodes, edges = select_targets(q, b)
    assert len(nodes) == 1 and len(edges) == (3 if science else 2)
    assert (q, b) == snapshot


@pytest.mark.parametrize("mutation", ["duplicate_queue", "duplicate_preimage", "extra_preimage", "unconfirmed", "wrong_round", "wrong_category", "node_hash", "wrong_id", "mutated", "missing_edge", "duplicate_edge", "boolean_ordinal", "edge_hash", "borrowed_science", "wrong_about", "wrong_roles"])
def test_selection_fails_closed(mutation):
    q, b = fixture()
    if mutation == "duplicate_queue": q.append(deepcopy(q[0]))
    elif mutation == "duplicate_preimage": b.append(deepcopy(b[0]))
    elif mutation == "extra_preimage": b.append({**deepcopy(b[0]), "claim_id": "CLM:other"})
    elif mutation == "unconfirmed": q[0]["classification"] = "additional_review"
    elif mutation == "wrong_round": q[0]["source_round"] = 6
    elif mutation == "wrong_category": q[0]["original_issue_category"] = "statistic_role"
    elif mutation == "node_hash": b[0]["before_node"]["metadata"]["negated"] = 0
    elif mutation == "wrong_id": b[0]["before_node"]["id"] = "CLM:other"
    elif mutation == "mutated": b[0]["actual_applied_mutations"] = ["x"]
    elif mutation == "missing_edge": b[0]["before_edges"].pop()
    elif mutation == "duplicate_edge": b[0]["before_edges"].append(deepcopy(b[0]["before_edges"][0]))
    elif mutation == "boolean_ordinal": b[0]["before_edges"][0]["candidate_edge_ordinal"] = True
    elif mutation == "edge_hash": b[0]["before_edges"][2]["edge"]["metadata"]["negated"] = 0
    elif mutation in ("borrowed_science", "wrong_about"):
        index = 2 if mutation == "borrowed_science" else 0
        edge = b[0]["before_edges"][index]["edge"]
        if mutation == "borrowed_science": edge["metadata"]["claim_id"] = "CLM:other"
        else: edge["source_id"] = "CLM:other"
        b[0]["before_edge_sha256"][str(index + 1)] = record_digest(edge)
    else: q[0]["owned_science_ordinals"] = []
    with pytest.raises((ValueError, KeyError)):
        select_targets(q, b)


@pytest.mark.parametrize("science", [True, False])
def test_full_stream_projection_keeps_concepts_and_other_claim(science):
    records, nodes, edges = graph(science)
    snapshot = deepcopy(records)
    result = scan(iter(records), nodes, edges)
    before, after = result["topology"]["before"], result["topology"]["if_deleted"]
    assert (before["nodes"], before["claims"], before["edges"]) == (4, 2, 4 if science else 3)
    assert (after["nodes"], after["claims"], after["edges"]) == (3, 1, 1)
    assert result["newly_isolated_retained_node_ids"] == ["b"]
    assert result["retained_exact_references"] == []
    assert after["connected_components"] == 2
    assert records == snapshot
    assert result["node_removal_preimages"][0]["before"] == nodes["CLM:x"]


@pytest.mark.parametrize("mutation", ["extra_owned_edge", "edge_afterimage", "node_afterimage", "missing_target", "duplicate_node", "dangling", "ordinal_gap"])
def test_stream_rejects_unsafe_changes(mutation):
    records, nodes, edges = graph()
    if mutation == "extra_owned_edge": records[-1][2]["metadata"]["claim_id"] = "CLM:x"
    elif mutation == "edge_afterimage": records[5][2]["target_id"] = "CLM:keep"
    elif mutation == "node_afterimage": records[1][2]["metadata"]["negated"] = 0
    elif mutation == "missing_target": records.pop(1)
    elif mutation == "duplicate_node": records.insert(2, deepcopy(records[1]))
    elif mutation == "dangling": records[-1][2]["target_id"] = "missing"
    else: records[-1] = ("edge", "9", records[-1][2])
    with pytest.raises(ValueError): scan(iter(records), nodes, edges)


def test_recursive_retained_reference_not_silently_ignored():
    records, nodes, edges = graph(refs=True)
    result = scan(iter(records), nodes, edges)
    assert result["retained_exact_references"][0]["key"] == "a"
    assert result["retained_exact_references"][0]["references"] == [{"json_path": ["metadata", "history", 0], "claim_id": "CLM:x"}]


def test_reference_types_embedded_text_and_dictionary_keys():
    hits = list(exact_references({"CLM:x": {"ids": ["CLM:x", "CLM:xyz", "prose CLM:x", None, False, 0]}}, {"CLM:x"}))
    assert len(hits) == 2
    assert hits[1]["json_path"] == ["CLM:x", "ids", 0]


@pytest.mark.parametrize("free, fits", [(FLOOR + 999, False), (FLOOR + 1000, True), (FLOOR + 1001, True)])
def test_reserve_is_not_silently_relaxed(free, fits):
    assert storage_check(free, 1000)["one_graph_fits_with_reserve"] is fits


@pytest.mark.parametrize("free,size", [(False, 1), (-1, 1), (2, 0), (2, True)])
def test_invalid_storage_numbers(free, size):
    with pytest.raises(ValueError): storage_check(free, size)
