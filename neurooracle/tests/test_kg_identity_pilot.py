from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import apply_kg_relation_identity as apply
from neurooracle.src.kg_identity_pilot import change_claim, change_edge, digest, nonidentity_claim
from neurooracle.src.relation_evidence import group_claim_evidence


def fixture():
    nodes = {nid: {"id": nid, "preferred_name": nid, "domain_tags": [], "metadata": {}}
             for nid in ("old", "target", "b")}
    for index, sid in ((1, "old"), (2, "target")):
        cid = "CLM:" + str(index)
        nodes[cid] = {"id": cid, "preferred_name": "measurement relates outcome", "metadata": {
            "id": cid, "subject_id": sid, "object_id": "b", "subject_name": "measurement",
            "object_name": "outcome", "predicate": "correlates_with", "negated": False,
            "source_paper": {"pmid": str(index)}, "raw_text": "Original evidence is preserved.",
            "metadata": {"subject_id": sid, "population": {"n": 0}},
            "scope_reaudit": {"frozen": True, "hash": "unchanged"}, "unknown": [0, False],
            "evidence": {"direction": "positive" if index == 1 else "negative"}}}
    edges = []
    for i, sid in ((1, "old"), (2, "target")):
        cid = "CLM:" + str(i)
        edges += [{"source_id": cid, "target_id": sid, "relation_type": "about", "metadata": {}},
                  {"source_id": sid, "target_id": "b", "relation_type": "correlates_with", "metadata": {"claim_id": cid, "negated": False}}]
    event = {"claim_id": "CLM:1", "side": "subject", "old_id": "old", "target_id": "target",
             "claim_sha256": digest(nodes["CLM:1"])}
    records = [("metadata", "", {"original": True})] + [("node", nid, row) for nid,row in nodes.items()]
    records += [("edge", str(i), row) for i,row in enumerate(edges, 1)]
    return nodes, edges, event, records


def test_only_routing_ids_change_no_science_audit_or_unknown_field_changes():
    nodes, _, event, _ = fixture()
    before = deepcopy(nodes["CLM:1"])
    after = change_claim(before, event)
    assert before == nodes["CLM:1"]
    assert digest(nonidentity_claim(after)) == digest(nonidentity_claim(before))
    assert after["metadata"]["subject_id"] == after["metadata"]["metadata"]["subject_id"] == "target"


@pytest.mark.parametrize("changes", [{"claim_sha256": "wrong"}, {"old_id": "stale"}, {"target_id": "b"}])
def test_stale_or_self_loop_endpoint_refused(changes):
    nodes, _, event, _ = fixture()
    with pytest.raises(ValueError):
        change_claim(nodes["CLM:1"], {**event, **changes})


def test_nested_conflict_refused_even_if_record_hash_matches():
    nodes, _, event, _ = fixture()
    node = nodes["CLM:1"]
    node["metadata"]["metadata"]["subject_id"] = "conflicting"
    event["claim_sha256"] = digest(node)
    with pytest.raises(ValueError, match="nested"):
        change_claim(node, event)


def test_owning_science_and_about_edges_follow_only_selected_claim():
    _, edges, event, _ = fixture()
    events = {"CLM:1": event}
    assert change_edge(edges[0], events)["target_id"] == "target"
    assert change_edge(edges[1], events)["source_id"] == "target"
    assert change_edge(edges[2], events) is edges[2]
    assert change_edge(edges[3], events) is edges[3]
    assert edges[1]["source_id"] == "old"


def test_science_edge_disagreement_refused():
    _, edges, event, _ = fixture()
    with pytest.raises(ValueError):
        change_edge({**edges[1], "source_id": "unexpected"}, {"CLM:1": event})


def test_stream_build_and_independent_verification(tmp_path):
    nodes, edges, event, records = fixture()
    stream = BytesIO()
    catalog = tmp_path / "relations.jsonl"
    expected = apply.stream_patch(iter(records), stream, [event], {"target": digest(nodes["target"])}, catalog)
    output = json.loads(stream.getvalue())
    parsed = [("metadata", "", output["metadata"])] + [("node", nid,row) for nid,row in output["concepts"].items()]
    parsed += [("edge",str(i),row) for i,row in enumerate(output["edges"],1)]
    checks = apply.verify_catalog_and_graph(iter(parsed), expected, catalog)
    assert checks["all_nonidentity_claim_fields_preserved"]
    assert expected["changed"] == {"claim_nodes": 1, "edge_records": 2}
    assert expected["counts"] == {"nodes": 5, "claims": 2, "edges": 4}
    indexed = [json.loads(line) for line in catalog.read_text().splitlines()]
    runtime = list(group_claim_evidence([output["concepts"]["CLM:1"]["metadata"], output["concepts"]["CLM:2"]["metadata"]]))
    assert indexed == runtime


@pytest.mark.parametrize("tamper", ["label", "source", "negated", "membership"])
def test_independent_catalog_validation_rejects_tampering(tmp_path, tamper):
    nodes, _, event, records = fixture()
    stream, catalog = BytesIO(), tmp_path / "relations.jsonl"
    expected = apply.stream_patch(iter(records), stream, [event], {"target":digest(nodes["target"])}, catalog)
    output = json.loads(stream.getvalue())
    values = [json.loads(line) for line in catalog.read_text().splitlines()]
    if tamper == "label": values[0]["subject_name"] = "different"
    if tamper == "source": values[0]["members"][0]["paper_key"] = "pmid:wrong"
    if tamper == "negated": values[0]["members"][0]["negated"] = 0
    if tamper == "membership": values[0]["members"][0]["claim_id"] = "CLM:missing"
    catalog.write_text("\n".join(json.dumps(v) for v in values), encoding="utf-8")
    parsed = [("metadata", "", output["metadata"])] + [("node",nid,row) for nid,row in output["concepts"].items()]
    parsed += [("edge",str(i),row) for i,row in enumerate(output["edges"],1)]
    with pytest.raises((ValueError, KeyError)):
        apply.verify_catalog_and_graph(iter(parsed), expected, catalog)
