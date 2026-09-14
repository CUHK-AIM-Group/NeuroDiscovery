from copy import deepcopy
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compact_current_kg_metadata as pipeline


def fixture():
    meta = {"stats": {"n_concepts": 3, "n_edges": 1, "domains": {"claim": 1},
            "sources": {"": 3}, "relations": {"causes": 1}, "connected_components": 2}}
    claim = {"id": "CLM:test", "preferred_name": "claim", "domain_tags": ["claim"],
        "metadata": {"id": "CLM:test", "claim_id": "CLM:test", "subject_id": "A", "object_id": "B",
            "subject_name": "Gene", "object_name": "Disease", "predicate": "causes", "confidence": 0.5,
            "negated": False, "raw_text": "Evidence.", "raw_sentence": "Evidence.", "population": [],
            "evidence": {"p_value": 0, "sample_size": None}, "source_paper": {"pmid": "123"},
            "scope_reaudit": {"decision": "finalized", "claim_evidence_sha256": "untouched"},
            "paper_case_study_ids": [], "claim_case_study_ids": [], "kg_injected": False,
            "metadata": {"population": [], "subject_id": "A", "object_id": "B",
                "paper_case_study_ids": [], "claim_case_study_ids": [],
                "case_study_membership_schema_version": "case_study_membership.v2"}}}
    nodes = {"A": {"id": "A", "metadata": {"unique": {"zero": 0}}},
             "B": {"id": "B", "metadata": {"unique": [False]}}, "CLM:test": claim}
    edges = [{"source_id": "A", "target_id": "B", "relation_type": "causes", "confidence": 0.5,
              "metadata": {"claim_id": "CLM:test", "claim_case_study_ids": [],
                           "case_study_membership_schema_version": "case_study_membership.v2"}}]
    baseline = {"counts": {"nodes": 3, "claims": 1, "edges": 1}, "connected_components": 2,
                "isolated_nodes": 1, "self_loops": 0}
    issue = {"claim_id": "CLM:test", "current_node_sha256": pipeline.record_digest(claim),
             "related_edge_ordinals": [1], "classification": "additional_review"}
    return {"metadata": meta, "concepts": nodes, "edges": edges}, baseline, [issue]


def records(data):
    yield "metadata", "", data["metadata"]
    for key, record in data["concepts"].items():
        yield "node", key, record
    for ordinal, record in enumerate(data["edges"], 1):
        yield "edge", str(ordinal), record


def built():
    data, baseline, issues = fixture()
    buffer = io.BytesIO()
    expected = pipeline.stream_compact(records(data), buffer, baseline, issues)
    return data, json.loads(buffer.getvalue()), expected


def test_source_output_structure_science_and_current_issue_continuity():
    source, output, expected = built()
    checks, coverage, denominators = pipeline.verify_compaction(records(output), expected)
    assert checks["protected_science_and_audits_identical"]
    assert checks["current_issue_hashes_verified"] == 1
    assert checks["counts"] == {"nodes": 3, "claims": 1, "edges": 1}
    assert output["metadata"][pipeline.LAYOUT_KEY] == pipeline.LAYOUT
    assert source["concepts"]["CLM:test"]["metadata"]["claim_id"] == "CLM:test"
    assert "claim_id" not in output["concepts"]["CLM:test"]["metadata"]
    assert "claim_id" in output["edges"][0]["metadata"]
    assert output["concepts"]["A"] == source["concepts"]["A"]
    assert expected["issues"][0]["related_edge_ordinals"] == [1]
    assert expected["issues"][0]["current_node_sha256"] == pipeline.record_digest(output["concepts"]["CLM:test"])
    assert "current_record" not in expected["issues"][0]
    assert expected["issues"][0]["rollback_preimages_saved"] is False
    assert denominators["node/claim"] == 1
    assert not any(row["scope"] == "node/claim" and row["field"] == "metadata.claim_id" for row in coverage)


@pytest.mark.parametrize("mutation", ["science", "audit", "missing_node", "dangling", "owner", "unknown_extension", "metadata", "edge_count"])
def test_independent_verifier_rejects_corruption(mutation):
    _, output, expected = built()
    if mutation == "science":
        output["concepts"]["CLM:test"]["metadata"]["evidence"]["p_value"] = False
    elif mutation == "audit":
        output["concepts"]["CLM:test"]["metadata"]["scope_reaudit"]["decision"] = "changed"
    elif mutation == "missing_node":
        del output["concepts"]["B"]
    elif mutation == "dangling":
        output["edges"][0]["target_id"] = "missing"
    elif mutation == "owner":
        output["edges"][0]["metadata"]["claim_id"] = "CLM:missing"
    elif mutation == "unknown_extension":
        output["concepts"]["A"]["metadata"]["unique"]["zero"] = False
    elif mutation == "metadata":
        output["metadata"]["stats"]["n_edges"] = 0
    else:
        output["edges"].append(deepcopy(output["edges"][0]))
    with pytest.raises(ValueError):
        pipeline.verify_compaction(records(output), expected)


def test_builder_rejects_changed_science_and_stale_issue(monkeypatch):
    source, baseline, issues = fixture()
    issues[0]["current_node_sha256"] = "wrong"
    with pytest.raises(ValueError, match="issue source"):
        pipeline.stream_compact(records(source), io.BytesIO(), baseline, issues)
    original = pipeline.compact_record
    def corrupt(kind, record, changes):
        result = deepcopy(original(kind, record, changes))
        if result.get("id") == "CLM:test":
            result["metadata"]["scope_reaudit"]["decision"] = "different"
        return result
    monkeypatch.setattr(pipeline, "compact_record", corrupt)
    with pytest.raises(ValueError, match="scientific/audit"):
        pipeline.stream_compact(records(source), io.BytesIO(), baseline)


def test_builder_rejects_layout_reentry_and_count_drift():
    source, baseline, _ = fixture()
    source["metadata"][pipeline.LAYOUT_KEY] = deepcopy(pipeline.LAYOUT)
    with pytest.raises(ValueError, match="already has"):
        pipeline.stream_compact(records(source), io.BytesIO(), baseline)
    del source["metadata"][pipeline.LAYOUT_KEY]
    baseline["counts"]["claims"] = 2
    with pytest.raises(ValueError, match="counts"):
        pipeline.stream_compact(records(source), io.BytesIO(), baseline)


def test_writer_output_has_no_record_preimages_or_extra_graph_sections():
    _, output, expected = built()
    assert set(output) == {"metadata", "concepts", "edges"}
    assert not any(key in expected for key in ("original_records", "deleted_records", "rollback", "preimages"))
    assert all(isinstance(n, int) for n in expected["removals"].values())
