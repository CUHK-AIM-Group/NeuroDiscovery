import hashlib
import json
from pathlib import Path

import pytest

from neurooracle.src.shared_relation_catalog import find_shared_relations


def fp(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def setup_files(tmp_path):
    graph, catalog, receipt, campaign = [tmp_path/name for name in ("kg.json", "relations.jsonl", "accepted.json", "campaign.json")]
    graph.write_text("{}")
    catalog.write_text(json.dumps(dict(subject_name="measurement", predicate="correlates_with", object_name="outcome", paper_count=2)) + "\n")
    accepted = dict(graph=fp(graph), shared_relations=fp(catalog), checks={"shared_relation_index_complete": True})
    receipt.write_text(json.dumps(accepted))
    control = dict(status="COMPLETED", active_process=None, current_graph=fp(graph), current_shared_relations=fp(catalog), current_acceptance=fp(receipt))
    campaign.write_text(json.dumps(control))
    return graph, catalog, receipt, campaign, control


def test_lookup_without_loading_large_graph(tmp_path):
    *_, campaign, control = setup_files(tmp_path)
    rows = find_shared_relations(campaign, subject="Measurement", minimum_papers=2)
    assert len(rows) == 1
    assert find_shared_relations(campaign, predicate="causes") == []


@pytest.mark.parametrize("index", [0,1,2])
def test_changed_graph_catalog_or_receipt_is_rejected(tmp_path, index):
    files = setup_files(tmp_path)
    path = files[index]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        find_shared_relations(files[3])


def test_active_rewrite_is_not_served_as_current(tmp_path):
    *_, campaign, control = setup_files(tmp_path)
    control["active_process"] = {"kind": "rewrite"}
    campaign.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="progress"):
        find_shared_relations(campaign)


def test_invalid_paper_threshold(tmp_path):
    *_, campaign, _ = setup_files(tmp_path)
    with pytest.raises(ValueError):
        find_shared_relations(campaign, minimum_papers=True)


def test_verified_canonical_index_searches_original_alias(tmp_path):
    graph,catalog,receipt,campaign,control=setup_files(tmp_path)
    catalog.write_text(json.dumps(dict(subject_id="CUI:test",subject_name="canonical measurement",predicate="correlates_with",object_id="B",object_name="outcome",paper_count=2))+"\n")
    terms=tmp_path/"terms.json"
    terms.write_text(json.dumps(dict(terms=[dict(name="original measurement",target_id="CUI:test",canonical_name="canonical measurement")])))
    control.update(current_shared_relations=fp(catalog),current_entity_terms=fp(terms))
    receipt.write_text(json.dumps(dict(graph=fp(graph),shared_relations=fp(catalog),entity_terms=fp(terms),
        checks={"shared_relation_index_complete":True,"verified_identity_proofs_complete":True})))
    control["current_acceptance"]=fp(receipt); campaign.write_text(json.dumps(control))
    assert len(find_shared_relations(campaign,subject="Original measurement",minimum_papers=2))==1
    terms.write_text(terms.read_text()+" ")
    with pytest.raises(ValueError): find_shared_relations(campaign,subject="Original measurement")
