"""Topic lookup over a real mini accepted graph; read-only and offline."""
from copy import deepcopy
import json
import sqlite3

import pytest

from core.topic_evidence import main, render_text, topic_evidence, topic_terms
from core.web.claim_layer_v8 import AcceptedClaimLayer
from neurooracle.tests.test_claim_evidence_query import setup, fingerprint


@pytest.fixture
def layer(tmp_path):
    campaign, records, _ = setup(tmp_path)
    return AcceptedClaimLayer(campaign), tmp_path, records


def indexed_setup(tmp_path, *, bad_offset=False, bad_cid=False):
    """Seal a real miniature original index through the existing layer format."""
    from neurooracle.src.kg_identity_pilot import digest
    from neurooracle.src.relation_evidence import relation_id, relation_key
    path, records, _ = setup(tmp_path)
    campaign = json.loads(path.read_text(encoding="utf-8"))
    graph = (tmp_path / "graph.json").read_bytes()
    database = tmp_path / "originals.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE observations(number INTEGER PRIMARY KEY,cid TEXT,node_sha TEXT,byte_offset INTEGER,"
                   "rid TEXT,subject_id TEXT,subject_name TEXT,predicate TEXT,object_id TEXT,object_name TEXT,"
                   "pmid TEXT,doi TEXT,source_key TEXT,paper_sig TEXT)")
        for i, record in enumerate(records):
            md = record["metadata"]
            cid = "CLM:fake" if bad_cid and i == 3 else record["id"]
            key = json.dumps(record["id"]).encode() + b":"
            offset = graph.index(key) + len(key)
            if bad_offset and i == 3:
                key = b'"CLM:0":'
                offset = graph.index(key) + len(key)
            db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (i, cid, digest(record), offset, relation_id(relation_key(md)),
                        *(md[k] for k in ("subject_id", "subject_name", "predicate", "object_id", "object_name")),
                        md["source_paper"]["pmid"], None, "pmid:" + md["source_paper"]["pmid"], "paper" + str(i)))
    def save(name, value):
        file = tmp_path / name
        file.write_text(json.dumps(value), encoding="utf-8")
        return fingerprint(file)
    index = save("original_index.json", dict(schema="kg.global_claim_index.acceptance.v1",
                 status="COMPLETE_RETRIEVAL_INDEX_NOT_SCIENTIFIC_APPROVAL", graph=campaign["current_graph"],
                 all_claim_hashes_verified=True, all_current_claims_indexed=len(records), database=fingerprint(database)))
    projection = save("projection.json", dict(groups=[], source_reviews={},
                      original_relation_extension=dict(global_index_acceptance=index, members=[])))
    acceptance = save("layer.json", dict(schema="kg.accepted_claim_layer.v6", status="ACCEPTED",
                      base_graph=campaign["current_graph"], base_acceptance=campaign["current_acceptance"],
                      base_dossiers=campaign["current_evidence_dossiers"], projection=projection,
                      checks=dict(source_anchors_and_whole_scopes_validated=True, original_observations_unchanged=True),
                      counts=dict(reviewed_multipaper_claims_after=1)))
    campaign["current_claim_layer"] = acceptance
    path.write_text(json.dumps(campaign), encoding="utf-8")
    return AcceptedClaimLayer(path), records


def test_singleton_topic_returns_the_same_complete_evidence_as_existing_lookup(tmp_path):
    from neurooracle.src.claim_evidence_query import query_claim_evidence
    service, records = indexed_setup(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = topic_evidence(service, "separate exposure", minimum_coverage=1)
    assert result["original_retrieval"] == dict(available=True, indexed=4, eligible=1, matched=1, matched_sources=1)
    assert result["claim_total"] == 2 and result["shared_claim_total"] == 1
    assert result["shared_claim_matched"] == 0 and result["study_returned"] == 1
    claim = result["claims"][0]
    assert claim["retrieval_origin"] == "original_singleton" and claim["query_id"] == "CLM:3"
    observation = result["studies"][0]["observations"][0]
    assert observation["metadata"] == records[3]["metadata"]
    assert observation["population"] == "Population 333" and observation["conditions"] == ["adjusted"]
    expected = query_claim_evidence(service.path, claim_id="CLM:3")
    actual = service.query_batch(queries=[{"claim_id": "CLM:3"}])["results"][0]
    assert all(actual[k] == v for k, v in expected.items())
    assert result["original_index_revision"]
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_original_recall_does_not_repeat_shared_observations_and_obeys_budget(tmp_path):
    service, _ = indexed_setup(tmp_path)
    result = topic_evidence(service, "exposure", limit=2, evidence_claims=2)
    assert result["claim_matched"] == 2 and result["study_returned"] == 3
    ids = [o["claim_id"] for s in result["studies"] for o in s["observations"]]
    assert sorted(ids) == ["CLM:0", "CLM:1", "CLM:2", "CLM:3"]
    result = topic_evidence(service, "separate", limit=1, evidence_claims=0)
    assert result["claims"][0]["counts"]["reviewed_supporting_article_count"] is None
    assert not result["claims"][0]["evidence_expanded"] and result["studies"] == []
    assert topic_evidence(service, "separate", minimum_papers=1)["study_returned"] == 1
    assert topic_evidence(service, "separate", minimum_papers=2)["claims"] == []
    assert topic_evidence(service, "separate", evidence_claims=0, minimum_papers=1)["claims"] == []


@pytest.mark.parametrize("fault", ["bad_offset", "bad_cid"])
def test_original_index_cannot_substitute_a_record_or_invent_an_id(tmp_path, fault):
    from core.web.claim_evidence import EvidenceUnavailable
    service, _ = indexed_setup(tmp_path, **{fault: True})
    with pytest.raises(EvidenceUnavailable):
        topic_evidence(service, "separate", minimum_coverage=1)


@pytest.mark.parametrize("file", ["originals.sqlite", "original_index.json"])
@pytest.mark.parametrize("expand", [0, 1])
def test_original_index_changes_during_a_query_reject_the_whole_response(tmp_path, monkeypatch, file, expand):
    from core.web.claim_evidence import EvidenceUnavailable
    service, _ = indexed_setup(tmp_path)
    original = service.original_claim_candidates
    def changed(*args, **kwargs):
        found = original(*args, **kwargs)
        with (tmp_path / file).open("ab") as stream:
            stream.write(b" ")
        return found
    monkeypatch.setattr(service, "original_claim_candidates", changed)
    with pytest.raises((EvidenceUnavailable, ValueError)):
        topic_evidence(service, "separate", evidence_claims=expand)


@pytest.mark.parametrize("topic,expected", [
    ("Hippocampal atrophy in MCI", ["hippocampal", "atrophy", "mci"]),
    ("The role of the cerebellum", ["cerebellum"]),
    ("\u6d77\u9a6c\u4f53\u840e\u7f29\u4e0e\u8ba4\u77e5", ["hippocampal", "atrophy", "cognition"]),
    ("Cerebellum \u5c0f\u8111 timing", ["cerebellum", "timing"]),
    ("海马与轻度认知障碍", ["hippocampal", "mci"]),
])
def test_topic_terms_keep_latin_words_and_cjk_ngrams(topic, expected):
    terms = topic_terms(topic)
    assert terms == expected
    assert topic_terms("the of and") == []


def test_topic_matches_and_expands_evidence_from_the_current_graph(layer):
    service, root, records = layer
    before = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
    result = topic_evidence(service, "exposure outcome")
    assert result["claim_total"] == 1 and result["claim_matched"] == 1
    claim = result["claims"][0]
    assert claim["match"]["coverage"] == 1.0
    assert claim["evidence_expanded"] is True
    assert claim["counts"]["reviewed_supporting_article_count"] == 2
    assert {study["pmid"] for study in claim["studies"]} == {"111", "222"}
    reviewed = [observation for study in claim["studies"] for observation in study["observations"]]
    assert {observation["proposition_support"] for observation in reviewed} == {"supports"}
    assert any(observation["observation_role"] == "own_result" for observation in reviewed)
    assert result["graph_revision"]
    assert before == {path: path.read_bytes() for path in root.iterdir() if path.is_file()}


def test_extra_terms_and_evidence_budget_are_bounded(layer):
    service, _, _ = layer
    result = topic_evidence(service, "\u6d77\u9a6c", extra_terms=["exposure"], evidence_claims=0)
    assert "exposure" in result["terms"]
    assert result["claims"][0]["evidence_expanded"] is False
    assert result["claims"][0]["studies"] == []


def test_a_topic_with_no_match_reports_the_gap_instead_of_failing(layer):
    service, _, _ = layer
    result = topic_evidence(service, "quantum entanglement")
    assert result["claims"] == [] and result["claim_matched"] == 0
    assert result["graph_revision"] == service.status()["graph_revision"]
    assert any("No shared claim matched" in note for note in result["notes"])
    assert "quantum" in render_text(result)


def test_coverage_and_paper_filters_exclude_weak_or_single_source_matches(layer):
    service, _, _ = layer
    assert topic_evidence(service, "exposure outcome", minimum_papers=3)["claims"] == []
    partial = topic_evidence(service, "exposure outcome absentterm", minimum_coverage=1.0)
    assert partial["claims"] == [] and partial["minimum_matches"] == 3
    partial = topic_evidence(service, "exposure outcome absentterm", minimum_coverage=0.6)
    assert partial["claim_matched"] == 1


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 51}, {"evidence_claims": -1}, {"minimum_papers": -1},
    {"minimum_coverage": 1.5},
])
def test_invalid_parameters_are_rejected(layer, kwargs):
    service, _, _ = layer
    with pytest.raises(ValueError):
        topic_evidence(service, "exposure outcome", **kwargs)


def test_empty_topic_is_rejected(layer):
    service, _, _ = layer
    with pytest.raises(ValueError):
        topic_evidence(service, "the of and")


def test_render_text_names_sources_and_limits(layer):
    service, _, _ = layer
    text = render_text(topic_evidence(service, "exposure outcome"))
    assert "exposure correlates_with outcome" in text
    assert "111" in text and "reviewed supporting 2" in text
    assert "not semantic, full-text or literature search" in text


def test_cli_writes_json_once_and_never_overwrites(layer, tmp_path, capsys):
    service, _, _ = layer
    campaign = service.path
    output = tmp_path / "lookup.json"
    main(["--topic", "exposure outcome", "--campaign", str(campaign), "--json", "--output", str(output)])
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["claims"][0]["shared_claim_id"] == service.claim_index()[0][0]["shared_claim_id"]
    assert json.loads(capsys.readouterr().out)["topic"] == "exposure outcome"
    with pytest.raises(FileExistsError):
        main(["--topic", "exposure outcome", "--campaign", str(campaign), "--output", str(output)])


def test_claim_index_cannot_mutate_the_cached_graph(layer):
    service, _, _ = layer
    snapshot = deepcopy(service.claim_index())
    assert snapshot and all(isinstance(text, str) for _, text in snapshot)
    service.claim_index()[0][0]["claim"]["subject_name"] = "corrupted"
    assert service.claim_index() == snapshot


def test_papers_keep_metadata_and_source_provenance(layer):
    service, _, records = layer
    result = topic_evidence(service, "exposure outcome")
    assert result["study_returned"] == 2
    originals = {r["id"]: r["metadata"] for r in records}
    for paper in result["studies"]:
        assert paper["related_claims"] and paper["publication_versions"]
        assert paper["bibliography"]["pmid"] == paper["pmid"]
        for observation in paper["observations"]:
            assert observation["metadata"] == originals[observation["claim_id"]]
            assert observation["shared_claim_id"] in {c["shared_claim_id"] for c in paper["related_claims"]}


def test_abbreviation_does_not_match_inside_an_unrelated_word(layer, monkeypatch):
    service, _, _ = layer
    index = service.claim_index()
    monkeypatch.setattr(service, "claim_index", lambda: [(index[0][0], "load subtle outcome")])
    assert topic_evidence(service, "AD")["claims"] == []
    assert topic_evidence(service, "TLE")["claims"] == []


def test_each_paper_must_match_the_topic_not_just_its_shared_claim(layer, monkeypatch):
    service, _, _ = layer
    index = service.claim_index()
    monkeypatch.setattr(service, "claim_index", lambda: [(index[0][0], "exposure outcome mci")])
    original = service.query_batch
    def labeled(**kwargs):
        batch = original(**kwargs)
        batch["results"][0]["papers"][0]["bibliography"]["title"] = "MCI exposure"
        return batch
    monkeypatch.setattr(service, "query_batch", labeled)
    result = topic_evidence(service, "exposure MCI")
    assert result["study_returned"] == 1 and result["context_study_count"] == 1
    assert result["studies"][0]["missing_terms"] == []


def test_topic_checks_all_dependencies_at_two_boundaries_not_for_each_nested_read(layer, monkeypatch):
    from core.web import claim_evidence
    service, _, _ = layer
    topic_evidence(service, "exposure outcome")
    original = claim_evidence.check_file
    checked = []
    def tracked(fp, **kwargs):
        checked.append(fp["path"])
        return original(fp, **kwargs)
    monkeypatch.setattr(claim_evidence, "check_file", tracked)
    topic_evidence(service, "exposure outcome")
    graph = service._campaign["current_graph"]["path"]
    assert checked.count(graph) == 2


@pytest.mark.parametrize("expand", [0, 1])
def test_revision_change_between_search_and_expansion_is_rejected(layer, monkeypatch, expand):
    from core.web.claim_evidence import EvidenceUnavailable
    service, root, _ = layer
    original = service.claim_index
    def changed():
        result = original()
        with (root / "graph.json").open("ab") as stream:
            stream.write(b" ")
        return result
    monkeypatch.setattr(service, "claim_index", changed)
    with pytest.raises((ValueError, EvidenceUnavailable)):
        topic_evidence(service, "exposure outcome", evidence_claims=expand)


def test_changed_layer_revision_or_fake_id_cannot_supply_evidence(layer, monkeypatch):
    from core.web.claim_evidence import EvidenceUnavailable
    service, _, _ = layer
    original = service.query_batch
    def changed(**kwargs):
        batch = original(**kwargs)
        batch["graph_revision"] = "other"
        return batch
    monkeypatch.setattr(service, "query_batch", changed)
    with pytest.raises(EvidenceUnavailable):
        topic_evidence(service, "exposure outcome")
    monkeypatch.setattr(service, "query_batch", original)
    summary, text = service.claim_index()[0]
    summary["shared_claim_id"] = "REL:missing"
    monkeypatch.setattr(service, "claim_index", lambda: [(summary, text)])
    with pytest.raises(KeyError):
        topic_evidence(service, "exposure outcome")


def test_http_topic_returns_evidence_and_reports_invalid_or_stale_input(layer):
    from fastapi.testclient import TestClient
    from core.web.server import create_app
    service, root, _ = layer
    app = create_app()
    app.state.accepted_claim_evidence = service
    with TestClient(app) as client:
        response = client.get("/api/kg/topic-evidence", params={"topic": "exposure outcome"})
        assert response.status_code == 200 and response.json()["study_returned"] == 2
        assert client.get("/api/kg/topic-evidence", params={"topic": "the and"}).status_code == 422
        assert client.get("/api/kg/topic-evidence", params={"topic": "exposure", "limit": 0}).status_code == 422
        with (root / "graph.json").open("ab") as stream:
            stream.write(b" ")
        response = client.get("/api/kg/topic-evidence", params={"topic": "exposure"})
        assert response.status_code == 503 and "studies" not in response.json()
