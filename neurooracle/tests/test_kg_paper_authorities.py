from copy import deepcopy
import hashlib
import json

import pytest

from neurooracle.src.kg_paper_identity import (
    VERSION, VerifiedPaperIdentities, authority_title, load_graph_papers, pubmed_record,
)


def source_record(pmid="1", doi="10.1234/example", title="A complete scientific title."):
    return dict(uid=pmid, title=title, pubdate="2021 Jan", epubdate="2020 Dec 28",
        articleids=[dict(idtype="pubmed", value=pmid), dict(idtype="doi", value=doi), dict(idtype="pmc", value="PMC"+pmid)])


def registry(*records, collisions=None):
    payload = dict(version=VERSION, records={r["uid"]: pubmed_record(r["uid"], r,
        dict(response_sha256="a"*64, url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi")) for r in records or [source_record()]},
        observed_collisions=collisions or {})
    return VerifiedPaperIdentities(payload)


def claim(**paper):
    return dict(id="CLM:1", source_paper=paper, raw_text="Unchanged scientific evidence", evidence={"n": 0, "negated": False})


def test_exact_cross_identifier_authority_preserves_original_claim():
    r = registry()
    a, b = claim(pmid="1"), claim(doi="https://doi.org/10.1234/EXAMPLE", title="A complete scientific title", year=2020)
    original = deepcopy([a,b])
    assert r.resolve(a) == r.resolve(b) == dict(paper_key="pmid:1", status="verified", reasons=[])
    assert [a,b] == original


@pytest.mark.parametrize("paper,reason", [
    ({"doi":"10.1234/example", "title":"A different scientific title"}, "complete_title_disagreement"),
    ({"pmid":"1", "doi":"10.1234/unknown"}, "authority_does_not_confirm_doi"),
    ({"pmid":"2", "doi":"10.1234/example"}, "pmid_disagreement"),
    ({"pmid":"GENMEDSUPP:1", "doi":"10.1234/example"}, "invalid_pmid"),
    ({"pmid":"1", "year":2022}, "publication_year_disagreement"),
    ({"pmid":"1", "doi":"not-a-doi"}, "invalid_doi"),
    ({"pmid":"1", "pmcid":"PMC1.1"}, "invalid_pmcid"),
    ({"pmid":"1", "pmcid":"PMC2"}, "authority_does_not_confirm_pmcid"),
])
def test_contradictory_or_invalid_sources_held_not_dropped(paper, reason):
    answer = registry().resolve(claim(**paper))
    assert answer["status"] == "conflict" and reason in answer["reasons"]
    assert answer["paper_key"] == "unresolved:CLM:1"


def test_shared_doi_cannot_merge_distinct_pubmed_papers():
    r = registry(source_record(), source_record("2"))
    for c in (claim(doi="10.1234/example"), claim(pmid="1", doi="10.1234/example")):
        assert r.resolve(c)["status"] == "conflict"


def test_observed_global_collision_blocks_doi_only_even_if_one_authority_returned():
    r = registry(collisions={"doi": ["10.1234/example"]})
    assert r.resolve(claim(doi="10.1234/example"))["status"] == "conflict"
    assert r.resolve(claim(pmid="1", doi="10.1234/example"))["status"] == "verified"


def test_title_only_is_never_an_authority_match():
    assert registry().resolve(claim(title="A complete scientific title.", year=2021))["status"] == "unverified"


def test_unrequested_pubmed_ids_are_not_reported_as_verified():
    answer = registry().resolve(claim(pmid="987"))
    assert answer["status"] == "unverified" and answer["paper_key"] == "pmid:987"


@pytest.mark.parametrize("where", ["outer", "inner"])
def test_redundant_pmid_disagreement_is_not_silently_overridden(where):
    c = claim(pmid="1")
    if where == "outer": c["pmid"] = "2"
    else: c["metadata"] = {"pmid":"2"}
    assert "alternate_pmid_conflict" in registry().resolve(c)["reasons"]


def test_complete_markup_title_and_two_publication_years():
    r = registry(source_record(title="A <i>complete</i> scientific title &amp; analysis."))
    for year in [2020,2021]:
        assert r.resolve(claim(pmid="1", title="A complete scientific title & analysis.", year=year))["status"] == "verified"
    assert authority_title("Not a measurement") != authority_title("A measurement")


def test_mutating_claim_or_export_does_not_return_stale_cached_identity():
    r = registry(); c = claim(pmid="1")
    assert r.resolve(c)["status"] == "verified"
    c["source_paper"]["title"] = "An unrelated paper"
    assert r.resolve(c)["status"] == "conflict"
    exported = r.export_payload(); exported["records"].clear()
    assert r.resolve(claim(pmid="1"))["status"] == "verified"


def test_pubmed_response_identity_and_proof_required():
    with pytest.raises(ValueError, match="mismatch"): pubmed_record("2", source_record(), {})
    payload = registry().export_payload(); payload["records"]["1"]["witness"] = {}
    with pytest.raises(ValueError, match="hash"): VerifiedPaperIdentities(payload)


def test_arxiv_and_journal_versions_not_merged_by_accompanying_doi():
    c = claim(doi="10.1234/example", arxiv_id="2106.03052")
    assert "unverified_arxiv_version" in registry().resolve(c)["reasons"]
    c = claim(doi="10.1234/example", journal="arXiv")
    assert "unverified_preprint_publication_version" in registry().resolve(c)["reasons"]
    record = source_record(); record["pubtype"] = ["Preprint"]
    assert registry(record).resolve(claim(doi="10.1234/example", journal="medRxiv"))["status"] == "verified"


def test_valid_legacy_doi_is_matched_using_pubmed_not_just_regex():
    doi = "10.1002/1098-2779(2000)6:3<180::AID-MRDD5>3.0.CO;2-I"
    assert registry(source_record(doi=doi)).resolve(claim(doi=doi))["status"] == "verified"
    assert registry().resolve(claim(doi=doi))["status"] == "unverified"


def test_normalized_census_projection_rechecks_raw_bibliography():
    import sqlite3
    from neurooracle.src.kg_paper_identity import census_identifier_projection
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE papers(sig TEXT,payload TEXT,pmid TEXT,doi TEXT,pmcid TEXT)")
    for p in ["1","2"]:
        db.execute("INSERT INTO papers VALUES (?,?,?,?,?)", (p, json.dumps(dict(pmid=p,doi="10.1234/<opaque>",pmcid="PMCID:PMC1")),p,"",""))
    result = census_identifier_projection(db)
    assert result["observed_collisions"] == {"doi":["10.1234/<opaque>"],"pmcid":["PMC1"]}
    assert len(result["normalized_column_overrides"]) == 2
    assert db.execute("SELECT COUNT(*) FROM papers WHERE doi=''").fetchone()[0] == 2


def test_graph_sidecar_hash_and_scope_guard(tmp_path):
    path = tmp_path / "nested" / "kg.json"; path.parent.mkdir()
    target = path.parent / "papers.json"
    raw = json.dumps(registry().export_payload()).encode(); target.write_bytes(raw)
    declaration = dict(version=VERSION, registry=target.name, sha256=hashlib.sha256(raw).hexdigest())
    assert load_graph_papers(path, {"paper_identity":declaration}).resolve(claim(pmid="1"))["status"] == "verified"
    with pytest.raises(ValueError, match="hash"):
        load_graph_papers(path, {"paper_identity":{**declaration, "sha256":"0"*64}})
    with pytest.raises(ValueError, match="escapes"):
        load_graph_papers(path, {"paper_identity":{**declaration, "registry":"../../escape.json"}})
    assert load_graph_papers(path, {}) is None


def test_relation_groups_distinguish_verified_papers_from_source_keys():
    from neurooracle.src.relation_evidence import group_claim_evidence
    from neurooracle.tests.test_relation_evidence import payload
    a = payload(pmid="1")
    b = {**payload("CLM:2"), "source_paper":{"doi":"10.1234/example"}, "negated":True}
    c = {**payload("CLM:3"), "source_paper":{"title":"Unverified title", "year":2021}}
    original = deepcopy([a,b,c])
    result = list(group_claim_evidence([a,b,c], papers=registry()))[0]
    assert result["claim_count"] == 3 and result["paper_count"] == 2
    assert result["verified_paper_count"] == 1 and result["unverified_claim_count"] == 1
    assert {m["negated"] for m in result["members"]} == {True,False}
    assert [a,b,c] == original


def test_mixed_or_unknown_member_statuses_rejected():
    from neurooracle.src.relation_evidence import evidence_member, relation_key, summarize_members
    from neurooracle.tests.test_relation_evidence import payload
    a = evidence_member(payload(pmid="1"), papers=registry())
    b = evidence_member(payload("CLM:2"))
    with pytest.raises(ValueError, match="mixed"): summarize_members(relation_key(payload()), [a,b])
    with pytest.raises(ValueError, match="status"): summarize_members(relation_key(payload()), [{**a,"paper_status":"unknown"}])


def test_cross_identifier_exact_observation_dedup_and_unknown_science_preserved():
    from neurooracle.src.claim_evidence_identity import evidence_dedup_key
    from neurooracle.tests.test_claim_evidence_identity import claim as scientific_claim
    a = scientific_claim(pmid="1").to_dict(); b = deepcopy(a)
    b.update(id="CLM:second", source_paper={"doi":"10.1234/example"})
    r = registry()
    assert evidence_dedup_key(a, papers=r) == evidence_dedup_key(b, papers=r)
    for field, value in [("negated", True), ("raw_text", "A different experiment."), ("conditions", {"age":0})]:
        assert evidence_dedup_key(a, papers=r) != evidence_dedup_key({**b,field:value}, papers=r)
    b["source_paper"]["cohort"] = "different sample"
    assert evidence_dedup_key(a, papers=r) != evidence_dedup_key(b, papers=r)
    unverified = {**a,"source_paper":{"pmid":"987"}}
    assert evidence_dedup_key(unverified, papers=r) == evidence_dedup_key(unverified)
    assert evidence_dedup_key({**a,"source_paper":{"title":"Unverified paper"}}, papers=r) is None


def test_real_ingestion_cross_identifier_dedup_survives_save_reload(tmp_path):
    from neurooracle.tests.test_claim_evidence_identity import graph, claim as scientific_claim, ingest
    from neurooracle.src.schema import PaperRef
    from neurooracle.src.storage import save_graph, load_graph, save_display_graph
    kg = graph(); kg.paper_identities = registry()
    assert ingest(kg, [scientific_claim(pmid="1")])["claims_added"] == 1
    b = scientific_claim("CLM:second", pmid=""); b.source_paper = PaperRef(doi="10.1234/example")
    assert ingest(kg, [deepcopy(b)])["claims_added"] == 0
    path = save_graph(kg, tmp_path / "graph.json")
    kg = load_graph(path)
    assert ingest(kg, [deepcopy(b)])["claims_added"] == 0
    b.negated = True
    assert ingest(kg, [deepcopy(b)])["claims_added"] == 1
    display = save_display_graph(kg, tmp_path / "display.json")
    assert "paper_identity" not in json.loads(display.read_text(encoding="utf-8"))["metadata"]
    assert list(kg.iter_relation_evidence())[0]["verified_paper_count"] == 1


def test_changing_registry_invalidates_ingestion_dedup_cache():
    from neurooracle.tests.test_claim_evidence_identity import graph, claim as scientific_claim, ingest
    from neurooracle.src.schema import PaperRef
    kg = graph()
    assert ingest(kg, [scientific_claim(pmid="1")])["claims_added"] == 1
    kg.paper_identities = registry()
    b = scientific_claim("CLM:second", pmid=""); b.source_paper = PaperRef(doi="10.1234/example")
    assert ingest(kg, [b])["claims_added"] == 0


def test_ordinary_save_drops_stale_paper_registry_pointer(tmp_path):
    from neurooracle.tests.test_claim_evidence_identity import graph
    from neurooracle.src.storage import save_graph
    kg = graph(); kg.serialization_metadata["paper_identity"] = {"registry":"stale.json"}
    assert "paper_identity" not in json.loads(save_graph(kg, tmp_path / "kg.json").read_text(encoding="utf-8"))["metadata"]


def test_authority_catalog_default_threshold_excludes_unverified_sources(tmp_path):
    from neurooracle.tests.test_shared_relation_catalog import setup_files, fp
    from neurooracle.src.shared_relation_catalog import find_shared_relations
    graph, catalog, receipt, campaign, control = setup_files(tmp_path)
    row = json.loads(catalog.read_text()); row.update(verified_paper_count=1, unverified_claim_count=1)
    catalog.write_text(json.dumps(row)+"\n")
    papers = tmp_path / "papers.json"; papers.write_text(json.dumps(registry().export_payload()))
    accepted = json.loads(receipt.read_text())
    accepted.update(shared_relations=fp(catalog), paper_identities=fp(papers))
    accepted["checks"]["paper_identity_witnesses_validated"] = True
    receipt.write_text(json.dumps(accepted))
    control.update(current_shared_relations=fp(catalog), current_paper_identities=fp(papers), current_acceptance=fp(receipt))
    campaign.write_text(json.dumps(control))
    assert find_shared_relations(campaign, minimum_papers=2) == []
    assert len(find_shared_relations(campaign, minimum_papers=2, verified_papers_only=False)) == 1
    papers.write_text(papers.read_text()+" ")
    with pytest.raises(ValueError): find_shared_relations(campaign)


def test_stream_paper_index_preserves_every_node_edge_and_context(tmp_path):
    from io import BytesIO
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
    from apply_kg_paper_identity import SourceProof
    from types import SimpleNamespace
    from neurooracle.tests.test_relation_evidence import payload
    from neurooracle.src.kg_identity_pilot import digest
    a = payload(pmid="1"); b = {**payload("CLM:2"), "source_paper":{"doi":"10.1234/example"}, "negated":True}
    nodes = {k:dict(id=k, preferred_name=k, metadata={}) for k in ("A","B")}
    for c in (a,b): nodes[c["id"]] = dict(id=c["id"], preferred_name=c["id"], metadata=c)
    edges = [dict(source_id=c["id"], target_id=target, relation_type="about") for c in (a,b) for target in ("A","B")]
    records = [("metadata","",{})] + [("node",k,v) for k,v in nodes.items()] + [("edge",str(i),e) for i,e in enumerate(edges,1)]
    catalog = tmp_path / "index.jsonl"; handle = BytesIO(); papers = registry()
    result = stream_patch(iter(records), handle, [], {}, catalog, papers=papers)
    proof = SourceProof(SimpleNamespace(entries={}), papers)
    list(proof.records(iter(records)))
    audit = dict(claim_status={"verified":2}, verified_key_changes=1)
    proof.verify(result, audit)
    with pytest.raises(ValueError, match="records changed"):
        proof.verify({**result, "record_digests":{"nodes":"0"*64,"edges":"0"*64}}, audit)
    output = json.loads(handle.getvalue())
    assert output["concepts"] == nodes and output["edges"] == edges
    assert result["catalog_stats"]["verified_multi_paper_shared_groups"] == 0
    assert result["catalog_stats"]["fully_verified_shared_groups"] == 1
    records[0] = ("metadata", "", output["metadata"])
    checks = verify_catalog_and_graph(iter(records), result, catalog, papers=papers)
    assert checks["shared_relation_index_complete"]
    row = json.loads(catalog.read_text()); row["members"][0]["paper_status"] = "unverified"
    catalog.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError): verify_catalog_and_graph(iter(records), result, catalog, papers=papers)
