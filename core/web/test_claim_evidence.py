"""Exercise the web boundary against a real, acceptance-bound small graph."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from core.web.claim_evidence import AcceptedClaimEvidence, EvidenceUnavailable, configured_campaign
from neurooracle.tests.test_claim_evidence_query import setup


def test_search_uses_reviewed_papers_not_duplicate_observations(tmp_path):
    path, records, _ = setup(tmp_path)
    service = AcceptedClaimEvidence(path)
    found = service.search("exposure", limit=1)
    assert found["total"] == 1
    row = found["claims"][0]
    assert row["article_count"] == row["reviewed_supporting_article_count"] == 2
    assert row["observation_count"] == 3
    assert row["own_result_article_count"] == 1
    assert service.search("exposure", minimum_papers=3)["total"] == 0
    assert service.search("missing")["total"] == 0
    assert service.search("exposure", offset=1)["claims"] == []
    assert service.status()["reviewed_multipaper_claim_count"] == 1


def test_original_ids_and_shared_id_preserve_the_complete_evidence(tmp_path):
    path, records, _ = setup(tmp_path)
    service = AcceptedClaimEvidence(path)
    first = service.query(claim_id="CLM:0")
    second = service.query(claim_id="CLM:1")
    shared = service.query(relation_id=first["shared_claim_id"])
    assert first["papers"] == second["papers"] == shared["papers"]
    assert first["requested_claim_id"] == "CLM:0" and second["requested_claim_id"] == "CLM:1"
    observed = {o["claim_id"]: o["original_claim"] for p in first["papers"] for o in p["observations"]}
    assert observed == {r["id"]: r["metadata"] for r in records[:3]}
    first["papers"].clear()
    assert len(service.query(claim_id="CLM:0")["papers"]) == 2
    single = service.query(claim_id="CLM:3")
    assert single["article_count"] == 1 and not single["consensus_inferred"]


@pytest.mark.parametrize("filename", ["graph.json", "census.sqlite", "dossiers.jsonl", "acceptance.json", "reviews.json"])
def test_cached_results_never_mask_changed_accepted_inputs(tmp_path, filename):
    path, _, _ = setup(tmp_path)
    service = AcceptedClaimEvidence(path)
    service.query(claim_id="CLM:0")
    with (tmp_path / filename).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        service.query(claim_id="CLM:0")
    with pytest.raises(ValueError):
        service.search()


def test_writer_state_rejects_the_previous_cached_result(tmp_path):
    path, _, _ = setup(tmp_path)
    service = AcceptedClaimEvidence(path)
    service.query(claim_id="CLM:0")
    campaign = json.loads(path.read_text())
    campaign.update(status="MANUAL_ACTIVE", active_process={"pid": 1})
    path.write_text(json.dumps(campaign))
    with pytest.raises(EvidenceUnavailable, match="updated"):
        service.query(claim_id="CLM:0")


def test_missing_configuration_is_distinct_from_missing_configured_source(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROCLAW_KG_CAMPAIGN", raising=False)
    assert configured_campaign(tmp_path) is None
    assert AcceptedClaimEvidence(None).status() == dict(configured=False, available=False)
    monkeypatch.setenv("NEUROCLAW_KG_CAMPAIGN", "missing.json")
    assert configured_campaign(tmp_path) == tmp_path / "missing.json"
    with pytest.raises(EvidenceUnavailable):
        AcceptedClaimEvidence(configured_campaign(tmp_path)).status()


@pytest.fixture
def client(tmp_path):
    from core.web.server import create_app
    path, _, _ = setup(tmp_path)
    app = create_app()
    app.state.accepted_claim_evidence = AcceptedClaimEvidence(path)
    with TestClient(app) as client:
        yield client, tmp_path


def test_http_routes_serve_current_paper_list_and_explicit_errors(client):
    http, tmp_path = client
    found = http.get("/api/kg/shared-claims").json()
    assert found["total"] == 1
    response = http.get("/api/kg/claim-evidence", params={"claim_id": "CLM:0"})
    assert response.status_code == 200 and len(response.json()["papers"]) == 2
    assert response.json()["graph_revision"] == found["graph_revision"]
    assert http.get("/api/kg/claim-evidence", params={"claim_id": "CLM:missing"}).status_code == 404
    assert http.get("/api/kg/claim-evidence").status_code == 422
    assert http.get("/api/kg/claim-evidence", params={"claim_id": "X"}).status_code == 422
    assert http.get("/api/kg/shared-claims", params={"limit": 101}).status_code == 422
    with (tmp_path / "graph.json").open("ab") as handle:
        handle.write(b" ")
    stale = http.get("/api/kg/claim-evidence", params={"claim_id": "CLM:0"})
    assert stale.status_code == 503 and stale.json()["available"] is False
