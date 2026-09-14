"""R33 bounded public XML witnesses for existing bibliographic holds only.

No KG edits, no private text sent, no model calls. Cached complete public XML
is evidence, not a KG or claim preimage. Never fetch through an abstract URL.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from xml.etree import ElementTree as ET

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_paper_identity import identifiers, normalize_pmid

OUTPUT = journal.OUTPUT / "round33_complete_title_evidence"
PREVIOUS = journal.OUTPUT / "round31_paper_identity"
API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def own_articles(raw):
    root = ET.fromstring(raw)
    require(root.tag == "PubmedArticleSet", "unexpected XML root")
    found = {}
    for article in root:
        require(article.tag in {"PubmedArticle", "PubmedBookArticle"}, "unexpected XML article kind")
        p = article.find("./MedlineCitation/PMID" if article.tag == "PubmedArticle" else "./BookDocument/PMID")
        pmid = normalize_pmid(p.text if p is not None else None)
        require(pmid and pmid not in found, "missing/duplicate owning PMID")
        found[pmid] = article
    return found


def main():
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["active_process"] is None and campaign["status"] == "COMPLETED", "KG writer active")
    journal.guards([campaign["current_graph"], campaign["current_detail_store"], campaign["formal_sources"]])
    for field in ("current_acceptance", "current_paper_identities", "current_paper_issues"):
        require(journal.fingerprint(Path(campaign[field]["path"])) == campaign[field], "current evidence changed")
    census = journal.read_json(PREVIOUS / "CENSUS.json"); journal.guards(census["database"])
    records = journal.read_json(campaign["current_paper_identities"]["path"])["records"]
    aliases = {kind: defaultdict(set) for kind in ("doi", "pmcid")}
    for pmid, record in records.items():
        for kind in aliases:
            for value in record[kind]: aliases[kind][value].add(pmid)
    db = sqlite3.connect(Path(census["database"]["path"]).as_uri()+"?mode=ro", uri=True)
    pmids = set(); affected = 0
    for line in Path(campaign["current_paper_issues"]["path"]).read_text(encoding="utf8").splitlines():
        item = json.loads(line)
        if "complete_title_disagreement" not in item["reasons"] and "alternate_pmid_conflict" not in item["reasons"]: continue
        affected += 1
        paper = json.loads(db.execute("SELECT p.payload FROM papers p JOIN claims c ON c.paper_sig=p.sig WHERE c.cid=?", (item["claim_id"],)).fetchone()[0])
        ids = identifiers(paper)
        if ids["pmid"] in records: pmids.add(ids["pmid"])
        for kind in aliases: pmids.update(aliases[kind].get(ids[kind], set()))
    for row in census["outer_pmid_conflicts"]:
        pmid = normalize_pmid(row["alternate_pmid"])
        if pmid in records: pmids.add(pmid)
    db.close()
    pmids = sorted(pmids, key=int)
    require(0 < len(pmids) <= 600, "unexpected broad fetch scope")
    request = dict(graph=campaign["current_graph"], source_registry=campaign["current_paper_identities"],
        source_issues=campaign["current_paper_issues"], census=journal.fingerprint(PREVIOUS / "CENSUS.json"),
        pmids=pmids, held_claims_in_scope=affected, endpoint=API, public_ids_only=True,
        policy="25 PMIDs per request; sequential <=1/s; no title-only search or models")
    OUTPUT.mkdir(exist_ok=True); plan = OUTPUT / "XML_REQUEST.json"
    if plan.exists(): require(journal.read_json(plan) == request, "fetch scope changed")
    else: journal.atomic_json(plan, request)
    cache = OUTPUT / "pubmed_xml"; cache.mkdir(exist_ok=True)
    witnesses = []; errors = []; downloaded = 0; reused = 0
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent":"NeuroClawKGIdentityAudit/1.0"}) as client:
        for start in range(0, len(pmids), 25):
            batch = pmids[start:start+25]; key = hashlib.sha256(",".join(batch).encode()).hexdigest()[:20]
            path = cache / (key + ".xml")
            params = dict(db="pubmed", id=",".join(batch), retmode="xml", tool="NeuroClawKGIdentityAudit")
            if path.exists():
                raw = path.read_bytes(); reused += 1
            else:
                raw = None
                for attempt in range(3):
                    try:
                        result = client.get(API, params=params); result.raise_for_status()
                        found = own_articles(result.content)
                        require(set(found) <= set(batch) and found, "unrequested/empty article response")
                        journal.atomic_text(path, result.text); raw = path.read_bytes(); downloaded += 1; break
                    except (httpx.HTTPError, ValueError, ET.ParseError) as error:
                        errors.append(dict(start=start, attempt=attempt+1, error=str(error)[:300]))
                        time.sleep(attempt+2)
                require(raw is not None, "XML retrieval failed; cached prior batches can resume")
            found = own_articles(raw); require(set(found) <= set(batch), "unrequested cached record")
            witnesses.append(dict(requested_pmids=batch, missing_pmids=sorted(set(batch)-set(found)),
                response=journal.fingerprint(path), url=str(httpx.URL(API, params=params))))
            state = dict(status="FETCHING", at=journal.utc_now(), processed=min(start+25,len(pmids)), total=len(pmids),
                downloaded_batches=downloaded, reused_batches=reused, witnesses=witnesses, errors=errors)
            journal.atomic_json(OUTPUT / "XML_FETCH_STATE.json", state)
            print(json.dumps({k:v for k,v in state.items() if k not in {"witnesses","errors"}}), flush=True)
            time.sleep(1.05)
    journal.guards([campaign["current_graph"], campaign["formal_sources"], census["database"]])
    require(journal.read_json(journal.OUTPUT / "CAMPAIGN.json") == campaign, "KG campaign changed during fetch")
    journal.atomic_json(OUTPUT / "XML_FETCH.json", dict(status="FETCHED_NOT_ADOPTED", at=journal.utc_now(),
        request=journal.fingerprint(plan), witnesses=witnesses, errors=errors, graph_modified=False,
        code=journal.fingerprint(Path(__file__))))
    print("COMPLETE_PUBLIC_XML_FETCH; no KG changes", flush=True)


if __name__ == "__main__": main()
