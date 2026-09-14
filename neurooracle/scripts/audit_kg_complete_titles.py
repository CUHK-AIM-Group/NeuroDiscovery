"""R33 exact complete-title witness audit. No KG writes or claim preimages."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact
from kg_accepted_candidate_lineage import require
from fetch_kg_complete_titles import OUTPUT, PREVIOUS, own_articles
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, identifiers
from neurooracle.src.pubmed_title_evidence import complete_title_evidence, presentation_key

FILES = [Path(__file__), Path(__file__).with_name("fetch_kg_complete_titles.py"),
    journal.REPO / "neurooracle/src/kg_paper_identity.py", journal.REPO / "neurooracle/src/pubmed_title_evidence.py"]


def reproduce_registry(source, fetched):
    payload = deepcopy(source); accepted = []; held = []; seen = set()
    for w in fetched["witnesses"]:
        require(journal.fingerprint(Path(w["response"]["path"])) == w["response"], "XML evidence changed")
        articles = own_articles(Path(w["response"]["path"]).read_bytes())
        require(set(articles) <= set(w["requested_pmids"]), "unrequested owning XML PMID")
        for pmid, article in articles.items():
            require(pmid not in seen and pmid in payload["records"], "duplicate or unbound XML PMID")
            seen.add(pmid); record = payload["records"][pmid]
            try:
                evidence = complete_title_evidence(article, record, dict(response_sha256=w["response"]["sha256"],
                    response_file=Path(w["response"]["path"]).name, url=w["url"]))
            except ValueError as error:
                held.append(dict(pmid=pmid, reason=str(error))); continue
            record["complete_title_evidence"] = evidence
            accepted.append(pmid)
    VerifiedPaperIdentities(payload)
    return payload, dict(accepted_pmids=sorted(accepted, key=int), held=held,
        document_kinds=dict(Counter(payload["records"][p]["complete_title_evidence"]["document_kind"] for p in accepted)))


def main():
    require(not (OUTPUT / "PAPER_IDENTITY_AUDIT.json").exists(), "audit already exists; inspect rather than overwrite")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    request = journal.read_json(OUTPUT / "XML_REQUEST.json"); fetched = journal.read_json(OUTPUT / "XML_FETCH.json")
    require(campaign["active_process"] is None and request["graph"] == campaign["current_graph"], "source changed/writer active")
    require(journal.fingerprint(OUTPUT / "XML_REQUEST.json") == fetched["request"], "request changed")
    for field in ("current_acceptance", "current_paper_identities", "current_paper_issues", "current_shared_relations"):
        require(journal.fingerprint(Path(campaign[field]["path"])) == campaign[field], "current artifact changed")
    census = journal.read_json(PREVIOUS / "CENSUS.json")
    journal.guards([census["database"], campaign["current_graph"], campaign["formal_sources"]])
    code = [journal.fingerprint(p) for p in FILES]
    source_payload = journal.read_json(campaign["current_paper_identities"]["path"])
    payload, xml_summary = reproduce_registry(source_payload, fetched)
    registry = VerifiedPaperIdentities(payload); old_registry = VerifiedPaperIdentities(source_payload)
    db = sqlite3.connect(Path(census["database"]["path"]).as_uri()+"?mode=ro", uri=True)
    outer = defaultdict(list)
    for row in census["outer_pmid_conflicts"]: outer[row["claim_id"]].append(row["alternate_pmid"])
    by_paper = {}; old_by_paper = {}; kinds = {}; bibliography_status = Counter()
    for sig, encoded in db.execute("SELECT sig,payload FROM papers"):
        paper = json.loads(encoded); c = dict(id="CLM:placeholder", source_paper=paper)
        answer = registry.resolve(c); by_paper[sig] = answer; old_by_paper[sig] = old_registry.resolve(c)
        bibliography_status[answer["status"]] += 1
        p = answer["paper_key"].removeprefix("pmid:") if answer["status"] == "verified" else ""
        kinds[sig] = payload["records"].get(p, {}).get("complete_title_evidence", {}).get("document_kind", "not_classified_in_this_batch")
    statuses = Counter(); old_statuses = Counter(); reasons = Counter(); transitions = Counter(); issues = []; key_changes = []
    newly_verified = []; shared_answers = {}; source_kinds = Counter(); held_parser_patterns = []
    for cid, node_sha, sig, legacy, shared in db.execute("SELECT cid,node_sha,paper_sig,legacy_key,shared FROM claims"):
        answers = []
        for r, cache in ((old_registry, old_by_paper), (registry, by_paper)):
            answer = dict(cache[sig])
            if cid in outer:
                p = json.loads(db.execute("SELECT payload FROM papers WHERE sig=?", (sig,)).fetchone()[0])
                observations = [r.resolve(dict(id=cid, source_paper=p, pmid=other)) for other in outer[cid]]
                if any(a["status"] == "conflict" for a in observations):
                    answer = dict(status="conflict", reasons=sorted({why for a in observations for why in a["reasons"]}), paper_key="unresolved:"+cid)
            if answer["status"] == "conflict": answer["paper_key"] = "unresolved:" + cid
            if answer["status"] == "unverified": answer["paper_key"] = legacy
            answers.append(answer)
        previous, answer = answers
        old_statuses[previous["status"]] += 1; statuses[answer["status"]] += 1
        reasons.update(answer["reasons"]); transitions[previous["status"]+"->"+answer["status"]] += 1
        require(previous["status"] != "verified" or answer == previous, "previously verified identity regressed")
        if answer["status"] == "verified":
            source_kinds[kinds[sig]] += 1
            if answer["paper_key"] != legacy:
                key_changes.append(dict(claim_id=cid, claim_sha256=node_sha, legacy_source_key=legacy, canonical_paper_key=answer["paper_key"]))
            if previous["status"] != "verified":
                require(previous["reasons"] == ["complete_title_disagreement"], "only complete-title false holds may be cleared")
                newly_verified.append(dict(claim_id=cid, claim_sha256=node_sha, paper_key=answer["paper_key"],
                    document_kind=kinds[sig], reason="exact_complete_authority_XML_rendering_with_all_identifier_year_version_guards"))
        if answer["status"] == "conflict":
            issues.append(dict(claim_id=cid, claim_sha256=node_sha, reasons=answer["reasons"], legacy_source_key=legacy, action="hold_no_claim_or_source_deletion"))
            if "complete_title_disagreement" in answer["reasons"]:
                p = json.loads(db.execute("SELECT payload FROM papers WHERE sig=?", (sig,)).fetchone()[0])
                pmid = identifiers(p)["pmid"]; evidence = payload["records"].get(pmid, {}).get("complete_title_evidence")
                if evidence:
                    from xml.etree import ElementTree as ET
                    first = ET.fromstring(evidence["title_xml"]).text
                    if first and presentation_key(p.get("title")) == presentation_key(first):
                        held_parser_patterns.append(dict(claim_id=cid, claim_sha256=node_sha, pmid=pmid,
                            reason="stored_title_matches_XML_first_text_only_not_accepted_as_complete_identity"))
        if shared: shared_answers[cid] = answer
    previous_audit = journal.read_json(PREVIOUS / "PAPER_IDENTITY_AUDIT.json")
    require(dict(old_statuses) == previous_audit["claim_status"], "baseline paper census no longer reproduces")
    groups = Counter(); group_changes = []
    for line in Path(campaign["current_shared_relations"]["path"]).read_text(encoding="utf8").splitlines():
        g = json.loads(line); answers = [shared_answers[m["claim_id"]] for m in g["members"]]
        verified = {a["paper_key"] for a in answers if a["status"] == "verified"}; unverified = sum(a["status"] != "verified" for a in answers)
        groups["shared_groups"] += 1; groups["verified_multi_paper_groups"] += len(verified)>1
        groups["fully_verified_groups"] += unverified == 0; groups["groups_with_unverified_members"] += unverified>0
        if any(a["paper_key"] != m["paper_key"] or a["status"] != m["paper_status"] for a,m in zip(answers,g["members"])):
            group_changes.append(dict(relation_id=g["id"], old_source_key_count=g["paper_count"], verified_paper_count=len(verified), unverified_claim_count=unverified))
    db.close(); journal.guards([census["database"], campaign["current_graph"], campaign["formal_sources"]])
    require(code == [journal.fingerprint(p) for p in FILES], "audit code changed")
    journal.atomic_json(OUTPUT / "CURRENT_PAPER_IDENTITIES.json", payload)
    for name, values in (("CURRENT_PAPER_ISSUES.jsonl",issues), ("VERIFIED_PAPER_KEY_CHANGES.jsonl",key_changes),
        ("CLEARED_TITLE_HOLDS.jsonl",newly_verified), ("PARSER_TRUNCATION_HOLDS.jsonl",held_parser_patterns)):
        journal.atomic_text(OUTPUT/name, "".join(compact(r)+"\n" for r in values))
    audit = dict(status="AUDITED_NOT_ADOPTED", at=journal.utc_now(), graph=campaign["current_graph"],
        authority_records=len(payload["records"]), claim_status=dict(statuses), bibliography_status=dict(bibliography_status),
        hold_reasons=dict(reasons), verified_key_changes=len(key_changes), shared_group_counts=dict(groups), changed_shared_groups=group_changes,
        transitions=dict(transitions), newly_verified=len(newly_verified), newly_verified_document_kinds=dict(Counter(r["document_kind"] for r in newly_verified)),
        verified_claim_document_kind=dict(source_kinds), xml_summary=xml_summary, parser_truncation_holds=len(held_parser_patterns),
        source_registry=campaign["current_paper_identities"], source_acceptance=campaign["current_acceptance"],
        registry=journal.fingerprint(OUTPUT/"CURRENT_PAPER_IDENTITIES.json"), paper_issues=journal.fingerprint(OUTPUT/"CURRENT_PAPER_ISSUES.jsonl"),
        key_changes=journal.fingerprint(OUTPUT/"VERIFIED_PAPER_KEY_CHANGES.jsonl"), cleared=journal.fingerprint(OUTPUT/"CLEARED_TITLE_HOLDS.jsonl"),
        parser_holds=journal.fingerprint(OUTPUT/"PARSER_TRUNCATION_HOLDS.jsonl"),
        xml_fetch=journal.fingerprint(OUTPUT/"XML_FETCH.json"), census=journal.fingerprint(PREVIOUS/"CENSUS.json"),
        authority_fetch=journal.fingerprint(PREVIOUS/"AUTHORITY_FETCH.json"),
        census_normalization=journal.fingerprint(PREVIOUS/"CENSUS_NORMALIZATION.json"), code=code,
        graph_modified=False, original_claims_preserved=True, record_preimages_saved=False,
        interpretation="Verified source records include Bookshelf documents; not a count of independent primary studies.")
    journal.atomic_json(OUTPUT/"PAPER_IDENTITY_AUDIT.json", audit)
    print(compact({k:v for k,v in audit.items() if k in {"status","claim_status","newly_verified","newly_verified_document_kinds","shared_group_counts","parser_truncation_holds","hold_reasons"}}), flush=True)


if __name__ == "__main__": main()
