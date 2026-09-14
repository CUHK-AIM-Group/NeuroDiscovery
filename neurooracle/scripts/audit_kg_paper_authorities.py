"""Validate retrieved bibliography, build exact crosswalk, and audit all current papers.

Derived compact indexes only. This phase neither edits nor adopts a KG.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_paper_identity import VERSION, VerifiedPaperIdentities, pubmed_record, census_identifier_projection

OUTPUT = journal.OUTPUT / "round31_paper_identity"


def main():
    code = [journal.fingerprint(Path(__file__)), journal.fingerprint(journal.REPO / "neurooracle/src/kg_paper_identity.py")]
    census = journal.read_json(OUTPUT / "CENSUS.json")
    fetched = journal.read_json(OUTPUT / "AUTHORITY_FETCH.json")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(census["graph"] == campaign["current_graph"] and campaign["active_process"] is None, "source advanced/writer active")
    journal.guards([census["database"], census["graph"], campaign["formal_sources"]])
    require(journal.fingerprint(OUTPUT / "AUTHORITY_REQUEST.json") == fetched["request"], "retrieval plan changed")
    requested = journal.read_json(OUTPUT / "AUTHORITY_REQUEST.json")
    require(journal.fingerprint(OUTPUT / "CENSUS.json") == requested["census"], "census changed")
    records = {}; retrieval_holds = []
    for witness in fetched["witnesses"]:
        require(journal.fingerprint(Path(witness["response"]["path"])) == witness["response"], "authority cache changed")
        data = journal.read_json(witness["response"]["path"])["result"]
        require(set(data["uids"]) <= set(witness["requested_pmids"]), "unexpected authority ID")
        for pmid in data["uids"]:
            require(pmid not in records and pmid in requested["pmids"], "duplicate/unrequested authority ID")
            if "error" in data[pmid]:
                retrieval_holds.append(dict(pmid=pmid, error=data[pmid]["error"])); continue
            records[pmid] = pubmed_record(pmid, data[pmid], dict(response_sha256=witness["response"]["sha256"],
                response_file=Path(witness["response"]["path"]).name, url=witness["url"]))
    db = sqlite3.connect(Path(census["database"]["path"]).as_uri() + "?mode=ro", uri=True)
    projection = census_identifier_projection(db)
    collisions = projection["observed_collisions"]
    journal.atomic_json(OUTPUT / "CENSUS_NORMALIZATION.json", dict(**projection, census=journal.fingerprint(OUTPUT / "CENSUS.json")))
    payload = dict(version=VERSION, source="NCBI PubMed ESummary", records=records, observed_collisions=collisions,
        policy="exact authority IDs; complete-title/year compatibility; no title-only, version or preprint-to-journal merging")
    registry = VerifiedPaperIdentities(payload)
    journal.atomic_json(OUTPUT / "CURRENT_PAPER_IDENTITIES.json", payload)
    outer = {}
    for row in census["outer_pmid_conflicts"]:
        outer.setdefault(row["claim_id"], []).append(row)
    claim_status = Counter(); paper_status = Counter(); reasons = Counter(); changed = []; issues = []
    by_paper = {}; total = 0
    for sig, encoded, number, example in db.execute("SELECT p.sig,p.payload,COUNT(*),MIN(c.cid) FROM papers p JOIN claims c ON c.paper_sig=p.sig GROUP BY p.sig"):
        paper = json.loads(encoded)
        answer = registry.resolve(dict(id=example, source_paper=paper))
        paper_status[answer["status"]] += 1
        by_paper[sig] = answer
        total += number
    require(total == census["counts"]["claims"], "incomplete paper audit")
    resolved_shared = {}
    for cid, node_sha, sig, legacy, shared in db.execute("SELECT cid,node_sha,paper_sig,legacy_key,shared FROM claims"):
        answer = by_paper[sig]
        if answer["status"] == "unverified":
            answer = {**answer, "paper_key": legacy}  # claim fallback must remain claim-specific
        if cid in outer:
            paper = json.loads(db.execute("SELECT payload FROM papers WHERE sig=?", (sig,)).fetchone()[0])
            # Multiple contradictory top-level/inner PMIDs must all remain held.
            candidates = [registry.resolve(dict(id=cid, source_paper=paper, pmid=row["alternate_pmid"])) for row in outer[cid]]
            if any(row["status"] == "conflict" for row in candidates):
                answer = dict(paper_key="unresolved:" + cid, status="conflict",
                    reasons=sorted({why for row in candidates for why in row["reasons"]}))
        if answer["status"] == "conflict":
            answer = {**answer, "paper_key": "unresolved:" + cid}
            issues.append(dict(claim_id=cid, claim_sha256=node_sha, reasons=answer["reasons"],
                legacy_source_key=legacy, action="hold_no_claim_or_source_deletion"))
        if answer["status"] == "verified" and answer["paper_key"] != legacy:
            changed.append(dict(claim_id=cid, claim_sha256=node_sha, legacy_source_key=legacy, canonical_paper_key=answer["paper_key"]))
        claim_status[answer["status"]] += 1
        reasons.update(answer["reasons"])
        if shared: resolved_shared[cid] = answer
    old = [json.loads(line) for line in Path(campaign["current_shared_relations"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    groups = Counter(); group_changes = []
    for group in old:
        verified = {resolved_shared[m["claim_id"]]["paper_key"] for m in group["members"] if resolved_shared[m["claim_id"]]["status"] == "verified"}
        unverified = sum(resolved_shared[m["claim_id"]]["status"] != "verified" for m in group["members"])
        groups["shared_groups"] += 1
        groups["verified_multi_paper_groups"] += len(verified) > 1
        groups["fully_verified_groups"] += not unverified
        groups["groups_with_unverified_members"] += bool(unverified)
        if any(resolved_shared[m["claim_id"]]["paper_key"] != m["paper_key"] for m in group["members"]):
            group_changes.append(dict(relation_id=group["id"], old_source_key_count=group["paper_count"],
                verified_paper_count=len(verified), unverified_claim_count=unverified))
    db.close()
    journal.guards([census["database"], census["graph"], campaign["formal_sources"]])
    require(code == [journal.fingerprint(Path(__file__)), journal.fingerprint(journal.REPO / "neurooracle/src/kg_paper_identity.py")], "audit code changed during run")
    journal.atomic_text(OUTPUT / "CURRENT_PAPER_ISSUES.jsonl", "".join(compact(row) + "\n" for row in issues))
    journal.atomic_text(OUTPUT / "VERIFIED_PAPER_KEY_CHANGES.jsonl", "".join(compact(row) + "\n" for row in changed))
    result = dict(status="AUDITED_NOT_ADOPTED", at=journal.utc_now(), graph=census["graph"],
        authority_records=len(records), retrieval_holds=retrieval_holds, claim_status=dict(claim_status),
        bibliography_status=dict(paper_status), hold_reasons=dict(reasons), verified_key_changes=len(changed),
        shared_group_counts=dict(groups), changed_shared_groups=group_changes,
        registry=journal.fingerprint(OUTPUT / "CURRENT_PAPER_IDENTITIES.json"),
        paper_issues=journal.fingerprint(OUTPUT / "CURRENT_PAPER_ISSUES.jsonl"),
        key_changes=journal.fingerprint(OUTPUT / "VERIFIED_PAPER_KEY_CHANGES.jsonl"),
        authority_fetch=journal.fingerprint(OUTPUT / "AUTHORITY_FETCH.json"), census=journal.fingerprint(OUTPUT / "CENSUS.json"),
        census_normalization=journal.fingerprint(OUTPUT / "CENSUS_NORMALIZATION.json"),
        graph_modified=False, original_claims_preserved=True, record_preimages_saved=False,
        code=code,
        compatibility_note="Census helper relocated from paper_identity.py to kg_paper_identity.py; original tracked legacy module restored exactly to HEAD before runtime tests. Historical census bindings retained, not rewritten.")
    journal.atomic_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json", result)
    print(compact({k:v for k,v in result.items() if k in {"status","authority_records","claim_status","bibliography_status","hold_reasons","verified_key_changes","shared_group_counts","retrieval_holds"}}), flush=True)


if __name__ == "__main__": main()
