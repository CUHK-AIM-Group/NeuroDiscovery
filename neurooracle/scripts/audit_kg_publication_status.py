"""R34: add publication review proofs, keeping the identity registry invariant."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require
from fetch_kg_complete_titles import own_articles
from fetch_kg_publication_notices import OUTPUT, PREVIOUS
from audit_kg_complete_titles import reproduce_registry
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, pubmed_record
from neurooracle.src.pubmed_title_evidence import complete_title_evidence
from neurooracle.src.kg_publication_status import owning_status_projection, make_publication_review

R31=journal.OUTPUT/"round31_paper_identity"
FILES=[Path(__file__),Path(__file__).with_name("fetch_kg_publication_notices.py"),
    *[journal.REPO/p for p in ("neurooracle/src/kg_publication_status.py","neurooracle/src/kg_paper_identity.py",
        "neurooracle/src/shared_relation_catalog.py","neurooracle/src/pubmed_title_evidence.py",
        "neurooracle/scripts/audit_kg_complete_titles.py","neurooracle/scripts/fetch_kg_complete_titles.py")]]


def validate_current_identity_source(source, authority_fetch, xml_fetch):
    base=deepcopy(source)
    require("publication_reviews" not in base,"source publication review already present; don't reapply")
    for row in base["records"].values():row.pop("complete_title_evidence",None)
    records={}
    for w in authority_fetch["witnesses"]:
        require(journal.fingerprint(Path(w["response"]["path"]))==w["response"],"ESummary evidence changed")
        data=journal.read_json(w["response"]["path"])["result"]
        require(set(data["uids"])<=set(w["requested_pmids"]),"unrequested ESummary ID")
        for pmid in data["uids"]:
            if "error" in data[pmid]:continue
            require(pmid not in records,"duplicate authority ID")
            records[pmid]=pubmed_record(pmid,data[pmid],dict(response_sha256=w["response"]["sha256"],response_file=Path(w["response"]["path"]).name,url=w["url"]))
    require(base["records"]==records,"base authority records changed")
    expected,_=reproduce_registry(base,xml_fetch)
    require(expected==source,"current identity not reproduced from original ESummary and complete XML")


def reviewed_payload(source, fetched):
    wanted=set(fetched["root_pmids"])|set(fetched["linked_pmids"]); articles={}; witnesses={}
    for w in fetched["witnesses"]:
        require(journal.fingerprint(Path(w["response"]["path"]))==w["response"],"publication XML changed")
        found=own_articles(Path(w["response"]["path"]).read_bytes())
        require(set(found)<=set(w["requested_pmids"]),"unrequested publication XML")
        for pmid,article in found.items():
            if pmid not in wanted:continue
            require(pmid not in articles,"duplicate selected publication witness")
            articles[pmid]=article
            witnesses[pmid]=dict(response_sha256=w["response"]["sha256"],response_file=Path(w["response"]["path"]).name,url=w["url"])
    require(set(articles)==wanted,"missing publication witness")
    projections={p:owning_status_projection(a,witnesses[p]) for p,a in articles.items()}
    reviews={}
    for pmid in fetched["root_pmids"]:
        require(pmid in source["records"],"new identity record not allowed")
        # Validate owning PMID, all own DOI/PMCID aliases and complete title.
        # This is a guard only: no new title variants or aliases are installed.
        complete_title_evidence(articles[pmid],source["records"][pmid],witnesses[pmid])
        peers=[projections[p] for p in sorted({r["pmid"] for r in projections[pmid]["links"] if r["pmid"]},key=int)]
        reviews[pmid]=make_publication_review(projections[pmid],peers)
    payload=deepcopy(source);payload["publication_reviews"]=reviews
    require({k:v for k,v in payload.items() if k!="publication_reviews"}==source,"bibliographic identity mutated")
    VerifiedPaperIdentities(payload)
    return payload


def main():
    require(not (OUTPUT/"PAPER_IDENTITY_AUDIT.json").exists(),"audit exists; inspect don't overwrite")
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json");require(c["active_process"] is None and c["status"]=="COMPLETED","writer active")
    journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"]])
    source_fp=c["current_paper_identities"]
    for fp in (source_fp,c["current_acceptance"],c["current_paper_issues"],c["current_shared_relations"]):
        require(journal.fingerprint(Path(fp["path"]))==fp,"current source advanced")
    fetched=journal.read_json(OUTPUT/"NOTICE_FETCH.json")
    require(journal.fingerprint(Path(fetched["request"]["path"]))==fetched["request"],"notice plan changed")
    request=journal.read_json(fetched["request"]["path"])
    require(request["graph"]==c["current_graph"],"notice scope not current")
    require(journal.fingerprint(Path(request["warning"]["path"]))==request["warning"],"warning baseline changed")
    source=journal.read_json(source_fp["path"]);authority_fp=journal.fingerprint(R31/"AUTHORITY_FETCH.json");xml_fp=journal.fingerprint(PREVIOUS/"XML_FETCH.json")
    validate_current_identity_source(source,journal.read_json(authority_fp["path"]),journal.read_json(xml_fp["path"]))
    payload=reviewed_payload(source,fetched); registry=VerifiedPaperIdentities(payload)
    previous=journal.read_json(PREVIOUS/"PAPER_IDENTITY_AUDIT.json")
    changed_groups=[]; inclusive=0; default=0
    from prune_current_kg import rows
    for group in rows(c["current_shared_relations"]["path"]):
        keys={m["paper_key"] for m in group["members"] if m["paper_status"]=="verified"}
        excluded={p for p in keys if registry.publication_review(p)["status"]=="retracted"}
        inclusive+=len(keys)>=2;default+=len(keys-excluded)>=2
        if excluded:changed_groups.append(dict(relation_id=group["id"],verified_source_count=len(keys),counted_source_count=len(keys-excluded),excluded_retracted_paper_keys=sorted(excluded)))
    require(inclusive==515 and default==513 and len(changed_groups)==3,"unreviewed exposure change")
    journal.atomic_json(OUTPUT/"CURRENT_PAPER_IDENTITIES.json",payload)
    audit=dict(status="AUDITED_NOT_ADOPTED",at=journal.utc_now(),graph=c["current_graph"],source_registry=source_fp,source_acceptance=c["current_acceptance"],
        registry=journal.fingerprint(OUTPUT/"CURRENT_PAPER_IDENTITIES.json"),paper_issues=c["current_paper_issues"],
        claim_status=previous["claim_status"],verified_key_changes=previous["verified_key_changes"],shared_group_counts=previous["shared_group_counts"],
        changed_shared_groups=[],identity_fields_completely_unchanged=True,claim_status_requires_full_source_recount_before_adoption=True,
        publication_status_counts=dict(Counter(v["status"] for v in payload["publication_reviews"].values())),
        affected_groups=changed_groups,inclusive_multi_source_groups=inclusive,default_multi_source_groups=default,
        authority_fetch=authority_fp,xml_fetch=xml_fp,notice_fetch=journal.fingerprint(OUTPUT/"NOTICE_FETCH.json"),
        census=journal.fingerprint(R31/"CENSUS.json"),census_normalization=journal.fingerprint(R31/"CENSUS_NORMALIZATION.json"),
        code=[journal.fingerprint(p) for p in FILES],graph_modified=False,original_claims_preserved=True,
        node_or_edge_metadata_fields_added=0,record_preimages_saved=False,
        policy="Separate identity and publication state. Only authority-confirmed retracted canonical source keys are excluded from default query thresholds; corrected republications stay distinct and included. Explicit include_retracted=True retains all-source views.")
    journal.atomic_json(OUTPUT/"PAPER_IDENTITY_AUDIT.json",audit)
    print(audit["publication_status_counts"],"default source threshold 2:",default,"all-source:",inclusive,flush=True)


if __name__=="__main__":main()
