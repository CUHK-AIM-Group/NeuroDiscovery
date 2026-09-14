"""Publish existing deep-verification evidence to the query-reader contract.

No graph writes or new scientific acceptance. The R43 validator executes the
entity, paper and publication proofs but does not emit the three capability
flags required by the existing fail-closed shared-relation reader.
"""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_explicit_pmid_provenance import authorities
from kg_accepted_candidate_lineage import require
from neurooracle.src.shared_relation_catalog import current_shared_relations, find_shared_relations

OUTPUT=journal.OUTPUT/"round43_gene_boundary"
CAPABILITIES=("verified_identity_proofs_complete","paper_identity_witnesses_validated","publication_status_witnesses_validated")


def flags_from_completed_proof(receipt,validated,state):
    require(receipt["status"]=="CURRENT_GENE_BOUNDARY_REPAIR_APPLIED","wrong batch")
    require(validated["status"]=="VALIDATED_NOT_ADOPTED","missing independent validation")
    require(receipt["graph"]["sha256"]==validated["graph"]["sha256"] and receipt["checks"]==validated["checks"],"validation/receipt differs")
    require(receipt["code"]==state["code"] and receipt["counts"]==state["result"]["counts"],"frozen build differs")
    require(receipt["checks"].get("current_census_all_claims_papers_and_shared_members_verified") and
        receipt["checks"].get("inverse_reproduces_all_source_node_and_edge_record_digests") and
        receipt["checks"].get("shared_relation_index_complete"),"required full proof missing")
    return dict.fromkeys(CAPABILITIES,True)


def main():
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(c["status"]=="COMPLETED" and c["active_process"] is None,"writer still active")
    receipt_path=OUTPUT/"CURRENT_ACCEPTANCE.json"
    require(journal.fingerprint(receipt_path)==c["current_acceptance"],"R43 not current")
    require(not (OUTPUT/"QUERY_CONTRACT_VALIDATION.json").exists(),"query contract already published; inspect/reuse")
    receipt=journal.read_json(receipt_path); validated=journal.read_json(OUTPUT/"VALIDATED.json"); state=journal.read_json(OUTPUT/"BUILD_STATE.json")
    require(journal.fingerprint(OUTPUT/"BUILD_STATE.json")==validated["build_state"],"build receipt changed")
    for fp in receipt["code"]: require(journal.fingerprint(fp["path"])==fp,"validated code changed")
    journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"]])
    flags=flags_from_completed_proof(receipt,validated,state)
    # SourceProof.verify and authority/publication reproduction are explicit in
    # this frozen validator. Recheck the small registry's owning public evidence;
    # reuse the just-completed full candidate record and native graph boundary.
    papers,_=authorities(c)
    audit=journal.read_json(receipt["identity_audit"]["path"])
    require(audit["claim_status"]["verified"]==receipt["checks"]["authority_verified_claims"],"full identity status differs")
    census=journal.read_json(c["current_paper_census"]["path"])
    require(census["graph"]==c["current_graph"] and census["independent_full_candidate_verified"],"current census binding differs")
    extra=[journal.fingerprint(Path(__file__)),journal.fingerprint(journal.REPO/"neurooracle/tests/test_kg_gene_boundary_query_contract.py")]
    query_tests=OUTPUT/"QUERY_CONTRACT_TEST_RESULTS.xml"
    from xml.etree import ElementTree as ET
    suite=ET.parse(query_tests).getroot().find("testsuite")
    require(int(suite.get("tests"))>=4 and all(int(suite.get(k))==0 for k in ("failures","errors","skipped")),"query contract tests not complete")
    new_receipt=deepcopy(receipt); new_receipt["checks"].update(flags)
    # Ephemeral small query fixtures, not historical KG/data copies.
    receipt_tmp=OUTPUT/"QUERY_RECEIPT.tmp.json"; control_tmp=OUTPUT/"QUERY_CONTROL.tmp.json"
    require(not receipt_tmp.exists() and not control_tmp.exists(),"query fixture exists; inspect")
    try:
        journal.atomic_json(receipt_tmp,new_receipt)
        probe=deepcopy(c); probe["current_acceptance"]=journal.fingerprint(receipt_tmp)
        journal.atomic_json(control_tmp,probe)
        groups=list(current_shared_relations(control_tmp))
        default=list(find_shared_relations(control_tmp,minimum_papers=2))
        inclusive=list(find_shared_relations(control_tmp,minimum_papers=2,include_retracted=True))
        require(len(groups)==c["relation_evidence_counts"]["shared_groups"] and
            len(inclusive)==c["relation_evidence_counts"]["verified_multi_paper_shared_groups"],"actual query differs")
        result=dict(at=journal.utc_now(),graph=c["current_graph"],source_receipt=c["current_acceptance"],
            independent_validation=journal.fingerprint(OUTPUT/"VALIDATED.json"),published_capability_flags=flags,
            raw_groups=len(groups),default_minimum_two_source_groups=len(default),inclusive_minimum_two_source_groups=len(inclusive),
            graph_modified=False,source_proof_reused_not_recomputed=True,authority_and_publication_witnesses_reproduced=True,
            code=extra,tests=journal.fingerprint(query_tests),test_count=int(suite.get("tests")))
        journal.atomic_json(OUTPUT/"QUERY_CONTRACT_VALIDATION.json",result)
        new_receipt["query_contract_validation"]=journal.fingerprint(OUTPUT/"QUERY_CONTRACT_VALIDATION.json")
        new_receipt["code"]+=extra
        require(journal.read_json(journal.OUTPUT/"CAMPAIGN.json")==c,"campaign advanced")
        journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"]])
        journal.atomic_json(receipt_path,new_receipt)
        runtime=journal.read_json(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json")
        runtime.update(acceptance=journal.fingerprint(receipt_path),code=new_receipt["code"],query_contract_validation=new_receipt["query_contract_validation"])
        journal.atomic_json(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json",runtime)
        c.update(current_acceptance=journal.fingerprint(receipt_path),current_runtime_acceptance=journal.fingerprint(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json"))
        journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",c)
        night=journal.read_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json")
        night.update(latest_acceptance=c["current_acceptance"],latest_runtime_acceptance=c["current_runtime_acceptance"])
        journal.atomic_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json",night)
    finally:
        for p in (receipt_tmp,control_tmp):
            require(p.resolve().parent==OUTPUT.resolve() and p.name in {"QUERY_RECEIPT.tmp.json","QUERY_CONTROL.tmp.json"},"unsafe small fixture cleanup")
            if p.exists(): p.unlink()
    require(len(list(current_shared_relations(journal.OUTPUT/"CAMPAIGN.json")))==len(groups),"published query differs")
    print("QUERY_CONTRACT_PUBLISHED",{k:result[k] for k in ("raw_groups","default_minimum_two_source_groups","inclusive_minimum_two_source_groups")},flush=True)


if __name__=="__main__": main()
