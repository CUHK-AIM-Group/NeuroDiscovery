from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from publish_kg_literal_query_contract import flags_from_completed_proof,CAPABILITIES


def fixture():
    checks=dict(current_census_all_claims_papers_and_shared_members_verified=True,
        inverse_reproduces_all_source_node_and_edge_record_digests=True,shared_relation_index_complete=True)
    r=dict(status="CURRENT_LITERAL_ENDPOINT_REPAIR_APPLIED",graph={"sha256":"proof"},checks=checks,code=["frozen"],counts={"claims":3})
    v=dict(status="VALIDATED_NOT_ADOPTED",graph={"sha256":"proof"},checks=deepcopy(checks))
    s=dict(code=["frozen"],result={"counts":{"claims":3}})
    return r,v,s


def test_complete_verified_boundary_publishes_only_required_flags():
    r,v,s=fixture(); before=deepcopy(r)
    assert flags_from_completed_proof(r,v,s)==dict.fromkeys(CAPABILITIES,True)
    assert r==before


@pytest.mark.parametrize("change",["graph","code","incomplete"])
def test_no_capability_claim_from_changed_or_incomplete_evidence(change):
    r,v,s=fixture()
    if change=="graph": r["graph"]["sha256"]="changed"
    if change=="code": r["code"]=[]
    if change=="incomplete":
        r["checks"]["shared_relation_index_complete"]=False
        v["checks"]["shared_relation_index_complete"]=False
    with pytest.raises(ValueError): flags_from_completed_proof(r,v,s)
