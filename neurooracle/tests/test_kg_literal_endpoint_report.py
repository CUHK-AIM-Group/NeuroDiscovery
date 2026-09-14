from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from report_kg_literal_endpoint_repair import comparison


def group(*ids): return {"members":[{"claim_id":cid} for cid in ids]}


def test_relabeling_relation_is_not_new_shared_evidence():
    result=comparison([group("a","b")],[group("b","a")])
    assert result["unchanged_member_sets"]==1
    assert result["newly_shared_claim_ids"]==[]
    assert result["no_longer_shared_claim_ids"]==[]


def test_unsharing_is_not_record_deletion():
    result=comparison([],[group("a","b")])
    assert result["no_longer_shared_claim_ids"]==["a","b"]
    assert result["changed_or_removed_member_sets"]==1


def test_rejoining_groups_does_not_invent_new_member_sources():
    result=comparison([group("a","b","c","d")],[group("a","b"),group("c","d")])
    assert result["changed_or_added_member_sets"]==1
    assert result["changed_or_removed_member_sets"]==2
    assert result["newly_shared_claim_ids"]==[]


def test_added_member_is_reported_exactly():
    result=comparison([group("a","b","c")],[group("a","b")])
    assert result["newly_shared_claim_ids"]==["c"]
    assert result["no_longer_shared_claim_ids"]==[]
