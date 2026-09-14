from copy import deepcopy
import pytest
from neurooracle.scripts.reconcile_kg_current_control import reconcile


def fixture():
    c = dict(status="COMPLETED", active_process=None, current_graph={"sha256":"a"}, counts={"nodes":2},
             current_coverage={"sha256":"b"}, relation_evidence_counts={"groups":1}, node_metadata_field_union=3,
             edge_metadata_field_union=1, retention_result={"path":"old"}, protected_science={"raw":"unchanged"})
    r = dict(graph=c["current_graph"], counts=c["counts"], metadata_coverage=c["current_coverage"],
             relation_counts=c["relation_evidence_counts"], metadata_key_unions={"nodes":["id","source"],"edges":["owner"]})
    return c, r


def test_stale_summary_reconciles_without_graph_changes():
    c,r = fixture(); before = deepcopy(c); out = reconcile(c,r,{"path":"current"})
    assert c == before and out["current_graph"] == before["current_graph"]
    assert out["node_metadata_field_union"] == 2 and out["retention_result"]["path"] == "current"
    assert out["protected_science"] == c["protected_science"]


@pytest.mark.parametrize("change", ["active","graph","counts","coverage","relations"])
def test_untrusted_or_active_control_cannot_reconcile(change):
    c,r = fixture(); r = deepcopy(r)
    if change == "active": c["active_process"] = {"pid":1}
    else: r[{"graph":"graph","counts":"counts","coverage":"metadata_coverage","relations":"relation_counts"}[change]] = {"bad":1}
    with pytest.raises((ValueError, RuntimeError, AssertionError)): reconcile(c,r,{})
