"""Reconcile small current-summary fields from an unchanged trusted acceptance."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require

OUTPUT = journal.OUTPUT / "round37_relation_scope"


def reconcile(campaign, receipt, retention):
    require(campaign["status"] == "COMPLETED" and campaign["active_process"] is None, "writer active")
    require(receipt["graph"] == campaign["current_graph"] and receipt["counts"] == campaign["counts"], "receipt not current")
    require(receipt["metadata_coverage"] == campaign["current_coverage"], "coverage binding differs")
    require(receipt["relation_counts"] == campaign["relation_evidence_counts"], "relation counts require separate investigation")
    out = deepcopy(campaign)
    out["node_metadata_field_union"] = len(receipt["metadata_key_unions"]["nodes"])
    out["edge_metadata_field_union"] = len(receipt["metadata_key_unions"]["edges"])
    out["retention_result"] = retention
    allowed = {"node_metadata_field_union", "edge_metadata_field_union", "retention_result"}
    require({k:v for k,v in out.items() if k not in allowed} == {k:v for k,v in campaign.items() if k not in allowed}, "expanded reconciliation")
    return out


def main():
    OUTPUT.mkdir(exist_ok=True)
    c = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    for key in ("current_acceptance", "current_runtime_acceptance", "current_coverage"):
        require(journal.fingerprint(Path(c[key]["path"])) == c[key], "current small evidence changed")
    receipt = journal.read_json(c["current_acceptance"]["path"])
    for fp in receipt["code"]:
        require(journal.fingerprint(Path(fp["path"])) == fp, "frozen code changed")
    validation = journal.read_json(Path(c["current_acceptance"]["path"]).parent / "REPORT_VALIDATION.json")
    require(validation["acceptance"] == c["current_acceptance"], "report binding differs")
    retention = validation["retention"]
    require(journal.fingerprint(Path(retention["path"])) == retention, "retention receipt differs")
    journal.guards([c["current_graph"], c["current_detail_store"], c["formal_sources"]])
    out = reconcile(c, receipt, retention)
    require(journal.read_json(journal.OUTPUT / "CAMPAIGN.json") == c, "campaign advanced")
    changes = {k:dict(previous=c.get(k), current=v) for k,v in out.items() if c.get(k) != v}
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", out)
    journal.atomic_json(OUTPUT / "CONTROL_RECONCILIATION.json", dict(status="COMPLETED", at=journal.utc_now(), changes=changes,
        graph=c["current_graph"], acceptance=c["current_acceptance"], last_deep_verification=c["last_deep_verification"],
        graph_or_runtime_modified=False, source_guards_passed=True, full_graph_rescan_required=False,
        code=journal.fingerprint(Path(__file__))))
    print("CURRENT_CONTROL_RECONCILED", sorted(changes))


if __name__ == "__main__": main()
