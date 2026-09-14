"""Review complete current endpoint/reference closure; persist hashes, not preimages."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import (GENES, VERSION, endpoint_gate,
    literal_node, reuse_gate, reviewed_claim, reviewed_edges, edge_owner)
from neurooracle.src.relation_evidence import name_key, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = journal.OUTPUT / "round37_relation_scope"


def progress(phase, **values):
    state = dict(status="READ_ONLY_PLANNING",phase=phase,at=journal.utc_now(),**values)
    journal.atomic_json(OUTPUT / "PLAN_STATE.json",state); print(compact(state),flush=True)


def expected_shared(db, changes, prior_catalog):
    """Project only exact relation IDs; never change source identity or evidence."""
    members = {m["claim_id"] for g in prior_catalog for m in g["members"]}
    old_rids = {e["old_relation_id"] for e in changes.values()}
    affected = old_rids | {e["new_relation_id"] for e in changes.values()}
    groups = {}
    for rid in sorted(affected):
        groups[rid] = {r[0] for r in db.execute("SELECT cid FROM claims WHERE relation_id=?",(rid,))}
        members.difference_update(groups[rid])
    for cid,event in changes.items():
        require(cid in groups[event["old_relation_id"]],"old census membership differs")
        groups[event["old_relation_id"]].remove(cid)
        groups[event["new_relation_id"]].add(cid)
    for group in groups.values():
        if len(group)>1: members.update(group)
    return sorted(members)


def main(replace_unapplied=False):
    require(not (OUTPUT / "PLAN.json").exists() or replace_unapplied,"plan exists; reuse/inspect")
    require(not (OUTPUT / "BUILD_STATE.json").exists() and not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(),"cannot revise built/applied plan")
    baseline=journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(baseline["status"]=="COMPLETED" and baseline["active_process"] is None,"active writer")
    inspection=journal.read_json(OUTPUT / "SOURCE_INSPECTION.json")
    require(inspection["graph"]==baseline["current_graph"],"inspection advanced")
    for fp in inspection["artifacts"].values(): require(journal.fingerprint(fp["path"])==fp,"inspection evidence changed")
    census=journal.read_json(baseline["current_paper_census"]["path"])
    journal.guards([baseline["current_graph"],baseline["formal_sources"],baseline["current_detail_store"],census["database"]])
    code=[journal.fingerprint(p) for p in (Path(__file__),journal.REPO/"neurooracle/src/kg_literal_endpoint_repair.py")]
    endpoints=rows(OUTPUT / "CURRENT_GENE_ENDPOINT_REVIEW.jsonl")
    source_ids={r["claim_id"] for r in endpoints}; names={name_key(r["name"]) for r in endpoints}
    generated={literal_node(n)["id"] for n in names}
    summaries={r["node_id"]:r for r in rows(OUTPUT / "CURRENT_EXACT_NAME_CANDIDATES.jsonl")}
    selected, candidates, refs, alias_candidates = {}, {}, defaultdict(list), defaultdict(set)
    counts=Counter(); seen_generated=set()
    progress("CURRENT_SOURCE_COMPLETE_NAME_ALIAS_AND_REFERENCE_SCAN",claims=len(source_ids))
    with hashed_reader(Path(baseline["current_graph"]["path"])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=="node":
                if key in generated: seen_generated.add(key)
                if key in source_ids: selected[key]=row
                if key in GENES:
                    require(row["preferred_name"]==GENES[key] and digest(row)==summaries[key]["node_sha256"],"gene identity changed")
                if not key.startswith("CLM:"):
                    if name_key(row.get("preferred_name")) in names:
                        candidates[key]=row
                        require(key in summaries and digest(row)==summaries[key]["node_sha256"],"candidate/incident census differs")
                    if key not in GENES:
                        for alias in row.get("aliases") or []:
                            if name_key(alias) in names: alias_candidates[name_key(alias)].add(key)
            elif kind=="edge" and edge_owner(row) in source_ids:
                refs[edge_owner(row)].append((int(key),row))
            if counts[kind]%750000==0: progress(kind,count=counts[kind])
        require(h.hexdigest()==baseline["current_graph"]["sha256"],"current full graph SHA differs")
    require(set(selected)==source_ids and not seen_generated,"scope incomplete or literal IDs already exist")
    by_name=defaultdict(list)
    for node in candidates.values(): by_name[name_key(node["preferred_name"])].append(node)
    events, edge_events, holds, added, targets = [], [], [], {}, {}
    terms=VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"]))
    for cid,row in sorted(selected.items()):
        changes=[]; pending=[]
        for side in ("subject","object"):
            md=row["metadata"]
            if md.get(side+"_id") not in GENES: continue
            reason=endpoint_gate(md,side); name=name_key(md.get(side+"_name"))
            possibilities=by_name.get(name,[])
            if reason is None:
                if len(possibilities)>1: reason="multiple_complete_name_candidates"
                elif possibilities:
                    target=possibilities[0]
                    reason=reuse_gate(target,name,summaries[target["id"]]["all_incident_claim_endpoints"])
                elif alias_candidates[name]: reason="existing_full_alias_requires_identity_proof"
                else: target=literal_node(name)
            if reason:
                holds.append(dict(claim_id=cid,claim_sha256=digest(row),side=side,name=name,current_node_id=md[side+"_id"],
                    reason=reason,candidate_ids=sorted(n["id"] for n in possibilities),alias_candidate_ids=sorted(alias_candidates[name])))
            else:
                changes.append(dict(side=side,old_id=md[side+"_id"],target_id=target["id"],name=name,
                    disposition="reuse_existing_exact_literal" if possibilities else "reuse_or_create_one_complete_literal"))
                pending.append(target)
        if not changes: continue
        event,current=reviewed_claim(row,changes)
        try: closure=reviewed_edges(cid,row,current,refs[cid])
        except ValueError as exc:
            holds.extend(dict(claim_id=cid,claim_sha256=digest(row),side=ch["side"],name=ch["name"],current_node_id=ch["old_id"],
                reason="reference_closure: "+str(exc),candidate_ids=[ch["target_id"]]) for ch in changes)
            continue
        event.update(old_relation_id=relation_id(terms.relation_key(row["metadata"])),
            new_relation_id=relation_id(terms.relation_key(current["metadata"])))
        events.append(event); edge_events.extend(closure)
        for target in pending:
            if target["id"] in candidates: targets[target["id"]]=digest(target)
            else: added[target["id"]]=dict(id=target["id"],name=target["preferred_name"],node_sha256=digest(target))
    require(sha256(Path(census["database"]["path"]))==census["database"]["sha256"],"current census SHA differs")
    db=sqlite3.connect(Path(census["database"]["path"]).as_uri()+"?mode=ro",uri=True)
    for ev in events:
        require(db.execute("SELECT node_sha,relation_id FROM claims WHERE cid=?",(ev["claim_id"],)).fetchone()==
            (ev["claim_sha256"],ev["old_relation_id"]),"source census claim differs")
    shared=expected_shared(db,{e["claim_id"]:e for e in events},rows(baseline["current_shared_relations"]["path"]))
    db.close()
    plan=dict(version=VERSION,status="REVIEWED_NOT_APPLIED",at=journal.utc_now(),graph=baseline["current_graph"],
        detail_store=baseline["current_detail_store"],source_acceptance=baseline["current_acceptance"],
        source_inspection=journal.fingerprint(OUTPUT / "SOURCE_INSPECTION.json"),source_full_sha_verified=True,
        source_census_full_sha_verified=True,source_census=census["database"],code=code,
        events=events,edge_events=sorted(edge_events,key=lambda x:x["ordinal"]),new_literals=sorted(added.values(),key=lambda x:x["id"]),
        existing_targets=targets,expected_shared_claim_ids=shared,
        changed_claims=len(events),changed_endpoints=sum(len(e["changes"]) for e in events),changed_edges=len(edge_events),
        added_literal_nodes=len(added),reused_existing_nodes=len(targets),held_endpoints=len(holds),
        held_reasons=dict(Counter(r["reason"] for r in holds)),
        boundaries=["complete original case-sensitive measurement surface; no shortening or fuzzy ontology mapping",
            "one literal per full name and explicit imaging role, not per claim/article",
            "unchanged original raw text, predicate, negation, roles, conditions, paper and independent source identity",
            "no deleted graph records, added metadata fields, formal graph, model or training writes"],
        graph_backups=0,record_preimages_saved=False)
    require(plan["changed_endpoints"]+len(holds)==len(endpoints),"endpoint decision closure differs")
    require(journal.read_json(journal.OUTPUT / "CAMPAIGN.json")==baseline,"campaign advanced")
    require([journal.fingerprint(fp["path"]) for fp in code]==code,"plan code changed")
    journal.guards([baseline["current_graph"],baseline["formal_sources"],baseline["current_detail_store"],census["database"]])
    write_rows(OUTPUT / "REMAINING_GENE_ENDPOINTS.jsonl",holds)
    plan["remaining_queue"]=journal.fingerprint(OUTPUT / "REMAINING_GENE_ENDPOINTS.jsonl")
    journal.atomic_json(OUTPUT / "PLAN.json",plan)
    progress("REVIEWED_NOT_APPLIED",**{k:plan[k] for k in ("changed_claims","changed_endpoints","changed_edges","added_literal_nodes","reused_existing_nodes","held_endpoints","held_reasons")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-unapplied",action="store_true")
    main(parser.parse_args().replace_unapplied)
