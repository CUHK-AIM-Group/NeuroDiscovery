"""Full current-KG exact-term census, planned routing and structural review."""
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_bulk_identity import change_claim,change_edge
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_structure_repairs import choose_repairs,repair_edge
from neurooracle.src.relation_evidence import relation_key,name_key,relation_id
from neurooracle.src.umls_audit_store import UmlsAuditStore
from neurooracle.src.verified_entity_terms import VERSION,load_candidates,approve_live_terms,VerifiedEntityTerms,endpoint_eligibility

OUTPUT=journal.OUTPUT/"round30_verified_terms"


def progress(phase,**details):
    state=dict(status="COLLECTING",phase=phase,at=journal.utc_now(),**details)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",state); print(compact(state),flush=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    campaign=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(campaign["status"]=="COMPLETED" and campaign["active_process"] is None,"active operation")
    journal.guards([campaign["current_graph"],campaign["current_detail_store"],campaign["formal_sources"]])
    previous=journal.read_json(journal.OUTPUT/"round29_bulk_identity/PLAN.json")
    old_events={e["claim_id"]:e for e in previous["events"]}
    blocked={json.loads(s)["claim_id"] for s in Path(campaign["current_issues"]["path"]).read_text(encoding="utf-8").splitlines() if s}
    structure_ids={json.loads(s)["claim_id"] for s in Path(campaign["current_structure_holds"]["path"]).read_text(encoding="utf-8").splitlines() if s}
    structure_targets=set()
    if (OUTPUT/"PLAN.json").exists():
        for item in journal.read_json(OUTPUT/"PLAN.json").get("structure_review",[]):
            structure_targets.update([item["subject_id"],item["object_id"]])
            for edge in item["edges"]: structure_targets.update([edge["source_id"],edge["target_id"]])
    with UmlsAuditStore(campaign["current_detail_store"]["path"]) as store:
        candidates,census=load_candidates(store.connection)
    progress("ATOMIC_TERM_CENSUS",counts=census)
    wanted={p[field] for proofs in candidates.values() for p in proofs for field in ("parent_id","atom_id","target_id")} | structure_targets
    refs={p["mapping_ref"] for proofs in candidates.values() for p in proofs}
    nodes,mappings,counts={}, {},Counter()
    db=sqlite3.connect(":memory:"); db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE claims(id TEXT PRIMARY KEY,payload TEXT,k TEXT,prior_k TEXT)")
    db.execute("CREATE TABLE edges(ord INTEGER PRIMARY KEY,owner TEXT,payload TEXT)")
    summaries,structure_edges={},defaultdict(list)
    with hashed_reader(Path(campaign["current_graph"]["path"])) as (reader,sha):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=="node":
                if key in wanted: nodes[key]=row
                if key.startswith("CLM:"):
                    md=row["metadata"]; k=relation_key(md); prior=list(k)
                    if key in old_events:
                        for c in old_events[key]["changes"]: prior[0 if c["side"]=="subject" else 3]=c["old_id"]
                    possible=any(name_key(md.get(side+"_name")) in candidates for side in ("subject","object"))
                    if possible or key in structure_ids:
                        db.execute("INSERT INTO claims VALUES (?,?,?,?)",(key,compact(row),compact(k),compact(prior)))
                    if key in structure_ids:
                        summaries[key]=dict(claim_id=key,claim_sha256=digest(row),
                            **{f:md.get(f) for f in ("subject_id","subject_name","predicate","object_id","object_name","source_paper","negated")},
                            subject_type=md.get("subject_type") or (md.get("metadata") or {}).get("subject_type"),
                            object_type=md.get("object_type") or (md.get("metadata") or {}).get("object_type"),
                            raw_text=str(md.get("raw_text") or "")[:1200])
            else:
                if kind!="edge": continue
                md=row.get("metadata") or {}; ref=md.get("audit_ref")
                if ref in refs: mappings[ref]=digest(row)
                owner=row["source_id"] if row["relation_type"]=="about" else md.get("claim_id")
                if owner and db.execute("SELECT 1 FROM claims WHERE id=?",(owner,)).fetchone():
                    db.execute("INSERT INTO edges VALUES (?,?,?)",(int(key),owner,compact(row)))
                if owner in structure_ids:
                    structure_edges[owner].append(dict(ordinal=int(key),edge_sha256=digest(row),source_id=row["source_id"],
                        target_id=row["target_id"],relation_type=row["relation_type"],claim_id=md.get("claim_id"),
                        nonendpoint_sha256=digest({k:v for k,v in row.items() if k not in {"source_id","target_id"}})))
            if counts[kind]%500000==0: progress(kind,counts=dict(counts))
        require(sha.hexdigest()==campaign["current_graph"]["sha256"],"source SHA mismatch")
    terms,term_holds=approve_live_terms(candidates,nodes,mappings)
    registry_payload=dict(version=VERSION,terms=[terms[k] for k in sorted(terms)],source_graph=campaign["current_graph"],
        scope="exact complete claim endpoint surfaces; never partial claim replacement")
    registry=VerifiedEntityTerms(registry_payload); registry.validate_nodes(nodes)
    structure_plan=choose_repairs([dict(**summaries[cid],edges=structure_edges[cid]) for cid in sorted(structure_ids)],
        {nid:{"name":row["preferred_name"]} for nid,row in nodes.items()})
    structural_edits={e["ordinal"]:e for e in structure_plan["edits"]}
    progress("LIVE_TERM_REGISTRY",approved_terms=len(terms),holds=term_holds)
    db.execute("CREATE INDEX edge_owner ON edges(owner)")
    events,holds,examples={},Counter(),defaultdict(list)
    for cid,payload,_,_ in db.execute("SELECT id,payload,k,prior_k FROM claims ORDER BY id"):
        record=json.loads(payload); md=record["metadata"]; changes=[]
        if cid in blocked: holds["current_semantic_issue"]+=1; continue
        for side in ("subject","object"):
            old=md.get(side+"_id"); term=terms.get(name_key(md.get(side+"_name")))
            if not term or not str(old).startswith("CLM_CONCEPT:"): continue
            reason=endpoint_eligibility(md,side,term)
            if reason:
                holds[reason]+=1; continue
            changes.append(dict(side=side,old_id=old,target_id=term["target_id"],term=term["name"],
                                missing_type_not_fabricated=not (md.get(side+"_type") or (md.get("metadata") or {}).get(side+"_type"))))
        if not changes: continue
        event=dict(claim_id=cid,claim_sha256=digest(record),changes=changes)
        try: after=change_claim(record,event)
        except ValueError: holds["claim_conflict_or_self_relation"]+=1; continue
        closure=Counter(); conflict=False
        for ordinal,payload in db.execute("SELECT ord,payload FROM edges WHERE owner=?",(cid,)):
            edge=json.loads(payload)
            if ordinal in structural_edits:
                edge=repair_edge(edge,structural_edits[ordinal])
                if edge is None: continue
            try: out=change_edge(edge,{cid:event})
            except ValueError: conflict=True; break
            if out is not edge: closure["about" if edge["relation_type"]=="about" else "science"]+=1
        if conflict or closure["about"]!=len(changes):
            holds["edge_closure_not_confirmed"]+=1; continue
        event["closure"]=dict(closure); events[cid]=event
        for change in changes:
            if len(examples[change["term"]])<3: examples[change["term"]].append(dict(claim_id=cid,
                paper=md.get("source_paper"),type=md.get(change["side"]+"_type") or (md.get("metadata") or {}).get(change["side"]+"_type")))
    # Do not introduce new fragmentation of an already shared full-name group.
    # All possible members share the same exact labels, hence were collected.
    db.execute("CREATE INDEX claim_key ON claims(k)")
    for k, in db.execute("SELECT k FROM claims GROUP BY k HAVING COUNT(*)>1"):
        members=list(db.execute("SELECT id,payload FROM claims WHERE k=?",(k,)))
        destinations=set()
        for cid,payload in members:
            record=json.loads(payload)
            if cid in events: record=change_claim(record,events[cid])
            destinations.add(compact(relation_key(record["metadata"])))
        if len(destinations)>1:
            for cid,_ in members:
                if cid in events: del events[cid]; holds["whole_group_incomplete_identity_hold"]+=1
    # A label is canonicalized consistently across an existing exact-name/ID
    # group. Conflicting study types hold the label globally, not just one
    # member; routing still requires endpoint-level proof independently.
    canonical_label_holds=set()
    for cid,payload in db.execute("SELECT id,payload FROM claims"):
        record=json.loads(payload)
        versions=[record,change_claim(record,events[cid])] if cid in events else [record]
        for version in versions:
            md=version["metadata"]
            for side in ("subject","object"):
                term=terms.get(name_key(md.get(side+"_name")))
                if term and md.get(side+"_id")==term["target_id"] and endpoint_eligibility(md,side,term):
                    canonical_label_holds.add(term["name"])
    for name in canonical_label_holds: terms[name]["canonicalize"]=False
    registry_payload["terms"]=[terms[k] for k in sorted(terms)]
    registry=VerifiedEntityTerms(registry_payload)
    prior_keys=defaultdict(set); before_keys=Counter(); after_keys=Counter(); canonical_before=Counter(); canonical_after=Counter()
    for cid,payload,k,prior in db.execute("SELECT id,payload,k,prior_k FROM claims"):
        record=json.loads(payload); prior_keys[prior].add(k); before_keys[k]+=1
        canonical_before[compact(registry.relation_key(record["metadata"]))]+=1
        if cid in events: record=change_claim(record,events[cid])
        after_keys[compact(relation_key(record["metadata"]))]+=1
        canonical_after[compact(registry.relation_key(record["metadata"]))]+=1
    split_groups=[]
    for prior,keys in prior_keys.items():
        if len(keys)<2: continue
        members=list(db.execute("SELECT id,payload FROM claims WHERE prior_k=?",(prior,)))
        dest=set()
        for cid,payload in members:
            record=json.loads(payload)
            if cid in events: record=change_claim(record,events[cid])
            dest.add(compact(registry.relation_key(record["metadata"])))
        split_groups.append(dict(prior_key=json.loads(prior),claim_ids=[c for c,_ in members],
                                 before_groups=len(keys),after_groups=len(dest)))
    used=Counter(c["term"] for e in events.values() for c in e["changes"])
    unique_changed_edges=0
    for ordinal,cid,payload in db.execute("SELECT ord,owner,payload FROM edges"):
        edge=json.loads(payload); out=repair_edge(edge,structural_edits[ordinal]) if ordinal in structural_edits else edge
        if out is None: continue
        if cid in events: out=change_edge(out,events)
        unique_changed_edges+=out is not edge
    result=dict(graph=campaign["current_graph"],detail_store=campaign["current_detail_store"],at=journal.utc_now(),
        source_full_sha_verified=True,counts=dict(counts),census=census,term_holds=term_holds,claim_holds=dict(holds),
        events=list(events.values()),changed_claims=len(events),changed_endpoints=sum(used.values()),
        changed_edges=sum(sum(e["closure"].values()) for e in events.values()),
        missing_type_identity_endpoints=sum(c["missing_type_not_fabricated"] for e in events.values() for c in e["changes"]),
        typed_annotation_values_written=0,original_evidence_deleted=False,canonical_label_holds=sorted(canonical_label_holds),
        structural_repairs=structure_plan,unique_changed_edge_records=unique_changed_edges,
        collected_relation_groups_before=len(before_keys),collected_relation_groups_after=len(after_keys),
        collected_canonical_groups_before=len(canonical_before),collected_canonical_groups_after=len(canonical_after),
        earlier_split_groups=split_groups,
        examples=[dict(name=k,endpoints=n,examples=examples[k]) for k,n in used.most_common(25)],
        structure_review=[dict(**summaries[cid],edges=structure_edges[cid]) for cid in sorted(structure_ids)],
        structure_target_nodes={nid:dict(name=nodes[nid]["preferred_name"],aliases=nodes[nid].get("aliases"),record_sha256=digest(nodes[nid]))
                                for nid in sorted(structure_targets) if nid in nodes and not nid.startswith("CLM:")},
        graph_backups=0,preimages_saved=False)
    journal.guards([campaign["current_graph"],campaign["current_detail_store"],campaign["formal_sources"]])
    journal.atomic_json(OUTPUT/"CURRENT_ENTITY_TERMS.json",registry_payload)
    journal.atomic_json(OUTPUT/"PLAN.json",result)
    progress("PLANNED",**{k:result[k] for k in ("changed_claims","changed_endpoints","changed_edges","missing_type_identity_endpoints",
        "collected_relation_groups_before","collected_relation_groups_after","collected_canonical_groups_before","collected_canonical_groups_after","earlier_split_groups")})
    db.close()


if __name__=="__main__": main()
