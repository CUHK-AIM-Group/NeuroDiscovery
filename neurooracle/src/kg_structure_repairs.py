"""Repair proved stale links; discard only same-owner identical edge payloads."""
from copy import deepcopy
from .kg_identity_pilot import digest
from .relation_evidence import name_key


def choose_repairs(reviews,nodes):
    edits=[]; repaired=[]; held=[]
    for item in reviews:
        cid=item["claim_id"]; sid=item["subject_id"]; oid=item["object_id"]
        science=[e for e in item["edges"] if e["relation_type"]!="about"]
        about=[e for e in item["edges"] if e["relation_type"]=="about"]
        if any(e.get("claim_id")!=cid for e in science) or any(e["source_id"]!=cid for e in about):
            held.append(cid); continue
        exact=[e for e in science if (e["source_id"],e["target_id"],e["relation_type"])==(sid,oid,item["predicate"])]
        stray=[e for e in about if e["target_id"] not in (sid,oid)]
        local=[]
        if len(science)==len(exact)==1 and len(about)==2 and len(stray)==1 and stray[0]["target_id"]=="CLM_CONCEPT:unnamed_da39a3ee5e6b":
            if any(e["target_id"]==sid for e in about) and name_key(nodes.get(oid,{}).get("name"))==name_key(item["object_name"]):
                edge=stray[0]
                local.append(dict(ordinal=edge["ordinal"],edge_sha256=edge["edge_sha256"],claim_id=cid,
                    action="retarget_placeholder_about",old_target=edge["target_id"],target_id=oid))
        elif len(exact)==1 and len(science)==2 and len(stray)==1 and len(about)==3:
            alternate=next(e for e in science if e not in exact)
            changed=[side for side,field,base in (("subject","source_id",sid),("object","target_id",oid)) if alternate[field]!=base]
            if len(changed)==1:
                side=changed[0]; field="source_id" if side=="subject" else "target_id"; base=sid if side=="subject" else oid
                other=alternate[field]; matching=[e for e in about if e["target_id"]==base]
                same_names=all(name_key(nodes.get(nid,{}).get("name"))==name_key(item[side+"_name"]) for nid in (base,other))
                if (same_names and len(matching)==1 and stray[0]["target_id"]==other
                    and alternate["nonendpoint_sha256"]==exact[0]["nonendpoint_sha256"]
                    and stray[0]["nonendpoint_sha256"]==matching[0]["nonendpoint_sha256"]):
                    for remove,keep in ((alternate,exact[0]),(stray[0],matching[0])):
                        local.append(dict(ordinal=remove["ordinal"],edge_sha256=remove["edge_sha256"],claim_id=cid,
                            action="remove_same_owner_duplicate",keep_ordinal=keep["ordinal"],keep_sha256=keep["edge_sha256"],
                            nonendpoint_sha256=remove["nonendpoint_sha256"]))
        if local: edits.extend(local); repaired.append(cid)
        else: held.append(cid)
    return dict(edits=edits,repaired_claim_ids=repaired,held_claim_ids=held,
                retargeted_edges=sum(e["action"]=="retarget_placeholder_about" for e in edits),
                removed_duplicate_edges=sum(e["action"]=="remove_same_owner_duplicate" for e in edits))


def repair_edge(record,edit):
    if digest(record)!=edit["edge_sha256"]: raise ValueError("structural witness changed")
    if edit["action"]=="remove_same_owner_duplicate":
        if digest({k:v for k,v in record.items() if k not in {"source_id","target_id"}})!=edit["nonendpoint_sha256"]:
            raise ValueError("duplicate scientific payload differs")
        return None
    if edit["action"]!="retarget_placeholder_about" or record["relation_type"]!="about" or record["source_id"]!=edit["claim_id"] or record["target_id"]!=edit["old_target"]:
        raise ValueError("unsafe structural retarget")
    result=deepcopy(record); result["target_id"]=edit["target_id"]
    return result
