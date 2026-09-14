"""R58 read-only source and reference closure for three qualified observations."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_claim_deletion import exact_references
from inspect_kg_remaining_scoped_literals import scope_projection,owned_projection
from inspect_kg_semantic_hold_sources import projection
from inspect_kg_scientific_definition_scope import field_hashes
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.kg_scoped_structure import source_documents

OUTPUT=j.OUTPUT/'round58_observational_semantics_review'
SELECTED={'CLM:a3ce9c23b246de1a','CLM:debb3aa8b8f253b0','CLM:a5442a6df750ab2d'}


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'review exists; inspect binding')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='scientific_definition_repair','expected pending R57')
    pending_fp=j.fingerprint(j.OUTPUT/'round57_scientific_definition_repair/PLAN.json');pending=j.read_json(pending_fp['path'])
    require(pending['graph']==c['current_graph'] and pending['source_acceptance']==c['current_acceptance'],'pending source differs')
    issues=rows(c['current_issues']['path']);expected={r['claim_id']:r['current_node_sha256'] for r in issues}
    require(len(expected)==17 and SELECTED<=set(expected) and not set(expected)&{e['claim_id'] for e in pending['events']},'semantic scope differs')
    old_fp=j.fingerprint(j.OUTPUT/'round56_scientific_definition_review/CURRENT_CLAIM_PROJECTIONS.jsonl')
    old={r['claim_id']:r for r in rows(old_fp['path']) if r['claim_id'] in expected}
    require(set(old)==set(expected) and all(old[cid]['claim_sha256']==h for cid,h in expected.items()),'current R56 projections differ')
    public_fp=j.fingerprint(j.OUTPUT/'round45_semantic_sources/PUBLIC_SOURCE_FETCH.json');public=j.read_json(public_fp['path'])
    response=public['response'];small_check(response)
    docs=source_documents([Path(response['path']).read_text(encoding='utf8')])
    public_rows={r['claim_id']:r for r in public['selected']}
    proof={}
    for cid,r in old.items():
        pmid=str(r['source_paper']['pmid']);paper=docs[pmid];saved=public_rows[cid]
        require(r['claim_sha256']==saved['claim_sha256'] and saved['pmid']==pmid and digest(paper['abstract'])==saved['public_abstract_sha256']
            and paper['title']==saved['public_title'] and saved['doi_matches'] and saved['title_matches'],'own public source changed')
        proof[cid]=dict(pmid=pmid,abstract_sha256=digest(paper['abstract']),title=paper['title'],dois=paper['dois'])
    targets={old[cid]['science'][s+'_id'] for cid in SELECTED for s in ('subject','object')}
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    claims={};nodes={};owned=defaultdict(list);exact=[];counts=Counter();needle=re.compile('|'.join(re.escape(cid) for cid in sorted(SELECTED)))
    progress('FULL_CURRENT_SEMANTIC_SOURCE_AND_ALL_EXACT_REFERENCES',claims=len(expected))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in expected:
                    require(digest(row)==expected[key] and projection(row)==old[key],'current scientific claim differs');claims[key]=row
                elif key in targets:nodes[key]=scope_projection(row)
            elif kind=='edge':
                owner=edge_owner(row)
                if owner in SELECTED:owned[owner].append(owned_projection(int(key),row))
            if needle.search(compact(row)):
                refs=list(exact_references(row,SELECTED))
                if refs:exact.append(dict(kind=kind,key=key,record_sha256=digest(row),references=refs))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full current source SHA differs')
    require(set(claims)==set(expected) and set(nodes)==targets and set(owned)==SELECTED,'scientific source closure incomplete')
    require((counts['node'],counts['edge'])==(c['counts']['nodes'],c['counts']['edges']),'source counts differ')
    data={'CURRENT_CLAIM_PROJECTIONS.jsonl':[projection(claims[cid]) for cid in sorted(claims)],
        'CURRENT_CLAIM_FIELD_HASHES.jsonl':[dict(claim_id=cid,claim_sha256=digest(claims[cid]),field_hashes=list(field_hashes(claims[cid]))) for cid in sorted(SELECTED)],
        'CURRENT_TARGET_WITNESSES.jsonl':[nodes[nid] for nid in sorted(nodes)],
        'CURRENT_OWNED_EDGE_PROJECTIONS.jsonl':[r for cid in sorted(owned) for r in owned[cid]],
        'CURRENT_EXACT_CLAIM_REFERENCES.jsonl':exact}
    for name,records in data.items():write_rows(OUTPUT/name,records)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    result=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        pending_disjoint_plan=pending_fp,original_projections=old_fp,public_sources=public_fp,public_response=response,public_proofs=proof,code=code,
        full_source_sha_verified=True,claims=len(claims),selected=sorted(SELECTED),nodes=len(nodes),owned_edges=sum(map(len,owned.values())),
        exact_reference_records=len(exact),graph_modified=False,record_preimages_saved=False,old_audit_not_revalidated=True,
        artifacts={name:j.fingerprint(OUTPUT/name) for name in data})
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',result)
    for cid in sorted(SELECTED):print(compact(dict(selected_current_record_for_review=claims[cid])),flush=True)
    progress('COMPLETED',claims=len(claims),selected=len(SELECTED),nodes=len(nodes),owned_edges=result['owned_edges'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',at=j.utc_now(),pid=os.getpid(),error=repr(error),graph_modified=False));raise
