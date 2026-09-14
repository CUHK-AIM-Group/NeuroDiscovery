"""R56 fresh scientific projections, whole-name candidates and exact references."""
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
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round56_scientific_definition_review'
REVIEW=j.OUTPUT/'round49_remaining_scoped_literals'
MEASUREMENT_PMIDS={'15380120','30384145','38223081','20735996','42300138','15038994','22504417','35145436','36847009'}
NAMES={'plasma bace1 concentration','bace1 concentration','bace1 amyloid-processing pathway','bilateral hippocampal volume',
    'mean bilateral hippocampal volume','left and right hippocampal volumes','depressive disorder severity in dpd',
    'severity of depressive disorder in dpd','depression severity in parkinson disease','depression severity in parkinson\'s disease',
    'hamilton depression rating scale score','hamilton depression rating scale score in depressed parkinson disease'}
DPD={'CLM:1e89987996da15c2c8921884e85f0ce7','CLM:0a6379df92d50ec2272354b3c523f2dc'}


def candidate_names(row):
    labels=[row.get('preferred_name'),*(row.get('aliases') or [])]
    return [label for label in labels if name_key(label).casefold() in NAMES or re.search(r'\bBACE1\b',str(label),re.I)]


def field_hashes(value,path=()):
    if isinstance(value,dict):
        yield dict(path=list(path),container_keys=sorted(value))
        for key,item in sorted(value.items()):yield from field_hashes(item,(*path,key))
    else:yield dict(path=list(path),sha256=digest(value))


def differences(a,b,path=()):
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:yield dict(path=list((*path,key)),left_present=key in a,right_present=key in b)
            else:yield from differences(a[key],b[key],(*path,key))
    elif a!=b:yield dict(path=list(path),left_sha256=digest(a),right_sha256=digest(b))


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'review already exists')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='existing_literal_reuse','expected pending R55')
    pending_fp=j.fingerprint(j.OUTPUT/'round55_existing_literal_reuse/PLAN.json');pending=j.read_json(pending_fp['path'])
    require(pending['graph']==c['current_graph'] and pending['source_acceptance']==c['current_acceptance'],'pending source differs')
    old_fp=j.fingerprint(REVIEW/'CURRENT_CLAIM_PROJECTIONS.jsonl');old=rows(old_fp['path'])
    issues=rows(c['current_issues']['path']);scope=j.read_json(c['current_scope_findings']['path'])
    selected={r['claim_id'] for r in old if str((r['source_paper'] or {}).get('pmid')) in MEASUREMENT_PMIDS}
    selected.update(r['claim_id'] for r in issues);selected.update(scope['current_claim_hashes'])
    require(DPD<=selected and not selected&{e['claim_id'] for e in pending['events']},'scientific source overlaps pending writer')
    expected={r['claim_id']:r['claim_sha256'] for r in old if r['claim_id'] in selected}
    expected.update({r['claim_id']:r['current_node_sha256'] for r in issues});expected.update(scope['current_claim_hashes'])
    public_fp=j.fingerprint(OUTPUT/'PUBLIC_SOURCE_FETCH.json');public=j.read_json(public_fp['path'])
    for fp in [public['original_abstract_response'],*([r['response'] for r in public['fulltexts'] if r['status']=='PUBLIC_FULLTEXT_FETCHED'])]:small_check(fp)
    known_ids={r['science'][s+'_id'] for r in old if r['claim_id'] in selected for s in ('subject','object')}
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    claims={};nodes={};name_matches=[];owned=defaultdict(list);exact=[];other=defaultdict(list);counts=Counter()
    needle=re.compile('|'.join(re.escape(cid) for cid in sorted(selected)))
    progress('FULL_CURRENT_SCIENTIFIC_SOURCE_AND_EXACT_REFERENCES',selected_claims=len(selected))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key.startswith('CLM:'):
                    counts['claims']+=1
                    if key in selected:
                        require(digest(row)==expected[key],'current scientific claim differs');claims[key]=row
                else:
                    labels=candidate_names(row)
                    if key in known_ids or labels:nodes[key]=scope_projection(row)
                    if labels:name_matches.append(dict(node_id=key,matched_labels=labels))
            elif kind=='edge':
                owner=edge_owner(row)
                if owner in selected:owned[owner].append(owned_projection(int(key),row))
                if not owner:
                    for nid in {row.get('source_id'),row.get('target_id')}&nodes.keys():
                        other[nid].append(dict(ordinal=int(key),edge_sha256=digest(row),relation_type=row['relation_type']))
            if needle.search(compact(row)):
                refs=list(exact_references(row,selected))
                if refs:exact.append(dict(kind=kind,key=key,record_sha256=digest(row),references=refs))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full current source SHA differs')
    require(set(claims)==selected,'current selected scope incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    incidences=[];second_nodes=0
    progress('CURRENT_EXISTING_TARGET_ALL_CLAIM_INCIDENCES_AFTER_FULL_SOURCE_SHA',nodes=len(nodes))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,unused_partial_sha):
        for kind,key,row in walk_graph(reader):
            if kind=='edge':break
            if kind!='node':continue
            second_nodes+=1
            if not key.startswith('CLM:'):continue
            md=row['metadata'];inner=md.get('metadata') or {}
            for side in ('subject','object'):
                if md.get(side+'_id') in nodes:
                    incidences.append(dict(node_id=md[side+'_id'],claim_id=key,side=side,name=md.get(side+'_name'),claim_sha256=digest(row),
                        nonidentity_sha256=digest(nonidentity_claim(row)),outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type')))
    require(second_nodes==c['counts']['nodes'],'second node incidence scan incomplete')
    pair=[claims[cid] for cid in sorted(DPD)]
    duplicate=dict(claim_ids=sorted(DPD),difference_paths=list(differences(*pair)),all_fields_identical=False,not_deletion_authorization=True)
    data={'CURRENT_CLAIM_PROJECTIONS.jsonl':[projection(claims[cid]) for cid in sorted(claims)],
        'CURRENT_CLAIM_FIELD_HASHES.jsonl':[dict(claim_id=cid,claim_sha256=digest(claims[cid]),field_hashes=list(field_hashes(claims[cid]))) for cid in sorted(claims)],
        'CURRENT_TARGET_WITNESSES.jsonl':[nodes[nid] for nid in sorted(nodes)],'CURRENT_NAME_MATCHES.jsonl':name_matches,
        'CURRENT_TARGET_INCIDENCES.jsonl':incidences,'CURRENT_OWNED_EDGE_PROJECTIONS.jsonl':[r for cid in sorted(owned) for r in owned[cid]],
        'CURRENT_EXACT_CLAIM_REFERENCES.jsonl':exact,'CURRENT_NONCLAIM_RELATIONS.jsonl':[dict(node_id=nid,edges=rs) for nid,rs in sorted(other.items())]}
    for name,records in data.items():write_rows(OUTPUT/name,records)
    j.atomic_json(OUTPUT/'DPD_DIFFERENCES.json',duplicate)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    result=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        pending_disjoint_plan=pending_fp,original_projections=old_fp,public_sources=public_fp,code=code,full_source_sha_verified=True,
        second_node_scan_guarded_by_unchanged_current_full_source_boundary=True,claims=len(claims),nodes=len(nodes),incidences=len(incidences),
        owned_edges=sum(map(len,owned.values())),exact_reference_records=len(exact),graph_modified=False,record_preimages_saved=False,
        old_audit_not_revalidated=True,artifacts={name:j.fingerprint(OUTPUT/name) for name in data},duplicate_differences=j.fingerprint(OUTPUT/'DPD_DIFFERENCES.json'))
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',result);progress('COMPLETED',claims=len(claims),nodes=len(nodes),incidences=len(incidences))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',at=j.utc_now(),pid=os.getpid(),error=repr(error),graph_modified=False));raise
