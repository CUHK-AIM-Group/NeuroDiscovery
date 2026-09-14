"""R40 current source boundary, whole held-endpoint triage and selected closures.

Writes compact evidence/hash registers only, never graph record preimages.
"""
from collections import Counter, defaultdict
from pathlib import Path
import re
import sqlite3
import sys
import httpx
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from fetch_kg_complete_titles import API, own_articles
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner, endpoint_gate
from neurooracle.src.claim_semantics import concept_atom_roles, declared_type_atoms
from neurooracle.src.relation_evidence import name_key
from inspect_kg_claim_deletion import exact_references

OUTPUT = j.OUTPUT/'round40_scoped_structure'
COLLISION = 'CLM:CASE1MAN:22306803:8568'
DIRECTION = 'CLM:ead63272cd396884'
NAMES = {
    'dorsal anterior cingulate and lateral prefrontal interference-related activation',
    'COMT genotype by externalizing behavior interaction',
    'fractional anisotropy in bilateral anterior thalamic radiation',
    'anterior cingulate cortex surface area',
    'bilateral uncinate fasciculus fractional anisotropy',
}


def progress(phase, **values):
    state = dict(status='READ_ONLY_INSPECTION', at=j.utc_now(), phase=phase, **values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json', state)
    print(compact(state), flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(), 'inspection already exists; inspect it')
    c = j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None, 'writer active')
    held = rows(c['current_gene_holds']['path'])
    require(len(held)==456 and len({r['claim_id'] for r in held})==455, 'held scope changed')
    candidate_ids = {nid for r in held if r['name'] in NAMES for nid in r.get('candidate_ids', [])}
    prior_nodes = rows(j.OUTPUT/'round38_measurement_reuse/CURRENT_NAME_CANDIDATES.jsonl')
    candidate_ids.update(r['node_id'] for r in prior_nodes if name_key(r['preferred_name']) in NAMES)
    selected = {r['claim_id'] for r in held if r['name'] in NAMES} | {COLLISION,DIRECTION}
    selected.update(i['claim_id'] for r in prior_nodes if r['node_id'] in candidate_ids for i in r['all_incident_claim_endpoints'])
    held_ids = {r['claim_id'] for r in held}
    wanted = candidate_ids | selected
    needle = re.compile('|'.join(re.escape(i) for i in sorted(wanted)))
    code = j.fingerprint(Path(__file__))
    nodes, claims, refs, exact, aliases = {}, {}, defaultdict(list), [], []
    incidents, found_held, counts = defaultdict(list), {}, Counter()
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    progress('FULL_CURRENT_SOURCE_AND_ENDPOINT_TRIAGE', held_endpoints=len(held), selected_claims=len(selected))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key.startswith('CLM:'):
                    md=row['metadata']
                    if key in selected: claims[key]=row
                    if key in held_ids:
                        found_held[key]=dict(claim_sha256=digest(row), endpoints={s:dict(
                            id=md.get(s+'_id'), name=md.get(s+'_name'),
                            types=sorted(a.value for a in declared_type_atoms(md.get(s+'_type') or (md.get('metadata') or {}).get(s+'_type'))),
                            gate=endpoint_gate(md,s)) for s in ('subject','object')})
                    for side in ('subject','object'):
                        if md.get(side+'_id') in candidate_ids:
                            require(key in selected, 'selected incident scope expanded')
                            incidents[md[side+'_id']].append(dict(claim_id=key,side=side,name=md.get(side+'_name'),
                                roles=sorted(a.value for a in declared_type_atoms(md.get(side+'_type') or (md.get('metadata') or {}).get(side+'_type'))),claim_sha256=digest(row)))
                else:
                    if name_key(row.get('preferred_name')) in NAMES:
                        require(key in candidate_ids, 'unexpected complete-name target; add it to read-only scope')
                        nodes[key]=row
                    for alias in row.get('aliases') or []:
                        if name_key(alias) in NAMES: aliases.append(dict(node_id=key,name=alias,node_sha256=digest(row)))
            if needle.search(compact(row)):
                links=list(exact_references(row,wanted))
                if links: exact.append(dict(kind=kind,key=key,record_sha256=digest(row),references=links))
            if kind=='edge' and edge_owner(row) in selected: refs[edge_owner(row)].append((int(key),row))
            if counts[kind] % 1000000 == 0: progress(kind,count=counts[kind])
        require(h.hexdigest()==c['current_graph']['sha256'], 'source full SHA differs')
    require(set(claims)==selected and set(found_held)==held_ids, 'current selected/held records missing')
    triage=[]
    for r in held:
        current=found_held[r['claim_id']]
        require(current['claim_sha256']==r['claim_sha256'], 'held claim hash changed')
        side=current['endpoints'][r['side']]
        require(side['id']==r['current_node_id'] and name_key(side['name'])==r['name'], 'held endpoint changed')
        triage.append(dict(**r,current_gate=side['gate'],current_declared_roles=side['types'],
            handling='selected_for_complete_reference_review' if r['claim_id'] in selected else 'held_not_approved_for_blanket_repair'))
    db=sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro',uri=True)
    deps={nid:dict(atoms=db.execute('SELECT COUNT(*) FROM atoms WHERE source_mention_id=?',(nid,)).fetchone()[0],
        mappings_as_source=db.execute('SELECT COUNT(*) FROM mappings WHERE source_id=?',(nid,)).fetchone()[0],
        mappings_as_target=db.execute('SELECT COUNT(*) FROM mappings WHERE target_id=?',(nid,)).fetchone()[0]) for nid in nodes}
    db.close()
    require(sha256(Path(c['current_detail_store']['path']))==c['current_detail_store']['sha256'], 'detail full SHA differs')
    write_rows(OUTPUT/'ENDPOINT_TRIAGE.jsonl',triage)
    write_rows(OUTPUT/'NODE_SCOPE.jsonl',[dict(node_id=nid,node_sha256=digest(n),name=n['preferred_name'],
        roles=sorted(a.value for a in concept_atom_roles(n)),incidents=incidents[nid],detail_dependencies=deps[nid]) for nid,n in sorted(nodes.items())])
    write_rows(OUTPUT/'CLAIM_SCOPE.jsonl',[dict(claim_id=cid,claim_sha256=digest(r),
        references=[dict(ordinal=i,edge_sha256=digest(e)) for i,e in refs[cid]]) for cid,r in sorted(claims.items())])
    write_rows(OUTPUT/'EXACT_REFERENCES.jsonl',exact)
    write_rows(OUTPUT/'ALIAS_CANDIDATES.jsonl',aliases)
    for nid,n in sorted(nodes.items()): print('NODE='+compact(n),flush=True)
    for cid,r in sorted(claims.items()):
        print('CLAIM='+compact(r),flush=True)
        for i,e in refs[cid]:print('EDGE='+compact(dict(ordinal=i,row=e)),flush=True)
    path=OUTPUT/'pubmed_36847009.xml'
    if not path.exists():
        params=dict(db='pubmed',id='36847009',retmode='xml',tool='NeuroClawKGReview')
        response=httpx.get(API,params=params,timeout=30,follow_redirects=True);response.raise_for_status()
        require(set(own_articles(response.content))=={'36847009'},'wrong public owner')
        j.atomic_text(path,response.text)
        j.atomic_json(OUTPUT/'PUBLIC_SOURCE_FETCH.json',dict(at=j.utc_now(),explicit_pmid='36847009',response=j.fingerprint(path),
            url=str(httpx.URL(API,params=params)),basis='explicit source_paper PMID, never synthetic suffix',code=code))
    root=ET.parse(path).getroot()
    print('PUBLIC_ABSTRACT='+' '.join(''.join(e.itertext()) for e in root.findall('.//AbstractText')),flush=True)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    summary=dict(status='CURRENT_SCOPED_STRUCTURE_INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],
        source_acceptance=c['current_acceptance'],detail_store=c['current_detail_store'],source_full_sha_verified=True,detail_full_sha_verified=True,
        code=code,counts=dict(counts),held_endpoints=len(triage),held_reason_counts=dict(Counter(r['reason'] for r in triage)),
        selected_claims=sorted(selected),graph_modified=False,record_preimages_saved=False,
        artifacts={n:j.fingerprint(OUTPUT/n) for n in ('ENDPOINT_TRIAGE.jsonl','NODE_SCOPE.jsonl','CLAIM_SCOPE.jsonl','EXACT_REFERENCES.jsonl','ALIAS_CANDIDATES.jsonl','PUBLIC_SOURCE_FETCH.json')})
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',summary)
    progress('COMPLETE',selected_claims=len(claims),nodes=len(nodes),held_reason_counts=summary['held_reason_counts'])


if __name__=='__main__':main()
