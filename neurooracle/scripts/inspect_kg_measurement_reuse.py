"""R38: current hash-only endpoint/reference census and owning source evidence."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
import httpx

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from fetch_kg_complete_titles import own_articles,API
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import GENES,edge_owner
from neurooracle.src.relation_evidence import name_key
from neurooracle.src.claim_semantics import concept_atom_roles,declared_type_atoms

OUTPUT=journal.OUTPUT/'round38_measurement_reuse'
PRIORITY={'CLM:1b4579c82247','CLM:951f2bb246a3'}
MATERIALIZED={'CLM:CASE1MAN:39829963:284','CLM:CASE1MAN:39829963:286'}


def progress(phase,**values):
    state=dict(status='READ_ONLY_INSPECTION',phase=phase,at=journal.utc_now(),**values)
    journal.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def public_sources(c):
    path=OUTPUT/'PUBLIC_SOURCE_FETCH.json'
    if path.exists():
        result=journal.read_json(path)
        require(result['graph']==c['current_graph'],'public request baseline changed')
        for w in result['witnesses']: require(journal.fingerprint(w['response']['path'])==w['response'],'public witness changed')
        return result
    witnesses=[]
    with httpx.Client(timeout=30,follow_redirects=True,headers={'User-Agent':'NeuroClawKGReview/1.0'}) as client:
        for db,identifier,file in [('pubmed','39829963','pubmed_39829963.xml'),('pmc','11740805','PMC11740805.xml')]:
            params=dict(db=db,id=identifier,retmode='xml',tool='NeuroClawKGReview')
            response=client.get(API,params=params);response.raise_for_status()
            if db=='pubmed': require(set(own_articles(response.content))=={'39829963'},'wrong PubMed owner')
            else:
                root=ET.fromstring(response.content);articles=root.findall('./article') if root.tag!='article' else [root]
                require(len(articles)==1 and articles[0].find("./front/article-meta/article-id[@pub-id-type='pmid']").text=='39829963','wrong PMC owner')
            journal.atomic_text(OUTPUT/file,response.text)
            witnesses.append(dict(database=db,explicit_id=identifier,url=str(httpx.URL(API,params=params)),response=journal.fingerprint(OUTPUT/file)))
    result=dict(at=journal.utc_now(),graph=c['current_graph'],witnesses=witnesses,public_identifiers_only=True,
        id_basis='explicit current PubMed and owning public PMC identifiers; never a synthetic-ID suffix')
    journal.atomic_json(path,result);return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'inspection exists; inspect/reuse')
    c=journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    journal.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    for key in ('current_acceptance','current_gene_holds','current_shared_relations','current_entity_terms'):
        require(journal.fingerprint(c[key]['path'])==c[key],'current evidence changed')
    public_sources(c)
    held=rows(c['current_gene_holds']['path']); claim_ids={r['claim_id'] for r in held}|MATERIALIZED
    names={name_key(r['name']) for r in held}
    candidates,claims,refs,incidents={}, {},defaultdict(list),defaultdict(list)
    # Candidate IDs are discovered before claims in some but not all sources;
    # keep all small endpoint tuples in RAM for a complete incident closure.
    import sqlite3
    db=sqlite3.connect(':memory:');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE endpoints(nid TEXT,cid TEXT,side TEXT,name TEXT,roles TEXT,sha TEXT)')
    batch=[];counts=Counter(); gene_rows=[]; code=journal.fingerprint(Path(__file__))
    progress('FULL_CURRENT_SOURCE_NAMES_TYPES_AND_REFERENCE_CLOSURE')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key.startswith('CLM:'):
                    md=row['metadata'];node_sha=digest(row)
                    if key in claim_ids:claims[key]=row
                    for side in ('subject','object'):
                        declared=md.get(side+'_type') or (md.get('metadata') or {}).get(side+'_type')
                        roles=sorted(a.value for a in declared_type_atoms(declared))
                        batch.append((md.get(side+'_id'),key,side,md.get(side+'_name'),compact(roles),node_sha))
                        if md.get(side+'_id') in GENES:
                            gene_rows.append(dict(claim_id=key,claim_sha256=node_sha,side=side,current_node_id=md[side+'_id'],
                                name=md.get(side+'_name'),declared_type=declared,declared_roles=roles))
                    if len(batch)>=5000:db.executemany('INSERT INTO endpoints VALUES (?,?,?,?,?,?)',batch);batch.clear()
                elif key in GENES or name_key(row.get('preferred_name')) in names:
                    candidates[key]=row
            elif kind=='edge':
                owner=edge_owner(row)
                if owner in claim_ids or row['source_id'] in candidates or row['target_id'] in candidates:
                    refs[owner].append(dict(ordinal=int(key),edge_sha256=digest(row),source_id=row['source_id'],target_id=row['target_id'],
                        relation_type=row['relation_type'],claim_id=owner,
                        nonendpoint_sha256=digest({k:v for k,v in row.items() if k not in {'source_id','target_id'}})))
            if counts[kind]%750000==0:progress(kind,count=counts[kind])
        require(h.hexdigest()==c['current_graph']['sha256'],'current graph full SHA differs')
    db.executemany('INSERT INTO endpoints VALUES (?,?,?,?,?,?)',batch);db.execute('CREATE INDEX endpoint_node ON endpoints(nid)')
    require(set(claims)==claim_ids,'selected current claims incomplete')
    require({(r['claim_id'],r['side'],r['claim_sha256']) for r in gene_rows}=={(r['claim_id'],r['side'],r['claim_sha256']) for r in held},'current gene queue does not reproduce')
    nodes=[]
    import json
    for nid,row in sorted(candidates.items()):
        ep=[dict(claim_id=cid,side=side,name=name,declared_roles=json.loads(roles),claim_sha256=sha)
            for cid,side,name,roles,sha in db.execute('SELECT cid,side,name,roles,sha FROM endpoints WHERE nid=? ORDER BY cid,side',(nid,))]
        nodes.append(dict(node_id=nid,node_sha256=digest(row),preferred_name=row.get('preferred_name'),
            declared_roles=sorted(a.value for a in concept_atom_roles(row)),semantic_types=row.get('semantic_types'),
            external_ids=row.get('external_ids'),definition=row.get('definition'),spatial_mapping=row.get('spatial_mapping'),
            all_incident_claim_endpoints=ep))
    db.close()
    write_rows(OUTPUT/'CURRENT_GENE_ENDPOINTS.jsonl',gene_rows)
    write_rows(OUTPUT/'CURRENT_NAME_CANDIDATES.jsonl',nodes)
    write_rows(OUTPUT/'CURRENT_REFERENCE_CLOSURE.jsonl',sorted([r for group in refs.values() for r in group],key=lambda r:r['ordinal']))
    current_priority=[]
    for cid in sorted(PRIORITY|MATERIALIZED):
        row=claims[cid];md=row['metadata']
        current_priority.append(dict(claim_id=cid,claim_sha256=digest(row),source_paper=md.get('source_paper'),
            subject_id=md.get('subject_id'),subject_name=md.get('subject_name'),subject_type=md.get('subject_type'),
            object_id=md.get('object_id'),object_name=md.get('object_name'),object_type=md.get('object_type'),
            predicate=md.get('predicate'),raw_text_sha256=__import__('hashlib').sha256(str(md.get('raw_text') or '').encode()).hexdigest()))
        print(compact(dict(console_only_current_claim=cid,raw_text=md.get('raw_text'),evidence=md.get('evidence'),metadata=md.get('metadata'))),flush=True)
    write_rows(OUTPUT/'SOURCE_REVIEW_TARGETS.jsonl',current_priority)
    # Console-only existing candidate definitions for exact-scope review.
    print(compact(dict(console_only_candidate_nodes=[row for nid,row in candidates.items() if nid not in GENES])),flush=True)
    require(journal.fingerprint(Path(__file__))==code and journal.read_json(journal.OUTPUT/'CAMPAIGN.json')==c,'inspection input/code advanced')
    journal.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    files=['CURRENT_GENE_ENDPOINTS.jsonl','CURRENT_NAME_CANDIDATES.jsonl','CURRENT_REFERENCE_CLOSURE.jsonl','SOURCE_REVIEW_TARGETS.jsonl','PUBLIC_SOURCE_FETCH.json']
    summary=dict(status='CURRENT_SCOPE_INSPECTED_NOT_APPLIED',at=journal.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        source_full_sha_verified=True,counts=dict(counts),current_gene_endpoints=len(gene_rows),candidate_nodes=len(nodes),
        artifacts={f:journal.fingerprint(OUTPUT/f) for f in files},code=code,graph_modified=False,record_preimages_saved=False)
    journal.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',summary);progress('COMPLETE',current_gene_endpoints=len(gene_rows),candidate_nodes=len(nodes))


if __name__=='__main__':main()
