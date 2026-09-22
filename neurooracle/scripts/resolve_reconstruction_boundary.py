"""Resolve provenance from the completed index; never reread the graph JSON."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import sqlite3
import zlib
from prepare_paper_reconstruction import dump,now,read,settings

EXTRA_FIXED_NODES={'UMLS_2026AA','UKB-Showcase','ADNI','HCP','UniProt'}
EXTRA_FIXED_EDGES={'NeuroClaw-IM-Outcome','NeuroClaw-IM-Reverse','OutcomeIM-Bridge',
  'ClinicalOutcomes-Bridge','IndividualData-Bridge','MedDRA-Bridge','DatasetVar-Bridge',
  'UKB-Showcase','ADNI','HCP'}
CURATED_SHARED={'NeuroClaw-ManualCanonical'}
MENTIONS={'claim_extraction_anchor','manual_claim_anchor','manual_general_claim_anchor',
          'replay_anchor_mint','manual_claim_anchor_repair','claim_extraction'}
ALIGNMENT_NODE='CLM_CONCEPT_atomic_projection'
ALIGNMENT_EDGE='UMLS_2026AA_normalized_exact_atomic_mention_alignment'


def resolved(kind,part,source):
    if part=='legacy_literature':return 'archived_old_paper_claim'
    if part=='fixed' or source in (EXTRA_FIXED_NODES if kind=='node' else EXTRA_FIXED_EDGES):return 'fixed_reference'
    if kind=='node' and source in CURATED_SHARED:return 'protected_shared_canonical'
    if (kind=='node' and source==ALIGNMENT_NODE) or (kind=='edge' and source==ALIGNMENT_EDGE):return 'protected_existing_alignment'
    if kind=='node' and source in MENTIONS:return 'archived_paper_mention'
    return 'unresolved_provenance_hold'


def stream_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(16*1024*1024):h.update(block)
    s=path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest()}


def main():
    _,out,_,corpus_binding=settings()
    target=out/'FIXED_BOUNDARY.json'
    if target.exists():print(json.dumps(read(target),ensure_ascii=False));return
    raw_receipt=read(out/'FIXED_PARTITION_READY.json')
    source=out/'GRAPH_PARTITION.sqlite'
    c=sqlite3.connect(f'file:{source.as_posix()}?mode=ro',uri=True)
    catalog=out/'REFERENCE_CATALOG.sqlite'
    resuming=catalog.exists()
    r=sqlite3.connect(catalog,uri=True)
    r.execute('attach database ? as original',(f'file:{source.as_posix()}?mode=ro',))
    r.executescript('''create table if not exists entities(id text primary key,preferred_name text,aliases_json text,
      provenance text,partition text,source_record_sha256 text);
      create index if not exists entity_names on entities(preferred_name collate nocase);
      create table if not exists protected_edges(ordinal integer primary key,source_id text,target_id text,
      provenance text,partition text,source_record_sha256 text);
      create table if not exists closure_nodes(id text primary key,partition text,source_record_sha256 text);
    ''')
    counts=Counter();provenance=[];digests={k:hashlib.sha256() for k in ['fixed_nodes','fixed_edges','shared_nodes','shared_edges']}
    for kind,table in [('node','nodes'),('edge','edges')]:
        counts[kind+':archived_old_paper_claim']=raw_receipt['counts'][kind+':legacy_literature']
        for part,src,n in c.execute(f"select partition,provenance,count(*) from {table} where partition!='legacy_literature' group by partition,provenance"):
            dest=resolved(kind,part,src);counts[kind+':'+dest]+=n
            if part!='legacy_literature':provenance.append({'kind':kind,'source':src,'decision':dest,'count':n})
    # Read full payloads only for the small fixed/reference vocabulary. All
    # original payloads, including existing alignments, remain in the index.
    if resuming:
        expected_nodes=counts['node:fixed_reference']+counts['node:protected_shared_canonical']
        expected_edges=counts['edge:fixed_reference']+counts['edge:protected_existing_alignment']
        if r.execute('select count(*) from entities').fetchone()[0]!=expected_nodes or r.execute('select count(*) from protected_edges').fetchone()[0]!=expected_edges:
            raise ValueError('Partial reference population requires explicit repair; not a completed-copy resume')
        if r.execute('''select count(*) from entities e left join original.nodes n using(id)
          where n.id is null or n.sha256!=e.source_record_sha256 or n.provenance!=e.provenance''').fetchone()[0]:
            raise ValueError('Reference nodes changed before resume')
        if r.execute('''select count(*) from protected_edges e left join original.edges o using(ordinal)
          where o.ordinal is null or o.sha256!=e.source_record_sha256 or o.source_id!=e.source_id or o.target_id!=e.target_id''').fetchone()[0]:
            raise ValueError('Reference edges changed before resume')
    for node,part,src,sha,payload in ([] if resuming else c.execute("select id,partition,provenance,sha256,payload from nodes where partition!='legacy_literature' order by id")):
        dest=resolved('node',part,src)
        if dest not in {'fixed_reference','protected_shared_canonical'}:continue
        value=json.loads(zlib.decompress(payload))
        r.execute('insert into entities values(?,?,?,?,?,?)',(node,value.get('preferred_name') or value.get('name'),json.dumps(value.get('aliases',[]),ensure_ascii=False),src,dest,sha))
    for ordinal,a,b,part,src,sha in ([] if resuming else c.execute("select ordinal,source_id,target_id,partition,provenance,sha256 from edges where partition!='legacy_literature' order by ordinal")):
        dest=resolved('edge',part,src)
        if dest not in {'fixed_reference','protected_existing_alignment'}:continue
        r.execute('insert into protected_edges values(?,?,?,?,?,?)',(ordinal,a,b,src,dest,sha))
    for node,part,sha in r.execute('select id,partition,source_record_sha256 from entities order by id'):
        digests['fixed_nodes' if part=='fixed_reference' else 'shared_nodes'].update((node+'\0'+sha+'\n').encode())
    for ordinal,part,sha in r.execute('select ordinal,partition,source_record_sha256 from protected_edges order by ordinal'):
        digests['fixed_edges' if part=='fixed_reference' else 'shared_edges'].update((str(ordinal)+'\0'+sha+'\n').encode())
    r.commit()
    r.execute('''insert or ignore into closure_nodes select n.id,n.partition,n.sha256 from original.nodes n
       join (select source_id id from protected_edges union select target_id from protected_edges) x using(id)''')
    absent=r.execute('''select count(*) from (select source_id id from protected_edges union select target_id from protected_edges) x
      left join closure_nodes n using(id) where n.id is null''').fetchone()[0]
    old_claim_endpoints=r.execute("select count(*) from closure_nodes where partition='legacy_literature'").fetchone()[0]
    protected_refs=r.execute('select count(*) from entities').fetchone()[0]
    closure_count=r.execute('select count(*) from closure_nodes').fetchone()[0]
    r.commit()
    if absent or old_claim_endpoints:raise ValueError('Protected boundary lacks valid reference endpoints')
    if r.execute('pragma quick_check').fetchone()[0]!='ok':raise ValueError('Reference catalog failed check')
    r.close();c.close()
    report={'at':now(),'status':'FIXED_BOUNDARY_VERIFIED','original_graph':raw_receipt['original_graph'],
      'source_corpus':corpus_binding,'original_graph_total_read_passes':1,
      'fixed_original_graph_hash_verified':raw_receipt['full_source_sha256_verified'],
      'counts':dict(counts),'reference_entities':protected_refs,'protected_edge_endpoint_nodes':closure_count,
      'missing_protected_endpoints':absent,'protected_edges_pointing_to_old_claims':old_claim_endpoints,
      'unresolved_provenance_records':sum(v for k,v in counts.items() if k.endswith('unresolved_provenance_hold')),
      'ordered_record_digest_roots':{k:v.hexdigest() for k,v in digests.items()},
      'partition_index':stream_hash(source),'reference_catalog':stream_hash(catalog),
      'provenance_decisions':provenance,
      'fixed_layer_modified':False,'existing_alignment_modified':False,'production_changed':False,
      'policy':'Fixed/reference records, shared canonical entities and existing UMLS mention alignments remain immutable. Old paper claims and old mentions remain archived; they are not extraction answers or scientific credit. Referenced alignment endpoints are preserved with their complete original payloads.',
      'integration_gate':'Future graph/view must preserve every protected record hash, edge ordinal/multiplicity, and endpoint; unresolved new entity mappings stay in the literature layer.',
      'resumed_existing_catalog_after_uri_open_fix':resuming}
    dump(target,report)
    state=read(out/'STATE.json');state.update(at=now(),fixed_partition_status=report['status'],fixed_boundary_receipt=str(target),active_process=None)
    dump(out/'STATE.json',state)
    print(json.dumps({k:v for k,v in report.items() if k not in {'provenance_decisions','source_corpus'}},ensure_ascii=False))


if __name__=='__main__':main()
