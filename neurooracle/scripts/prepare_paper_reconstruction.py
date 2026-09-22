"""Prepare source-only paper work; never run the old extractor or edit a graph."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,sqlite3,sys,zlib

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'neurooracle/configs/kg_reconstruction.json'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from neurooracle.src.kg_storage import stat_matches, storage_path, staging_directory

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,obj):
    p=Path(p);temp=p.with_suffix(p.suffix+'.writing')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def settings():
    config=read(CONFIG);out=staging_directory(config['execution']['output_directory'],ROOT)
    if config['old_claims_are_extraction_input'] or config['fixed_neuroscience_layer']['allow_writes']:
        raise ValueError('This pipeline supports paper-source input and a read-only fixed layer only')
    corpus=storage_path(ROOT/config['paper_materials']['corpus'])
    manifest=read(ROOT/config['paper_materials']['integrity_manifest'])
    binding=next(x for x in manifest['files'] if storage_path(x['path'])==corpus)
    if not stat_matches(binding,corpus):raise ValueError('Preserved source corpus changed')
    return config,out,corpus,binding

def bootstrap():
    config,out,corpus,binding=settings();out.mkdir(parents=True,exist_ok=True)
    complete=out/'QUEUE_READY.json'
    if complete.exists():
        report=read(complete)
        if report['config_sha256']!=file_hash(CONFIG):
            auth=out/'BATCH_MODEL/AUTHORIZATION.json'
            previous=out/'BATCH_MODEL/CONFIG_BEFORE_AUTHORIZATION.json'
            if not auth.exists() or not previous.exists():raise ValueError('Existing queue belongs to another configuration')
            binding=read(auth);before=read(previous)
            unchanged={k:v for k,v in before.items() if k!='execution'}=={k:v for k,v in config.items() if k!='execution'}
            if not (unchanged and report['config_sha256']==binding['source_config_before_sha256'] and file_hash(CONFIG)==binding['source_config_after_sha256'] and before['execution']['output_directory']==config['execution']['output_directory']):
                raise ValueError('Source scope changed; do not reuse/rebuild queue silently')
        print(json.dumps(report,ensure_ascii=False));return
    database=out/'PAPERS.sqlite'
    if database.exists():
        with sqlite3.connect(database) as probe:
            if probe.execute("select count(*) from sqlite_master where type='table'").fetchone()[0]:
                raise ValueError('Partial queue exists: inspect/resume it, never overwrite or start a duplicate')
        # A failed opening before schema creation has no committed input rows.
        # Reuse the empty database instead of deleting or duplicating it.
    c=sqlite3.connect(database,uri=True);c.row_factory=sqlite3.Row
    c.execute('pragma journal_mode=WAL')
    c.execute('attach database ? as material',(f'file:{corpus.as_posix()}?mode=ro',))
    c.executescript('''
      create table publications(pub_id text primary key,job_id text,identity_status text,ids_json text,title text);
      create index publication_jobs on publications(job_id);
      create table paper_jobs(job_id text primary key,work_key text,identity_status text,status text,selected_source_sha text);
      create table source_options(job_id text,document_sha text,pub_id text,quality text,word_count integer,primary key(job_id,document_sha));
      create index source_jobs on source_options(job_id,quality,word_count);
      create table potential_versions(abstract_key text,job_id text,primary key(abstract_key,job_id));
      create table extraction_results(job_id text primary key,source_sha text,result_json text,validation_status text,at text);
    ''')
    aliases=defaultdict(set)
    for alias,pid in c.execute('select distinct alias,pub_id from material.owner_aliases'):aliases[alias].add(pid)
    count=0
    # Raw observations establish corpus membership only; their scientific claims
    # and historical review content are never selected into extraction packets.
    query='select p.* from material.publications p where exists(select 1 from material.originals o where o.pub_id=p.pub_id) order by p.pub_id'
    for p in c.execute(query):
        ids=json.loads(p['ids_json']);owners=set();conflict=False
        for kind,value in ids.items():
            found=aliases.get(kind.upper()+':'+value,set());owners.update(found);conflict|=len(found)>1
        conflict|=len(owners)>1 or bool(ids.get('pmid') and owners and 'PMID:'+ids['pmid'] not in owners)
        work=next(iter(owners)) if len(owners)==1 and not conflict else p['pub_id']
        unknown=work.startswith('UNRESOLVED:')
        identity='conflict' if conflict else 'unresolved' if unknown else 'own_record_identifiers' if owners else 'declared_identifiers'
        key=None if unknown or conflict else work
        job='PAPER:'+digest(key or {'unresolved_publication':p['pub_id']})
        c.execute('insert into publications values(?,?,?,?,?)',(p['pub_id'],job,identity,p['ids_json'],p['title']))
        c.execute('insert or ignore into paper_jobs values(?,?,?,?,null)',(job,key,identity,'identity_review' if unknown or conflict else 'source_inventory'))
        count+=1
        if count%50000==0:
            c.commit();dump(out/'STATE.json',{'status':'PREPARING_PAPER_QUEUE','at':now(),'publication_records':count,'extracted_papers':0,'fixed_layer_modified':False,'pid':os.getpid()})
    # The corpus has immutable source snapshots and owning-record identifiers.
    c.execute('insert or ignore into source_options select p.job_id,s.document_sha,s.pub_id,s.quality,s.word_count from publications p join material.sources s using(pub_id)')
    c.execute('insert or ignore into source_options select j.job_id,s.document_sha,s.pub_id,s.quality,s.word_count from paper_jobs j join material.sources s on s.pub_id=j.work_key')
    for job in c.execute('select job_id,identity_status from paper_jobs').fetchall():
        selected=c.execute("select document_sha from source_options where job_id=? order by case quality when 'OWN_RAW_RECORD_CACHE' then 0 else 1 end,word_count desc,document_sha limit 1",(job['job_id'],)).fetchone()
        status='identity_review' if job['identity_status'] in {'conflict','unresolved'} else 'ready_for_source_review' if selected else 'missing_cached_source'
        c.execute('update paper_jobs set status=?,selected_source_sha=? where job_id=?',(status,selected[0] if selected else None,job['job_id']))
    c.execute("insert or ignore into potential_versions select s.abstract_key,o.job_id from source_options o join material.sources s using(document_sha) where s.word_count>=60 and s.abstract_key!='' and s.abstract_key in (select s2.abstract_key from source_options o2 join material.sources s2 using(document_sha) where s2.word_count>=60 and s2.abstract_key!='' group by s2.abstract_key having count(distinct o2.job_id)>1)")
    c.execute("update paper_jobs set status='version_identity_review' where status='ready_for_source_review' and job_id in (select job_id from potential_versions)")
    c.commit();c.execute('pragma wal_checkpoint(TRUNCATE)')
    report={'at':now(),'status':'PAPER_SOURCE_QUEUE_READY','publication_records':count,
      'paper_jobs_before_final_version_resolution':c.execute('select count(*) from paper_jobs').fetchone()[0],
      'status_counts':dict(c.execute('select status,count(*) from paper_jobs group by status')),
      'source_snapshot_references':c.execute('select count(*) from source_options').fetchone()[0],
      'extracted_papers':0,'fixed_layer_modified':False,'fixed_partition_status':'pending_provenance_partition; original graph is read-only',
      'parent_release':None,'legacy_patch_route':'retired','old_claims_used_as_extraction_answers':False,
      'config_sha256':file_hash(CONFIG),'source_corpus':binding,'external_models':0,'agents':0}
    c.close();dump(complete,report);dump(out/'STATE.json',{**report,'active_process':None,'next':'fixed_provenance_partition_and_source_based_extraction_validation'})
    print(json.dumps(report,ensure_ascii=False))

def packet(job_id):
    _,out,corpus,_=settings()
    c=sqlite3.connect(f'file:{(out/"PAPERS.sqlite").as_posix()}?mode=ro',uri=True);c.row_factory=sqlite3.Row
    c.execute('attach database ? as material',(f'file:{corpus.as_posix()}?mode=ro',))
    job=c.execute('select * from paper_jobs where job_id=?',(job_id,)).fetchone()
    if job is None:raise KeyError(job_id)
    selected=c.execute('select payload from material.sources where document_sha=?',(job['selected_source_sha'],)).fetchone()
    raw_source=json.loads(zlib.decompress(selected[0])) if selected else None
    source=source_fields(raw_source) if raw_source else None
    if not c.execute("select 1 from sqlite_master where name='bibliography_refs'").fetchone():
        raise RuntimeError('Build the bibliography reference index once with --bibliography-index before packet access')
    publications=[dict(r) for r in c.execute('select pub_id,identity_status,ids_json,title from publications where job_id=?',(job_id,))]
    bibliographies=[]
    for pub in publications:
        for row in c.execute('select b.bib_sha,b.payload from bibliography_refs r join material.bibliography_versions b on b.bib_sha=r.bib_sha where r.pub_id=? order by r.bib_sha',(pub['pub_id'],)):
            bibliographies.append({'bib_sha256':row['bib_sha'],'pub_id':pub['pub_id'],
                                  'fields':bibliography_fields(json.loads(zlib.decompress(row['payload'])))})
    locations=[dict(r) for r in c.execute('select path,byte_offset,line_sha from material.source_locations where document_sha=?',(job['selected_source_sha'],))]
    answer={'job':dict(job),'publications':publications,'bibliography_versions':bibliographies,
      'source_snapshot':source,'source_sha256':job['selected_source_sha'],'material_level':'abstract' if source else 'unavailable',
      'source_locations':locations,'complete_original_metadata_reference':{'corpus':str(corpus),'document_sha256':job['selected_source_sha']},
      'old_claims_included':False,'fixed_layer_write_permission':False,'scientific_extraction_status':'not_extracted'}
    c.close();return answer

def source_fields(raw):
    """Source-only envelope; cached review/claim annotations stay in archive."""
    allowed={'pmid','doi','pmcid','title','authors','year','journal','abstract_text','abstract',
      'publication_types','language','article_ids','own_article_ids','notices','update_relations',
      'url','source_url','fetched_at','retrieved_at','license','comments_corrections',
      'source_kind','source','original_cache_status','cache_quality'}
    result={k:v for k,v in raw.items() if k in allowed}
    if isinstance(raw.get('paper'),dict):result['paper']=bibliography_fields(raw['paper'])
    return result


def bibliography_fields(raw):
    allowed={'pmid','doi','pmcid','title','authors','year','journal','volume','issue','pages',
             'publication_types','language','url','license','article_ids','own_article_ids',
             'notices','update_relations','comments_corrections'}
    return {k:v for k,v in raw.items() if k in allowed}


def bibliography_index():
    _,out,corpus,binding=settings()
    receipt=out/'BIBLIOGRAPHY_INDEX_READY.json'
    if receipt.exists():return read(receipt)
    c=sqlite3.connect(out/'PAPERS.sqlite',uri=True)
    c.execute('attach database ? as material',(f'file:{corpus.as_posix()}?mode=ro',))
    with c:
        c.execute('create table bibliography_refs(pub_id text,bib_sha text,primary key(pub_id,bib_sha))')
        c.execute('insert into bibliography_refs select pub_id,bib_sha from material.bibliography_versions')
    report={'at':now(),'bibliography_references':c.execute('select count(*) from bibliography_refs').fetchone()[0],
            'source_corpus_sha256':binding['sha256'],'source_corpus_modified':False}
    c.close();dump(receipt,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',help='Print source-only packet for an existing PAPER ID')
    p.add_argument('--bibliography-index',action='store_true')
    a=p.parse_args()
    if a.bibliography_index:print(json.dumps(bibliography_index(),ensure_ascii=False))
    elif a.packet:print(json.dumps(packet(a.packet),ensure_ascii=False,indent=2))
    else:bootstrap()
