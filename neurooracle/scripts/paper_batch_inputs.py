"""Compile the ready source queue once; no model calls or graph writes.

Outputs are provider-independent so a credential or model change never requires
rescanning the graph. Full bibliography/notice metadata stays in the source
archive; compact model inputs are whitelisted and retain exact abstract text.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib
from prepare_paper_reconstruction import dump, now, read, settings, source_fields, stat_matches


def compact_source(work_key, title, raw):
    source=source_fields(raw)
    nested=source.get('paper') or {}
    abstract=source.get('abstract_text') or source.get('abstract')
    types=[]
    for value in (source.get('publication_types'),nested.get('publication_types')):
        if isinstance(value,list):types.extend(value)
        elif value:types.append(value)
    unique={json.dumps(v,sort_keys=True,ensure_ascii=False):v for v in types}
    return {'paper_id':work_key,'title':source.get('title') or nested.get('title') or title,
            'material_level':'abstract','abstract':abstract,
            'publication_types':list(unique.values()),
            'publication_notices':{k:source[k] for k in ('notices','update_relations','comments_corrections') if k in source},
            'nested_publication_notices':{k:nested[k] for k in ('notices','update_relations','comments_corrections') if k in nested}}


def encode_input(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest(),zlib.compress(raw),len(raw)


def compile_inputs():
    _,out,corpus,binding=settings();d=out/'BATCH_MODEL';receipt=d/'INPUTS_READY.json'
    database=d/'INPUTS.sqlite';lock=d/'INPUTS.lock'
    if receipt.exists():
        report=read(receipt)
        if not stat_matches(report['database'],database):
            raise ValueError('Completed source inputs changed; inspect instead of rebuilding')
        return report
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    c=None;s=None
    try:
        protocol=read(d/'EVALUATION_PROTOCOL.json')
        heldout={p['job_id'] for p in protocol['heldout_papers']}
        s=sqlite3.connect(f'file:{(out/"PAPERS.sqlite").as_posix()}?mode=ro',uri=True)
        s.execute('attach database ? as material',(f'file:{corpus.as_posix()}?mode=ro',))
        c=sqlite3.connect(database);c.execute('pragma journal_mode=WAL')
        c.executescript('''
          create table if not exists inputs(job_id text primary key,work_key text,source_sha256 text,
            input_sha256 text,input_zlib blob,utf8_bytes integer,abstract_words integer,status text,fold text);
          create table if not exists metadata(key text primary key,value text);
        ''')
        binding_json=json.dumps({'corpus_sha256':binding['sha256'],'contract_version':1,
                                 'evaluation_protocol_sha256':hashlib.sha256((d/'EVALUATION_PROTOCOL.json').read_bytes()).hexdigest()},sort_keys=True)
        saved=c.execute("select value from metadata where key='binding'").fetchone()
        if saved and saved[0]!=binding_json:raise ValueError('Partial inputs have another source/contract binding')
        c.execute("insert or ignore into metadata values('binding',?)",(binding_json,));c.commit()
        cursor=c.execute('select max(job_id) from inputs').fetchone()[0] or ''
        initial=c.execute('select count(*) from inputs').fetchone()[0]
        query='''select j.job_id,j.work_key,j.selected_source_sha,
          (select title from publications p where p.job_id=j.job_id order by pub_id limit 1),src.payload
          from paper_jobs j left join material.sources src on src.document_sha=j.selected_source_sha
          where j.status='ready_for_source_review' and j.job_id>? order by j.job_id'''
        count=initial
        for job,work,sha,title,blob in s.execute(query,(cursor,)):
            if blob is None:raise ValueError('Ready paper lacks its bound source snapshot')
            raw=json.loads(zlib.decompress(blob));value=compact_source(work,title,raw)
            input_sha,compressed,nbytes=encode_input(value)
            abstract=value['abstract'];status='READY' if isinstance(abstract,str) and abstract.strip() else 'HOLD_EMPTY_ABSTRACT'
            c.execute('insert into inputs values(?,?,?,?,?,?,?,?,?)',(job,work,sha,input_sha,compressed,nbytes,
              len(abstract.split()) if isinstance(abstract,str) else 0,status,'heldout' if job in heldout else 'corpus'))
            count+=1
            if count%2000==0:
                c.commit();dump(d/'INPUT_PROGRESS.json',{'at':now(),'status':'COMPILING','papers':count,'pid':os.getpid(),
                   'model_calls':0,'scientific_extractions':0,'fixed_graph_reads':0})
        c.commit()
        expected=s.execute("select count(*) from paper_jobs where status='ready_for_source_review'").fetchone()[0]
        actual=c.execute('select count(*) from inputs').fetchone()[0]
        if expected!=actual:raise ValueError('Queue coverage changed during compilation')
        report={'at':now(),'status':'SOURCE_INPUTS_READY','papers':actual,
           'states':dict(c.execute('select status,count(*) from inputs group by status')),
           'folds':dict(c.execute('select fold,count(*) from inputs group by fold')),
           'utf8_bytes':c.execute('select sum(utf8_bytes) from inputs').fetchone()[0],
           'abstract_words':c.execute('select sum(abstract_words) from inputs').fetchone()[0],
           'max_input_bytes':c.execute('select max(utf8_bytes) from inputs').fetchone()[0],
           'queue_statuses':dict(s.execute('select status,count(*) from paper_jobs group by status')),
           'source_corpus_sha256':binding['sha256'],'model_calls':0,'scientific_extractions':0,
           'fixed_graph_reads':0,'fixed_layer_modified':False,'metadata_retention':'Full source and all bibliography variants remain referenced in CORPUS.sqlite/PAPERS.sqlite; no old claims in model input.',
           'exclusions':'Previously root-extracted papers, missing sources, identity holds and version holds are not dispatched.',
           'next':'Validate credentials and frozen scientific evaluation before priced whole-corpus dispatch.'}
        c.execute('pragma wal_checkpoint(TRUNCATE)');c.close();c=None
        stat=database.stat();report['database']={'path':str(database),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        dump(receipt,report);dump(d/'INPUT_PROGRESS.json',report)
        return report
    finally:
        if c:c.close()
        if s:s.close()
        lock.unlink()


if __name__=='__main__':
    print(json.dumps(compile_inputs(),ensure_ascii=False))
