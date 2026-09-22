"""Read-only queries over the reconstruction's new source-derived literature."""
from __future__ import annotations
import argparse
import json
import sqlite3
from prepare_paper_reconstruction import settings
from paper_evidence_store import metrics


def connect():
    _,out,_,_=settings()
    return sqlite3.connect(f'file:{(out/"LITERATURE.sqlite").as_posix()}?mode=ro',uri=True)


def inspect_paper(c,identifier):
    work=identifier if ':' in identifier else 'PMID:'+identifier
    rows=c.execute('select job_id,work_key,source_packet_json,extraction_json from papers where work_key=? or job_id=?',(work,identifier)).fetchall()
    return [{'job_id':job,'work_key':wk,'source':json.loads(source),'extraction':json.loads(extraction),
      'claim_links':[dict(observation_id=o,claim_id=cid,statement=stmt,relation=rel,rationale=why,scope=scope)
       for o,cid,stmt,rel,why,scope in c.execute('''select e.observation_id,c.claim_id,c.statement,e.relation,e.rationale,e.scope_match
         from evidence e join claims c using(claim_id) join observations o using(observation_id)
         where o.job_id=? order by o.observation_id,c.claim_id''',(job,))]}
      for job,wk,source,extraction in rows]


def inspect_claim(c,identifier):
    rows=c.execute('select claim_id,statement,definition_json from claims where claim_id=? or claim_id like ?',(identifier,identifier+'%')).fetchall()
    if len(rows)>1:raise ValueError('Ambiguous claim prefix; use a full ID')
    if not rows:return None
    cid,statement,definition=rows[0]
    evidence=[]
    for work,job,role,payload,relation,why in c.execute('''select p.work_key,p.job_id,o.role,o.payload_json,e.relation,e.rationale
      from evidence e join observations o using(observation_id) join papers p using(job_id)
      where e.claim_id=? order by p.work_key,o.observation_id''',(cid,)):
        evidence.append({'work_key':work,'job_id':job,'passage_role':role,'relation':relation,
                         'rationale':why,'observation':json.loads(payload)})
    return {'claim_id':cid,'statement':statement,'definition':json.loads(definition),'evidence':evidence,
            'staging_only':True,'independent_cohorts':'unknown'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--paper');g.add_argument('--claim');g.add_argument('--summary',action='store_true')
    args=p.parse_args();c=connect()
    result=inspect_paper(c,args.paper) if args.paper else inspect_claim(c,args.claim) if args.claim else metrics(c)
    print(json.dumps(result,ensure_ascii=False,indent=2));c.close()
