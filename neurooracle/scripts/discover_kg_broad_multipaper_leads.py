"""Full-graph retrieval for the next batch; lexical matches never approve claims.

Runs read-only against the accepted source while R78 builds a different file.
All selected observations must be rebound and fully reviewed after R78 adoption.
Only scientific projections, not graph or full-record preimages, are retained.
"""
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import json, mmap, os, re, sqlite3, sys, unicodedata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from project_kg_systematic_review import record_at, science
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import normalize_pmid
from neurooracle.src.shared_relation_catalog import check_file

OUTPUT=j.OUTPUT/'multipaper_12h_20260912/broad_semantic_batch'
BASE=j.OUTPUT/'round78_multipaper_semantics'
CLAIMS=re.compile(rb'"(CLM:[^"\\]+)":\{')

# Retrieval vocabulary only. Short forms can be ambiguous; source review must
# reject a wrong expansion. Keep negation, laterality, modality, age and subtype.
EXPANSIONS={
 'adhd':'attention deficit hyperactivity disorder','ocd':'obsessive compulsive disorder',
 'ptsd':'posttraumatic stress disorder','scz':'schizophrenia','sz':'schizophrenia',
 'asd':'autism spectrum disorder','mdd':'major depressive disorder','bd':'bipolar disorder',
 'pd':'parkinson disease','ad':'alzheimer disease','als':'amyotrophic lateral sclerosis',
 'dlb':'dementia with lewy bodies','ftd':'frontotemporal dementia','bvftd':'behavioral variant frontotemporal dementia',
 'mci':'mild cognitive impairment','amci':'amnestic mild cognitive impairment',
 'ssri':'selective serotonin reuptake inhibitor','ssris':'selective serotonin reuptake inhibitor',
 'snri':'serotonin norepinephrine reuptake inhibitor','snris':'serotonin norepinephrine reuptake inhibitor',
 'tca':'tricyclic antidepressant','tcas':'tricyclic antidepressant',
 'ect':'electroconvulsive therapy','cbt':'cognitive behavioral therapy',
 'dbs':'deep brain stimulation','rtms':'repetitive transcranial magnetic stimulation',
 'tdcs':'transcranial direct current stimulation','vns':'vagus nerve stimulation',
 'tms':'transcranial magnetic stimulation','stn':'subthalamic nucleus',
 'gpcr':'g protein coupled receptor','nmda':'n methyl d aspartate',
 'nmdar':'n methyl d aspartate receptor','nmdars':'n methyl d aspartate receptor',
 'bdnf':'brain derived neurotrophic factor','nfl':'neurofilament light chain',
 'dmn':'default mode network','dlpfc':'dorsolateral prefrontal cortex',
 'vmpfc':'ventromedial prefrontal cortex','pfc':'prefrontal cortex',
 'acc':'anterior cingulate cortex','pcc':'posterior cingulate cortex',
 'ofc':'orbitofrontal cortex','vta':'ventral tegmental area',
 'fa':'fractional anisotropy','gmv':'gray matter volume','gm':'gray matter','wm':'white matter',
 'wmh':'white matter hyperintensity','wmhs':'white matter hyperintensity',
 'fc':'functional connectivity','rsfc':'resting state functional connectivity',
 'dfc':'dynamic functional connectivity','fmri':'functional magnetic resonance imaging',
 'mri':'magnetic resonance imaging','pet':'positron emission tomography',
 'csf':'cerebrospinal fluid','mmse':'mini mental state examination','moca':'montreal cognitive assessment',
 'panss':'positive and negative syndrome scale',
}
WORDS={
 'behaviour':'behavior','behavioural':'behavioral','ageing':'aging','grey':'gray',
 'disorders':'disorder','diseases':'disease','inhibitors':'inhibitor','receptors':'receptor',
 'antidepressants':'antidepressant','antipsychotics':'antipsychotic','symptoms':'symptom',
 'volumes':'volume','levels':'level','concentrations':'concentration','networks':'network',
 'hyperintensities':'hyperintensity','differences':'difference','gaps':'gap',
 'decreased':'reduced','lower':'reduced','reduction':'reduced','reductions':'reduced',
 'increased':'elevated','higher':'elevated','elevation':'elevated','elevations':'elevated',
 'post':'post','traumatic':'traumatic',
}
PHRASES=[
 ('social phobia','social anxiety disorder'),('major depression','major depressive disorder'),
 ('post traumatic stress','posttraumatic stress'),('brainpad','brain predicted age difference'),
 ('brain pad','brain predicted age difference'),('brain age gap','brain predicted age difference'),
 ('brainage gap','brain predicted age difference'),
 ('amyloid beta pet imaging','amyloid beta pet'),('amyloid β','amyloid beta'),('β amyloid','amyloid beta'),
 ('aβ','amyloid beta'),('α synuclein','alpha synuclein'),
 ('selective serotonin re uptake','selective serotonin reuptake'),
]


def normalized(text):
    text=unicodedata.normalize('NFKC',str(text)).casefold().replace('’',"'")
    text=re.sub(r"\b(alzheimer|parkinson)'s\b",r'\1',text)
    text=re.sub(r'[-‐‑–—/]',' ',text)
    text=' '.join(text.split())
    for before,after in PHRASES:text=re.sub(r'(?<!\w)'+re.escape(before)+r'(?!\w)',after,text)
    tokens=re.findall(r'[^\W_]+',text)
    expanded=[]
    for token in tokens:expanded.extend(EXPANSIONS.get(token,token).split())
    return tuple(sorted(WORDS.get(t,t) for t in expanded if t not in ('the','a','an')))


def progress(phase,**values):
    state=dict(status='RUNNING',phase=phase,pid=os.getpid(),at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'DISCOVERY_STATE.json',state)
    print(json.dumps(state,ensure_ascii=False),flush=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    require(not (OUTPUT/'RETRIEVAL_ACCEPTANCE.json').exists(),'retrieval already completed')
    contract=j.read_json(j.OUTPUT/'multipaper_12h_20260912/CONTRACT.json')
    require(datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline']),'window ended')
    base=j.read_json(BASE/'SOURCE_BASELINE.json')
    require(base['current_graph']==j.read_json(j.OUTPUT/'CAMPAIGN.json')['current_graph'],'accepted source advanced')
    check_file(base['current_graph']);check_file(base['current_acceptance'],full_hash=True)
    census=j.read_json(base['current_paper_census']['path']);check_file(census['database'])
    con=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    bindings={cid:(seal,rid) for cid,seal,rid in con.execute('SELECT cid,node_sha,relation_id FROM claims')};con.close()
    active_scopes=[set(g['member_ids']) for g in rows(BASE/'GLOBAL_MULTIPAPER_PROPOSALS.jsonl')]
    cached={r['pmid']:r for r in rows(BASE/'PRIMARY_SOURCES.jsonl')}
    buckets=defaultdict(list);seen=set();features={};text_buckets=defaultdict(list)
    progress('ALL_CURRENT_CLAIMS')
    with open(base['current_graph']['path'],'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        begin=mm.find(b',"concepts":{');end=mm.find(b'},"edges":[')
        require(0<=begin<end,'source layout differs')
        for index,match in enumerate(CLAIMS.finditer(mm,begin,end),1):
            cid=match[1].decode();offset=match.end()-1;record=record_at(mm,offset)
            require(record['id']==cid and cid in bindings and cid not in seen,'claim census differs')
            seen.add(cid);md=record['metadata'];pmid=normalize_pmid(md['source_paper'].get('pmid'))
            names=tuple(md[k] for k in ('subject_name','predicate','object_name'))
            key=(normalized(names[0]),names[1],normalized(names[2]))
            buckets[key].append(cid)
            features[cid]=(pmid,bindings[cid][1],offset,names,md['negated'])
            raw=' '.join(unicodedata.normalize('NFKC',str(md.get('raw_text') or '')).casefold().split())
            if 80<=len(raw)<=1600:text_buckets[(digest(raw),names[1])].append(cid)
            if index%100000==0:progress('ALL_CURRENT_CLAIMS',claims=index,lexical_keys=len(buckets),sentence_keys=len(text_buckets))
        require(seen==bindings.keys(),'incomplete current census')
        candidates={};large=[]
        def add(kind,key,members):
            cids=tuple(sorted(members));pmids={features[c][0] for c in cids}-{None,''}
            rids={features[c][1] for c in cids};variants=sorted({features[c][3] for c in cids})
            if len(pmids)<2 or len(rids)<2 or len(variants)<2:return
            if any(set(cids)<=scope for scope in active_scopes):return
            if len(cids)>100:
                large.append(dict(kind=kind,key_digest=digest(key),observations=len(cids),source_pmids=len(pmids),decision='FULL_SCOPE_EXCEEDS_THIS_REVIEW_BATCH'))
                return
            if cids in candidates:
                candidates[cids]['retrieval_methods'].append(kind);return
            candidates[cids]=dict(probe_id='BROAD:'+digest([kind,key])[:16],retrieval_methods=[kind],
                retrieval_key=key,member_ids=list(cids),source_pmids=sorted(pmids),current_relations=sorted(rids),
                variants=variants,negation_status=dict(Counter(str(features[c][4]) for c in cids)),
                cached_pmids=sorted(pmids&cached.keys()),approved=False,
                priority=min(len(pmids),8)*3+len(pmids&cached.keys())*2-len(variants),
                owning_source_review_required=True,complete_existing_relation_closure_required=True,
                must_rebase_after_R78=True)
        for key,members in buckets.items():add('LEXICAL_EXPANSION_AND_WORD_ORDER',key,members)
        for key,members in text_buckets.items():add('EXACT_NORMALIZED_EVIDENCE_SENTENCE',key,members)
        selected=sorted(candidates.values(),key=lambda r:(-r['priority'],r['probe_id']))
        chosen={cid for g in selected for cid in g['member_ids']};projected=[]
        progress('BOUND_SELECTED_SCIENTIFIC_FIELDS',groups=len(selected),observations=len(chosen))
        for cid in sorted(chosen):
            record=record_at(mm,features[cid][2]);require(digest(record)==bindings[cid][0],'selected source seal differs')
            projected.append(science(record,bindings[cid][1]))
    check_file(base['current_graph'],full_hash=True)
    require(base['current_graph']==j.read_json(j.OUTPUT/'CAMPAIGN.json')['current_graph'],'source changed during retrieval')
    write_rows(OUTPUT/'BROAD_LEADS.jsonl',selected)
    write_rows(OUTPUT/'INITIAL_SCIENTIFIC_PROJECTIONS.jsonl',projected)
    write_rows(OUTPUT/'PRIMARY_SOURCES.jsonl',[cached[p] for p in sorted(cached)])
    write_rows(OUTPUT/'LARGE_SCOPES_NOT_PARTLY_SELECTED.jsonl',large)
    j.atomic_json(OUTPUT/'SOURCE_BASELINE.json',base)
    ids=sorted({p for g in selected for p in g['source_pmids']})
    proof=dict(status='RETRIEVAL_ONLY_NOT_APPROVED',at=j.utc_now(),graph=base['current_graph'],
        all_current_claims_enumerated=len(seen),groups=len(selected),selected_observations=len(chosen),source_pmids=ids,
        cached_pmids=sorted(set(ids)&cached.keys()),scientific_projection=j.fingerprint(OUTPUT/'INITIAL_SCIENTIFIC_PROJECTIONS.jsonl'),
        leads=j.fingerprint(OUTPUT/'BROAD_LEADS.jsonl'),source_sha_verified=True,graph_mutations=0,approved_groups=0,
        must_rebase_after_R78_adoption=True,full_record_preimages_saved=False,code=j.fingerprint(Path(__file__)))
    j.atomic_json(OUTPUT/'RETRIEVAL_ACCEPTANCE.json',proof)
    j.atomic_json(OUTPUT/'DISCOVERY_STATE.json',dict(status='COMPLETED',at=j.utc_now(),groups=len(selected),source_pmids=len(ids)))
    print(json.dumps(dict(groups=len(selected),source_pmids=len(ids),cached=len(set(ids)&cached.keys()),large_scopes=len(large))),flush=True)


if __name__=='__main__':main()
