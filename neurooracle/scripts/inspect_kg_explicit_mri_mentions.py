"""R40 supplemental read-only explicit-MRI review; reuse this turn's source SHA."""
from collections import defaultdict
from pathlib import Path
import sys
import httpx
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact,walk_graph,IncrementalJsonReader
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from fetch_kg_complete_titles import API,own_articles
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round40_scoped_structure'
NAMES={
    'bilateral pulvinar high signal on MRI',
    'contralateral amygdala T2 relaxation signal',
    'leftward dorsolateral prefrontal cortex fMRI lateralization for pleasant words',
    'symmetrical increased T2 signal in posterior or posterior-lateral cervical and thoracic spinal cord columns',
    'frontotemporal diffusion MRI laterality indexes',
    'lateral prefrontal cortex qMRI myelin content',
}


def main():
    require(not (OUTPUT/'MRI_SOURCE_INSPECTION.json').exists(),'supplement already exists')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');prior=j.read_json(OUTPUT/'SOURCE_INSPECTION.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and prior['graph']==c['current_graph'] and prior['source_full_sha_verified'],'no trusted current source')
    require(j.fingerprint(prior['code']['path'])==prior['code'],'prior check code differs')
    held=[r for r in rows(c['current_gene_holds']['path']) if r['name'] in NAMES]
    require(len(held)==6 and {r['name'] for r in held}==NAMES,'six complete mentions required')
    expected={r['claim_id']:r for r in held};claims={};refs=defaultdict(list);matches=[];aliases=[]
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    print('READ_ONLY_SIX_MRI_SOURCE_REVIEW',flush=True)
    with Path(c['current_graph']['path']).open('r',encoding='utf8') as handle:
        for kind,key,row in walk_graph(IncrementalJsonReader(handle)):
            if kind=='node':
                if key in expected:
                    require(digest(row)==expected[key]['claim_sha256'],'current held record differs');claims[key]=row
                if not key.startswith('CLM:'):
                    if name_key(row.get('preferred_name')) in NAMES:matches.append(dict(node_id=key,node_sha256=digest(row),name=name_key(row['preferred_name'])))
                    for a in row.get('aliases') or []:
                        if name_key(a) in NAMES:aliases.append(dict(node_id=key,name=name_key(a)))
            elif kind=='edge' and edge_owner(row) in expected:refs[edge_owner(row)].append((int(key),row))
    require(set(claims)==set(expected),'missing selected claim')
    pmids=sorted({str(r['metadata']['source_paper']['pmid']) for r in claims.values()})
    require(len(pmids)==6 and all(p.isdecimal() and len(p)>=5 for p in pmids),'explicit owning PMIDs required')
    xml_path=OUTPUT/'pubmed_explicit_mri.xml'
    if not xml_path.exists():
        params=dict(db='pubmed',id=','.join(pmids),retmode='xml',tool='NeuroClawKGReview')
        response=httpx.get(API,params=params,timeout=30,follow_redirects=True);response.raise_for_status()
        require(set(own_articles(response.content))==set(pmids),'own public source set differs')
        j.atomic_text(xml_path,response.text)
        j.atomic_json(OUTPUT/'MRI_SOURCE_FETCH.json',dict(at=j.utc_now(),explicit_pmids=pmids,response=j.fingerprint(xml_path),url=str(httpx.URL(API,params=params)),code=code,basis='explicit current source_paper.pmid fields'))
    for cid,r in sorted(claims.items()):
        md=r['metadata'];print('MRI_CLAIM='+compact(dict(id=cid,md=md)),flush=True)
        for i,e in refs[cid]:print('MRI_EDGE='+compact(dict(ordinal=i,row=e)),flush=True)
    for article in ET.parse(xml_path).getroot().findall('PubmedArticle'):
        print('MRI_PUBLIC='+compact(dict(pmid=article.findtext('./MedlineCitation/PMID'),
            title=''.join(article.find('./MedlineCitation/Article/ArticleTitle').itertext()),
            abstract=' '.join(''.join(e.itertext()) for e in article.findall('./MedlineCitation/Article/Abstract/AbstractText')))),flush=True)
    write_rows(OUTPUT/'MRI_CLAIM_SCOPE.jsonl',[dict(claim_id=cid,claim_sha256=digest(r),explicit_pmid=r['metadata']['source_paper']['pmid'],
        references=[dict(ordinal=i,edge_sha256=digest(e)) for i,e in refs[cid]]) for cid,r in sorted(claims.items())])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    j.atomic_json(OUTPUT/'MRI_SOURCE_INSPECTION.json',dict(at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        reused_full_source_sha_boundary=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),source_native_guards_passed=True,
        scope=j.fingerprint(OUTPUT/'MRI_CLAIM_SCOPE.jsonl'),public_fetch=j.fingerprint(OUTPUT/'MRI_SOURCE_FETCH.json'),
        existing_full_name_matches=matches,full_alias_matches=aliases,code=code,graph_modified=False,record_preimages_saved=False))
    print('MRI_REVIEW_COMPLETE',dict(selected=len(claims),existing_names=matches,aliases=aliases),flush=True)


if __name__=='__main__':main()
