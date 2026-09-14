"""R56 own fulltexts for duplicate and bilateral-measurement follow-up."""
from pathlib import Path
from xml.etree import ElementTree as ET
import sys
import requests

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from fetch_kg_remaining_scope_sources import owning_fulltext_ids
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_scoped_structure import source_documents

OUTPUT=j.OUTPUT/'round56_scientific_definition_review'
SOURCE=j.OUTPUT/'round49_remaining_scoped_literals/PUBLIC_SOURCE_FETCH.json'
PMIDS={'42300138','20735996','38223081'}


def own_pmcs(content,required):
    found={}
    for article in ET.fromstring(content).findall('PubmedArticle'):
        pmid=article.findtext('./MedlineCitation/PMID')
        if pmid not in required:continue
        require(pmid not in found,'duplicate own PMID')
        ids=[n.text for n in article.findall('./PubmedData/ArticleIdList/ArticleId') if n.get('IdType')=='pmc']
        require(len(ids)<=1 and all(str(v).startswith('PMC') and str(v)[3:].isdigit() for v in ids),'invalid own PMCID')
        found[pmid]=ids[0] if ids else None
    require(set(found)==set(required),'explicit own PMID coverage differs')
    return found


def main():
    require(not (OUTPUT/'PUBLIC_SOURCE_FETCH.json').exists(),'public evidence exists')
    manifest_fp=j.fingerprint(SOURCE);source=j.read_json(SOURCE)
    require(j.fingerprint(source['response']['path'])==source['response'],'public abstract evidence changed')
    selected=[r for r in source['selected'] if r['pmid'] in PMIDS]
    require({r['pmid'] for r in selected}==PMIDS,'claims do not provide all explicit own PMIDs')
    content=Path(source['response']['path']).read_text(encoding='utf8')
    docs=source_documents([content]);pmcs=own_pmcs(content,PMIDS);code=j.fingerprint(Path(__file__))
    records=[]
    for pmid,pmcid in sorted(pmcs.items()):
        if pmcid is None:
            records.append(dict(pmid=pmid,status='NO_OWN_PMC_IDENTIFIER',no_paywall_bypass=True));continue
        try:
            r=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                params=dict(db='pmc',id=pmcid[3:],retmode='xml',tool='NeuroClawKGReview'),timeout=30)
            r.raise_for_status();body=r.content.decode('utf8');ids=owning_fulltext_ids(body)
            require(ids.get('pmid')==pmid,'wrong owning PMID')
            require(ids.get('doi','').casefold() in {d.casefold() for d in docs[pmid]['dois']},'owning DOI differs')
            file=OUTPUT/(pmcid+'.xml');j.atomic_text(file,body)
            records.append(dict(pmid=pmid,pmcid=pmcid,status='PUBLIC_FULLTEXT_FETCHED',identifiers=ids,response=j.fingerprint(file),url=r.url))
        except (requests.RequestException,ValueError,ET.ParseError) as error:
            records.append(dict(pmid=pmid,pmcid=pmcid,status='UNAVAILABLE',error=str(error),no_paywall_bypass=True))
    require(j.fingerprint(SOURCE)==manifest_fp and j.fingerprint(Path(__file__))==code,'source/code advanced')
    result=dict(status='PUBLIC_SOURCES_FETCHED_NOT_ADJUDICATED',at=j.utc_now(),prior_public_manifest=manifest_fp,
        original_abstract_response=source['response'],selected=selected,fulltexts=records,code=code,
        graph_modified=False,registry_not_automatically_upgraded=True,no_claim_suffix_inference=True,record_preimages_saved=False)
    j.atomic_json(OUTPUT/'PUBLIC_SOURCE_FETCH.json',result)
    print([(r['pmid'],r['status']) for r in records],flush=True)


if __name__=='__main__':main()
