"""Fetch own public abstracts/fulltexts and a species-specific protein witness."""
from pathlib import Path
from xml.etree import ElementTree as ET
import json
import sys
import requests

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_scoped_structure import source_documents
from neurooracle.src.kg_paper_identity import title_key

OUTPUT=j.OUTPUT/'round49_remaining_scoped_literals'
FULLTEXT_PMIDS={'40448221','37721751','36716140','36990674','36847009','15885483'}


def owning_fulltext_ids(content):
    root=ET.fromstring(content)
    article=root if root.tag=='article' else root.find('article')
    require(article is not None,'no fulltext article')
    return {node.get('pub-id-type'):''.join(node.itertext()).strip()
        for node in article.findall('./front/article-meta/article-id')}


def main():
    require(not (OUTPUT/'PUBLIC_SOURCE_FETCH.json').exists(),'public source manifest already exists')
    inspection_fp=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json');inspection=j.read_json(inspection_fp['path'])
    require(inspection['status']=='INSPECTED_NOT_APPLIED','inspection incomplete')
    fp=inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl'];require(j.fingerprint(fp['path'])==fp,'current projections changed')
    selected=[]
    for row in rows(fp['path']):
        paper=row['source_paper'];pmid=str(paper.get('pmid') or '')
        if pmid.isdigit():selected.append(dict(claim_id=row['claim_id'],claim_sha256=row['claim_sha256'],pmid=pmid,doi=paper.get('doi') or '',
            current_title=paper.get('title') or '',paper_sha256=digest(paper)))
    pmids=sorted({r['pmid'] for r in selected});require(pmids,'no explicit owning PMIDs')
    response=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
        params=dict(db='pubmed',id=','.join(pmids),retmode='xml',tool='NeuroClawKGReview'),timeout=45)
    response.raise_for_status();content=response.content.decode('utf8');docs=source_documents([content])
    require(set(docs)==set(pmids),'public PMID set differs')
    j.atomic_text(OUTPUT/'pubmed_remaining_scope.xml',content)
    pmcs={}
    for article in ET.fromstring(content).findall('PubmedArticle'):
        pmid=article.findtext('./MedlineCitation/PMID')
        for node in article.findall('./PubmedData/ArticleIdList/ArticleId'):
            if node.get('IdType')=='pmc':pmcs[pmid]=node.text
    for row in selected:
        d=docs[row['pmid']]
        row.update(public_title=d['title'],public_dois=d['dois'],public_abstract_sha256=digest(d['abstract']),pmcid=pmcs.get(row['pmid']),
            title_matches=title_key(row['current_title'])==title_key(d['title']),
            doi_matches=not row['doi'] or row['doi'].casefold() in {v.casefold() for v in d['dois']})
    fulltexts=[]
    for pmid in sorted(FULLTEXT_PMIDS):
        pmcid=pmcs.get(pmid)
        if not pmcid:
            fulltexts.append(dict(pmid=pmid,status='NO_OWN_PMC_IDENTIFIER',no_paywall_bypass=True));continue
        try:
            full=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                params=dict(db='pmc',id=pmcid.removeprefix('PMC'),retmode='xml',tool='NeuroClawKGReview'),timeout=45)
            full.raise_for_status();body=full.content.decode('utf8');identifiers=owning_fulltext_ids(body)
            require(identifiers.get('pmid')==pmid,'fulltext owning PMID mismatch')
            if identifiers.get('doi'):require(identifiers['doi'].casefold() in {v.casefold() for v in docs[pmid]['dois']},'fulltext owning DOI mismatch')
            file=OUTPUT/(pmcid+'.xml');j.atomic_text(file,body)
            fulltexts.append(dict(pmid=pmid,pmcid=pmcid,status='PUBLIC_FULLTEXT_FETCHED',identifiers=identifiers,response=j.fingerprint(file),url=full.url))
        except (requests.RequestException,ValueError,ET.ParseError) as error:
            fulltexts.append(dict(pmid=pmid,pmcid=pmcid,status='UNAVAILABLE',error=str(error),no_paywall_bypass=True))
    protein={}
    try:
        response_protein=requests.get('https://rest.uniprot.org/uniprotkb/Q01853.json',timeout=45)
        response_protein.raise_for_status();payload=response_protein.json()
        require(payload.get('primaryAccession')=='Q01853' and payload['organism']['taxonId']==10090,'protein species identity mismatch')
        j.atomic_json(OUTPUT/'uniprot_Q01853.json',payload)
        protein=dict(status='PUBLIC_PROTEIN_WITNESS_FETCHED',response=j.fingerprint(OUTPUT/'uniprot_Q01853.json'),url=response_protein.url,
            accession=payload['primaryAccession'],entry=payload.get('uniProtkbId'),organism=payload['organism'],
            protein_description=payload.get('proteinDescription'),genes=payload.get('genes'),not_a_graph_identity_change=True)
    except (requests.RequestException,ValueError,KeyError) as error:
        protein=dict(status='UNAVAILABLE',error=str(error),not_a_graph_identity_change=True)
    require(j.fingerprint(inspection_fp['path'])==inspection_fp,'source inspection changed')
    j.atomic_json(OUTPUT/'PUBLIC_SOURCE_FETCH.json',dict(status='PUBLIC_SOURCES_FETCHED_NOT_ADJUDICATED',at=j.utc_now(),
        source_inspection=inspection_fp,source_graph=inspection['graph'],selected=selected,
        response=j.fingerprint(OUTPUT/'pubmed_remaining_scope.xml'),url=response.url,fulltexts=fulltexts,protein_witness=protein,
        code=j.fingerprint(Path(__file__)),public_identifiers_only=True,no_claim_suffix_inference=True,graph_modified=False,
        record_preimages_saved=False,registry_not_automatically_upgraded=True))
    print('R49_PUBLIC_SOURCES_FETCHED',dict(claims=len(selected),papers=len(pmids),title_differences=sum(not r['title_matches'] for r in selected),
        doi_differences=sum(not r['doi_matches'] for r in selected),fulltexts=[(r['pmid'],r['status']) for r in fulltexts],protein=protein['status']),flush=True)


if __name__=='__main__':main()
