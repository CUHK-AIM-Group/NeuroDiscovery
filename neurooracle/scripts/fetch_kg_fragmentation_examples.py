"""Fetch only named primary-source review cases; do not add papers to the KG."""
from pathlib import Path
from xml.etree import ElementTree as ET
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_identity_pilot import digest

OUTPUT=j.OUTPUT/'round68_fragmentation_root_census'
WANTED={'20641291','20641338','15056502','17023001','26122586','30234890','27137745','35648832',
        '41992767','32853481','33190123','28169089','41076152','30445512','30819549'}


def documents(content):
    out={}
    for article in ET.fromstring(content):
        book=article.tag=='PubmedBookArticle'
        holder=article.find('BookDocument' if book else 'MedlineCitation')
        if holder is None:continue
        pmid=holder.findtext('PMID')
        body=holder if book else holder.find('Article')
        if not pmid or body is None:continue
        title=body.find('ArticleTitle')
        if title is None and book:title=body.find('./Book/BookTitle')
        require(title is not None,'primary title missing')
        text=' '.join(' '.join(''.join(x.itertext()).split()) for x in body.findall('./Abstract/AbstractText'))
        own_ids=article.findall('./PubmedBookData/ArticleIdList/ArticleId' if book else './PubmedData/ArticleIdList/ArticleId')
        dois=sorted({x.text for x in own_ids if x.get('IdType')=='doi' and x.text})
        types=sorted({''.join(x.itertext()) for x in body.findall('./PublicationTypeList/PublicationType')})
        out[pmid]=dict(title=''.join(title.itertext()),abstract=text,dois=dois,publication_types=types,
            source_kind='book_chapter' if book else 'journal_record',pmid=pmid)
    return out


def main():
    require(not (OUTPUT/'PRIMARY_CASE_SOURCES.json').exists(),'primary case receipt frozen')
    found={};bindings={};failures=[]
    for path in sorted((j.OUTPUT/'round66_relation_semantic_review').glob('pubmed*.xml')):
        for pmid,doc in documents(path.read_text(encoding='utf8')).items():
            if pmid in WANTED and pmid not in found:
                fp=j.fingerprint(path);bindings[str(path)]=fp;found[pmid]=dict(**doc,response=fp)
    missing=sorted(WANTED-set(found));path=OUTPUT/'primary_case_pubmed.xml'
    if missing:
        if path.exists():content=path.read_text(encoding='utf8')
        else:
            content=None
            for attempt in range(2):
                try:
                    response=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                        params=dict(db='pubmed',id=','.join(missing),retmode='xml',tool='NeuroClawKGReview'),timeout=35)
                    response.raise_for_status();content=response.content.decode('utf8')
                    require(set(documents(content))<=set(missing),'unexpected primary source ID')
                    j.atomic_text(path,content);break
                except requests.RequestException as exc:
                    failures.append(dict(attempt=attempt+1,error=type(exc).__name__))
        if content:
            fp=j.fingerprint(path);bindings[str(path)]=fp
            for pmid,doc in documents(content).items():
                if pmid in WANTED:found[pmid]=dict(**doc,response=fp)
    # The cache contains source XML; this receipt contains hashes/locators, not
    # copied full abstracts or any authority upgrade for current KG records.
    receipt={pmid:{k:v for k,v in d.items() if k!='abstract'}|dict(abstract_sha256=digest(d['abstract'])) for pmid,d in found.items()}
    j.atomic_json(OUTPUT/'PRIMARY_CASE_SOURCES.json',dict(at=j.utc_now(),status='FETCHED_NOT_BLANKET_SCIENTIFIC_APPROVAL',
        requested=sorted(WANTED),found=receipt,missing=sorted(WANTED-set(found)),bindings=list(bindings.values()),failures=failures,
        code=j.fingerprint(Path(__file__)),kg_modified=False,paper_registry_modified=False,model_calls=0))
    print('PRIMARY_CASE_SOURCES',dict(found=len(found),missing=sorted(WANTED-set(found))),flush=True)


if __name__=='__main__':main()
