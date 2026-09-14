"""Cache owning NCBI publication records for the R73 source review."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest

OUTPUT=j.OUTPUT/'round73_systematic_relation_consolidation'


def text(element):return ''.join(element.itertext()) if element is not None else ''


def parsed(path):
    result=[];root=ET.parse(path).getroot();fp=j.fingerprint(path)
    for article in root.findall('PubmedArticle'):
        pmid=article.findtext('./MedlineCitation/PMID')
        ids={e.get('IdType'):text(e) for e in article.findall('./PubmedData/ArticleIdList/ArticleId')}
        require(ids.get('pubmed',pmid)==pmid,'owning PMID mismatch')
        result.append(dict(pmid=pmid,own_article_ids=ids,
            title=text(article.find('./MedlineCitation/Article/ArticleTitle')),
            abstract=[dict(label=e.get('Label',''),text=text(e)) for e in article.findall('./MedlineCitation/Article/Abstract/AbstractText')],
            publication_types=[text(e) for e in article.findall('./MedlineCitation/Article/PublicationTypeList/PublicationType')],
            mesh_terms=[text(e) for e in article.findall('./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName')],
            comments_corrections=[dict(ref_type=e.get('RefType'),pmid=e.findtext('PMID'),source=e.findtext('RefSource')) for e in article.findall('./MedlineCitation/CommentsCorrectionsList/CommentsCorrections')],
            source=fp,url='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/'))
    return result


def fetch(ids):
    folder=OUTPUT/'source_review';folder.mkdir(exist_ok=True)
    existing={r['pmid']:r for r in rows(OUTPUT/'PRIMARY_SOURCES.jsonl')} if (OUTPUT/'PRIMARY_SOURCES.jsonl').exists() else {}
    base=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    prior=Path(base['current_acceptance']['path']).parent/'source_review'
    for path in sorted(prior.glob('PRIMARY_PUBMED_*.xml')):
        for r in parsed(path):existing.setdefault(r['pmid'],r)
    for path in sorted(folder.glob('NCBI_*.xml')):
        for r in parsed(path):existing.setdefault(r['pmid'],r)
    missing=sorted(set(ids)-set(existing));artifacts=[]
    for start in range(0,len(missing),25):
        batch=missing[start:start+25];path=folder/('NCBI_'+digest(batch)[:16]+'.xml')
        if not path.exists():
            query=urllib.parse.urlencode(dict(db='pubmed',id=','.join(batch),retmode='xml',tool='NeuroClaw_source_review'))
            request=urllib.request.Request('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+query,headers={'User-Agent':'NeuroClawSourceReview/1.0'})
            with urllib.request.urlopen(request,timeout=50) as response:payload=response.read()
            ET.fromstring(payload);path.write_bytes(payload)
        entries=parsed(path);returned={r['pmid'] for r in entries}
        require(returned<=set(batch),'unexpected article in requested source response')
        existing.update({r['pmid']:r for r in entries});artifacts.append(j.fingerprint(path))
        write_rows(OUTPUT/'PRIMARY_SOURCES.jsonl',[existing[k] for k in sorted(existing)])
        print(json.dumps(dict(phase='PRIMARY_RECORDS',requested=len(batch),returned=len(entries),remaining=max(0,len(missing)-start-25))),flush=True)
        time.sleep(.4)
    write_rows(OUTPUT/'PRIMARY_SOURCES.jsonl',[existing[k] for k in sorted(existing)])
    receipt=dict(at=j.utc_now(),requested=sorted(set(ids)),available=len(set(ids)&set(existing)),missing=sorted(set(ids)-set(existing)),
        all_ids_from_owning_pubmed_records=True,reference_article_ids_excluded=True,sources=artifacts,
        parsed=j.fingerprint(OUTPUT/'PRIMARY_SOURCES.jsonl'),code=j.fingerprint(Path(__file__)))
    j.atomic_json(OUTPUT/'PRIMARY_FETCH_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ids_file');args=p.parse_args()
    print(json.dumps(fetch(j.read_json(Path(args.ids_file))),ensure_ascii=False),flush=True)
