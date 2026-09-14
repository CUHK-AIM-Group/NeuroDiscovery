"""Exact supplemental PubMed identity witnesses from existing raw own records."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, pubmed_record


def authority_record(article, source):
    pmid = article.findtext('./MedlineCitation/PMID')
    title_node = article.find('./MedlineCitation/Article/ArticleTitle')
    if not pmid or title_node is None:
        raise ValueError('Own PubMed article identity is incomplete')
    article_ids = [dict(idtype=n.get('IdType'), value=n.text or '')
                   for n in article.findall('./PubmedData/ArticleIdList/ArticleId')]
    article_ids.append(dict(idtype='pubmed', value=pmid))
    pub = article.find('./MedlineCitation/Article/Journal/JournalIssue/PubDate')
    publication_date = ' '.join(pub.itertext()) if pub is not None else ''
    electronic = ' '.join(' '.join(n.itertext()) for n in article.findall('./MedlineCitation/Article/ArticleDate'))
    record = dict(uid=pmid, title=''.join(title_node.itertext()), articleids=article_ids,
                  pubdate=publication_date, epubdate=electronic,
                  pubtype=[n.text for n in article.findall('./MedlineCitation/Article/PublicationTypeList/PublicationType')],
                  fulljournalname=article.findtext('./MedlineCitation/Article/Journal/Title'))
    witness = dict(response_sha256=source['sha256'], response_path=source['path'],
                   source='cached_own_PubmedArticle', own_record_pmid=pmid)
    return pubmed_record(pmid, record, witness)


def extend(payload, docs, *, check_file):
    result = deepcopy(payload)
    wanted = set(docs) - set(result['records'])
    files = {docs[p]['source']['path']: docs[p]['source'] for p in wanted}
    added, unresolved = [], []
    for path, fingerprint in files.items():
        check_file(fingerprint, full_hash=True)
        tree = ET.parse(path)
        for article in tree.getroot().findall('./PubmedArticle'):
            pmid = article.findtext('./MedlineCitation/PMID')
            if pmid not in wanted or docs[pmid]['source']['path'] != path:
                continue
            own_id = (docs[pmid].get('own_article_ids') or {}).get('pubmed')
            if own_id is not None and own_id != pmid:
                unresolved.append(dict(pmid=pmid, reason='cached_own_id_disagrees'));continue
            record = authority_record(article, fingerprint)
            if not record['title'] or not record['years']:
                unresolved.append(dict(pmid=pmid, reason='own_bibliography_incomplete'));continue
            result['records'][pmid] = record
            added.append(pmid)
    VerifiedPaperIdentities(result)
    if any(result['records'][p] != record for p, record in payload['records'].items()):
        raise ValueError('Previously accepted identity record changed')
    return result, dict(added_pmids=sorted(added), unresolved=unresolved,
                       original_records_unchanged=True, new_network_requests=0,
                       raw_authority_files=list(files.values()))
