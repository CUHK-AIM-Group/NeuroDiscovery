"""Append exact owning PubMed XML identities without weakening the old registry."""
from copy import deepcopy
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_paper_identity import (VerifiedPaperIdentities, authority_title,
    normalize_pmid, normalize_doi, normalize_pmcid)
from neurooracle.src.pubmed_title_evidence import complete_title_evidence
from neurooracle.src.shared_relation_catalog import check_file

VERSION = 'kg.owning_pubmed_extension.r76.v1'
OUTPUT = j.OUTPUT/'round76_multipaper_claims'

def own_record(article, source):
    pmid = normalize_pmid(article.findtext('./MedlineCitation/PMID'))
    require(bool(pmid), 'missing own PMID')
    root = article.find('./MedlineCitation/Article')
    require(root is not None, 'missing own article')
    title = root.find('ArticleTitle')
    require(title is not None, 'missing complete title')
    ids = {'doi':set(), 'pmcid':set()}
    for el in article.findall('./PubmedData/ArticleIdList/ArticleId'):
        kind = el.get('IdType')
        if kind == 'pubmed': require(normalize_pmid(el.text) == pmid, 'own identifier mismatch')
        if kind in ('doi','pmc'):
            field, normalizer = ('doi',normalize_doi) if kind=='doi' else ('pmcid',normalize_pmcid)
            value = normalizer(el.text)
            require(bool(value), 'invalid own identifier'); ids[field].add(value)
    # Publication dates only. Submission/acceptance/indexing history is excluded.
    dates = [*root.findall('./Journal/JournalIssue/PubDate'), *root.findall('./ArticleDate')]
    years = sorted({y for el in dates for y in re.findall(r'\b(?:18|19|20|21)[0-9]{2}\b', ' '.join(el.itertext()))})
    require(bool(years), 'missing publication date')
    types = [''.join(el.itertext()) for el in root.findall('./PublicationTypeList/PublicationType')]
    retractions = [el.get('RefType') for el in article.findall('./MedlineCitation/CommentsCorrectionsList/CommentsCorrections')]
    require(not {'Retracted Publication','Retraction of Publication'} & set(types) and 'RetractionIn' not in retractions,
        'retracted source requires separate publication review')
    journal = root.findtext('./Journal/Title') or ''
    witness = dict(response_file=source['path'], response_sha256=source['sha256'],
        url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='+pmid+'&retmode=xml',
        response_scope='own article within retained batch XML; URL identifies an equivalent single-record retrieval')
    record = dict(pmid=pmid, title=authority_title(''.join(title.itertext())), years=years,
        preprint='Preprint' in types or bool(re.search(r'\b(?:biorxiv|medrxiv|arxiv|research square|preprints?)\b',journal,re.I)),
        **{k:sorted(v) for k,v in ids.items()}, witness=witness)
    record['complete_title_evidence'] = complete_title_evidence(article, record, witness)
    return record

def reconstruct(base, selected, sources):
    result = deepcopy(base)
    requested = set(selected)
    require(not requested & set(base['records']), 'extension overwrites existing authority')
    found = {}
    for fp in sources:
        check_file(fp, full_hash=True)
        for article in ET.parse(fp['path']).getroot().findall('PubmedArticle'):
            pmid = normalize_pmid(article.findtext('./MedlineCitation/PMID'))
            if pmid not in requested: continue
            require(pmid not in found, 'duplicate extension authority')
            found[pmid] = own_record(article, fp)
    require(set(found) == requested, 'extension missing own source')
    result['records'].update(found)
    VerifiedPaperIdentities(result)
    return result

def prepare(selected):
    base = j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    registry = j.read_json(base['current_paper_identities']['path'])
    missing = sorted(set(selected)-set(registry['records']))
    docs = {r['pmid']:r for r in rows(OUTPUT/'PRIMARY_SOURCES.jsonl')}
    source_files = {docs[p]['source']['path']:docs[p]['source'] for p in missing}
    payload = reconstruct(registry, missing, list(source_files.values()))
    j.atomic_json(OUTPUT/'CURRENT_PAPER_IDENTITIES.json',payload)
    manifest = dict(version=VERSION, base_registry=base['current_paper_identities'],
        base_extension=base.get('current_paper_identity_extension'), added_pmids=missing,
        sources=list(source_files.values()), result=j.fingerprint(OUTPUT/'CURRENT_PAPER_IDENTITIES.json'),
        existing_records_and_publication_reviews_preserved=True, references_excluded=True,
        fuzzy_or_title_only_identity_used=False, selected_pmids=sorted(set(selected)))
    j.atomic_json(OUTPUT/'PAPER_IDENTITY_EXTENSION.json',manifest)
    return VerifiedPaperIdentities(payload)

def validate_extension(baseline, manifest_path):
    manifest = j.read_json(manifest_path)
    require(manifest['version'] == VERSION, 'unsupported paper extension')
    for fp in (manifest['base_registry'],manifest['result']): check_file(fp,full_hash=True)
    base_context = dict(baseline,current_paper_identities=manifest['base_registry'])
    if manifest.get('base_extension'):
        fp=manifest['base_extension'];check_file(fp,full_hash=True)
        old,_=validate_extension(base_context,fp['path'])
    else:
        from apply_kg_explicit_pmid_provenance import authorities
        old,_=authorities(base_context)
    reconstructed = reconstruct(old.export_payload(),manifest['added_pmids'],manifest['sources'])
    require(reconstructed == j.read_json(manifest['result']['path']), 'own XML registry reconstruction differs')
    return VerifiedPaperIdentities(reconstructed), manifest

def authorities(baseline):
    return validate_extension(baseline,OUTPUT/'PAPER_IDENTITY_EXTENSION.json')

