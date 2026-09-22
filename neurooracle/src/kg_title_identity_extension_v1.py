"""Append complete owning-title evidence without rewriting accepted identities.

Only an absent ``complete_title_evidence`` field may be added to an existing
record. All previous fields, publication reviews and collision guards remain
exactly equal. Raw XML is independently checked before the new proof is used.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import xml.etree.ElementTree as ET

from .pubmed_title_evidence import complete_title_evidence, validated_title_keys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def without_title_evidence(record):
    return {k:deepcopy(v) for k,v in record.items() if k != 'complete_title_evidence'}


def require_record_extension(previous, current):
    if previous == current:
        return
    require('complete_title_evidence' not in previous, 'Previously accepted title evidence changed')
    require(set(current) == set(previous) | {'complete_title_evidence'},
            'Only an owning complete-title witness may be appended')
    require(without_title_evidence(current) == previous,
            'Previously accepted identity fields changed')
    validated_title_keys(current)


def require_monotonic_registry(previous, current):
    require(isinstance(previous.get('records'),dict) and isinstance(current.get('records'),dict),
            'Invalid authority registry')
    require({k:v for k,v in current.items() if k != 'records'} ==
            {k:v for k,v in previous.items() if k != 'records'},
            'Previously accepted registry metadata changed')
    for pmid, old_record in previous['records'].items():
        require(pmid in current['records'], 'Previously accepted authority record removed')
        require_record_extension(old_record,current['records'][pmid])


class OwningTitleWitnesses:
    """Bounded raw-XML cache shared by one source-identity extension operation."""

    def __init__(self, files, check_file):
        self.files = {}
        for fp in files:
            require(fp['path'] not in self.files or self.files[fp['path']] == fp,
                    'Conflicting owning-title file fingerprints')
            self.files[fp['path']] = fp
        self.check_file = check_file
        self.checked = set()
        self.used = {}
        self.verified = set()

    def _file(self, witness):
        fp = self.files.get(witness.get('response_path'))
        require(fp is not None and fp['sha256'] == witness.get('response_sha256'),
                'Complete-title witness has no declared owning raw file')
        path = self.check_file(fp,full_hash=fp['path'] not in self.checked)
        self.checked.add(fp['path'])
        self.used[fp['path']] = fp
        return path

    @lru_cache(maxsize=4)
    def _articles(self, path):
        root = ET.parse(path).getroot()
        result = {}
        for article in root:
            if article.tag == 'PubmedArticle':
                pmid = article.findtext('./MedlineCitation/PMID')
            elif article.tag == 'PubmedBookArticle':
                pmid = article.findtext('./BookDocument/PMID')
            else:
                continue
            if not pmid:
                continue
            require(pmid not in result, 'Repeated owning PMID in raw title file')
            result[pmid] = article
        return result

    def article(self, pmid, witness):
        path = self._file(witness)
        article = self._articles(path).get(pmid)
        require(article is not None, 'Owning PMID missing from raw title file')
        self.check_file(self.files[witness['response_path']],full_hash=False)
        return article

    def verify(self, record):
        evidence = record.get('complete_title_evidence')
        require(evidence is not None, 'Missing complete-title evidence')
        validated_title_keys(record)
        witness = evidence['witness']
        # Even a cached proof checks the frozen raw file's current fingerprint.
        self._file(witness)
        key = hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,
                                       separators=(',',':')).encode()).hexdigest()
        if key in self.verified:
            return
        article = self.article(record['pmid'],witness)
        require(complete_title_evidence(article,record,witness) == evidence,
                'Complete-title evidence differs from its own raw XML')
        self.verified.add(key)

    def require_authority(self, record, accepted_records):
        """Verify new raw authorities or additive proof on an accepted record."""
        from neurooracle.scripts.extend_cached_kg_paper_authority_20260913 import authority_record
        if record.get('complete_title_evidence') is not None:
            self.verify(record)
        witness = record.get('witness') or {}
        if witness.get('response_path') in self.files:
            article = self.article(record['pmid'],witness)
            if article.tag == 'PubmedArticle' and authority_record(article,self.files[witness['response_path']]) == without_title_evidence(record):
                return
        previous = accepted_records.get(record['pmid'])
        require(previous is not None and record.get('complete_title_evidence') is not None,
                'New authority differs from its owning PubMed record')
        require_record_extension(previous,record)

    @property
    def inputs(self):
        return list(self.used.values())
