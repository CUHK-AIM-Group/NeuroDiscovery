"""Complete owning PubMed records for every PMID in the accepted graph.

All existing raw XML is reused once. Requests batch 500 public IDs, use a single
connection at a time, and checkpoint by source, without reading the graph.
Missing abstracts, books and notices retain explicit states; no paper credit is
granted by this process. Credentials are neither needed nor read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import whole_graph_claim_batch_20260914 as work
from extend_cached_kg_paper_authority_20260913 import authority_record

OUT = work.OUT / 'source_completion'
ENDPOINT = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'


def text(node):
    return ''.join(node.itertext()) if node is not None else ''


def own_records(data, witness):
    root = ET.fromstring(data)
    for article in root.findall('./PubmedArticle'):
        pmid = article.findtext('./MedlineCitation/PMID')
        if not pmid:
            continue
        own_ids = {n.get('IdType'): text(n) for n in article.findall('./PubmedData/ArticleIdList/ArticleId')}
        work.index.require(own_ids.get('pubmed', pmid) == pmid, 'Own publication ID disagrees')
        abstract = [dict(label=n.get('Label', ''), text=text(n)) for n in article.findall('./MedlineCitation/Article/Abstract/AbstractText')]
        types = [text(n) for n in article.findall('./MedlineCitation/Article/PublicationTypeList/PublicationType')]
        notices = [dict(ref_type=n.get('RefType'), pmid=n.findtext('PMID'), source=n.findtext('RefSource'))
                   for n in article.findall('./MedlineCitation/CommentsCorrectionsList/CommentsCorrections')]
        doc = dict(pmid=pmid, own_article_ids=own_ids,
                   title=text(article.find('./MedlineCitation/Article/ArticleTitle')),
                   abstract=abstract, publication_types=types, comments_corrections=notices,
                   source=witness, url='https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/')
        authority = authority_record(article, witness)
        state = 'NOTICE_REVIEW_REQUIRED' if notices else ('OWN_RECORD_CACHED' if abstract else 'FULL_TEXT_REQUIRED')
        yield pmid, doc, authority, state
    for article in root.findall('./PubmedBookArticle'):
        own = article.find('BookDocument')
        if own is None or not own.findtext('PMID'):
            continue
        pmid = own.findtext('PMID')
        doc = dict(pmid=pmid, title=text(own.find('ArticleTitle')), source_kind='book_chapter',
                   abstract=[dict(label=n.get('Label', ''), text=text(n)) for n in own.findall('./Abstract/AbstractText')],
                   source=witness)
        yield pmid, doc, None, 'BOOK_IDENTITY_REVIEW_REQUIRED'


def progress(db, phase, out=OUT, **extra):
    counts = dict(db.execute('SELECT state,count(*) FROM papers GROUP BY state'))
    data = dict(status='RUNNING', phase=phase, at=work.index.journal.utc_now(),
                source_states=counts, total=sum(counts.values()),
                scientific_approvals=0, graph_reads=0, **extra)
    work.index.journal.atomic_json(out / 'STATE.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


def ingest(db, data, witness, wanted=None):
    seen = set()
    for pmid, doc, authority, status in own_records(data, witness):
        if wanted is not None and pmid not in wanted:
            continue
        existing = db.execute('SELECT state FROM papers WHERE pmid=?', (pmid,)).fetchone()
        if existing is None:
            continue
        seen.add(pmid)
        # Existing cached source choices are stable; retain later snapshots separately.
        sha = work.index.digest(doc)
        db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?)',
                   (pmid, sha, work.packed(doc), work.packed(authority), witness['path']))
        if existing[0] == 'PENDING':
            db.execute('UPDATE papers SET state=?,document_sha=? WHERE pmid=?', (status, sha, pmid))
    return seen


def prepare(out=OUT):
    out = Path(out)
    work.index.require(not out.exists(), 'Source completion version already exists')
    accepted = work.read(work.index.OUTPUT / 'INDEX_ACCEPTANCE.json')
    work.index.check_file(accepted['database']); work.index.check_file(accepted['graph'])
    out.mkdir(parents=True)
    db = work.connect(out / 'SOURCES.sqlite')
    db.executescript('''
      CREATE TABLE papers(pmid TEXT PRIMARY KEY,state TEXT NOT NULL DEFAULT 'PENDING',document_sha TEXT);
      CREATE INDEX paper_state ON papers(state,pmid);
      CREATE TABLE records(pmid TEXT,document_sha TEXT,payload BLOB,authority BLOB,path TEXT,PRIMARY KEY(pmid,document_sha));
      CREATE TABLE requests(request_id TEXT PRIMARY KEY,state TEXT NOT NULL,pmids TEXT,path TEXT);
      CREATE TABLE cached_files(path TEXT PRIMARY KEY,bytes INTEGER,mtime_ns INTEGER,sha256 TEXT);
    ''')
    with work.connect(accepted['database']['path'], True) as source:
        db.executemany('INSERT INTO papers(pmid) VALUES(?)', [(r[0],) for r in source.execute('SELECT DISTINCT pmid FROM observations WHERE pmid IS NOT NULL ORDER BY pmid')])
    db.commit()
    progress(db, 'REUSE_OWN_RAW_RECORDS', out)
    paths = [ROOT / p for p in subprocess.check_output(
        ['rg', '--files', '--hidden', '--no-ignore', str(work.index.journal.OUTPUT), '-g', '*.xml'],
        cwd=ROOT, text=True).splitlines()]
    for i, path in enumerate(sorted(set(paths)), 1):
        before = path.stat(); data = path.read_bytes(); after = path.stat()
        work.index.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Cached XML changed')
        fp = dict(path=str(path.resolve()), bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=hashlib.sha256(data).hexdigest())
        # PMC full text and non-PubMed XML are not used as PubMed identity records.
        try:
            ingest(db, data, fp)
        except ET.ParseError:
            continue
        db.execute('INSERT INTO cached_files VALUES(?,?,?,?)', (fp['path'], fp['bytes'], fp['mtime_ns'], fp['sha256']))
        if i % 25 == 0:
            db.commit(); progress(db, 'REUSE_OWN_RAW_RECORDS', out, cached_files=i, total_cached_files=len(paths))
    db.commit()
    protocol = dict(schema='kg.whole_graph_own_source_completion.v1', at=work.index.journal.utc_now(),
        code=work.index.fingerprint(Path(__file__)), shared_code=work.index.fingerprint(Path(work.__file__)),
        authority_code=work.index.fingerprint(Path(authority_record.__code__.co_filename)),
        graph=accepted['graph'], input_index=accepted['database'],
        total_pmids=db.execute('SELECT count(*) FROM papers').fetchone()[0],
        endpoint=ENDPOINT, batch_size=500, minimum_request_interval_seconds=0.5,
        parallel_requests=1, request_timeout_seconds=60, transport_attempts_per_batch=2,
        auto_resume_model_calls=False, model_calls=0, source_request_cap=None,
        cached_files=[dict(r) for r in db.execute('SELECT * FROM cached_files ORDER BY path')],
        publication_policy='Source records only; identity/scope/notices/version and scientific adjudication still required.',
        graph_read_policy='No graph reads; accepted index reused with native fingerprint checks.')
    work.index.journal.atomic_json(out / 'PROTOCOL.json', protocol)
    work.index.journal.atomic_json(out / 'PREPARATION_ACCEPTANCE.json', dict(
        status='ALL_GRAPH_PMIDS_QUEUED', protocol=work.index.fingerprint(out / 'PROTOCOL.json'),
        source_states=dict(db.execute('SELECT state,count(*) FROM papers GROUP BY state'))))
    progress(db, 'ALL_SOURCES_QUEUED', out); db.close()


def run(out=OUT):
    out = Path(out); protocol = work.read(out / 'PROTOCOL.json')
    for field in ('code', 'shared_code', 'authority_code'):
        work.index.check_file(protocol[field], full_hash=True)
    work.index.check_file(protocol['input_index']); work.index.check_file(protocol['graph'])
    db = work.connect(out / 'SOURCES.sqlite')
    work.index.require(not db.execute("SELECT 1 FROM requests WHERE state='DISPATCHING'").fetchone(), 'Prior source fetch needs reconciliation')
    started = time.monotonic(); requests = 0; consecutive_failures = 0
    while True:
        if (out / 'STOP_REQUEST.json').exists():
            progress(db, 'PAUSED_AFTER_DURABLE_BATCH', out); break
        pmids = [r[0] for r in db.execute("SELECT pmid FROM papers WHERE state='PENDING' ORDER BY pmid LIMIT ?", (protocol['batch_size'],))]
        if not pmids:
            result = progress(db, 'SOURCE_PASS_COMPLETE', out, elapsed_seconds=round(time.monotonic()-started, 2))
            result['status'] = 'COMPLETE_SOURCE_PASS_WITH_EXPLICIT_HOLDS'
            work.index.journal.atomic_json(out / 'STATE.json', result)
            work.index.journal.atomic_json(out / 'SOURCE_PASS_ACCEPTANCE.json', result)
            break
        request_id = work.index.digest(dict(endpoint=ENDPOINT, pmids=pmids, retmode='xml'))
        folder = out / 'requests' / request_id; folder.mkdir(parents=True, exist_ok=False)
        work.index.journal.atomic_json(folder / 'REQUEST.json', dict(endpoint=ENDPOINT, pmids=pmids, at=work.index.journal.utc_now()))
        db.execute('INSERT INTO requests VALUES(?,?,?,?)', (request_id, 'DISPATCHING', json.dumps(pmids), str(folder)))
        db.commit(); attempts = []; response = None
        for attempt in range(protocol['transport_attempts_per_batch']):
            if requests:
                time.sleep(protocol['minimum_request_interval_seconds'])
            requests += 1
            query = urllib.parse.urlencode(dict(db='pubmed', id=','.join(pmids), retmode='xml', tool='NeuroClawSourceReview'))
            request = urllib.request.Request(ENDPOINT, data=query.encode(), headers={
                'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': 'NeuroClawSourceReview/2.0'})
            try:
                with urllib.request.urlopen(request, timeout=protocol['request_timeout_seconds']) as handle:
                    response = handle.read()
                # Validate complete XML before marking any requested source complete.
                ET.fromstring(response)
                attempts.append(dict(attempt=attempt + 1, state='RESPONSE_CAPTURED', bytes=len(response)))
                break
            except Exception as exc:
                response = None
                attempts.append(dict(attempt=attempt + 1, state='READ_ONLY_FETCH_FAILED', error_type=type(exc).__name__, http_status=getattr(exc, 'code', None)))
                if attempt + 1 < protocol['transport_attempts_per_batch']:
                    time.sleep(2)
        work.index.journal.atomic_json(folder / 'ATTEMPTS.json', attempts)
        if response is None:
            consecutive_failures += 1
            db.executemany("UPDATE papers SET state='FETCH_ERROR_RETRYABLE' WHERE pmid=? AND state='PENDING'", [(p,) for p in pmids])
            db.execute("UPDATE requests SET state='FETCH_ERROR_RETRYABLE' WHERE request_id=?", (request_id,)); db.commit()
            progress(db, 'SOURCE_FETCH_ERROR_RETAINED', out, request_id=request_id)
            # A provider outage should not send hundreds of repeated failing requests.
            if consecutive_failures >= 3:
                result = progress(db, 'BLOCKED_PUBLIC_SOURCE_TRANSPORT', out)
                result['status'] = 'BLOCKED_PUBLIC_SOURCE_TRANSPORT'
                work.index.journal.atomic_json(out / 'STATE.json', result); break
            continue
        consecutive_failures = 0
        path = folder / 'OWN_PUBMED.xml'; path.write_bytes(response)
        fp = dict(path=str(path.resolve()), bytes=len(response), mtime_ns=path.stat().st_mtime_ns, sha256=hashlib.sha256(response).hexdigest())
        found = ingest(db, response, fp, set(pmids))
        missing = sorted(set(pmids) - found)
        db.executemany("UPDATE papers SET state='NOT_RETURNED_BY_AUTHORITY' WHERE pmid=? AND state='PENDING'", [(p,) for p in missing])
        db.execute("UPDATE requests SET state='COMPLETE' WHERE request_id=?", (request_id,)); db.commit()
        work.index.journal.atomic_json(folder / 'RECEIPT.json', dict(status='OWN_SOURCE_RECORDS_BOUND', source=fp,
            requested=len(pmids), returned=len(found), not_returned=missing, scientific_approvals=0))
        progress(db, 'FETCH_ALL_REMAINING_OWN_SOURCES', out, requests_this_run=requests,
                 elapsed_seconds=round(time.monotonic()-started, 2))
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'all'])
    parser.add_argument('--out', type=Path, default=OUT); args = parser.parse_args()
    if args.command in {'prepare', 'all'}:
        prepare(args.out)
    if args.command in {'run', 'all'}:
        run(args.out)
