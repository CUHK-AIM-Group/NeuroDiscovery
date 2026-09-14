"""One complete, reusable retrieval index over the accepted claim observations.

This is a separately versioned continuation, not a graph/claim approval. Every
claim is census-bound; byte locations allow subsequent reviews to read complete
observations without another graph scan. Retrieval never grants paper credit.
"""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import os
import re
import sqlite3
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kg_overnight_report as journal
from project_kg_systematic_review import record_at, science
from discover_kg_broad_multipaper_leads import EXPANSIONS, WORDS, PHRASES
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import identifiers
from neurooracle.src.relation_evidence import name_key
from neurooracle.src.shared_relation_catalog import check_file

OUTPUT = journal.OUTPUT / 'global_claim_aggregation_20260913'
CLAIMS = re.compile(rb'"(CLM:[^"\\]+)":\{')
PHRASE_MAP = dict(PHRASES)
PHRASE_RE = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(PHRASE_MAP, key=len, reverse=True)) + r')(?!\w)')
METHODS = {
    'literal_surface': 'surface_key',
    'lexical_endpoint_and_predicate': 'lexical_key',
    'lexical_endpoint_predicate_omitted': 'scope_key',
    'endpoint_ids_predicate_omitted': 'endpoint_key',
    'matching_extracted_sentence': 'sentence_key',
    'existing_relation': 'rid',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'file changed while binding')
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=h.hexdigest())


def progress(phase, **values):
    state = dict(schema='kg.global_claim_index.state.v1', status='RUNNING', phase=phase,
                 at=journal.utc_now(), pid=os.getpid(), **values)
    journal.atomic_json(OUTPUT / 'STATE.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


@lru_cache(maxsize=200000)
def lexical_name(value):
    """Ambiguous expansion is a retrieval feature, never an identity decision."""
    text = unicodedata.normalize('NFKC', value).casefold().replace('’', "'")
    text = re.sub(r"\b(alzheimer|parkinson)'s\b", r'\1', text)
    text = ' '.join(re.sub(r'[-‐‑–—/]', ' ', text).split())
    text = PHRASE_RE.sub(lambda match: PHRASE_MAP[match[0]], text)
    tokens = []
    for token in re.findall(r'[^\W_]+', text):
        tokens.extend(EXPANSIONS.get(token, token).split())
    return tuple(sorted(WORDS.get(t, t) for t in tokens if t not in ('the', 'a', 'an')))


def retrieval_keys(md):
    sn, pred, on = (name_key(md.get(k)) for k in ('subject_name', 'predicate', 'object_name'))
    neg = md.get('negated')
    surface = digest([sn, pred, on, neg]) if sn and pred and on else None
    ns, no = lexical_name(sn), lexical_name(on)
    lexical = digest([ns, pred, no, neg]) if ns and pred and no else None
    scope = digest([ns, no, neg]) if ns and no else None
    sid, oid = md.get('subject_id'), md.get('object_id')
    endpoint = digest([sid, oid, neg]) if sid and oid else None
    raw = ' '.join(unicodedata.normalize('NFKC', str(md.get('raw_text') or '')).casefold().split())
    sentence = digest([raw, pred, neg]) if len(raw) >= 80 else None
    return surface, lexical, scope, endpoint, sentence


def create_schema(db):
    db.executescript('''
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        PRAGMA cache_size=-65536;
        CREATE TABLE observations(
            number INTEGER PRIMARY KEY, cid TEXT NOT NULL UNIQUE, node_sha TEXT NOT NULL,
            byte_offset INTEGER NOT NULL, rid TEXT NOT NULL, paper_sig TEXT NOT NULL,
            source_key TEXT, pmid TEXT, doi TEXT, pmcid TEXT,
            subject_id TEXT, subject_name TEXT, subject_type TEXT, predicate TEXT,
            object_id TEXT, object_name TEXT, object_type TEXT, negated_json TEXT,
            conditions_sha TEXT, population_sha TEXT, evidence_sha TEXT,
            surface_key TEXT, lexical_key TEXT, scope_key TEXT, endpoint_key TEXT, sentence_key TEXT
        );
        CREATE TABLE candidates(
            candidate_id TEXT PRIMARY KEY, method TEXT NOT NULL, retrieval_key TEXT NOT NULL,
            observation_count INTEGER NOT NULL, source_key_count INTEGER NOT NULL,
            current_relation_count INTEGER NOT NULL, missing_source_count INTEGER NOT NULL,
            decision TEXT NOT NULL DEFAULT 'OWN_SOURCE_AND_EQUIVALENCE_REVIEW_REQUIRED',
            UNIQUE(method,retrieval_key)
        );
        CREATE TABLE queue_entries(
            queue TEXT NOT NULL, entry_id TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(queue,entry_id)
        );
    ''')


def import_prior_queues(db, base):
    window = journal.OUTPUT / 'multipaper_12h_20260913'
    paths = [window / 'EXISTING_SHARED_REVIEW_QUEUE.json', window / 'NEXT_BATCH_METADATA_QUEUE.json',
             window / 'shared_review_source_prefetch/PREFETCH_RECEIPT.json']
    bound = []
    for path in paths:
        payload = journal.read_json(path)
        require(payload['graph'] == base['current_graph'], 'queue belongs to another graph')
        bound.append(fingerprint(path))
    shared, remaining, prefetched = (journal.read_json(p) for p in paths)
    entries = shared['entries']
    db.executemany('INSERT INTO queue_entries VALUES (?,?,?)',
                   [('existing_shared_source_review', e['shared_claim_id'], json.dumps(e, ensure_ascii=False)) for e in entries])
    # Import every ready and held scope: the old 150-item display is not a limit.
    counts = {}
    for name in ('proposed_review_groups', 'other_ready_groups', 'held_metadata_groups'):
        rows = prefetched[name]
        db.executemany('INSERT INTO queue_entries VALUES (?,?,?)',
                       [('source_cache_' + name, e['shared_claim_id'], json.dumps(e, ensure_ascii=False)) for e in rows])
        counts[name] = len(rows)
    # Preserve the complete prior queue payload even if its entry schema changes.
    db.execute('INSERT INTO queue_entries VALUES (?,?,?)',
               ('prior_endpoint_queue', 'complete_queue', json.dumps(remaining, ensure_ascii=False)))
    db.commit()
    return dict(inputs=bound, existing_shared_review_scopes=len(entries), source_cache=counts,
                prior_endpoint_queue_complete=True, cached_sources_are_unreviewed=True)


def build_catalog(db):
    stats = {}
    for method, column in METHODS.items():
        progress('BUILD_COMPLETE_RETRIEVAL_CATALOG', method=method)
        db.execute(f'CREATE INDEX observations_{column} ON observations({column})')
        rows = db.execute(f'''
            SELECT {column},COUNT(*),COUNT(DISTINCT source_key),COUNT(DISTINCT rid),
                   SUM(source_key IS NULL)
            FROM observations WHERE {column} IS NOT NULL GROUP BY {column}
            HAVING COUNT(*) >= 2 AND COUNT(DISTINCT source_key) >= 2
        ''').fetchall()
        db.executemany('INSERT INTO candidates(candidate_id,method,retrieval_key,observation_count,source_key_count,current_relation_count,missing_source_count) VALUES (?,?,?,?,?,?,?)',
                       [('GLOBAL:' + digest([method, key]), method, key, n, p, r, missing) for key, n, p, r, missing in rows])
        db.commit()
        stats[method] = dict(scopes=len(rows), member_occurrences=sum(r[1] for r in rows),
                             scopes_larger_than_60=sum(r[1] > 60 for r in rows),
                             largest_scope=max((r[1] for r in rows), default=0))
    db.execute('CREATE INDEX observations_source ON observations(source_key)')
    db.execute('CREATE INDEX candidates_rank ON candidates(current_relation_count,source_key_count)')
    db.commit()
    # Every member remains addressable. No transitive union of overlapping leads.
    conditions = ' OR '.join(f'o.{column} IN (SELECT retrieval_key FROM candidates WHERE method=\'{method}\')'
                             for method, column in METHODS.items())
    covered = db.execute(f'SELECT COUNT(*) FROM observations o WHERE {conditions}').fetchone()[0]
    return dict(methods=stats, unique_observations_with_cross_source_retrieval_lead=covered,
                observations_without_current_retrieval_match=db.execute('SELECT COUNT(*) FROM observations').fetchone()[0] - covered,
                scientific_approval_count=0, oversized_scopes_excluded=0,
                caveat='Source-key differences are unverified retrieval counts, not distinct reviewed works. No-match is limited to these retrieval methods; it is not proof of absent evidence.')


def build():
    require(not OUTPUT.exists(), 'versioned output already exists; inspect its state instead of overwriting')
    OUTPUT.mkdir(parents=True)
    started = time.monotonic()
    base = journal.read_json(journal.OUTPUT / 'CAMPAIGN.json')
    require(base['status'] == 'COMPLETED' and base['active_process'] is None, 'accepted graph is not stable')
    check_file(base['current_graph'])
    for key in ('current_acceptance', 'current_paper_census', 'current_paper_identities'):
        check_file(base[key], full_hash=True)
    census = journal.read_json(base['current_paper_census']['path'])
    require(census['graph'] == base['current_graph'], 'census does not bind current graph')
    check_file(census['database'])
    journal.atomic_json(OUTPUT / 'SOURCE_BASELINE.json', base)
    code_files = [Path(__file__), Path(record_at.__code__.co_filename),
                  Path(sys.modules['discover_kg_broad_multipaper_leads'].__file__),
                  Path(sys.modules['neurooracle.src.kg_identity_pilot'].__file__),
                  Path(sys.modules['neurooracle.src.kg_paper_identity'].__file__),
                  Path(sys.modules['neurooracle.src.relation_evidence'].__file__)]
    code = [fingerprint(p) for p in code_files]
    journal.atomic_json(OUTPUT / 'CONTRACT.json', dict(
        schema='kg.global_claim_aggregation.v1', at=journal.utc_now(), goal='Increase source-reviewed, distinct-paper support for the same scientific claim.',
        complete_input_claims=base['counts']['claims'], input_graph=base['current_graph'],
        scope='All current observations, all retrieval scopes irrespective of size, and all prior unfinished review queues.',
        approval_requires=['Complete original-relation membership', 'Source-owning text and publication identity',
                           'Equivalent proposition with qualifiers preserved', 'Version deduplication and source-role review'],
        limits=dict(review_batch_cap=None, retrieval_scope_size_cap=None, automatic_scientific_approvals=0),
        full_record_preimages_saved=False, code=code,
        integrity='All claim record hashes compared with accepted independent census; graph and census database fully hashed once at the index sealing boundary. Routine status checks use atomic state and native fingerprints.'))
    source = sqlite3.connect(Path(census['database']['path']).as_uri() + '?mode=ro', uri=True)
    bindings = {cid: (seal, rid, sig) for cid, seal, rid, sig in source.execute('SELECT cid,node_sha,relation_id,paper_sig FROM claims')}
    source.close()
    db = sqlite3.connect(OUTPUT / 'GLOBAL_CLAIM_INDEX.sqlite', uri=True)
    create_schema(db)
    queue = import_prior_queues(db, base)
    seen = set()
    counts = Counter()
    batch = []
    progress('INDEX_ALL_CURRENT_OBSERVATIONS', claims=0, total=len(bindings))
    with open(base['current_graph']['path'], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        begin, end = mm.find(b',"concepts":{'), mm.find(b'},"edges":[')
        require(0 <= begin < end, 'accepted graph layout changed')
        for number, match in enumerate(CLAIMS.finditer(mm, begin, end), 1):
            cid = match[1].decode()
            require(cid in bindings and cid not in seen, 'missing or duplicate census claim')
            seal, rid, sig = bindings[cid]
            offset = match.end() - 1
            record = record_at(mm, offset)
            require(record['id'] == cid and digest(record) == seal, 'claim record differs from independent census: ' + cid)
            md = record['metadata']
            ids = identifiers(md.get('source_paper'))
            source_key = next((kind + ':' + ids[kind] for kind in ('pmid', 'doi', 'pmcid') if ids.get(kind)), None)
            counts['source_' + (source_key.split(':', 1)[0] if source_key else 'unresolved')] += 1
            counts['missing_endpoint_id'] += int(not md.get('subject_id') or not md.get('object_id'))
            inner = md.get('metadata') or {}
            def field(key):
                value = md.get(key)
                return value if value not in (None, '', [], {}) else inner.get(key)
            row = (number, cid, seal, offset, rid, sig, source_key, ids.get('pmid'), ids.get('doi'), ids.get('pmcid'),
                   md.get('subject_id'), md.get('subject_name'), json.dumps(field('subject_type'), ensure_ascii=False), md.get('predicate'),
                   md.get('object_id'), md.get('object_name'), json.dumps(field('object_type'), ensure_ascii=False), json.dumps(md.get('negated')),
                   digest(field('conditions')), digest(field('population')), digest(md.get('evidence')), *retrieval_keys(md))
            batch.append(row)
            seen.add(cid)
            if len(batch) >= 5000:
                db.executemany('INSERT INTO observations VALUES (' + ','.join('?' for _ in row) + ')', batch)
                db.commit()
                batch.clear()
                progress('INDEX_ALL_CURRENT_OBSERVATIONS', claims=number, total=len(bindings), elapsed_seconds=round(time.monotonic() - started, 1))
        if batch:
            db.executemany('INSERT INTO observations VALUES (' + ','.join('?' for _ in batch[0]) + ')', batch)
            db.commit()
    require(seen == bindings.keys() and len(seen) == base['counts']['claims'], 'full claim enumeration failed')
    del bindings, seen
    lexical_name.cache_clear()
    catalog = build_catalog(db)
    progress('VERIFY_COMPLETE_INDEX_AND_INPUT_BOUNDARY')
    require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite index integrity failure')
    db.execute('ATTACH DATABASE ? AS census', (Path(census['database']['path']).as_uri() + '?mode=ro',))
    require(db.execute('SELECT COUNT(*) FROM observations o JOIN census.claims c ON o.cid=c.cid WHERE o.node_sha<>c.node_sha OR o.rid<>c.relation_id OR o.paper_sig<>c.paper_sig').fetchone()[0] == 0, 'index binding mismatch')
    require(db.execute('SELECT COUNT(*) FROM census.claims c LEFT JOIN observations o ON o.cid=c.cid WHERE o.cid IS NULL').fetchone()[0] == 0, 'census claims omitted')
    db.execute('DETACH DATABASE census')
    db.row_factory = sqlite3.Row
    with (OUTPUT / 'ALL_RETRIEVAL_SCOPES.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in db.execute('SELECT * FROM candidates ORDER BY method,source_key_count DESC,candidate_id'):
            stream.write(json.dumps(dict(row), ensure_ascii=False) + '\n')
    total_sources = db.execute('SELECT COUNT(DISTINCT source_key) FROM observations').fetchone()[0]
    db.close()
    check_file(base['current_graph'], full_hash=True)
    check_file(census['database'], full_hash=True)
    require(base == journal.read_json(journal.OUTPUT / 'CAMPAIGN.json'), 'accepted graph advanced during index build')
    for fp in code:
        check_file(fp, full_hash=True)
    receipt = dict(schema='kg.global_claim_index.acceptance.v1', at=journal.utc_now(),
                   status='COMPLETE_RETRIEVAL_INDEX_NOT_SCIENTIFIC_APPROVAL', graph=base['current_graph'],
                   input_acceptance=base['current_acceptance'], independent_census=census['database'],
                   all_current_claims_indexed=base['counts']['claims'], all_claim_hashes_verified=True,
                   normalized_source_keys=total_sources, source_coverage=dict(counts), catalog=catalog,
                   prior_queues=queue, graph_mutations=0, newly_approved_shared_claims=0,
                   original_graph_and_observations_retained=True, raw_record_preimages_saved=False,
                   source_full_sha_verified=True, census_full_sha_verified=True,
                   elapsed_seconds=round(time.monotonic() - started, 2), code=code,
                   database=fingerprint(OUTPUT / 'GLOBAL_CLAIM_INDEX.sqlite'),
                   catalog_file=fingerprint(OUTPUT / 'ALL_RETRIEVAL_SCOPES.jsonl'))
    journal.atomic_json(OUTPUT / 'INDEX_ACCEPTANCE.json', receipt)
    journal.atomic_json(OUTPUT / 'STATE.json', dict(status='COMPLETED', phase=receipt['status'], at=journal.utc_now(),
                        claims=receipt['all_current_claims_indexed'], elapsed_seconds=receipt['elapsed_seconds'],
                        retrieval_catalog=catalog, newly_approved_shared_claims=0))
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def review_packet(candidate_id):
    """Read a whole candidate plus complete existing relation closure by offset."""
    accepted = journal.read_json(OUTPUT / 'INDEX_ACCEPTANCE.json')
    check_file(accepted['graph'])
    check_file(accepted['database'])
    require(accepted['graph'] == journal.read_json(journal.OUTPUT / 'CAMPAIGN.json')['current_graph'], 'index is stale')
    db = sqlite3.connect(Path(accepted['database']['path']).as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    candidate = db.execute('SELECT * FROM candidates WHERE candidate_id=?', (candidate_id,)).fetchone()
    require(candidate is not None, 'candidate not found')
    column = METHODS[candidate['method']]
    selected = db.execute(f'SELECT * FROM observations WHERE rid IN (SELECT DISTINCT rid FROM observations WHERE {column}=?) ORDER BY cid', (candidate['retrieval_key'],)).fetchall()
    db.close()
    members = []
    with open(accepted['graph']['path'], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for row in selected:
            record = record_at(mm, row['byte_offset'])
            require(record['id'] == row['cid'] and digest(record) == row['node_sha'], 'review location no longer matches')
            members.append(science(record, row['rid']))
    check_file(accepted['graph'])
    require(accepted['graph'] == journal.read_json(journal.OUTPUT / 'CAMPAIGN.json')['current_graph'], 'graph advanced during packet read')
    return dict(candidate=dict(candidate), complete_current_relation_closure=True, observations=members,
                scientific_approval=False, graph=accepted['graph'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'packet'))
    parser.add_argument('--candidate-id')
    args = parser.parse_args()
    if args.action == 'packet':
        print(json.dumps(review_packet(args.candidate_id), ensure_ascii=False))
    else:
        try:
            build()
        except Exception as exc:
            if OUTPUT.exists() and not (OUTPUT / 'INDEX_ACCEPTANCE.json').exists():
                journal.atomic_json(OUTPUT / 'STATE.json', dict(status='FAILED', at=journal.utc_now(), error=str(exc)))
            raise
