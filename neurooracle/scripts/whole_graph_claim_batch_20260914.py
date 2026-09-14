"""One durable workset for the whole accepted graph, with no review-count cap.

This is retrieval and work scheduling, never scientific approval. Existing
observations, accepted decisions, full scopes and source variants stay intact.
The graph is not scanned to rebuild an index that already exists. Review packets
reuse a persistent, hash-checked original-record cache, so each original is read
from the graph at most once by this continuation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import mmap
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import index_kg_global_claims_20260913 as index
from neurooracle.src.kg_paper_identity import identifiers

OUT = index.OUTPUT / 'whole_graph_batch_v1_20260914'
OLD = ROOT / 'tmp/kg_group_review_local_v2'
PREVIOUS = ROOT / 'tmp/kg_group_review_batch_20260914'
PHRASES = index.PHRASE_MAP
EXPANSIONS, WORDS = index.EXPANSIONS, index.WORDS
# Only retrieval stop terms; they never rewrite a scientific claim.
GENERIC = set('effect effects treatment treatments patient patients disorder disease '
              'symptom symptoms level levels change changes study studies outcome '
              'outcomes function functions group groups control controls increased '
              'decreased elevated reduced risk association activity mechanism '
              'mechanisms response responses measure measures performance'.split())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def packed(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode(), 1)


def unpacked(value):
    return json.loads(zlib.decompress(value))


def connect(path, readonly=False):
    con = sqlite3.connect(Path(path).resolve().as_uri() + ('?mode=ro' if readonly else ''), uri=True)
    con.row_factory = sqlite3.Row
    if not readonly:
        con.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-131072;')
    return con


@lru_cache(maxsize=200000)
def ordered_terms(text):
    value = unicodedata.normalize('NFKC', str(text or '')).casefold().replace('’', "'")
    value = re.sub(r"\b(alzheimer|parkinson)'s\b", r'\1', value)
    value = ' '.join(re.sub(r'[-‐‑–—/]', ' ', value).split())
    value = index.PHRASE_RE.sub(lambda m: PHRASES[m[0]], value)
    tokens = []
    for token in re.findall(r'[^\W_]+', value):
        tokens.extend(EXPANSIONS.get(token, token).split())
    return tuple(WORDS.get(t, t) for t in tokens if t not in {'the', 'a', 'an'})


def find_spans(tokens, vocabulary):
    """Longest attested endpoint spans; all original qualifiers stay in storage."""
    found = []
    for start in range(len(tokens)):
        for end in range(min(len(tokens), start + 8), start, -1):
            span = tokens[start:end]
            if span in vocabulary:
                if not any(left <= start and end <= right for left, right, _ in found):
                    found.append((start, end, span))
                break
    # No vocabulary match still has an exact full endpoint retrieval path.
    if not found:
        return {tokens} if tokens else set()
    return {span for _, _, span in found}


def schema(db):
    db.executescript('''
      CREATE TABLE observations(number INTEGER PRIMARY KEY,cid TEXT UNIQUE NOT NULL,
        node_sha TEXT NOT NULL,byte_offset INTEGER NOT NULL,rid TEXT NOT NULL,
        source_key TEXT,pmid TEXT,doi TEXT,pmcid TEXT,subject_name TEXT,predicate TEXT,
        object_name TEXT,negated_json TEXT,surface_key TEXT,lexical_key TEXT,scope_key TEXT,
        endpoint_key TEXT,sentence_key TEXT);
      CREATE INDEX observations_rid ON observations(rid);
      CREATE INDEX observations_source ON observations(source_key);
      CREATE TABLE scopes(scope_id TEXT PRIMARY KEY,method TEXT NOT NULL,label TEXT,
        state TEXT NOT NULL DEFAULT 'PENDING_SOURCE_CHECK',observation_count INTEGER DEFAULT 0,
        source_count INTEGER DEFAULT 0,cached_source_count INTEGER DEFAULT 0,
        previous_state TEXT,request_id TEXT);
      CREATE TABLE members(scope_id TEXT NOT NULL,number INTEGER NOT NULL,
        PRIMARY KEY(scope_id,number));
      CREATE INDEX members_number ON members(number,scope_id);
      CREATE INDEX scope_state ON scopes(state,method);
      CREATE TABLE source_documents(document_sha TEXT PRIMARY KEY,payload BLOB NOT NULL,
        abstract_sha TEXT NOT NULL,title TEXT NOT NULL,quality TEXT NOT NULL);
      CREATE TABLE source_keys(source_key TEXT NOT NULL,document_sha TEXT NOT NULL,
        PRIMARY KEY(source_key,document_sha));
      CREATE TABLE source_locations(document_sha TEXT NOT NULL,path TEXT NOT NULL,
        byte_offset INTEGER NOT NULL,line_sha TEXT NOT NULL,
        PRIMARY KEY(document_sha,path,byte_offset));
      CREATE TABLE files(path TEXT PRIMARY KEY,bytes INTEGER,mtime_ns INTEGER,sha256 TEXT,
        input_rows INTEGER,relevant_rows INTEGER);
      CREATE TABLE originals(cid TEXT PRIMARY KEY,node_sha TEXT NOT NULL,payload BLOB NOT NULL);
      CREATE TABLE original_reads(cid TEXT PRIMARY KEY,at TEXT NOT NULL);
      CREATE TABLE prior_decisions(scope_id TEXT PRIMARY KEY,ledger TEXT NOT NULL,decision BLOB NOT NULL);
      CREATE TABLE old_packets(scope_id TEXT PRIMARY KEY,packet BLOB NOT NULL);
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE observation_work(number INTEGER PRIMARY KEY,state TEXT NOT NULL);
      CREATE TABLE requests(request_id TEXT PRIMARY KEY,state TEXT NOT NULL,path TEXT NOT NULL);
    ''')


def state(out, phase, **extra):
    item = dict(status='RUNNING', phase=phase, at=index.journal.utc_now(),
                scientific_approvals=0, graph_mutations=0) | extra
    index.journal.atomic_json(Path(out) / 'STATE.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def corpus_files():
    names = ['papers.jsonl', 'abstract_cache.jsonl', 'abstracts_ready_for_extraction.jsonl',
             'PRIMARY_SOURCES.jsonl', 'OWN_PRIMARY_RECORDS.jsonl']
    args = ['rg', '--files', '--hidden', '--no-ignore', 'neurooracle/data']
    for name in names:
        args.extend(['-g', name])
    paths = []
    for name in subprocess.check_output(args, cwd=ROOT, text=True).splitlines():
        path = (ROOT / name).resolve()
        parts = path.relative_to(ROOT).parts
        if any(p == 'archive' or p.startswith('rejected_') for p in parts):
            continue
        paths.append(path)
    return sorted(set(paths))


def source_document(row):
    if not isinstance(row, dict):
        return None
    paper = dict(row.get('paper') or {})
    paper.update({k: v for k, v in row.items() if k in {
        'pmid', 'pmcid', 'doi', 'title', 'year', 'journal', 'authors', 'publication_types'}})
    abstract = row.get('abstract')
    if isinstance(abstract, list):
        text = ' '.join(str(a.get('text') or '') for a in abstract if isinstance(a, dict))
    elif isinstance(abstract, str):
        text = abstract
    else:
        return None
    if not text.strip():
        return None
    ids = identifiers(paper)
    keys = [k + ':' + ids[k] for k in ('pmid', 'doi', 'pmcid') if ids.get(k)]
    if not keys:
        return None
    # Flags from old extraction inputs do not confer verified paper identity.
    quality = 'OWN_RAW_RECORD_CACHE' if isinstance(abstract, list) and row.get('source') else 'HISTORICAL_ABSTRACT_CACHE'
    document = dict(paper=paper, abstract=abstract, abstract_text=text,
                    own_article_ids=row.get('own_article_ids'),
                    publication_types=row.get('publication_types', paper.get('publication_types', [])),
                    comments_corrections=row.get('comments_corrections'),
                    source_kind=row.get('source_kind'), source=row.get('source'),
                    original_cache_status=row.get('status'), cache_quality=quality)
    return keys, document


def ingest_sources(db, paths, out):
    wanted = {r[0] for r in db.execute('SELECT DISTINCT source_key FROM observations WHERE source_key IS NOT NULL')}
    # Include lower-priority aliases, without assuming those aliases establish identity.
    for field in ('pmid', 'doi', 'pmcid'):
        wanted.update(field + ':' + r[0] for r in db.execute(f'SELECT DISTINCT {field} FROM observations WHERE {field} IS NOT NULL'))
    for number, path in enumerate(paths, 1):
        before = path.stat()
        h = hashlib.sha256(); n = kept = offset = 0
        with path.open('rb') as handle:
            for line in handle:
                h.update(line); n += 1
                if line.strip():
                    parsed = source_document(json.loads(line))
                    if parsed and wanted.intersection(parsed[0]):
                        keys, document = parsed
                        sha = index.digest(document)
                        db.execute('INSERT OR IGNORE INTO source_documents VALUES(?,?,?,?,?)',
                                   (sha, packed(document), index.digest(document['abstract_text']),
                                    document['paper'].get('title') or '', document['cache_quality']))
                        db.executemany('INSERT OR IGNORE INTO source_keys VALUES(?,?)', [(key, sha) for key in keys])
                        db.execute('INSERT OR IGNORE INTO source_locations VALUES(?,?,?,?)',
                                   (sha, str(path), offset, hashlib.sha256(line).hexdigest()))
                        kept += 1
                offset += len(line)
        after = path.stat()
        index.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source cache changed during ingestion')
        db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)', (str(path), after.st_size, after.st_mtime_ns, h.hexdigest(), n, kept))
        db.commit()
        if number % 10 == 0 or number == len(paths):
            state(out, 'IMPORT_ALL_RELEVANT_SOURCE_CACHES', files=number, total_files=len(paths))


def import_existing(db, accepted):
    catalog = read(OLD / 'CATALOG.json')
    ids = {r['cid']: r['number'] for r in db.execute('SELECT number,cid FROM observations')}
    old_states = {r['group_id']: r['state'] for r in connect(OLD / 'QUEUE.sqlite', True).execute('SELECT group_id,state FROM groups')}
    for item in catalog:
        previous = old_states.get(item['group_id'], item['disposition'])
        current = 'REVIEW_PENDING' if previous == 'PENDING' else ('PRESERVED_DECISION' if previous in {'PUBLISHED', 'HELD'} else 'PENDING_SOURCE_CHECK')
        db.execute('INSERT INTO scopes(scope_id,method,label,state,previous_state) VALUES(?,?,?,?,?)',
                   (item['group_id'], 'existing_' + item['method'], item['candidate_id'], current, previous))
        db.executemany('INSERT INTO members VALUES(?,?)', [(item['group_id'], ids[cid]) for cid in item['claim_ids']])
    for ledger in (OLD / 'LOCAL_DECISIONS.json', PREVIOUS / 'LOCAL_DECISIONS.json'):
        content = read(ledger)
        decisions = content.get('decisions', content)
        for gid, decision in decisions.items():
            if not str(gid).startswith('GROUP:'):
                continue
            db.execute('INSERT INTO prior_decisions VALUES(?,?,?)', (gid, str(ledger), packed(decision)))
    with (OLD / 'GROUP_PACKETS.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            packet = json.loads(line)
            db.execute('INSERT INTO old_packets VALUES(?,?)', (packet['group_id'], packed(packet)))
    db.commit()
    return old_states


def retrieve_spans(db, out):
    # The vocabulary comes from observed endpoints in multiple source records.
    # It is a retrieval dictionary, not a new ontology or a claim-identity rule.
    first_source, repeated = {}, set()
    for row in db.execute('SELECT subject_name,object_name,source_key FROM observations'):
        for side in ('subject_name', 'object_name'):
            terms = ordered_terms(row[side])
            if not terms or len(terms) > 8 or (len(terms) == 1 and (terms[0] in GENERIC or terms[0].isnumeric())):
                continue
            source = row['source_key']
            if source and terms in first_source and first_source[terms] != source:
                repeated.add(terms)
            elif source:
                first_source[terms] = source
    del first_source
    state(out, 'RETRIEVE_ALL_OBSERVATIONS', endpoint_vocabulary=len(repeated), processed=0)
    db.executescript('CREATE TABLE span_proposals(scope_id TEXT,number INTEGER,subject TEXT,object TEXT,PRIMARY KEY(scope_id,number));')
    pending = []
    for number, row in enumerate(db.execute('SELECT number,subject_name,object_name FROM observations ORDER BY number'), 1):
        subjects = find_spans(ordered_terms(row['subject_name']), repeated)
        objects = find_spans(ordered_terms(row['object_name']), repeated)
        for s in sorted(subjects):
            for o in sorted(objects):
                sid = 'SPAN:' + index.digest([s, o])
                pending.append((sid, row['number'], ' '.join(s), ' '.join(o)))
        if number % 10000 == 0:
            db.executemany('INSERT OR IGNORE INTO span_proposals VALUES(?,?,?,?)', pending)
            db.commit(); pending.clear()
        if number % 100000 == 0:
            state(out, 'RETRIEVE_ALL_OBSERVATIONS', endpoint_vocabulary=len(repeated), processed=number)
    if pending:
        db.executemany('INSERT OR IGNORE INTO span_proposals VALUES(?,?,?,?)', pending); db.commit()
    db.executescript('''
      INSERT INTO scopes(scope_id,method,label)
        SELECT s.scope_id,'attested_endpoint_spans',json_object('subject',s.subject,'object',s.object)
        FROM span_proposals s JOIN observations o ON o.number=s.number
        GROUP BY s.scope_id HAVING count(DISTINCT o.source_key)>=2;
      INSERT INTO members SELECT s.scope_id,s.number FROM span_proposals s JOIN scopes c USING(scope_id);
      DROP TABLE span_proposals;
    ''')
    db.commit()
    ordered_terms.cache_clear()


def classify(db, out):
    state(out, 'CLASSIFY_COMPLETE_WORKSET')
    db.executescript('''
      CREATE TEMP TABLE source_available AS SELECT DISTINCT source_key FROM source_keys;
      CREATE UNIQUE INDEX source_available_key ON source_available(source_key);
      CREATE TEMP TABLE counts AS
        SELECT m.scope_id,count(*) n,count(DISTINCT o.source_key) p,
        count(DISTINCT CASE WHEN a.source_key IS NOT NULL THEN o.source_key END) cached,
        sum(o.source_key IS NULL) missing
        FROM members m JOIN observations o USING(number)
        LEFT JOIN source_available a ON a.source_key=o.source_key GROUP BY m.scope_id;
      CREATE UNIQUE INDEX counts_scope ON counts(scope_id);
      UPDATE scopes SET observation_count=(SELECT n FROM counts c WHERE c.scope_id=scopes.scope_id),
        source_count=(SELECT p FROM counts c WHERE c.scope_id=scopes.scope_id),
        cached_source_count=(SELECT cached FROM counts c WHERE c.scope_id=scopes.scope_id);
      UPDATE scopes SET state=CASE
        WHEN cached_source_count=source_count AND source_count>=2
          AND (SELECT missing FROM counts c WHERE c.scope_id=scopes.scope_id)=0
        THEN 'CACHED_SOURCE_IDENTITY_AND_SEMANTIC_REVIEW_PENDING'
        ELSE 'SOURCE_COMPLETION_PENDING' END WHERE state='PENDING_SOURCE_CHECK';
      INSERT INTO observation_work SELECT number,CASE WHEN EXISTS
        (SELECT 1 FROM members m WHERE m.number=o.number)
        THEN 'CROSS_SOURCE_CANDIDATE_REVIEW_PENDING' ELSE 'BROADER_SEMANTIC_RETRIEVAL_PENDING' END
        FROM observations o;
    ''')
    db.commit()


def counts(db):
    return dict(observations=db.execute('SELECT count(*) FROM observations').fetchone()[0],
        scopes=db.execute('SELECT count(*) FROM scopes').fetchone()[0],
        scope_states=dict(db.execute('SELECT state,count(*) FROM scopes GROUP BY state')),
        retrieval_methods=dict(db.execute('SELECT method,count(*) FROM scopes GROUP BY method')),
        observation_states=dict(db.execute('SELECT state,count(*) FROM observation_work GROUP BY state')),
        cached_source_keys=db.execute('SELECT count(DISTINCT source_key) FROM source_keys').fetchone()[0],
        observations_with_cached_source=db.execute('SELECT count(*) FROM observations o WHERE EXISTS(SELECT 1 FROM source_keys s WHERE s.source_key=o.source_key)').fetchone()[0],
        source_document_variants=db.execute('SELECT count(*) FROM source_documents').fetchone()[0],
        source_keys_with_distinct_abstracts=db.execute('SELECT count(*) FROM (SELECT k.source_key FROM source_keys k JOIN source_documents d USING(document_sha) GROUP BY k.source_key HAVING count(DISTINCT d.abstract_sha)>1)').fetchone()[0],
        preserved_prior_decisions=db.execute('SELECT count(*) FROM prior_decisions').fetchone()[0],
        graph_original_record_reads=db.execute('SELECT count(*) FROM original_reads').fetchone()[0],
        new_scientific_approvals=0)


def build(out=OUT):
    out = Path(out)
    index.require(not out.exists(), 'Version exists; inspect status, never rebuild frozen inputs')
    accepted = read(index.OUTPUT / 'INDEX_ACCEPTANCE.json')
    base = read(index.journal.OUTPUT / 'CAMPAIGN.json')
    index.require(base['current_graph'] == accepted['graph'], 'Accepted graph differs from global index')
    index.check_file(accepted['graph']); index.check_file(accepted['database'])
    out.mkdir()
    start = time.monotonic()
    index.journal.atomic_json(out / 'SOURCE_BASELINE.json', base)
    db = connect(out / 'WORKSET.sqlite'); schema(db)
    state(out, 'REUSE_COMPLETE_ACCEPTED_INDEX', expected_observations=accepted['all_current_claims_indexed'])
    db.execute('ATTACH DATABASE ? AS frozen', (Path(accepted['database']['path']).resolve().as_uri() + '?mode=ro',))
    cols = [r['name'] for r in db.execute('PRAGMA table_info(observations)')]
    db.execute('INSERT INTO observations SELECT ' + ','.join(cols) + ' FROM frozen.observations')
    db.commit(); db.execute('DETACH DATABASE frozen')
    index.require(db.execute('SELECT count(*) FROM observations').fetchone()[0] == accepted['all_current_claims_indexed'], 'Full input coverage differs')
    import_existing(db, accepted)
    paths = corpus_files()
    ingest_sources(db, paths, out)
    retrieve_spans(db, out)
    classify(db, out)
    result = counts(db)
    index.require(result['observations'] == sum(result['observation_states'].values()), 'Observation coverage is incomplete')
    index.require(result['preserved_prior_decisions'] == 73, 'Prior adjudication coverage differs')
    index.require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Workset integrity failure')
    files = [dict(r) for r in db.execute('SELECT * FROM files ORDER BY path')]
    db.execute('INSERT INTO metadata VALUES(?,?)', ('graph', json.dumps(accepted['graph'])))
    db.execute('INSERT INTO metadata VALUES(?,?)', ('baseline_layer', json.dumps(base['current_claim_layer'])))
    db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    # New consequential index is verified against its already accepted input;
    # this reads the 1.71 GB index once, not the 6.76 GB graph.
    index.check_file(accepted['database'], full_hash=True)
    index.check_file(accepted['graph'])
    protocol = dict(schema='kg.whole_graph_claim_batch.v1', at=index.journal.utc_now(),
        graph=accepted['graph'], input_index=accepted['database'],
        source_baseline=index.fingerprint(out / 'SOURCE_BASELINE.json'),
        code=index.fingerprint(Path(__file__)), source_files=files,
        previous_ledgers=[index.fingerprint(OLD / 'LOCAL_DECISIONS.json'), index.fingerprint(PREVIOUS / 'LOCAL_DECISIONS.json')],
        scope='Every original observation; all old scopes; every attested endpoint span scope; unmatched observations remain queued.',
        reviewed_group_cap=None, original_observation_cap=None, scientific_approvals=0,
        graph_full_scans=0, graph_hash_policy='Reuse accepted unchanged graph fingerprint; full hash once at final publication boundary or an integrity trigger.',
        retrieval_warning='Nested terms and omitted predicate/negation are retrieval only. They do not assert equivalence or approve support. Original qualifiers and complete relation closure are mandatory at review.',
        source_warning='Historical abstracts, source variants and author IDs are preserved. Cached text does not establish identity, publication validity, independence or scientific support.',
        publication_policy='Reuse accepted decisions; adjudicate each owning source; retain holds and singletons explicitly; assemble one complete release after all executable work is processed.',
        model_policy=dict(model='deepseek-v4.1-flash:cloud',think='high',temperature=0,num_predict=65536,upstream_concurrency=1),
        previous_uncertain_request=str(ROOT / 'tmp/kg_group_review_flash_v1'),
        uncertain_request_policy='Never resend the previous request or its observations automatically. Preserve its unresolved disposition.',
        elapsed_seconds=round(time.monotonic()-start, 2), counts=result)
    index.journal.atomic_json(out / 'PROTOCOL.json', protocol)
    index.journal.atomic_json(out / 'WORKSET_ACCEPTANCE.json', dict(
        status='COMPLETE_WHOLE_GRAPH_RETRIEVAL_WORKSET_NOT_SCIENTIFIC_APPROVAL',
        at=index.journal.utc_now(), protocol=index.fingerprint(out / 'PROTOCOL.json'),
        initial_workset=index.fingerprint(out / 'WORKSET.sqlite'), **result))
    state(out, 'WHOLE_GRAPH_WORKSET_READY', status='REVIEW_AND_SOURCE_COMPLETION_PENDING',
          elapsed_seconds=protocol['elapsed_seconds'], **result)
    return protocol


def cached_original(db, row, graph_fp, reader=None):
    cached = db.execute('SELECT node_sha,payload FROM originals WHERE cid=?', (row['cid'],)).fetchone()
    if cached:
        record = unpacked(cached['payload'])
        index.require(cached['node_sha'] == row['node_sha'] == index.digest(record), 'Original cache binding differs')
        return record
    index.check_file(graph_fp)
    if reader is None:
        with Path(graph_fp['path']).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            record = index.record_at(mm, row['byte_offset'])
    else:
        record = reader(row['byte_offset'])
    index.require(record['id'] == row['cid'] and index.digest(record) == row['node_sha'], 'Original record differs from accepted census')
    db.execute('INSERT INTO originals VALUES(?,?,?)', (row['cid'], row['node_sha'], packed(record)))
    db.execute('INSERT INTO original_reads VALUES(?,?)', (row['cid'], index.journal.utc_now()))
    return record


def packet(scope_id, out=OUT):
    protocol = read(Path(out) / 'PROTOCOL.json')
    index.check_file(protocol['code'], full_hash=True)
    index.check_file(protocol['graph'])
    db = connect(Path(out) / 'WORKSET.sqlite')
    scope = db.execute('SELECT * FROM scopes WHERE scope_id=?', (scope_id,)).fetchone()
    index.require(scope is not None, 'Unknown workset scope')
    records = db.execute('''SELECT * FROM observations WHERE rid IN
        (SELECT o.rid FROM members m JOIN observations o USING(number) WHERE m.scope_id=?)
        ORDER BY number''', (scope_id,)).fetchall()
    originals, docs = [], {}
    with Path(protocol['graph']['path']).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for row in records:
            originals.append(cached_original(db, row, protocol['graph'], lambda offset: index.record_at(mm, offset)))
            for document in db.execute('SELECT document_sha,payload FROM source_keys k JOIN source_documents d USING(document_sha) WHERE k.source_key=?', (row['source_key'],)):
                docs[document['document_sha']] = unpacked(document['payload'])
    db.commit(); db.close()
    return dict(scope=dict(scope), observations=originals, source_variants=docs,
                complete_original_relation_closure=True, scientific_approval=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'status', 'packet'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--scope-id')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.out)
    elif args.command == 'status':
        with connect(args.out / 'WORKSET.sqlite', True) as connection:
            print(json.dumps(counts(connection), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(packet(args.scope_id, args.out), ensure_ascii=False))
