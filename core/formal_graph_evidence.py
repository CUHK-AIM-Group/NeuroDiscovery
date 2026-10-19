"""Read original claims from the formal KG, without inventing review receipts.

CURRENT_STATE binds the graph bytes. A disposable SQLite cache stores retrieval
fields and byte locations only; every returned observation comes from the KG.
Scope membership reviews are never treated as proposition-support reviews.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

from core.web.claim_evidence import AcceptedClaimEvidence, EvidenceUnavailable

INDEX_VERSION = 'formal-claims-v2'
_locks = {}
_locks_guard = threading.Lock()


def _signature(path):
    info = Path(path).stat()
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def formal_state_path(root):
    path = Path(root) / 'neurooracle/data/full_v2/CURRENT_STATE.json'
    return path if path.is_file() else None


class FormalGraphEvidence(AcceptedClaimEvidence):
    def __init__(self, state_path, *, graph_path=None, cache_directory=None):
        super().__init__(Path(state_path))
        self.graph = Path(graph_path) if graph_path else self.path.parent / 'knowledge_graph.json'
        cache = Path(cache_directory) if cache_directory else Path(tempfile.gettempdir()) / 'neurodiscovery-kg-cache'
        self.cache_directory = cache
        self._verified_graph = self._verified_index = None

    def _read_current(self):
        state = json.loads(self.path.read_text(encoding='utf-8'))
        if state.get('schema_version') == 'neurodiscovery.graph-publication.v1':
            publication = state
            if (publication.get('source_status') != 'canonical_current' or
                    not re.fullmatch('[a-f0-9]{64}', publication.get('source_state_sha256', ''))):
                raise EvidenceUnavailable('The published formal KG source is invalid')
            state = dict(status=publication['source_status'], canonical_files=dict(knowledge_graph=dict(
                         bytes=publication['graph']['uncompressed_bytes'],
                         sha256=publication['graph']['uncompressed_sha256'])),
                         formal_kg_statistics=dict(general=dict(claims=publication['counts']['claims'],
                                                                papers=publication['counts']['papers'])))
        if state.get('status') != 'canonical_current':
            raise EvidenceUnavailable('The formal KG state is not canonical_current')
        graph = state['canonical_files']['knowledge_graph']
        if (type(graph.get('bytes')) is not int or
                not re.fullmatch('[a-f0-9]{64}', graph.get('sha256', '')) or
                _signature(self.graph)[0] != graph['bytes']):
            raise EvidenceUnavailable('The formal KG differs from CURRENT_STATE.json')
        return state

    def _verify_graph(self, state):
        signature = _signature(self.graph)
        identity = (state['canonical_files']['knowledge_graph']['sha256'], signature)
        if self._verified_graph != identity:
            if _sha(self.graph) != identity[0] or _signature(self.graph) != signature:
                raise EvidenceUnavailable('The formal KG hash differs from CURRENT_STATE.json')
            self._verified_graph = identity

    def _revision(self, state):
        return dict(graph_revision=state['canonical_files']['knowledge_graph']['sha256'],
                    source='formal_current_graph', evidence_kind='original_extraction',
                    independence_established=False, consensus_inferred=False)

    @contextmanager
    def read_snapshot(self):
        with self._lock:
            state = self._read_current()
            self._verify_graph(state)
            signature = _signature(self.graph)
            yield self._revision(state)
            if state != self._read_current() or signature != _signature(self.graph):
                raise EvidenceUnavailable('The formal KG changed during evidence retrieval')

    def status(self):
        with self.read_snapshot() as revision:
            state = self._read_current()
            index = self._index_path(state)
            return dict(configured=True, available=True, shared_claim_count=0,
                        original_claim_count=state['formal_kg_statistics']['general']['claims'],
                        reviewed_multipaper_claim_count=0, index_ready=index.is_file(),
                        **revision)

    def _index_path(self, state):
        sha = state['canonical_files']['knowledge_graph']['sha256']
        return self.cache_directory / f'{INDEX_VERSION}-{sha}.sqlite'

    @staticmethod
    def _source_key(md):
        paper = md.get('source_paper') or {}
        if paper.get('pmid'):
            return 'PMID:' + str(paper['pmid'])
        if paper.get('doi'):
            return 'DOI:' + str(paper['doi']).casefold()
        return 'RECORD:' + md['id']

    def _index(self):
        state = self._read_current()
        self._verify_graph(state)
        path = self._index_path(state)
        with _locks_guard:
            lock = _locks.setdefault(str(path.resolve()), threading.RLock())
        with lock:
            if not path.exists():
                self._build_index(state, path)
            receipt_path = path.with_suffix('.json')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            expected = dict(version=INDEX_VERSION, graph_sha256=self._revision(state)['graph_revision'],
                            claims=state['formal_kg_statistics']['general']['claims'])
            if any(receipt.get(k) != v for k, v in expected.items()):
                raise EvidenceUnavailable('The formal claim cache belongs to another source')
            signature = _signature(path)
            if self._verified_index != (signature, receipt['sha256']):
                if _sha(path) != receipt['sha256'] or _signature(path) != signature:
                    raise EvidenceUnavailable('The formal claim cache hash changed')
                self._verified_index = (signature, receipt['sha256'])
            return path

    def _build_index(self, state, path):
        from neurooracle.scripts.streaming_graph_json import IncrementalJsonReader

        class OffsetReader(IncrementalJsonReader):
            byte_offset = char_position = 0

            def tell(self):
                self.byte_offset += len(self.buffer[self.char_position:self.position].encode('utf-8'))
                self.char_position = self.position
                return self.byte_offset

            def _compact(self):
                if self.position > self.chunk_size:
                    self.tell()
                    super()._compact()
                    self.char_position = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.formal-index-', suffix='.sqlite', dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        signature = _signature(self.graph)
        claims = nodes = 0
        try:
            with closing(sqlite3.connect(temporary)) as db, self.graph.open(encoding='utf-8') as handle:
                db.execute('PRAGMA journal_mode=OFF')
                db.execute('CREATE TABLE nodes (id TEXT PRIMARY KEY, offset INTEGER, length INTEGER)')
                db.execute('CREATE TABLE claims (id TEXT PRIMARY KEY, subject_id TEXT, subject_name TEXT, '
                           'predicate TEXT, object_id TEXT, object_name TEXT, subject_type TEXT, object_type TEXT, '
                           'source_key TEXT, search TEXT)')
                reader = OffsetReader(handle)
                reader.expect('{')
                while True:
                    key = reader.value()
                    reader.expect(':')
                    if key == 'concepts':
                        break
                    reader.value()
                    reader.expect(',')
                reader.expect('{')
                while reader.peek() != '}':
                    key = reader.value()
                    reader.expect(':')
                    reader.skip_whitespace()
                    start = reader.tell()
                    node = reader.value()
                    length = reader.tell() - start
                    if node.get('id') != key:
                        raise EvidenceUnavailable('The formal KG has a concept key/ID mismatch')
                    db.execute('INSERT INTO nodes VALUES (?,?,?)', (key, start, length))
                    nodes += 1
                    if 'claim' in (node.get('domain_tags') or []):
                        md = node['metadata']
                        if md.get('id') != key or not key.startswith('CLM:'):
                            raise EvidenceUnavailable('The formal KG claim identity is inconsistent')
                        fields = [str(md.get(k) or '') for k in
                                  ('subject_id', 'subject_name', 'predicate', 'object_id', 'object_name')]
                        inner = md.get('metadata') or {}
                        paper = md.get('source_paper') or {}
                        # Opaque hash IDs are exact lookup keys, not keyword
                        # evidence (e.g. the random hex fragment "cd44").
                        text = ' '.join([fields[1], fields[2], fields[4],
                                         *(str(paper.get(k) or '') for k in ('pmid', 'doi', 'title'))])
                        db.execute('INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?)',
                                   (key, *fields, json.dumps(inner.get('subject_type')),
                                    json.dumps(inner.get('object_type')), self._source_key(md), text.casefold()))
                        claims += 1
                    if reader.peek() == '}':
                        break
                    reader.expect(',')
                if claims != state['formal_kg_statistics']['general']['claims']:
                    raise EvidenceUnavailable('The formal claim count differs from CURRENT_STATE.json')
                db.execute('CREATE INDEX claim_predicate ON claims(predicate)')
                db.commit()
            if signature != _signature(self.graph) or state != self._read_current():
                raise EvidenceUnavailable('The formal KG changed while its cache was built')
            receipt = dict(version=INDEX_VERSION, graph_sha256=self._revision(state)['graph_revision'],
                           claims=claims, nodes=nodes, sha256=_sha(temporary),
                           status='RETRIEVAL_CACHE_NOT_SCIENTIFIC_REVIEW')
            receipt_temp = temporary.with_suffix('.json')
            receipt_temp.write_text(json.dumps(receipt), encoding='utf-8')
            os.replace(receipt_temp, path.with_suffix('.json'))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _record(self, db, cid):
        row = db.execute('SELECT offset,length FROM nodes WHERE id=?', (cid,)).fetchone()
        if row is None:
            raise KeyError('claim or concept not found: ' + cid)
        with self.graph.open('rb') as handle:
            handle.seek(row[0])
            node = json.loads(handle.read(row[1]))
        if node.get('id') != cid:
            raise EvidenceUnavailable('The formal cache location does not match its graph record')
        return node

    @staticmethod
    def _counts():
        return {name: (1 if name in ('article_count', 'publication_record_count') else 0)
                for name in AcceptedClaimEvidence.COUNTS}

    def _document(self, db, cid, revision):
        node = self._record(db, cid)
        if 'claim' not in (node.get('domain_tags') or []):
            raise KeyError('claim not found: ' + cid)
        md = node['metadata']
        inner = md.get('metadata') or {}
        bibliography = md.get('source_paper') or {}
        source = self._source_key(md)
        observation = dict(claim_id=cid, original_claim=md, raw_text=md.get('raw_text'),
                           evidence=md.get('evidence'), negated=md.get('negated'),
                           population=inner.get('population'), conditions=inner.get('conditions'),
                           observation_role='not_reviewed', proposition_support='not_reviewed',
                           observation_origin='original_graph', source_review=None)
        paper = dict(paper_key=source, work_key=source, bibliography=bibliography,
                     publication_versions=[dict(bibliography=bibliography)], publication_review=None,
                     source_identity=dict(status='declared_in_formal_graph'), verified=False,
                     counted_toward_default=False, supports_reviewed_proposition=False,
                     own_result=False, observations=[observation])
        return dict(shared_claim_id=cid, requested_claim_id=cid, original_claim_ids=[cid],
                    claim={k: md.get(k) for k in ('subject_id', 'subject_name', 'predicate', 'object_id', 'object_name')},
                    observation_count=1, papers=[paper], **self._counts(), **revision,
                    scope='Original extraction in the formal KG; proposition support and publication identity are not newly reviewed')

    def query_batch(self, *, queries):
        from core.web.claim_layer_v8 import validate_queries
        queries = validate_queries(queries)
        with self.read_snapshot() as revision, closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            results = []
            for query in queries:
                if 'claim_id' not in query:
                    raise KeyError('The formal KG has no reviewed shared relation: ' + query['relation_id'])
                results.append(self._document(db, query['claim_id'], revision))
            return dict(results=results, query_count=len(results), **revision)

    def query(self, *, claim_id=None, relation_id=None):
        if bool(claim_id) == bool(relation_id):
            raise ValueError('Provide exactly one claim or relation ID')
        query = {'claim_id': claim_id} if claim_id else {'relation_id': relation_id}
        return self.query_batch(queries=[query])['results'][0]

    def search(self, query='', *, minimum_papers=0, offset=0, limit=30):
        if not isinstance(query, str) or len(query) > 300 or type(minimum_papers) is not int or not 0 <= minimum_papers <= 10000:
            raise ValueError('Invalid formal claim search')
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid claim search page')
        with self.read_snapshot() as revision, closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            words = query.casefold().split()
            clause = ' AND '.join('(instr(search,?)>0 OR lower(id)=? OR lower(subject_id)=? OR lower(object_id)=?)'
                                  for _ in words) or '1'
            arguments = [word for word in words for _ in range(4)]
            if minimum_papers:
                clause = '0'  # No source-support review is implied by formal publication.
                arguments = []
            total = db.execute('SELECT count(*) FROM claims WHERE ' + clause, arguments).fetchone()[0]
            rows = db.execute('SELECT id FROM claims WHERE ' + clause + ' ORDER BY rowid LIMIT ? OFFSET ?',
                              [*arguments, limit, offset]).fetchall()
            documents = [self._document(db, row[0], revision) for row in rows]
            summaries = [{k: v for k, v in document.items() if k not in ('papers', 'scope')} for document in documents]
            return dict(total=total, claims=summaries, offset=offset, limit=limit,
                        minimum_reviewed_supporting_papers=minimum_papers, **revision)

    def claim_index(self):
        return []  # The deleted experimental shared/reviewed catalog is not formal KG evidence.

    def original_claim_candidates(self, matchers, *, minimum_matches, limit):
        any_term = re.compile('|'.join(pattern.pattern for _, pattern in matchers), re.IGNORECASE)
        best, matched = {}, 0
        with self.read_snapshot(), closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            for cid, sid, subject, predicate, oid, obj, source, text in db.execute(
                    'SELECT id,subject_id,subject_name,predicate,object_id,object_name,source_key,search FROM claims ORDER BY rowid'):
                if not any_term.search(text):
                    continue
                hits = [term for term, pattern in matchers if pattern.search(text)]
                if len(hits) < minimum_matches:
                    continue
                matched += 1
                rank = (-len(hits), subject.casefold(), cid)
                if source not in best or rank < best[source][0]:
                    best[source] = (rank, dict(shared_claim_id=cid, query_id=cid, original_claim_ids=[cid],
                        retrieval_origin='formal_original', matched_terms=hits, observation_count=1,
                        claim=dict(subject_id=sid, subject_name=subject, predicate=predicate, object_id=oid, object_name=obj),
                        **self._counts()))
            count = db.execute('SELECT count(*) FROM claims').fetchone()[0]
        return dict(available=True, indexed=count, eligible=count, matched=matched, matched_sources=len(best),
                    candidates=[row for _, row in sorted(best.values(), key=lambda item: item[0])[:limit]])

    def graph_nodes(self, node_ids):
        nodes = sorted(set(node_ids))
        if len(nodes) > 128:
            raise ValueError('Too many chain nodes')
        with self.read_snapshot(), closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            found = {}
            for cid in nodes:
                try:
                    found[cid] = self._record(db, cid)
                except KeyError:
                    pass
            return found

    def chain_index(self, *, predicates):
        records = []
        with self.read_snapshot(), closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            for predicate in sorted(set(predicates)):
                for cid, sid, sn, pred, oid, on, st, ot, source in db.execute(
                        'SELECT id,subject_id,subject_name,predicate,object_id,object_name,subject_type,object_type,source_key '
                        'FROM claims WHERE predicate=? ORDER BY rowid', (predicate,)):
                    records.append(dict(id=cid, subject_id=sid, subject_name=sn, predicate=pred,
                                        object_id=oid, object_name=on, subject_type=json.loads(st),
                                        object_type=json.loads(ot), source_key=source, pmid=None, doi=None))
            count = db.execute('SELECT count(*) FROM claims').fetchone()[0]
        return dict(available=True, indexed=count, records=records)

    def pair_support_counts(self, pairs):
        sources = {tuple(sorted(pair)): set() for pair in pairs}
        with self.read_snapshot(), closing(sqlite3.connect(self._index().as_uri() + '?mode=ro', uri=True)) as db:
            for sid, oid, source in db.execute('SELECT subject_id,object_id,source_key FROM claims'):
                key = tuple(sorted((sid, oid)))
                if key in sources:
                    sources[key].add(source)
        return {key: len(value) for key, value in sources.items()}
