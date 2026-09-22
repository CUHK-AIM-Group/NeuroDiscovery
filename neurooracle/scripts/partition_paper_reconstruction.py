"""Index the existing graph once, protecting fixed/uncertain provenance verbatim.

No ingestion modules or model providers are imported. The original graph remains
read-only. Legacy claim payloads already live in the frozen corpus; this index
stores their identity and digest, and stores complete fixed/uncertain records.
"""
from __future__ import annotations

from collections import Counter
import codecs
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib

from prepare_paper_reconstruction import ROOT, dump, now, read, settings
from streaming_graph_json import IncrementalJsonReader

FIXED_SOURCES = {
    'NeuroNames', 'MeSH', 'MeSH_hierarchy', 'DisGeNET', 'CognitiveAtlas',
    'BrainMap', 'BrainMap_approx', 'ATC', 'ClinicalOutcomes', 'MedDRA-SOC',
    'IndividualDataAnchor', 'visual_functional_roi', 'experiment_infra',
    'HPO', 'HGNC', 'AHBA', 'HansenReceptor2022', 'curated_pharmacology',
    'Neurosynth_v0.7', 'ENIGMA', 'NeuroClaw-IM', 'NeuroClaw-GeneSet',
    'NeuroClaw-GM', 'spatial_mapping',
}
FIXED_PREFIXES = ('spatial_mapping.v', 'visual_stimulus_taxonomy')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def classify(kind, key, record):
    provenance = record.get('source_vocab' if kind == 'node' else 'source') or ''
    if not isinstance(provenance, str):
        return 'protected_unclassified', json.dumps(provenance, ensure_ascii=False)
    # A bibliographic citation does not make a fixed atlas or map a paper claim.
    if provenance in FIXED_SOURCES or provenance.startswith(FIXED_PREFIXES):
        return 'fixed', provenance
    if kind == 'node':
        tags = set(record.get('domain_tags') or [])
        if provenance == 'claim_extraction' and tags == {'claim'}:
            return 'legacy_literature', provenance
    elif provenance == 'claim_extraction' or provenance.startswith('claim:'):
        return 'legacy_literature', provenance
    # Mixed/shared entities and unrecognized bridges are protected, never dropped.
    return 'protected_unclassified', provenance


class HashingTextReader:
    """Decode UTF-8 while hashing every original byte in the same read pass."""
    def __init__(self, handle):
        self.handle = handle
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.hash = hashlib.sha256()
        self.bytes = 0
        self.eof = False

    def read(self, size):
        if self.eof:
            return ''
        block = self.handle.read(size)
        self.hash.update(block)
        self.bytes += len(block)
        self.eof = not block
        return self.decoder.decode(block, final=self.eof)


def graph_records(handle):
    """One top-level pass; preserve all array edges, including duplicates."""
    r = IncrementalJsonReader(handle)
    r.expect('{')
    seen = set()
    while r.peek() != '}':
        name = r.value()
        if name in seen:
            raise ValueError('Duplicate graph section: ' + str(name))
        seen.add(name)
        r.expect(':')
        if name in {'concepts', 'edges'}:
            is_nodes = name == 'concepts'
            r.expect('{' if is_nodes else '[')
            end = '}' if is_nodes else ']'
            ordinal = 0
            while r.peek() != end:
                key = r.value() if is_nodes else ordinal
                if is_nodes:
                    r.expect(':')
                value = r.value()
                if not isinstance(value, dict):
                    raise ValueError('Graph record is not an object')
                yield ('node' if is_nodes else 'edge'), key, value
                ordinal += 1
                if r.peek() != end:
                    r.expect(',')
                    if r.peek() == end:
                        raise ValueError('Trailing comma in graph records')
            r.expect(end)
        else:
            yield 'header', name, r.value()
        if r.peek() != '}':
            r.expect(',')
            if r.peek() == '}':
                raise ValueError('Trailing graph comma')
    r.expect('}')
    # Drain trailing whitespace so the source SHA covers the complete file.
    if r.unread_text().strip():
        raise ValueError('Unexpected data after graph')
    while block := handle.read(1024 * 1024):
        if block.strip():
            raise ValueError('Unexpected data after graph')
    if not {'concepts', 'edges'} <= seen:
        raise ValueError('Missing graph sections')


def run():
    _, out, _, _ = settings()
    completion = out / 'FIXED_PARTITION_READY.json'
    if completion.exists():
        print(json.dumps({k:v for k,v in read(completion).items() if k!='provenance_counts'}, ensure_ascii=False))
        return
    binding = read(ROOT / 'tmp/kg_rebuild_full_20260920_v1/INPUTS.json')['preserved_base_graph']
    original = Path(binding['path'])
    stat = original.stat()
    if (stat.st_size, stat.st_mtime_ns) != (binding['bytes'], binding['mtime_ns']):
        raise ValueError('Original graph changed since preservation')
    db = out / 'GRAPH_PARTITION.sqlite'
    if db.exists():
        raise RuntimeError('Partition already started; inspect its receipt/process before recovery')
    c = sqlite3.connect(db)
    c.execute('pragma journal_mode=WAL')
    c.executescript('''
      create table nodes(id text primary key, partition text, provenance text,
                         name text, sha256 text, payload blob);
      create table edges(ordinal integer primary key, source_id text, target_id text,
                         partition text, provenance text, sha256 text, payload blob);
      create table headers(name text primary key, sha256 text, payload blob);
    ''')
    counts, sources = Counter(), Counter()
    begun = now()
    try:
        with original.open('rb') as raw:
            stream = HashingTextReader(raw)
            for kind, key, record in graph_records(stream):
                data = canonical(record)
                sha = hashlib.sha256(data).hexdigest()
                if kind == 'header':
                    c.execute('insert into headers values(?,?,?)', (key, sha, zlib.compress(data)))
                    continue
                part, source = classify(kind, key, record)
                payload = None if part == 'legacy_literature' else zlib.compress(data, 1)
                if kind == 'node':
                    c.execute('insert into nodes values(?,?,?,?,?,?)',
                              (key, part, source, record.get('preferred_name') or record.get('name'), sha, payload))
                else:
                    c.execute('insert into edges values(?,?,?,?,?,?,?)',
                              (key, record.get('source_id'), record.get('target_id'), part, source, sha, payload))
                counts[kind + ':' + part] += 1
                sources[(kind, source, part)] += 1
                if sum(counts.values()) % 50000 == 0:
                    c.commit()
                    dump(out / 'PARTITION_PROGRESS.json', {
                        'status': 'RUNNING', 'at': now(), 'began': begun,
                        'pid': os.getpid(), 'counts': dict(counts), 'bytes_read': stream.bytes,
                        'source_bytes': binding['bytes'], 'fixed_layer_modified': False})
        end_stat = original.stat()
        if stream.hash.hexdigest() != binding['sha256'] or stream.bytes != binding['bytes']:
            raise ValueError('Full graph identity mismatch at partition boundary')
        if (end_stat.st_size, end_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Source changed during partition')
        c.executescript('''
          create index node_partition on nodes(partition,provenance);
          create index node_name on nodes(name);
          create index edge_partition on edges(partition,provenance);
          create index edge_source on edges(source_id);
          create index edge_target on edges(target_id);
        ''')
        c.commit()
        endpoints = c.execute("""select count(*) from edges e
          left join nodes s on s.id=e.source_id left join nodes t on t.id=e.target_id
          where e.partition='fixed' and (s.id is null or t.id is null
            or s.partition='legacy_literature' or t.partition='legacy_literature')""").fetchone()[0]
        if c.execute('pragma quick_check').fetchone()[0] != 'ok':
            raise ValueError('Partition SQLite check failed')
        c.execute('pragma wal_checkpoint(TRUNCATE)')
        report = {
            'status': 'PARTITION_INDEXED_WITH_PROVENANCE_HOLDS', 'at': now(), 'began': begun,
            'original_graph': binding, 'graph_read_passes': 1, 'full_source_sha256_verified': True,
            'counts': dict(counts), 'fixed_edges_with_unresolved_or_legacy_endpoints': endpoints,
            'provenance_counts': [dict(kind=k, source=s, partition=p, count=n)
                                  for (k, s, p), n in sorted(sources.items())],
            'preservation': 'complete payloads for fixed and protected_unclassified records; legacy originals remain in frozen corpus/source graph',
            'fixed_layer_modified': False, 'production_changed': False,
            'ready_for_integrated_replacement': False,
            'next': 'review protected provenance using this index; do not rescan original graph',
        }
        c.close()
        dump(completion, report)
        dump(out / 'PARTITION_PROGRESS.json', {**report, 'active_process': None})
        print(json.dumps({k: v for k, v in report.items() if k != 'provenance_counts'}, ensure_ascii=False))
    except BaseException as e:
        c.close()
        dump(out / 'PARTITION_FAILURE.json', {'at': now(), 'error': repr(e), 'counts': dict(counts)})
        raise


if __name__ == '__main__':
    run()
