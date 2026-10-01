"""Read-only, source/pair-separated preparation for graph reconstruction training.

Labels mean recorded graph relations, not source-adjudicated scientific truth.
This intentionally does not manufacture the reviewed bundle of experiment.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from neurooracle.scripts.streaming_graph_json import iter_concepts, iter_edges
from neurooracle.src.kge.triple_loader import _DROP_RELATIONS, _INFRA_DOMAINS

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / ('neurooracle/data/umls_mapping/umls_2026AA_atomic_mentions_v1_20260906/'
                     'kg_overnight_20260907/global_claim_aggregation_20260913/INDEX_ACCEPTANCE.json')
CORPUS = ROOT / 'tmp/kg_rebuild_full_20260920_v1/CORPUS.sqlite'
PAPERS = ROOT / 'tmp/kg_paper_reconstruction_20260920/PAPERS.sqlite'
INPUTS = ROOT / 'tmp/kg_paper_reconstruction_20260920/BATCH_MODEL/INPUTS.sqlite'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_name(path.name + '.writing')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def fingerprint(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def provenance():
    """Identity metadata only; never fetch abstracts or heldout source payloads."""
    with connect(INPUTS) as db:
        protected = {r[0] for r in db.execute("SELECT work_key FROM inputs WHERE fold!='corpus'")}
    if len(protected) != 40:
        raise ValueError('Frozen heldout inventory changed; review before preparation')
    with connect(PAPERS) as db:
        pub_work = dict(db.execute('SELECT p.pub_id,COALESCE(j.work_key,j.job_id) FROM publications p '
                                   'JOIN paper_jobs j ON p.job_id=j.job_id'))
        versions = db.execute('SELECT v.abstract_key,COALESCE(j.work_key,j.job_id) '
                              'FROM potential_versions v JOIN paper_jobs j ON v.job_id=j.job_id').fetchall()
    families = Union()
    join_version_groups(families,versions)
    with connect(CORPUS) as db:
        # All known version, correction and related-publication links are kept
        # together conservatively, including comment/retraction relations.
        for a, b in db.execute('SELECT a,b FROM publication_relations'):
            families.join(pub_work.get(a, a), pub_work.get(b, b))
        claims = {cid: pub_work.get(pub, pub or key.upper())
                  for cid, pub, key in db.execute('SELECT cid,pub_id,source_key FROM originals')
                  if pub or key}
    protected = {families.find(x) for x in protected}
    claims = {cid: families.find(work) for cid, work in claims.items()}
    pub_work = {pub: families.find(work) for pub, work in pub_work.items()}
    return claims, pub_work, families, protected


def join_version_groups(families,rows):
    """Conservatively join known same-abstract candidates, without reading text."""
    first = {}
    for abstract_key,work in rows:
        if abstract_key and work:
            if abstract_key in first:
                families.join(first[abstract_key],work)
            else:
                first[abstract_key] = work


def split_records(records, fixed_groups, protected_pairs, seed=123):
    """Group sources and ALL relations/directions of each endpoint pair.

    No evaluation rows are moved to train to hide OOV. Fixed reference groups
    are training context; components touching them remain training-only.
    """
    groups, pairs = Union(), {}
    retained, exclusions = [], Counter()
    for s, r, t, family in records:
        pair = (min(s, t), max(s, t))
        if pair in protected_pairs:
            exclusions['protected_endpoint_proxy'] += 1
            continue
        groups.find(family)
        if pair in pairs:
            groups.join(family, pairs[pair])
        else:
            pairs[pair] = family
        retained.append((s, r, t, family))
    train_roots = {groups.find(g) for g in fixed_groups}
    assignments = {}
    for _, _, _, family in retained:
        root = groups.find(family)
        if root not in assignments:
            bucket = int.from_bytes(hashlib.sha256(f'{seed}:{root}'.encode()).digest()[:8], 'big') % 100
            assignments[root] = 0 if root in train_roots or bucket < 80 else 1 if bucket < 90 else 2
    rows = np.asarray([(s, r, t, assignments[groups.find(g)]) for s, r, t, g in retained], dtype=np.int64)
    if not len(rows):
        raise ValueError('Empty eligible graph')
    rows = np.unique(rows, axis=0)
    # Every source and endpoint pair must be confined to one fold.
    source_folds = {g: assignments[groups.find(g)] for g in groups.parent}
    counts = Counter(source_folds.values())
    return rows[:, :3], rows[:, 3].astype(np.int8), source_folds, dict(exclusions), dict(counts)


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seal = read(ACCEPTANCE)
    graph = Path(seal['graph']['path'])
    before = {str(p): fingerprint(p) for p in (graph, CORPUS, PAPERS, INPUTS, ACCEPTANCE)}

    def progress(stage, **kw):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **kw)
        write(output / 'PREPARATION_PROGRESS.json', row, replace=True)
        print(json.dumps(row), flush=True)

    progress('verify_current_graph')
    if fingerprint(graph)['bytes'] != seal['graph']['bytes'] or sha(graph) != seal['graph']['sha256']:
        raise ValueError('Current graph binding failed')
    claim_work, pub_work, families, protected = provenance()
    write(output / 'PROTECTED_WORKS.json', sorted(protected))
    counts, domains, node_info = Counter(), {}, {}
    progress('read_node_domains')
    for i, (nid, node) in enumerate(iter_concepts(graph), 1):
        tags = node.get('domain_tags') or []
        domain = str(tags[0]) if tags else 'unknown'
        domains.setdefault(domain, len(domains))
        node_info[nid] = (domains[domain], bool(_INFRA_DOMAINS.intersection(tags)))
        if i % 250000 == 0:
            progress('read_node_domains', nodes=i)
    counts['raw_nodes'] = len(node_info)
    entities, relations = {}, {}
    records, fixed_groups, protected_pairs = [], set(), set()
    source_counts = Counter()
    # Provenance-qualified fixed namespaces seen in the existing protected layer.
    fixed_sources = {'MeSH_hierarchy', 'NeuroLex', 'UBERON', 'FMA', 'HPO', 'DisGeNET',
                     'Allen_Human_Brain_Atlas', 'Allen_Brain_Atlas', 'Neurotransmitters',
                     'CognitiveAtlas', 'Cognitive_Atlas', 'HGNC', 'GO', 'KEGG', 'DrugBank',
                     'ChEBI', 'ATC', 'ClinicalOutcomes', 'UMLS', 'UMLS_alignment'}
    # Read the existing fixed boundary's provenance inventory, not its old
    # partition names (which were refined after the original database build).
    boundary = read(ROOT / 'tmp/kg_paper_reconstruction_20260920/FIXED_BOUNDARY.json')
    if boundary['original_graph']['sha256'] != seal['graph']['sha256']:
        raise ValueError('Fixed boundary belongs to another graph')
    for row in boundary['provenance_decisions']:
        if row['kind'] == 'edge' and row['decision'] in {'fixed_reference', 'protected_existing_alignment'}:
            fixed_sources.add(row['source'])
    progress('read_all_edges')
    for i, edge in enumerate(iter_edges(graph), 1):
        counts['raw_edges'] += 1
        s, r, t = (str(edge.get(k) or '') for k in ('source_id', 'relation_type', 'target_id'))
        meta = edge.get('metadata') or {}
        confidence = float(edge.get('confidence', 0) or 0)
        reason = ('missing' if not s or not r or not t else
                  'self_loop' if s == t else
                  'infrastructure_relation' if r in _DROP_RELATIONS else
                  'dangling' if s not in node_info or t not in node_info else
                  'infrastructure_node' if node_info[s][1] or node_info[t][1] else
                  'low_nonfinite_confidence' if not math.isfinite(confidence) or confidence < .2 else
                  'negated' if edge.get('negated') or meta.get('negated') else None)
        if reason:
            counts[reason] += 1
            continue
        source = str(edge.get('source') or '')
        cid = str(meta.get('claim_id') or '')
        family = claim_work.get(cid)
        if not family and source.startswith('claim:'):
            identifier = source[6:]
            if identifier.isdigit():
                family = families.find(pub_work.get('PMID:' + identifier, 'PMID:' + identifier))
        if not family and source in fixed_sources:
            family = 'FIXED:' + source
            fixed_groups.add(family)
        if not family:
            counts['unresolved_provenance'] += 1
            source_counts[source] += 1
            continue
        si = entities.setdefault(s, len(entities))
        ti = entities.setdefault(t, len(entities))
        ri = relations.setdefault(r, len(relations))
        if family in protected:
            counts['protected_work'] += 1
            protected_pairs.add((min(si, ti), max(si, ti)))
            continue
        records.append((si, ri, ti, family))
        counts['eligible_records_before_proxy_filter'] += 1
        if i % 250000 == 0:
            progress('read_all_edges', edges=i, eligible=len(records))
    progress('source_and_endpoint_components', eligible=len(records))
    triples, split, source_folds, excluded, source_counts_split = split_records(records, fixed_groups, protected_pairs)
    counts.update(excluded)
    counts['unique_eligible_triples'] = len(triples)
    counts['duplicate_records_collapsed'] = len(records) - sum(excluded.values()) - len(triples)
    del records
    # Compact vocabulary to retained graph, without adding evaluation IDs to training.
    used = np.unique(triples[:, [0, 2]])
    remap = np.full(len(entities), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    entity_list = list(entities)
    entity_list = [entity_list[int(i)] for i in used]
    triples[:, 0] = remap[triples[:, 0]]
    triples[:, 2] = remap[triples[:, 2]]
    domain_ids = np.asarray([node_info[e][0] for e in entity_list], dtype=np.int16)
    train = triples[split == 0]
    trained_nodes = np.zeros(len(entity_list), dtype=bool)
    trained_nodes[train[:, [0, 2]]] = True
    trained_relations = np.zeros(len(relations), dtype=bool)
    trained_relations[train[:, 1]] = True
    measure = trained_nodes[triples[:, 0]] & trained_nodes[triples[:, 2]] & trained_relations[triples[:, 1]]
    fold_counts = {str(f): dict(total=int((split == f).sum()), measurable=int(((split == f) & measure).sum())) for f in range(3)}
    if any(fold_counts[str(f)]['measurable'] == 0 for f in range(3)):
        raise ValueError(f'Empty measurable fold: {fold_counts}')
    with (output / 'data.npz').open('xb') as stream:
        np.savez(stream, triples=triples, split=split, domain_ids=domain_ids,
                 trained_nodes=trained_nodes, trained_relations=trained_relations)
    write(output / 'vocabulary.json', dict(entities=entity_list, relations=list(relations), domains=list(domains)))
    write(output / 'source_folds.json', source_folds)
    after = {str(p): fingerprint(p) for p in (graph, CORPUS, PAPERS, INPUTS, ACCEPTANCE)}
    if before != after:
        raise ValueError('Read-only source changed while preparing')
    manifest = dict(schema='idea-full-graph-reconstruction.v1', graph_revision=seal['graph']['sha256'],
                    graph=str(graph), counts=dict(counts), entities=len(entity_list), relations=len(relations),
                    folds=fold_counts, source_fold_counts=source_counts_split, protected_work_count=len(protected),
                    unresolved_provenance_by_source=dict(source_counts), source_fingerprint=before,
                    files={p.name: sha(p) for p in output.iterdir() if p.name in
                           {'data.npz', 'vocabulary.json', 'source_folds.json', 'PROTECTED_WORKS.json'}},
                    preparation_seconds=time.monotonic()-started, max_nodes=None, max_edges=None,
                    splitting='source/version connected components joined by unordered endpoint pairs; fixed context train only; 80/10/10 hash seed123',
                    negative_semantics='unobserved corruptions, not scientifically false statements',
                    scientific_validation=False, unknown_source_family_links_possible=True,
                    evaluation_oov_moved_to_training=False, graph_modified=False)
    write(output / 'DATASET.json', manifest)
    progress('prepared', **{k: manifest[k] for k in ('counts','entities','relations','folds')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
