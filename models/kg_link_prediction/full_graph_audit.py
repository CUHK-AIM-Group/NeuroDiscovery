"""Read-only reconciliation of raw graph records against a frozen training set."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .full_graph_data import (ROOT, CORPUS, PAPERS, INPUTS, fingerprint, provenance,
                              read, sha, write)
from neurooracle.scripts.streaming_graph_json import iter_concepts, iter_edges
from neurooracle.src.kge.triple_loader import _DROP_RELATIONS, _INFRA_DOMAINS


def exclusion(edge, nodes):
    """The documented eligibility policy, audited before consulting saved rows."""
    s, r, t = (str(edge.get(k) or '') for k in ('source_id', 'relation_type', 'target_id'))
    if not all((s, r, t)):
        return 'missing'
    if s == t:
        return 'self_loop'
    if r in _DROP_RELATIONS:
        return 'infrastructure_relation'
    if s not in nodes or t not in nodes:
        return 'dangling'
    if nodes[s][1] or nodes[t][1]:
        return 'infrastructure_node'
    confidence = float(edge.get('confidence', 0) or 0)
    if not math.isfinite(confidence) or confidence < .2:
        return 'low_nonfinite_confidence'
    if edge.get('negated') or (edge.get('metadata') or {}).get('negated'):
        return 'negated'
    return None


def check_pair_folds(triples, folds, entity_count):
    pairs = np.sort(triples[:, [0, 2]], axis=1).astype(np.uint64)
    keys = pairs[:, 0] * np.uint64(entity_count) + pairs[:, 1]
    order = np.argsort(keys)
    leaks = int(((keys[order][1:] == keys[order][:-1]) &
                 (folds[order][1:] != folds[order][:-1])).sum())
    if leaks:
        raise ValueError(f'Endpoint pairs cross folds: {leaks}')
    return leaks


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dataset = read(run/'data/DATASET.json')
    vocabulary = read(run/'data/vocabulary.json')
    data = np.load(run/'data/data.npz', allow_pickle=False)
    triples, folds = data['triples'], data['split']
    entities = {v: k for k, v in enumerate(vocabulary['entities'])}
    relations = {v: k for k, v in enumerate(vocabulary['relations'])}
    # A tuple index is intentionally separate from the preparer's sort/dedup code.
    row_index = {tuple(map(int, row)): i for i, row in enumerate(triples)}
    if len(row_index) != len(triples):
        raise ValueError('Duplicate frozen triples')
    for name, digest in dataset['files'].items():
        if sha(run/'data'/name) != digest:
            raise ValueError(f'Frozen data changed: {name}')
    graph = Path(dataset['graph'])
    boundary_path = ROOT/'tmp/kg_paper_reconstruction_20260920/FIXED_BOUNDARY.json'
    inputs = [Path(p) for p in dataset['source_fingerprint']] + [boundary_path]
    before = {str(p): fingerprint(p) for p in inputs}
    for p, expected in dataset['source_fingerprint'].items():
        if before[p] != expected:
            raise ValueError(f'Input identity changed: {p}')
    hashes = {str(p): sha(p) for p in inputs}
    if hashes[str(graph)] != dataset['graph_revision']:
        raise ValueError('Graph digest changed')
    claims, pubs, families, protected = provenance()
    if protected != set(read(run/'data/PROTECTED_WORKS.json')):
        raise ValueError('Protected inventory changed')
    saved_source_folds = read(run/'data/source_folds.json')
    boundary = read(boundary_path)
    if boundary['original_graph']['sha256'] != dataset['graph_revision']:
        raise ValueError('Fixed boundary mismatch')
    fixed = {x['source'] for x in boundary['provenance_decisions']
             if x['kind'] == 'edge' and x['decision'] in {'fixed_reference', 'protected_existing_alignment'}}
    fixed |= {'MeSH_hierarchy', 'NeuroLex', 'UBERON', 'FMA', 'HPO', 'DisGeNET',
              'Allen_Human_Brain_Atlas', 'Allen_Brain_Atlas', 'Neurotransmitters',
              'CognitiveAtlas', 'Cognitive_Atlas', 'HGNC', 'GO', 'KEGG', 'DrugBank',
              'ChEBI', 'ATC', 'ClinicalOutcomes', 'UMLS', 'UMLS_alignment'}
    print(json.dumps(dict(stage='audit_nodes', elapsed=time.monotonic()-started)), flush=True)
    nodes = {}
    for nid, node in iter_concepts(graph):
        tags = node.get('domain_tags') or []
        nodes[nid] = (str(tags[0]) if tags else 'unknown', bool(_INFRA_DOMAINS.intersection(tags)))
    domain_names = np.asarray(vocabulary['domains'])
    domain_ids = data['domain_ids']  # NPZ indexing reloads an entire member each time.
    if any(nodes[name][0] != domain_names[domain_ids[i]] for name, i in entities.items()):
        raise ValueError('Training node domain mismatch')
    records, exclusions = Counter(), Counter()
    per_relation = defaultdict(Counter)
    protected_pairs, protected_works_seen = set(), Counter()
    source_ids = {}
    ordinals, indices, sources = array('I'), array('I'), array('I')
    missing_records = []
    for ordinal, edge in enumerate(iter_edges(graph), 1):
        relation = str(edge.get('relation_type') or '')
        tally = per_relation[relation]
        tally['raw_records'] += 1
        reason = exclusion(edge, nodes)
        if reason:
            tally[reason] += 1
            exclusions[reason] += 1
            continue
        s, t = str(edge['source_id']), str(edge['target_id'])
        meta, source = edge.get('metadata') or {}, str(edge.get('source') or '')
        family = claims.get(str(meta.get('claim_id') or ''))
        if not family and source.startswith('claim:') and source[6:].isdigit():
            pub = 'PMID:' + source[6:]
            family = families.find(pubs.get(pub, pub))
        if not family and source in fixed:
            family = 'FIXED:' + source
        if not family:
            tally['unresolved_provenance'] += 1
            exclusions['unresolved_provenance'] += 1
            continue
        pair = tuple(sorted((s, t)))
        if family in protected:
            tally['protected_work'] += 1
            exclusions['protected_work'] += 1
            protected_pairs.add(pair)
            protected_works_seen[family] += 1
            continue
        if s not in entities or t not in entities or relation not in relations:
            missing_records.append((pair, relation))
            continue
        index = row_index.get((entities[s], relations[relation], entities[t]))
        if index is None:
            missing_records.append((pair, relation))
            continue
        if family not in saved_source_folds or saved_source_folds[family] != int(folds[index]):
            raise ValueError(f'Source/fold mismatch at raw edge {ordinal}')
        if family.startswith('FIXED:') and folds[index] != 0:
            raise ValueError('Fixed context in evaluation')
        source_id = source_ids.setdefault(family, len(source_ids))
        ordinals.append(ordinal)
        indices.append(index)
        sources.append(source_id)
        records[index] += 1
        tally['included_records'] += 1
        if ordinal % 500000 == 0:
            print(json.dumps(dict(stage='audit_edges', raw_edges=ordinal,
                                  elapsed=time.monotonic()-started)), flush=True)
    for pair, relation in missing_records:
        if pair not in protected_pairs:
            raise ValueError(f'Eligible raw edge missing from dataset: {relation}, {pair}')
        per_relation[relation]['protected_endpoint_proxy'] += 1
        exclusions['protected_endpoint_proxy'] += 1
    for s, _, t in triples:
        if tuple(sorted((vocabulary['entities'][s], vocabulary['entities'][t]))) in protected_pairs:
            raise ValueError('Protected endpoint proxy retained')
    if len(records) != len(triples):
        raise ValueError('Frozen triples lack raw graph provenance')
    trained_nodes = np.zeros(len(entities), dtype=bool)
    trained_relations = np.zeros(len(relations), dtype=bool)
    train = triples[folds == 0]
    trained_nodes[train[:, [0, 2]]] = True
    trained_relations[train[:, 1]] = True
    np.testing.assert_array_equal(trained_nodes, data['trained_nodes'])
    np.testing.assert_array_equal(trained_relations, data['trained_relations'])
    measurable = trained_nodes[triples[:, 0]] & trained_nodes[triples[:, 2]] & trained_relations[triples[:, 1]]
    for relation, ri in relations.items():
        tally = per_relation[relation]
        mask = triples[:, 1] == ri
        tally['unique_included'] = int(mask.sum())
        tally['duplicate_records'] = tally['included_records'] - tally['unique_included']
        for fold, label in enumerate(('train', 'validation', 'test')):
            tally[label] = int((mask & (folds == fold)).sum())
            tally[label+'_measurable'] = int((mask & (folds == fold) & measurable).sum())
    for tally in per_relation.values():
        for key in ('included_records', 'unique_included', 'duplicate_records', 'train', 'validation', 'test',
                    'train_measurable', 'validation_measurable', 'test_measurable'):
            tally.setdefault(key, 0)
    raw_count = sum(row['raw_records'] for row in per_relation.values())
    if raw_count != len(indices) + sum(exclusions.values()):
        raise ValueError('Raw relation accounting incomplete')
    for key, count in exclusions.items():
        if count != dataset['counts'].get(key, 0):
            raise ValueError(f'Exclusion discrepancy: {key}')
    if raw_count != dataset['counts']['raw_edges'] or len(nodes) != dataset['counts']['raw_nodes']:
        raise ValueError('Full graph inventory discrepancy')
    pair_leaks = check_pair_folds(triples, folds, len(entities))
    if before != {str(p): fingerprint(p) for p in inputs}:
        raise ValueError('Source changed during read-only audit')
    write(output/'RELATION_ACCOUNTING.json', dict(sorted(per_relation.items())))
    write(output/'SOURCE_FAMILIES.json', list(source_ids))
    np.savez_compressed(output/'EDGE_PROVENANCE.npz', raw_edge_ordinal=np.asarray(ordinals, dtype=np.uint32),
                        dataset_row=np.asarray(indices, dtype=np.uint32), source_family=np.asarray(sources, dtype=np.uint32))
    result = dict(status='FULL_RAW_GRAPH_RECONCILIATION_PASSED', graph_sha256=dataset['graph_revision'],
                  dataset_sha256=sha(run/'data/DATASET.json'), raw_nodes=len(nodes), raw_edges=raw_count,
                  unique_included=len(triples), included_raw_records=len(indices), relation_types_raw=len(per_relation),
                  relation_types_included=len(relations), exclusions=dict(exclusions),
                  every_raw_edge_accounted=True, every_dataset_row_has_raw_provenance=True,
                  source_and_fixed_fold_mismatches=0, endpoint_pair_leaks=pair_leaks,
                  protected_family_count=len(protected), protected_families_seen=dict(protected_works_seen),
                  protected_endpoint_pairs=len(protected_pairs), protected_source_or_pair_leaks=0,
                  trained_exposure_masks_recomputed=True, input_hashes=hashes, input_stats_unchanged=True,
                  seconds=time.monotonic()-started, source_abstracts_read=0, scientific_validation=False,
                  verifier_code_sha256=sha(__file__),
                  files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write(output/'AUDIT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'raw_edges', 'unique_included', 'relation_types_raw', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
