"""Finite R65 nominal identity consolidation, never a measurement equivalence.

All names remain case-sensitive and complete. Original role labels, contexts,
audit, raw text and source evidence remain per claim. Six source anchors remain
because the immutable detail store references them; only two reference-free
duplicates can be retired. This is not a general name-based merge rule.
"""
from collections import defaultdict
import hashlib
import re

from .kg_bulk_identity import change_claim
from .kg_gene_boundary_repair import word_interior_hits
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import edge_owner, reviewed_edges, apply_edge, reverse_claim
from neurooracle.scripts.build_umls_simplification_candidate import compact
from neurooracle.scripts.inspect_kg_claim_deletion import exact_references
from neurooracle.scripts.kg_accepted_candidate_lineage import require

VERSION = 'kg.finite_nominal_consolidation.v1'
ROLE_SCOPE = {
    'hippocampal and amygdala volume': {'imaging_marker'},
    'incident dementia': {'outcome', 'clinical_outcome', 'disease', 'disease_outcome',
        'dementia_outcome', 'dementia_risk', 'neurocognitive_outcome', 'prediction_outcome', 'clinical_event'},
    'major depressive disorder severity': {'outcome', 'clinical_feature', 'clinical_phenotype',
        'disease', 'mood_disorder_clinical_feature', 'symptom_severity'},
    'neuromelanin-sensitive MRI substantia nigra signal': {'imaging_marker'},
    'resting-state functional connectivity patterns': {'imaging_marker', 'biomarker',
        'migraine_rsfc_biomarker', 'rs_fmri_biomarker', 'resting_state_functional_connectivity_pattern'},
    'supplementary motor area activation': {'imaging_marker', 'brain_region_measure'},
    'total gray matter volume': {'imaging_marker', 'biomarker', 'brain_structure', 'outcome'},
}
IMPORT_KEYS = {'anchor_role', 'atom_type', 'atom_types', 'curation_scope', 'staging_source'}


def check_types(name, outer, inner):
    require(name in ROLE_SCOPE, 'unreviewed complete name')
    require(all(not v or isinstance(v, str) and v.casefold() in ROLE_SCOPE[name]
                for v in (outer, inner)), 'unreviewed scoped role')


def check_node(witness, name):
    require(witness['name'] == name, 'complete primary name differs')
    require(not any(witness[k] for k in ('aliases', 'semantic_types', 'external_ids')),
            'external or alias identity requires separate proof')
    require(witness['definition_sha256'] == digest('') and witness['spatial_mapping_sha256'] == digest(None),
            'measurement or spatial definition requires separate proof')
    require(set(witness['metadata_keys']) <= IMPORT_KEYS, 'unreviewed concept metadata')
    require(witness['source_vocab'] in {'replay_anchor_mint', 'manual_claim_anchor', 'manual_general_claim_anchor'},
            'unreviewed concept origin')
    require(set(witness['domain_tags']) <= {'external', 'claim_concept', 'treatment_outcome', 'imaging_feature', 'biomarker'},
            'unreviewed domain')
    md = witness['reviewed_metadata']
    for role in [md.get('atom_type'), *(md.get('atom_types') or [])]:
        check_types(name, role, None)


def reviewed_claim(record, event, plan):
    require(record['id'] == event['claim_id'] and digest(record) == event['claim_sha256'], 'reviewed claim changed')
    require(digest(nonidentity_claim(record)) == event['nonidentity_sha256'], 'scientific context changed')
    seen = set(); md = record['metadata']; inner = md.get('metadata') or {}
    for ch in event['changes']:
        side = ch['side']; key = record['id'] + '|' + side
        require(side in {'subject', 'object'} and side not in seen, 'duplicate or invalid side'); seen.add(side)
        review = plan['endpoint_reviews'][key]
        require(review['claim_sha256'] == event['claim_sha256'] and review['change'] == ch, 'finite endpoint differs')
        require(md[side + '_id'] == ch['old_id'] and md[side + '_name'] == ch['name'], 'complete source endpoint differs')
        require(inner.get(side + '_id', ch['old_id']) == ch['old_id'], 'nested endpoint conflict')
        require((md.get(side + '_type'), inner.get(side + '_type')) == (review['outer_type'], review['inner_type']),
                'original type labels changed')
        check_types(ch['name'], review['outer_type'], review['inner_type'])
        require(plan['canonical_targets'][ch['name']] == ch['target_id'] != ch['old_id'], 'unapproved target')
        check_node(plan['target_witnesses'][ch['target_id']], ch['name'])
        if ch['reason'] == 'same_nominal_concept':
            require(plan['redirects'].get(ch['old_id']) == ch['target_id'], 'unreviewed redirect')
            check_node(plan['target_witnesses'][ch['old_id']], ch['name'])
        else:
            require(ch['reason'] == 'gene_endpoint_to_existing_nominal_concept', 'unreviewed repair reason')
            gene = plan['gene_witnesses'][ch['old_id']]
            require(gene['node_id'] == ch['old_id'] and 'T028' in gene['semantic_types'], 'canonical gene witness missing')
            labels = [gene['name'], *gene['aliases']]
            require(word_interior_hits(ch['name'], labels) == review['word_interior_alias_hits']
                    and review['word_interior_alias_hits'], 'gene word-interior mismatch not reproduced')
    require(seen, 'empty endpoint mutation')
    current = change_claim(record, event)
    require(digest(current) == event['current_node_sha256']
            and digest(nonidentity_claim(current)) == event['nonidentity_sha256'], 'identity-only output differs')
    return current


def no_retired_references(kind, key, row, plan):
    removed = {r['node_id'] for r in plan['removed_nodes']}
    require(not (kind == 'node' and key in removed), 'retired node remains')
    payload = compact(row)
    if any(nid in payload for nid in removed):
        require(not list(exact_references(row, removed)), 'retired exact reference remains')
    losers = set(plan['redirects'])
    if kind == 'node' and key.startswith('CLM:'):
        md = row['metadata']; inner = md.get('metadata') or {}
        require(not {v.get(s + '_id') for v in (md, inner) for s in ('subject', 'object')} & losers,
                'old nominal scientific endpoint remains')
    elif kind == 'edge':
        require(not {row.get('source_id'), row.get('target_id')} & losers, 'old nominal edge endpoint remains')


def incidence_rows(key, row, targets):
    if not key.startswith('CLM:'): return []
    md = row['metadata']; inner = md.get('metadata') or {}
    return [dict(node_id=md[s + '_id'], claim_id=key, side=s, name=md.get(s + '_name'),
                 claim_sha256=digest(row), outer_type=md.get(s + '_type'), inner_type=inner.get(s + '_type'))
            for s in ('subject', 'object') if md.get(s + '_id') in targets]


def ordered_incidences(values):
    return sorted(values, key=lambda r: (r['node_id'], r['claim_id'], r['side']))


class NominalTransform:
    """Source rolling hashes omit only the two hash-bound retired concepts.

    The independent scan reverses endpoint changes and reproduces every retained
    original record, without saving or reconstructing deleted record preimages.
    """
    def __init__(self, plan):
        self.plan = plan
        self.events = {e['claim_id']: e for e in plan['events']}
        self.edges = {e['ordinal']: e for e in plan['edge_events']}
        self.removed = {r['node_id']: r for r in plan['removed_nodes']}
        require(len(self.events) == len(plan['events']) and len(self.edges) == len(plan['edge_events'])
                and len(self.removed) == len(plan['removed_nodes']), 'duplicate mutation identity')
        self.digests = {k: hashlib.sha256() for k in ('nodes', 'edges')}
        self.originals = {}; self.refs = defaultdict(list)
        self.seen_edges = set(); self.seen_targets = set(); self.seen_removed = set(); self.incidences = []

    def records(self, records):
        for kind, key, row in records:
            if kind == 'node' and key in self.plan['existing_targets']:
                require(digest(row) == self.plan['existing_targets'][key], 'hash-bound concept changed')
                self.seen_targets.add(key)
            if kind == 'node' and key in self.removed:
                r = self.removed[key]
                require(digest(row) == r['node_sha256'] and not r['reasons']
                        and not any(r['detail_references'].values()), 'unsafe concept retirement')
                self.seen_removed.add(key); continue
            if kind != 'metadata': self.digests[kind + 's'].update(compact(row).encode() + b'\n')
            out = row
            if kind == 'node':
                self.incidences.extend(incidence_rows(key, row, self.plan['target_witnesses']))
                if key in self.events:
                    out = reviewed_claim(row, self.events[key], self.plan); self.originals[key] = row
            elif kind == 'edge':
                owner = edge_owner(row)
                if owner in self.events: self.refs[owner].append((int(key), row))
                if int(key) in self.edges:
                    out = apply_edge(row, self.edges[int(key)]); self.seen_edges.add(int(key))
            no_retired_references(kind, key, out, self.plan)
            yield kind, key, out
        require(set(self.originals) == set(self.events) and self.seen_edges == set(self.edges), 'changed record closure differs')
        require(self.seen_targets == set(self.plan['existing_targets']) and self.seen_removed == set(self.removed), 'concept closure differs')
        require(ordered_incidences(self.incidences) == self.plan['source_incidences'], 'complete source incidence scope differs')
        reproduced = []
        for cid, row in self.originals.items():
            reproduced.extend(reviewed_edges(cid, row, reviewed_claim(row, self.events[cid], self.plan), self.refs[cid]))
        require(sorted(reproduced, key=lambda e: e['ordinal']) == self.plan['edge_events'], 'complete owned edge review differs')
