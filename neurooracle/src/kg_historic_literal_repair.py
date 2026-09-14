"""Finite, reviewed restoration of complete mentions from wrong gene identities.

These are ordinary literal concepts, not inferred imaging markers, synonyms,
single molecular entities, new claims, or new scientific/source validations.
The R47 allowlist binds every original claim, side, complete name and gene hash.
"""
import hashlib

from .claim_semantics import declared_type_atoms
from .kg_bulk_identity import change_claim
from .kg_gene_boundary_repair import word_interior_hits
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import (edge_owner, reviewed_edges, apply_edge,
    reverse_claim, literal_node as imaging_literal_node)
from .relation_evidence import name_key
from .schema import ConceptNode

VERSION = 'kg.historic_complete_literal.v1'
GENES = {'CUI:C1414531', 'CUI:C1421437'}
REUSABLE_NAMES = {'frontal cortical surface area', 'parietal cortical surface area'}


def review_key(cid, side):
    return cid + '|' + side


def literal_node(name):
    name = name_key(name)
    if not name:
        raise ValueError('missing complete name')
    key = hashlib.sha256((VERSION + '|' + name).encode('utf8')).hexdigest()
    return ConceptNode(id='CLM_CONCEPT:complete_mention_' + key,
        preferred_name=name, domain_tags=['claim_concept'],
        source_vocab='claim_extraction').to_dict()


def endpoint_gate(record, side, witness, review):
    if side not in {'subject', 'object'}:
        return 'invalid_side'
    md = record['metadata']; nid = md.get(side + '_id')
    if not review or review.get('claim_id') != record['id'] or review.get('side') != side:
        return 'not_in_finite_reviewed_scope'
    if review.get('claim_sha256') != digest(record):
        return 'reviewed_claim_changed'
    if review.get('current_gate') is not None:
        return 'historic_review_still_held'
    if nid not in GENES or nid != witness.get('node_id') or review.get('current_node_id') != nid:
        return 'wrong_reviewed_gene'
    if 'T028' not in (witness.get('semantic_types') or []) or not witness.get('name'):
        return 'not_confirmed_gene_witness'
    if not witness.get('node_sha256') or review.get('source_gene_node_sha256') != witness['node_sha256']:
        return 'unbound_gene_witness'
    name = md.get(side + '_name')
    if not isinstance(name, str) or not name_key(name) or name != review.get('name'):
        return 'complete_original_name_changed'
    hits = word_interior_hits(name, witness.get('labels') or [])
    if not hits or hits != review.get('word_interior_alias_hits'):
        return 'not_reviewed_word_interior_only'
    if review.get('identity_scope') != 'complete_original_mention_not_single_VCP_or_FANCE_gene':
        return 'unreviewed_identity_scope'
    inner = md.get('metadata') or {}; field = side + '_id'
    if inner.get(field, nid) != nid:
        return 'nested_id_conflict'
    outer_type, inner_type = md.get(side + '_type'), inner.get(side + '_type')
    if (outer_type, inner_type) != (review.get('outer_type'), review.get('inner_type')):
        return 'reviewed_types_changed'
    if outer_type and inner_type and declared_type_atoms(outer_type) != declared_type_atoms(inner_type):
        return 'nested_type_conflict'
    return None


def reuse_gate(node, name, incidents):
    # Other same-name concepts include source-scoped metadata, differing
    # measurement definitions or broad/specific mixtures. Hold them separately.
    if name_key(name) not in REUSABLE_NAMES or node != imaging_literal_node(name):
        return 'existing_literal_needs_separate_scope_review'
    if not incidents:
        return 'existing_incidence_proof_missing'
    for row in incidents:
        if name_key(row.get('name')) != name_key(name):
            return 'existing_incident_scope_differs'
        if set(row.get('declared_roles') or []) not in (set(), {'imaging_marker'}):
            return 'existing_incident_type_conflict'
    return None


def reviewed_claim(record, changes, witnesses, reviews):
    if not changes:
        raise ValueError('empty repair')
    seen = set()
    for change in changes:
        side = change['side']
        if side in seen:
            raise ValueError('duplicate side')
        seen.add(side)
        reason = endpoint_gate(record, side, witnesses.get(change['old_id'], {}),
            reviews.get(review_key(record['id'], side)))
        if reason:
            raise ValueError(reason)
        if change['old_id'] != record['metadata'][side + '_id'] or change['name'] != name_key(record['metadata'][side + '_name']):
            raise ValueError('complete endpoint differs')
        expected = literal_node(change['name'])['id']
        reused = imaging_literal_node(change['name'])['id'] if change['name'] in REUSABLE_NAMES else None
        allowed = {expected}
        if reused is not None:
            allowed.add(reused)
        if change['target_id'] not in allowed:
            raise ValueError('unreviewed literal target')
    event = dict(claim_id=record['id'], claim_sha256=digest(record), changes=changes,
        nonidentity_sha256=digest(nonidentity_claim(record)))
    out = change_claim(record, event)
    event['current_node_sha256'] = digest(out)
    return event, out
