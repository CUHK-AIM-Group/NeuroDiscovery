"""Restore full imaging-mention identity after a word-interior gene alias hit.

Only explicit imaging markers with concrete measurement surfaces qualify.
Actual gene mentions, complete-token aliases, molecular/assay/procedure scopes,
missing types and conflicting nested fields stay held. This is an identity
repair, not a new extraction, a source verdict, or a semantic synonym merge.
"""
from .claim_semantics import (declared_type_atoms, looks_like_concrete_imaging_measurement,
    looks_like_non_imaging_assay_entity, looks_like_method_or_procedure_entity)
from .kg_bulk_identity import change_claim
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import (MOLECULAR, literal_node, reviewed_edges,
    edge_owner, apply_edge, reverse_claim)
from .relation_evidence import name_key
import re

VERSION='kg.gene_boundary_repair.v1'


def word_interior_hits(name, labels):
    if not isinstance(name,str) or not name.strip():return []
    text=name.strip().casefold()
    labels={s.strip().casefold() for s in labels if isinstance(s,str) and s.strip()}
    if text in labels or any(s in text and re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text) for s in labels):return []
    return sorted(s for s in labels if s in text)


def endpoint_gate(md, side, witness):
    if side not in {'subject','object'}:return 'invalid_side'
    nid=md.get(side+'_id')
    if not nid or nid!=witness.get('node_id') or not nid.startswith('CUI:'):return 'wrong_gene_witness'
    if 'T028' not in (witness.get('semantic_types') or []):return 'not_confirmed_gene_type'
    if not witness.get('node_sha256') or not witness.get('name'):return 'unbound_gene_witness'
    inner=md.get('metadata') or {};f=side+'_id'
    if inner.get(f,nid)!=nid:return 'nested_id_conflict'
    outer_type,inner_type=md.get(side+'_type'),inner.get(side+'_type')
    roles=lambda v:{a.value for a in declared_type_atoms(v)}
    if outer_type and inner_type and roles(outer_type)!=roles(inner_type):return 'nested_type_conflict'
    if roles(outer_type or inner_type)!={'imaging_marker'}:return 'not_explicit_imaging_type'
    name=md.get(side+'_name')
    if not isinstance(name,str) or not name_key(name):return 'missing_complete_name'
    if not word_interior_hits(name,witness.get('labels') or []):return 'not_word_interior_only'
    if MOLECULAR.search(name):return 'molecular_or_genetic_context'
    if looks_like_non_imaging_assay_entity(name) or looks_like_method_or_procedure_entity(name):return 'assay_or_method_context'
    if not looks_like_concrete_imaging_measurement(name):return 'not_unambiguous_measurement_surface'
    return None


def reviewed_claim(record,changes,witnesses):
    md=record['metadata'];seen=set()
    if not changes:raise ValueError('empty repair')
    for ch in changes:
        side=ch['side']
        if side in seen:raise ValueError('duplicate side')
        seen.add(side)
        witness=witnesses.get(ch['old_id'],{})
        reason=endpoint_gate(md,side,witness)
        if reason:raise ValueError(reason)
        if ch['old_id']!=md[side+'_id'] or ch['name']!=name_key(md[side+'_name']):raise ValueError('source complete identity differs')
        if not ch['target_id'].startswith('CLM_CONCEPT:') or ch['target_id']==ch['old_id']:raise ValueError('not literal concept target')
    event=dict(claim_id=record['id'],claim_sha256=digest(record),changes=changes,
        nonidentity_sha256=digest(nonidentity_claim(record)))
    out=change_claim(record,event);event['current_node_sha256']=digest(out)
    return event,out
