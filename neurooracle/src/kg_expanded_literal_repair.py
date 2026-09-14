"""Finite non-molecular complete-mention repairs from R52 current-source evidence."""
from .claim_semantics import declared_type_atoms
from .kg_bulk_identity import change_claim
from .kg_gene_boundary_repair import word_interior_hits
from .kg_historic_literal_repair import literal_node,review_key
from .kg_identity_pilot import digest,nonidentity_claim
from .kg_literal_endpoint_repair import edge_owner,reviewed_edges,apply_edge,reverse_claim
from .relation_evidence import name_key
from neurooracle.scripts.classify_kg_expanded_literal_candidates import candidate_gate

VERSION='kg.expanded_nonmolecular_literals.v1'


def endpoint_gate(record,side,witness,review):
    if side not in {'subject','object'}:return 'invalid_side'
    md=record['metadata'];nid=md.get(side+'_id');inner=md.get('metadata') or {}
    if not review or review.get('claim_id')!=record['id'] or review.get('side')!=side:return 'not_in_finite_reviewed_scope'
    if review.get('claim_sha256')!=digest(record):return 'reviewed_claim_changed'
    if review.get('literal_candidate_gate') is not None or candidate_gate(review,set()) is not None:return 'finite_nonmolecular_scope_not_reproduced'
    if nid!=witness.get('node_id') or nid!=review.get('current_node_id') or 'T028' not in witness.get('semantic_types',[]):return 'source_gene_witness_differs'
    if not witness.get('node_sha256') or witness['node_sha256']!=review.get('source_gene_node_sha256'):return 'gene_hash_unbound'
    if md.get(side+'_name')!=review.get('name'):return 'complete_original_name_changed'
    hits=word_interior_hits(md[side+'_name'],witness.get('labels',[]))
    if not hits or hits!=review.get('word_interior_alias_hits'):return 'word_interior_only_proof_changed'
    if inner.get(side+'_id',nid)!=nid:return 'nested_id_conflict'
    outer_type,inner_type=md.get(side+'_type'),inner.get(side+'_type')
    if (outer_type,inner_type)!=(review.get('outer_type'),review.get('inner_type')):return 'source_types_changed'
    roles=sorted(a.value for a in declared_type_atoms(outer_type or inner_type))
    if roles!=review.get('declared_roles'):return 'source_roles_changed'
    if outer_type and inner_type and declared_type_atoms(outer_type)!=declared_type_atoms(inner_type):return 'nested_type_conflict'
    return None


def reviewed_claim(record,changes,witnesses,reviews):
    if not changes:raise ValueError('empty repair')
    seen=set()
    for ch in changes:
        side=ch['side']
        if side in seen:raise ValueError('duplicate side')
        seen.add(side)
        reason=endpoint_gate(record,side,witnesses.get(ch['old_id'],{}),reviews.get(review_key(record['id'],side)))
        if reason:raise ValueError(reason)
        if ch['old_id']!=record['metadata'][side+'_id'] or ch['name']!=name_key(record['metadata'][side+'_name']):
            raise ValueError('complete endpoint differs')
        if ch['target_id']!=literal_node(ch['name'])['id']:raise ValueError('nonliteral or paper-salted target')
    event=dict(claim_id=record['id'],claim_sha256=digest(record),changes=changes,nonidentity_sha256=digest(nonidentity_claim(record)))
    out=change_claim(record,event);event['current_node_sha256']=digest(out)
    return event,out
