"""R55 exact existing whole-name reuse; no new concepts or scientific edits."""
from .kg_bulk_identity import change_claim
from .kg_expanded_literal_repair import endpoint_gate as source_endpoint_gate
from .kg_historic_literal_repair import review_key
from .kg_identity_pilot import digest,nonidentity_claim
from .kg_literal_endpoint_repair import edge_owner,reviewed_edges,apply_edge,reverse_claim
from .relation_evidence import name_key
from neurooracle.scripts.inspect_kg_existing_literal_reuse import candidate_node_gate

VERSION='kg.existing_literal_reuse.v1'


def literal_node(name):
    raise ValueError('R55 never creates literal nodes')


def endpoint_gate(record,side,witness,review):
    if not review or review.get('reuse_candidate_gate') is not None:return 'not_a_reviewed_reuse'
    target=review.get('target_witness') or {}
    reason=candidate_node_gate(review,target)
    if reason:return reason
    if review.get('existing_name_or_alias_candidates')!=[review.get('target_id')]:return 'target_not_the_unique_existing_candidate'
    if review.get('target_id')!=target.get('node_id') or review.get('target_node_sha256')!=target.get('node_sha256'):
        return 'existing_target_not_hash_bound'
    cleared=dict(review,literal_candidate_gate=None,existing_name_or_alias_candidates=[])
    return source_endpoint_gate(record,side,witness,cleared)


def reviewed_claim(record,changes,witnesses,reviews):
    if not changes:raise ValueError('empty reuse')
    seen=set()
    for ch in changes:
        side=ch['side'];review=reviews.get(review_key(record['id'],side))
        if side in seen:raise ValueError('duplicate side')
        seen.add(side)
        reason=endpoint_gate(record,side,witnesses.get(ch['old_id'],{}),review)
        if reason:raise ValueError(reason)
        if ch['old_id']!=record['metadata'][side+'_id'] or ch['name']!=name_key(record['metadata'][side+'_name']):
            raise ValueError('original complete endpoint differs')
        if ch['target_id']!=review['target_id'] or ch['target_id']==ch['old_id']:
            raise ValueError('unreviewed or unchanged existing target')
    event=dict(claim_id=record['id'],claim_sha256=digest(record),changes=changes,nonidentity_sha256=digest(nonidentity_claim(record)))
    current=change_claim(record,event);event['current_node_sha256']=digest(current)
    return event,current
