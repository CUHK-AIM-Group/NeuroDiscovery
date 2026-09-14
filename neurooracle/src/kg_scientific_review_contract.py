"""Proposed review-card acceptance policy; NOT imported by KG ingestion/runtime.

Inputs are source-reviewed facts, not facts inferred from labels or confidence.
This evaluates a human-auditable decision record; it cannot certify an unseen
paper, resolve an alias, or estimate literature-wide scientific equivalence.
"""
SCOPE_FIELDS=('laterality','aggregation','measure','modality','biotype','variant')
VERSION='kg.scientific_identity_review.v1'
ASSERTION_ISSUES={None,'causal_overstatement','classifier_overstatement','aim_not_result','source_internal_conflict','hypothesis_not_result','analysis_scope_overstatement','insufficient_source'}


def evaluate(card):
    if card.get('assertion_issue') not in ASSERTION_ISSUES:raise ValueError('unknown scientific issue, do not silently approve')
    if len(card['observations'])!=2:raise ValueError('exactly two observations required')
    left,right=card['observations']
    for observation in (left,right):
        if type(observation['negated']) is not bool:raise ValueError('typed negation required')
        for side in ('subject','object'):
            ep=observation[side]
            if type(ep.get('identity_proof')) is not bool:raise ValueError('explicit review boolean required')
            if ep.get('node_kind') not in {'entity','claim'}:raise ValueError('unknown endpoint kind')
            if set(ep['scope'])!=set(SCOPE_FIELDS) or any(not isinstance(v,str) or not v for v in ep['scope'].values()):raise ValueError('complete explicit scope required')
    blockers=[];unknown=[]
    for side in ('subject','object'):
        a,b=left[side],right[side]
        if a.get('node_kind')=='claim' or b.get('node_kind')=='claim':
            blockers.append(side+':claim_node_is_not_entity')
        if not a.get('identity_proof') or not b.get('identity_proof'):
            unknown.append(side+':identity_unproven')
        elif a['concept']!=b['concept']:
            blockers.append(side+':different_concept')
        for field in SCOPE_FIELDS:
            x,y=a['scope'][field],b['scope'][field]
            if x=='unknown' or y=='unknown':unknown.append(side+':'+field+':unknown')
            elif x!=y:blockers.append(side+':'+field+':different')
    if blockers:relation='keep_distinct_or_repair_invalid_endpoint'
    elif unknown:relation='hold_identity_or_scope'
    elif left['predicate']!=right['predicate']:
        # No global associated/correlated/causal/diagnostic equivalence. Any
        # normalization must be explicitly source-reviewed for this pair.
        if card.get('predicate_equivalence_reviewed') is True and {left['predicate'],right['predicate']}=={'is_associated_with','correlates_with'}:
            relation='share_relation'
        else:relation='keep_distinct_predicates'
    else:relation='share_relation'
    assertion='eligible_at_reviewed_scope'
    if card.get('assertion_issue') in {'causal_overstatement','classifier_overstatement','aim_not_result','source_internal_conflict','hypothesis_not_result','analysis_scope_overstatement'}:
        assertion='hold_or_repair_scientific_assertion'
    elif card.get('assertion_issue')=='insufficient_source':assertion='hold_source_evidence'
    if assertion!='eligible_at_reviewed_scope' and relation=='share_relation':relation='hold_scientific_assertion'
    return dict(relation_decision=relation,assertion_decision=assertion,blockers=blockers,unknown=unknown,
        evidence_decision='preserve_each_observation',automatic_evidence_deletion=False,
        independent_studies_inferred=False,consensus_inferred=False,
        polarity_difference=left['negated']!=right['negated'],
        same_source_key=left.get('paper_key')==right.get('paper_key'),
        denominator_must_remain_per_analysis=True)
