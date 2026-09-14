"""R72 finite relation repairs and opt-in, source-bound reimport consistency.

Rules never classify an unseen article or strip endpoint qualifiers. The exact
reviewed source sentence, polarity, complete relation and declared roles must
match. Target nodes are checked again on every applicable import. Unknown
claim fields and original scientific evidence remain untouched.
"""
from copy import deepcopy

from .kg_identity_pilot import digest
from .kg_root_repairs import event, owned_endpoint_changes
from .kg_literal_endpoint_repair import reviewed_edges

VERSION = 'kg.reviewed_relation_repairs.r72.v1'
APPROVED_RULESET_DIGEST = 'bd39416c0ab023272e6216d9b9cc2608aa677bc32137ec93a939ce85e2ced5f2'
FIELDS = ('subject_id','subject_name','predicate','object_id','object_name')


def role_values(metadata, side):
    inner=metadata.get('metadata') or {}
    return sorted({str(v) for v in (metadata.get(side+'_type'),inner.get(side+'_type')) if v not in (None,'')})


def revise_claim(record, replacements, *, review_id):
    if not replacements or not set(replacements)<=set(FIELDS):
        raise ValueError('unapproved claim field')
    out=deepcopy(record);md=out['metadata'];inner=md.get('metadata') or {}
    for field,new in replacements.items():
        old=md[field]
        if field in inner:
            if inner[field]!=old:raise ValueError('conflicting nested claim field')
            inner[field]=new
        md[field]=new
    if md['subject_id']==md['object_id']:raise ValueError('new endpoint collapse')
    expected=' '.join(record['metadata'][k] for k in ('subject_name','predicate','object_name'))
    if record['preferred_name']!=expected:raise ValueError('claim label has unreviewed content')
    out['preferred_name']=' '.join(md[k] for k in ('subject_name','predicate','object_name'))
    proposal=event(record,out,claim_id=record['id'],review_id=review_id)
    allowed={('preferred_name',)}|{('metadata',k) for k in replacements}|{('metadata','metadata',k) for k in replacements}
    if any(tuple(c['path']) not in allowed for c in proposal['field_changes']):raise ValueError('scientific field outside finite scope')
    return proposal,out


def revise_edges(before,after,owned):
    reviewed_edges(before['id'],before,before,owned)
    events=owned_endpoint_changes(before,after,owned)
    by_ordinal={e['ordinal']:e for e in events}
    from .kg_root_repairs import apply_event
    revised=[(i,apply_event(row,by_ordinal[i]) if i in by_ordinal else row) for i,row in owned]
    reviewed_edges(before['id'],after,after,revised)
    return events


def reimport_rule(before,after,target_nodes,review_id):
    md=before['metadata'];new=after['metadata']
    return dict(review_id=review_id,pmid=str(md['source_paper']['pmid']),
        raw_text_sha256=digest(md['raw_text']),negated=md['negated'],
        old_relation={k:md[k] for k in FIELDS},new_relation={k:new[k] for k in FIELDS},
        roles={side:role_values(md,side) for side in ('subject','object')},
        target_node_sha256={new[side+'_id']:digest(target_nodes[new[side+'_id']]) for side in ('subject','object')})


def apply_reviewed_reimport(kg,claim):
    """Return the claim when a reviewed source rule handles both endpoints.

    No opt-in declaration means no change to existing ingestion behavior.
    An applicable rule with a stale/missing target fails before mutation.
    Evidence-key matching also covers the same observation assigned a new ID.
    """
    policy=kg.serialization_metadata.get('reviewed_relation_repairs')
    if policy is None:return None
    if policy.get('version')!=VERSION or digest(policy.get('rules'))!=APPROVED_RULESET_DIGEST:
        raise ValueError('unapproved reviewed relation policy')
    md=claim.to_dict();hits=[]
    for rule in policy['rules']:
        if str(md.get('source_paper',{}).get('pmid') or '')!=rule['pmid']:continue
        if digest(md.get('raw_text'))!=rule['raw_text_sha256'] or md.get('negated') is not rule['negated']:continue
        labels=('subject_name','predicate','object_name')
        if not any(all(md[k]==rule[state][k] for k in labels) for state in ('old_relation','new_relation')):continue
        if any(role_values(md,side)!=rule['roles'][side] for side in ('subject','object')):continue
        for field in ('subject_id','object_id'):
            if md.get(field) not in ('',None,rule['old_relation'][field],rule['new_relation'][field]):
                raise ValueError('conflicting supplied endpoint for reviewed observation')
        hits.append(rule)
    if not hits:return None
    if len(hits)!=1:raise ValueError('ambiguous source-bound repair')
    rule=hits[0]
    for nid,seal in rule['target_node_sha256'].items():
        node=kg.get_concept(nid)
        if node is None or digest(node.to_dict())!=seal:raise ValueError('reviewed target is missing or changed')
    for field,new in rule['new_relation'].items():
        if field in claim.metadata and claim.metadata[field] not in (rule['old_relation'][field],new):
            raise ValueError('conflicting nested endpoint in reviewed import')
    for field,new in rule['new_relation'].items():
        setattr(claim,field,new)
        if field in claim.metadata:claim.metadata[field]=new
    return claim
