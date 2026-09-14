"""R73 source-bound changes with complete rule hits and typed literal targets.

Discovery probes are never runtime rules. This module can replay only the
frozen reviewed source sentences, complete triples, polarity and annotations.
Entity type updates preserve the entity's identity and every other field.
"""
from copy import deepcopy
import hashlib
import re

from .kg_identity_pilot import digest
from .kg_reviewed_relation_repair import FIELDS, role_values
from .kg_root_application import RootTransform, unique_index
from .kg_root_repairs import event, apply_event
from .schema import ConceptNode
from .kg_literal_endpoint_repair import edge_owner

VERSION='kg.systematic_consolidation.r73.v1'
APPROVED_RULESET_DIGEST='b5e7c3ed294c71c0af22e5897dcd8e777c9334fce0cfee4fe2f5c44a2583fe99'


def literal(name,scientific_type='imaging_feature'):
    if not isinstance(name,str) or not name.strip() or scientific_type!='imaging_feature':
        raise ValueError('unapproved literal definition')
    key=hashlib.sha256((VERSION+'|'+scientific_type+'|'+name).encode()).hexdigest()
    return ConceptNode(id='CLM_CONCEPT:systematic_imaging_mention_'+key,preferred_name=name,
        domain_tags=[scientific_type],source_vocab='claim_extraction').to_dict()


def type_event(before,domains,*,review_id):
    if not before['id'].startswith('CLM_CONCEPT:unclassified_mention_'):
        raise ValueError('type scope is a reviewed unclassified mention')
    if before.get('domain_tags')!=['external'] or before.get('semantic_types') or before.get('external_ids'):
        raise ValueError('existing external/type identity cannot be overwritten')
    if not domains or any(not re.fullmatch('[a-z_]+',d) for d in domains):raise ValueError('invalid scientific type')
    after=deepcopy(before);after['domain_tags']=list(domains)
    return event(before,after,claim_id=before['id'],entity_id=before['id'],review_id=review_id),after


def revise_owned_edges(before,after,owned):
    """Also repair false original self-links only with explicit about roles."""
    cid=before['id'];old=before['metadata'];new=after['metadata'];seen=set();scientific=0;out=[]
    if not owned:raise ValueError('missing owned edges')
    for ordinal,edge in owned:
        if edge_owner(edge)!=cid:raise ValueError('foreign owned edge')
        revised=deepcopy(edge)
        if edge['relation_type']=='about':
            md=edge.get('metadata') or {}
            if md.get('claim_id') not in ('',None,cid):raise ValueError('about owner conflict')
            candidates=[s for s in ('subject','object') if old[s+'_id']==edge['target_id']]
            role=md.get('anchor_role')
            if role in ('subject','object'):
                if role not in candidates:raise ValueError('explicit about role disagrees with endpoint')
                side=role
            elif role in ('',None) and len(candidates)==1:side=candidates[0]
            else:raise ValueError('ambiguous original about endpoint requires role evidence')
            if side in seen:raise ValueError('duplicate about role')
            seen.add(side);revised['target_id']=new[side+'_id']
        else:
            scientific+=1
            if (edge['source_id'],edge['relation_type'],edge['target_id'])!=(old['subject_id'],old['predicate'],old['object_id']):
                raise ValueError('scientific edge differs from source claim')
            revised.update(source_id=new['subject_id'],target_id=new['object_id'],relation_type=new['predicate'])
            if revised['source_id']==revised['target_id']:raise ValueError('new scientific self-link')
        if revised!=edge:out.append(event(edge,revised,ordinal=ordinal,claim_id=cid))
    if seen!={'subject','object'} or scientific>1:raise ValueError('incomplete or duplicated owned-edge closure')
    return out


class SystematicTransform(RootTransform):
    """Use the existing full-stream inverse engine with separately gated nodes."""
    def __init__(self,claims,edges,deleted_claims,deleted_edges,new_nodes,entity_events):
        if deleted_claims or deleted_edges:raise ValueError('R73 does not delete observations or edges')
        super().__init__(claims,edges,deleted_claims,deleted_edges,[])
        self.entity_events=unique_index(entity_events,'entity_id')
        for nid,proposal in self.entity_events.items():
            if proposal.get('claim_id')!=nid or not nid.startswith('CLM_CONCEPT:unclassified_mention_'):
                raise ValueError('entity event identity differs')
            if len(proposal['field_changes'])!=1 or proposal['field_changes'][0]['path']!=['domain_tags']:
                raise ValueError('entity event exceeds type-only scope')
            if nid in self.claims:raise ValueError('overlapping node operations')
        self.claim_events=dict(self.claims)
        # The inherited engine checks/inverts all keyed node changes uniformly.
        self.claims.update(self.entity_events)
        self.new_nodes=unique_index(new_nodes,'id')
        for nid,row in self.new_nodes.items():
            if row!=literal(row.get('preferred_name')):raise ValueError('new literal does not match full-name definition')


def apply_systematic_reimport(kg,claim):
    policy=kg.serialization_metadata.get('systematic_consolidation')
    if policy is None:return None
    if policy.get('version')!=VERSION or digest(policy.get('rules'))!=APPROVED_RULESET_DIGEST:
        raise ValueError('unapproved systematic consolidation policy')
    md=claim.to_dict();matches=[];matched_scope=False
    for rule in policy['rules']:
        if str(md.get('source_paper',{}).get('pmid') or '')!=rule['pmid']:continue
        if digest(md.get('raw_text'))!=rule['raw_text_sha256'] or md.get('negated') is not rule['negated']:continue
        if any(role_values(md,s)!=rule['roles'][s] for s in ('subject','object')):continue
        labels=('subject_name','predicate','object_name')
        if not any(all(md[k]==rule[state][k] for k in labels) for state in ('old_relation','new_relation')):continue
        matched_scope=True
        if any(md.get(k) not in ('',None,rule['old_relation'][k],rule['new_relation'][k]) for k in ('subject_id','object_id')):
            continue
        matches.append(rule)
    if not matches:
        if matched_scope:raise ValueError('conflicting supplied endpoint for systematic observation')
        return None
    outputs={digest((r['new_relation'],r['target_node_sha256'])) for r in matches}
    if len(outputs)!=1:raise ValueError('ambiguous source-bound systematic repair')
    rule=matches[0]
    for nid,seal in rule['target_node_sha256'].items():
        node=kg.get_concept(nid)
        if node is None or digest(node.to_dict())!=seal:raise ValueError('systematic target missing or changed')
    for field,new in rule['new_relation'].items():
        if field in claim.metadata and claim.metadata[field] not in (rule['old_relation'][field],new):
            raise ValueError('conflicting nested field in systematic import')
    for field,new in rule['new_relation'].items():
        setattr(claim,field,new)
        if field in claim.metadata:claim.metadata[field]=new
    return claim
