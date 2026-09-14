"""Finite source-reviewed identities that let a claim gather multiple papers."""
import re
from .kg_identity_pilot import digest
from .kg_reviewed_relation_repair import role_values
from .kg_root_application import RootTransform, unique_index
from .schema import ConceptNode

VERSION = 'kg.multipaper_claims.r91.v1'
APPROVED_RULESET_DIGEST = 'c2715f18f6514e5395157d4b3620b201acd00f4533509b4e804f0cb292f90d63'

def shared_literal(name, domain):
    if not isinstance(name, str) or not name.strip() or not re.fullmatch(r'[a-z][a-z_]*', domain):
        raise ValueError('invalid reviewed shared concept')
    key = digest((VERSION, domain, name))
    return ConceptNode(id='CLM_CONCEPT:reviewed_shared_'+key, preferred_name=name,
        domain_tags=[domain], source_vocab='claim_extraction',
        definition='Source-reviewed shared '+domain+' concept: '+name+'. Identity is approved only for the finite source-bound observations in the graph policy. Each observation retains its own population, measurement, protocol, qualifiers, polarity and evidence. Shared identity does not imply equal effects or independent studies.').to_dict()

class MultipaperTransform(RootTransform):
    def __init__(self, claims, edges, deleted_claims, deleted_edges, new_nodes, entity_events):
        if deleted_claims or deleted_edges or entity_events:
            raise ValueError('R91 preserves all existing entities and observations')
        super().__init__(claims, edges, [], [], [])
        self.entity_events = {}
        self.claim_events = dict(self.claims)
        self.new_nodes = unique_index(new_nodes, 'id')
        for row in self.new_nodes.values():
            domains = row.get('domain_tags') or []
            if len(domains) != 1 or row != shared_literal(row.get('preferred_name'), domains[0]):
                raise ValueError('added concept differs from reviewed shared identity')

def apply_multipaper_reimport(kg,claim):
    policy=kg.serialization_metadata.get('multipaper_claim_accretion')
    if policy is None:return None
    if policy.get('version')!=VERSION or digest(policy.get('rules'))!=APPROVED_RULESET_DIGEST:
        raise ValueError('unapproved multipaper policy')
    md=claim.to_dict();matches=[];matched_scope=False
    for rule in policy['rules']:
        if str(md.get('source_paper',{}).get('pmid') or '')!=rule['pmid']:continue
        if digest(md.get('raw_text'))!=rule['raw_text_sha256'] or md.get('negated') is not rule['negated']:continue
        if any(role_values(md,s)!=rule['roles'][s] for s in ('subject','object')):continue
        labels=('subject_name','predicate','object_name')
        if not any(all(md[k]==rule[state][k] for k in labels) for state in ('old_relation','new_relation')):continue
        matched_scope=True
        if any(md.get(k) not in ('',None,rule['old_relation'][k],rule['new_relation'][k]) for k in ('subject_id','object_id')):continue
        matches.append(rule)
    if not matches:
        if matched_scope:raise ValueError('conflicting supplied endpoint for multipaper observation')
        return None
    if len({digest((r['new_relation'],r['target_node_sha256'])) for r in matches})!=1:
        raise ValueError('ambiguous source-bound multipaper repair')
    rule=matches[0]
    for nid,seal in rule['target_node_sha256'].items():
        node=kg.get_concept(nid)
        if node is None or digest(node.to_dict())!=seal:raise ValueError('multipaper target missing or changed')
    for field,new in rule['new_relation'].items():
        if field in claim.metadata and claim.metadata[field] not in (rule['old_relation'][field],new):
            raise ValueError('conflicting nested field in multipaper import')
    for field,new in rule['new_relation'].items():
        setattr(claim,field,new)
        if field in claim.metadata:claim.metadata[field]=new
    return claim
