"""R74 exact source mention identity, without cross-paper assay equivalence."""
import hashlib
import re

from .kg_identity_pilot import digest
from .kg_reviewed_relation_repair import role_values
from .kg_root_application import RootTransform,unique_index
from .schema import ConceptNode

VERSION='kg.semantic_measurement.r74.v1'
APPROVED_RULESET_DIGEST='182ce93e9f52280bc05d7fb2d7f2b384448096e1ef021f049d508d11aa810757'
DOMAINS={'clinical_measure','clinical_phenotype','exposure','imaging_feature',
         'biochemical_measure','biological_process','pathology_state','molecular_mention','intervention'}


def literal(name,domain,pmid):
    if not isinstance(name,str) or not name.strip() or domain not in DOMAINS or not re.fullmatch(r'[1-9][0-9]*',pmid):
        raise ValueError('invalid reviewed source mention')
    key=hashlib.sha256((VERSION+'|'+domain+'|'+pmid+'|'+name).encode()).hexdigest()
    return ConceptNode(id='CLM_CONCEPT:source_scoped_mention_'+pmid+'_'+key,preferred_name=name,
        domain_tags=[domain],source_vocab='claim_extraction',
        definition='Exact mention in owning publication PMID '+pmid+'. Observation-specific scale, assay, dose, species and population remain in the source evidence; no cross-publication equivalence is asserted.').to_dict()


class SemanticTransform(RootTransform):
    def __init__(self,claims,edges,deleted_claims,deleted_edges,new_nodes,entity_events):
        if deleted_claims or deleted_edges or entity_events:raise ValueError('R74 preserves all existing entities and records')
        super().__init__(claims,edges,[],[],[])
        self.entity_events={};self.claim_events=dict(self.claims)
        self.new_nodes=unique_index(new_nodes,'id')
        for nid,row in self.new_nodes.items():
            match=re.fullmatch(r'CLM_CONCEPT:source_scoped_mention_([1-9][0-9]*)_[0-9a-f]{64}',nid)
            domains=row.get('domain_tags') or []
            if not match or len(domains)!=1 or row!=literal(row.get('preferred_name'),domains[0],match[1]):
                raise ValueError('added mention differs from exact source scope')


def apply_semantic_reimport(kg,claim):
    policy=kg.serialization_metadata.get('semantic_measurement_repair')
    if policy is None:return None
    if policy.get('version')!=VERSION or digest(policy.get('rules'))!=APPROVED_RULESET_DIGEST:
        raise ValueError('unapproved semantic measurement policy')
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
        if matched_scope:raise ValueError('conflicting supplied endpoint for semantic observation')
        return None
    if len({digest((r['new_relation'],r['target_node_sha256'])) for r in matches})!=1:
        raise ValueError('ambiguous source-bound semantic repair')
    rule=matches[0]
    for nid,seal in rule['target_node_sha256'].items():
        node=kg.get_concept(nid)
        if node is None or digest(node.to_dict())!=seal:raise ValueError('semantic target missing or changed')
    for field,new in rule['new_relation'].items():
        if field in claim.metadata and claim.metadata[field] not in (rule['old_relation'][field],new):
            raise ValueError('conflicting nested field in semantic import')
    for field,new in rule['new_relation'].items():
        setattr(claim,field,new)
        if field in claim.metadata:claim.metadata[field]=new
    return claim
