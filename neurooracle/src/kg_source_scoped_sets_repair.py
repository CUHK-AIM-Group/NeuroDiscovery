"""R75 exact source mention identity, without cross-paper assay equivalence."""
import hashlib
import re

from .kg_identity_pilot import digest
from .kg_reviewed_relation_repair import role_values
from .kg_root_application import RootTransform,unique_index
from .schema import ConceptNode

VERSION='kg.source_scoped_sets.r75.v1'
APPROVED_RULESET_DIGEST='018844d140e750364d4a7414ba4e9e0911a357cae4eb36e400311cdd924a427a'
DOMAINS={'gene_set','expression_profile','transcript_set','genetic_variant_mention','genetic_construct','biomarker_panel',
 'microbiome_measure','biochemical_measure','molecular_set','computational_model','risk_profile','intervention'}
WORK_PMIDS=('40463528','41167554')
WORK_NAME='transcranial temporal interference stimulation'


def literal(name,domain,pmid):
    if not isinstance(name,str) or not name.strip() or domain not in DOMAINS or not re.fullmatch(r'[1-9][0-9]*',pmid):
        raise ValueError('invalid reviewed source mention')
    key=hashlib.sha256((VERSION+'|'+domain+'|'+pmid+'|'+name).encode()).hexdigest()
    return ConceptNode(id='CLM_CONCEPT:source_scoped_mention_'+pmid+'_'+key,preferred_name=name,
        domain_tags=[domain],source_vocab='claim_extraction',
        definition='Exact '+domain+' mention in owning publication PMID '+pmid+'. This preserves the source-defined set, profile, variant, panel or method without asserting members, alleles, assays or equivalence to another publication; source observations retain all scientific qualifiers.').to_dict()

def work_literal():
    key=digest((VERSION,'verified_work_versions',WORK_PMIDS,'intervention',WORK_NAME))
    return ConceptNode(id='CLM_CONCEPT:verified_work_41167554_'+key,preferred_name=WORK_NAME,
        domain_tags=['intervention'],source_vocab='claim_extraction',
        definition='Exact tTIS intervention mention in verified versions of one systematic review, PMIDs 40463528 and 41167554, linked bidirectionally by NCBI UpdateIn/UpdateOf. Version records are not independent primary studies; this does not validate a disease-specific stimulation target or therapeutic effect.').to_dict()


def reviewed_literal(name,domain,pmid):
    if name==WORK_NAME and domain=='intervention' and pmid in WORK_PMIDS:return work_literal()
    return literal(name,domain,pmid)


class SourceSetsTransform(RootTransform):
    def __init__(self,claims,edges,deleted_claims,deleted_edges,new_nodes,entity_events):
        if deleted_claims or deleted_edges or entity_events:raise ValueError('R75 preserves all existing entities and records')
        super().__init__(claims,edges,[],[],[])
        self.entity_events={};self.claim_events=dict(self.claims)
        self.new_nodes=unique_index(new_nodes,'id')
        for nid,row in self.new_nodes.items():
            if row==work_literal():continue
            match=re.fullmatch(r'CLM_CONCEPT:source_scoped_mention_([1-9][0-9]*)_[0-9a-f]{64}',nid)
            domains=row.get('domain_tags') or []
            if not match or len(domains)!=1 or row!=literal(row.get('preferred_name'),domains[0],match[1]):
                raise ValueError('added mention differs from exact source scope')


def apply_source_sets_reimport(kg,claim):
    policy=kg.serialization_metadata.get('source_scoped_sets_repair')
    if policy is None:return None
    if policy.get('version')!=VERSION or digest(policy.get('rules'))!=APPROVED_RULESET_DIGEST:
        raise ValueError('unapproved source sets policy')
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
        if matched_scope:raise ValueError('conflicting supplied endpoint for source sets observation')
        return None
    if len({digest((r['new_relation'],r['target_node_sha256'])) for r in matches})!=1:
        raise ValueError('ambiguous source-bound source sets repair')
    rule=matches[0]
    for nid,seal in rule['target_node_sha256'].items():
        node=kg.get_concept(nid)
        if node is None or digest(node.to_dict())!=seal:raise ValueError('source sets target missing or changed')
    for field,new in rule['new_relation'].items():
        if field in claim.metadata and claim.metadata[field] not in (rule['old_relation'][field],new):
            raise ValueError('conflicting nested field in source sets import')
    for field,new in rule['new_relation'].items():
        setattr(claim,field,new)
        if field in claim.metadata:claim.metadata[field]=new
    return claim
