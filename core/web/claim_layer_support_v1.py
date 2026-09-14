"""Explicit, independently reviewed evidence from papers without old KG nodes.

Supplemental evidence never acquires the provenance of an original graph node.
Owning XML, full abstracts, authority identities and host decisions are bound.
"""
from collections import defaultdict
from copy import deepcopy
import json

from core.web.claim_evidence import EvidenceUnavailable
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.claim_evidence_query import paper_evidence as original_paper_evidence
from neurooracle.src.shared_relation_catalog import check_file
from neurooracle.scripts.whole_graph_sources_20260914 import own_records

ORIGIN='supplemental_own_abstract'
ROLES={'own_result','review_synthesis','background_assertion'}
ARTICLE_ROLES={'primary_research_article','review_or_evidence_synthesis','commentary_or_opinion','methods_article','unresolved_article_design'}
UNSAFE_TYPES={'Retracted Publication','Retraction of Publication','Expression of Concern','Corrected and Republished Article','Preprint'}


def require(condition,message):
    if not condition:raise EvidenceUnavailable(message)


def normalized(text):return ' '.join(str(text).split())


def make_observation(target_id,target,original_ids,document,action,review_id,note,provenance,registry):
    require(original_ids and len(original_ids)==len(set(original_ids)),'Supplemental source lacks original claim binding')
    require(action['role'] in ROLES and action['source_role'] in ARTICLE_ROLES,'Ineligible supplemental source role')
    require(action.get('support')=='supports','Only explicitly supporting supplemental evidence is admitted')
    require(note and provenance and review_id,'Explicit independent source adjudication is missing')
    require(not document.get('comments_corrections') and not document.get('source_kind') and not UNSAFE_TYPES.intersection(document['publication_types']),'Supplemental source needs publication-status review')
    abstract=' '.join(a['text'] for a in document['abstract'])
    require(action['anchor'] and normalized(action['anchor']) in normalized(abstract),'Supplemental anchor is not in its own complete abstract')
    identity_binding=dict(target_shared_claim_id=target_id,anchor_original_claim_ids=sorted(original_ids),pmid=document['pmid'],abstract_sha256=digest(document['abstract']),anchor=action['anchor'])
    cid='CLM:SUPPLEMENTAL:'+digest(identity_binding)
    paper=dict(pmid=document['pmid'],title=document['title'])
    if document['own_article_ids'].get('doi'):paper['doi']=document['own_article_ids']['doi']
    if document['own_article_ids'].get('pmc'):paper['pmcid']=document['own_article_ids']['pmc']
    metadata=dict(id=cid,**deepcopy(target),negated=False,raw_text=action['anchor'],source_paper=paper,
        source='supplemental_own_source_review',metadata=dict(observation_origin=ORIGIN,supports_original_claim_ids=sorted(original_ids)))
    identity=registry.resolve(metadata)
    require(identity['status']=='verified' and identity['paper_key']=='pmid:'+document['pmid'],'Supplemental bibliography lacks verified own identity')
    publication=registry.publication_review(identity['paper_key'])
    require(publication['status']!='retracted','Retracted supplemental evidence cannot support a claim')
    sha=digest(dict(id=cid,metadata=metadata))
    review=dict(claim_sha256=sha,pmid=document['pmid'],source_role=action['source_role'],observation_role=action['role'],proposition_support='supports',
        review_id=review_id,primary_source=document['source'],publication_types=document['publication_types'],source_abstract_sha256=digest(document['abstract']),
        source_anchor=action['anchor'],scope_note=note,source_specific_note=action.get('note',''),cohort_independence_not_inferred=True,
        reviewed_extent='complete_own_PubMed_abstract_and_bound_original_claim',reading_provenance=provenance,observation_origin=ORIGIN)
    return dict(claim_id=cid,claim_sha256=sha,claim=metadata,source_identity=identity,publication_review=publication,source_review=review,observation_origin=ORIGIN)


def load_support_records(layer):
    records=layer.get('supplemental_source_records',{})
    if not records:return {}
    declared={fp['path']:fp for fp in layer['source_support_inputs']}
    def checked(fp):
        require(declared.get(fp['path'])==fp,'Supplemental input not declared in the sealed layer')
        return check_file(fp,full_hash=True)
    registry_fp=layer['original_relation_extension']['supplemental_identity_registry']
    registry_payload=json.loads(checked(registry_fp).read_text(encoding='utf-8'))
    registry=VerifiedPaperIdentities(registry_payload)
    documents,ledgers,raw_sources={}, {}, {}
    by_target=defaultdict(list)
    for cid,record in records.items():
        document_fp=record['owning_documents'];binding=record['host_decision'];ledger_fp=binding['ledger']
        if document_fp['path'] not in documents:
            documents[document_fp['path']]=json.loads(checked(document_fp).read_text(encoding='utf-8'))
        if ledger_fp['path'] not in ledgers:
            ledgers[ledger_fp['path']]=json.loads(checked(ledger_fp).read_text(encoding='utf-8'))
        decision=ledgers[ledger_fp['path']][binding['key']]
        require(not decision.get('hold') and decision.get('adjudicator')=='current_task_host','Supplemental evidence lacks independent host approval')
        pmid=record['pmid'];doc=documents[document_fp['path']]['records'][pmid]
        require(doc['pmid']==pmid and pmid in decision['owning_abstracts_read'],'Incomplete own-source reading declaration')
        action=decision['supplemental_sources'][pmid]
        require(action['own_abstract_sha256']==digest(doc['abstract']),'Host decision used a different abstract')
        require(record['target_shared_claim_id']==decision['target_shared_claim_id'] and record['anchor_original_claim_ids']==decision['original_claim_ids'],'Supplemental original-target binding changed')
        fp=doc['source']
        if fp['path'] not in raw_sources:
            raw_sources[fp['path']]={p:(d,a,s) for p,d,a,s in own_records(checked(fp).read_bytes(),fp)}
        raw=raw_sources[fp['path']].get(pmid)
        require(raw is not None and raw[0]==doc and raw[2]=='OWN_RECORD_CACHED','Source document is not the complete owning PubMed article')
        own_registry=VerifiedPaperIdentities(dict(version='kg.paper_identity.v1',records={pmid:raw[1]}))
        observation=make_observation(record['target_shared_claim_id'],decision['canonical_claim'],record['anchor_original_claim_ids'],doc,action,
            decision['review_id'],decision['note'],decision['reading_provenance'],registry)
        require(own_registry.resolve(observation['claim'])==observation['source_identity'],'Raw owning XML does not independently verify supplemental bibliography')
        require(cid==observation['claim_id'] and record['observation']==observation,'Supplemental record differs from the explicit host source decision')
        by_target[record['target_shared_claim_id']].append(dict(observation=observation,original_claim_ids=record['anchor_original_claim_ids']))
    return dict(by_target)


def paper_evidence(dossier):
    evidence=original_paper_evidence(dossier)
    supplemental={o['claim_id'] for o in dossier['observations'] if o.get('observation_origin')==ORIGIN}
    for paper in evidence['papers']:
        for observation in paper['observations']:
            if observation['claim_id'] in supplemental:
                observation['observation_origin']=ORIGIN
                observation['original_graph_observation']=False
                observation['source_derived_claim']=observation.pop('original_claim')
    return evidence
