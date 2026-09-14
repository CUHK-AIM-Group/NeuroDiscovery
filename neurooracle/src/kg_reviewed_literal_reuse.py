"""Finite complete-name reuse, three obsolete about branches and mouse VCP.

Scientific claim fields and old audits are immutable. Existing source anchors,
their metadata and offline atoms stay intact; provenance is not a new identity.
"""
from copy import deepcopy

from .kg_bulk_identity import change_claim
from .kg_gene_boundary_repair import word_interior_hits
from .kg_historic_literal_repair import literal_node as complete_literal
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import (edge_owner, reviewed_edges as ordinary_edges,
    apply_edge, reverse_claim)
from .kg_scoped_structure import source_documents
from .relation_evidence import name_key
from .schema import ConceptNode

VERSION='kg.reviewed_literal_reuse.v1'
REGIONAL='CLM_CONCEPT:regional_brain_structural_alterations_in_reward_and_salience_networks'
BRAIN='CLM_CONCEPT:brain_structural_alterations_c23f31249265'
GRAY='CLM_CONCEPT:0a55ca2778fecaaf'
GUT='CLM_CONCEPT:gut_microbiota_alterations_ed671b8aed43'
CHRONIC='CLM_CONCEPT:chronic_mild_traumatic_brain_injury_in_iraq_and_afghanistan_veterans_669e946d5069'
MOUSE='CLM:efe7514ed7b72a53'
MOUSE_NAME='Valosin-containing protein (Mus musculus)'
MOUSE_ID='CLM_CONCEPT:protein_uniprot_Q01853'
PRENATAL='CLM:CASE1MAN:36716140:8023'
CHORIO='CLM:734ebe395d0e'
BRANCH_CLAIMS={CHORIO,PRENATAL,'CLM:a7e04d8b2c7b'}
OLD_COGNITIVE='CLM_CONCEPT:cognitive_performance'
COGNITIVE='NCL_OUTCOME:cognitive_performance'
BROAD_REASSIGN={'CLM:CASE1MAN:35252949:892','CLM:CASE1MAN:35739320:479'}
SPECS={
    'CLM:0ca422df2989':('object','functional brain alterations','CLM_CONCEPT:functional_brain_alterations_03d0d5161517'),
    'CLM:247ad12bad55':('object','structural brain alterations','CLM_CONCEPT:structural_brain_alterations_1bfbbf12df79'),
    'CLM:30d68bc29e1f':('object','structural brain alterations','CLM_CONCEPT:structural_brain_alterations_1bfbbf12df79'),
    'CLM:2d81439e0314':('subject','fractional anisotropy in bilateral anterior thalamic radiation','CLM_CONCEPT:fractional_anisotropy_in_bilateral_anterior_thalamic_radiation_d3ced270396e'),
    'CLM:31c209454a43':('subject','collybistin-gephyrin interaction','CLM_CONCEPT:collybistin_gephyrin_interaction_6edc69d2763b'),
    'CLM:585022a2f24a':('subject','gut microbiota alterations',GUT),
    'CLM:7b53f4437ee6':('subject','gut microbiota alterations',GUT),
    'CLM:f2e8c6257e4d':('subject','gut microbiota alterations',GUT),
    'CLM:a7e04d8b2c7b':('subject','gut microbiota alterations',GUT),
    'CLM:60a7bc1def1d':('subject','brain structural alterations',BRAIN),
    'CLM:d8d5f4a85909':('object','gray matter volume alterations',GRAY),
    'CLM:ae86c11dfaff':('object','left dorsolateral prefrontal cortex activation','CLM_CONCEPT:left_dorsolateral_prefrontal_cortex_activation_d96a64856905'),
    'CLM:CASE1MAN:18335036:11485':('subject','bilateral prefrontal activation','CLM_CONCEPT:bilateral_prefrontal_activation_9d3518bb851c'),
    'CLM:CASE1MAN:23150548:9604':('subject','anterior cingulate cortex surface area','CLM_CONCEPT:anterior_cingulate_cortex_surface_area_7b88655e3f49'),
    'CLM:CASE1MAN:34998125:8980':('subject','bilateral uncinate fasciculus fractional anisotropy','CLM_CONCEPT:bilateral_uncinate_fasciculus_fractional_anisotropy_c89819772ca5'),
    'CLM:CASE1MAN:22706988:6561':('object','chronic mild traumatic brain injury in Iraq and Afghanistan veterans',CHRONIC),
    CHORIO:('subject','chorio-scleral interface thickness',complete_literal('chorio-scleral interface thickness')['id']),
    PRENATAL:('subject','childhood cortical volume and surface area',complete_literal('childhood cortical volume and surface area')['id']),
    MOUSE:('subject','VCP',MOUSE_ID),
    'CLM:CASE1MAN:35252949:892':('subject','brain structural alterations',BRAIN),
    'CLM:CASE1MAN:35739320:479':('subject','brain structural alterations',BRAIN),
}


def literal_node(name):
    if name==MOUSE_NAME:
        return ConceptNode(id=MOUSE_ID,preferred_name=MOUSE_NAME,semantic_types=['T116'],
            domain_tags=['gene'],source_vocab='UniProt',
            external_ids={'UniProt':'Q01853','NCBI_Taxonomy':'10090'}).to_dict()
    if name not in {'childhood cortical volume and surface area','chorio-scleral interface thickness'}:
        raise ValueError('unreviewed new literal')
    return complete_literal(name)


def public_scope_proof(abstract_xml,protein):
    docs=source_documents([abstract_xml]);d=docs.get('15885483')
    fragments=('MC3T3-E1 mouse osteoblast-like cells','VCP protein expression','untransformed osteoblastic cells')
    if not d or '10.1016/j.orthres.2004.12.012' not in d['dois'] or not all(f in d['abstract'] for f in fragments):
        raise ValueError('own mouse-cell protein source differs')
    if protein.get('primaryAccession')!='Q01853' or protein.get('uniProtkbId')!='TERA_MOUSE' or protein.get('organism',{}).get('taxonId')!=10090:
        raise ValueError('not the reviewed mouse protein')
    if protein['organism'].get('scientificName')!='Mus musculus' or not any(r.get('geneName',{}).get('value')=='Vcp' for r in protein.get('genes',[])):
        raise ValueError('mouse protein gene/species witness differs')
    alternatives=protein.get('proteinDescription',{}).get('alternativeNames',[])
    if not any(r.get('fullName',{}).get('value')=='Valosin-containing protein' and any(s.get('value')=='VCP' for s in r.get('shortNames',[])) for r in alternatives):
        raise ValueError('reviewed protein expansion missing')
    return dict(pmid='15885483',doi='10.1016/j.orthres.2004.12.012',abstract_sha256=digest(d['abstract']),
        protein_source_sha256=digest(protein),accession='Q01853',taxon_id=10090,source_is_protein_not_human_gene=True)


def reviewed_claim(record,changes,witnesses,reviews,source_proof):
    cid=record['id'];md=record['metadata'];review=reviews.get(cid)
    if cid not in SPECS or not review or review.get('claim_sha256')!=digest(record):
        raise ValueError('not the exact reviewed current claim')
    side,name,target=SPECS[cid]
    if len(changes)!=1 or changes[0]!=dict(side=side,name=name,old_id=md.get(side+'_id'),target_id=target):
        raise ValueError('finite complete-name target differs')
    if md.get(side+'_name')!=name or md.get('metadata',{}).get(side+'_id',md[side+'_id'])!=md[side+'_id']:
        raise ValueError('original full name/nested ID differs')
    if cid in BROAD_REASSIGN:
        if md[side+'_id']!=REGIONAL:raise ValueError('not reviewed broad/regional mixture')
    else:
        nid=md[side+'_id'];w=witnesses.get(nid,{})
        if nid not in {'CUI:C1414531','CUI:C1421437'} or w.get('node_id')!=nid or 'T028' not in w.get('semantic_types',[]):
            raise ValueError('source gene witness differs')
        if cid==MOUSE:
            if nid!='CUI:C1421437' or str(md.get('source_paper',{}).get('pmid'))!='15885483' or source_proof.get('source_is_protein_not_human_gene') is not True:
                raise ValueError('actual mouse VCP scope unproved')
            if md.get('raw_text')!='MC3T3-E1 cells showed high VCP expression mainly in cytoplasm and mild physiological stress did not change VCP levels or distribution.':
                raise ValueError('mouse protein statement changed')
        elif not word_interior_hits(name,w.get('labels',[])):
            raise ValueError('not a reviewed word-interior gene error')
    event=dict(claim_id=cid,claim_sha256=digest(record),changes=changes,nonidentity_sha256=digest(nonidentity_claim(record)))
    out=change_claim(record,event);event['current_node_sha256']=digest(out)
    return event,out


def split_owned(cid,original,edge_rows):
    """A retired about is an exact duplicate except for its obsolete target."""
    if cid not in BRANCH_CLAIMS:return edge_rows,[]
    md=original['metadata']
    if md['object_id']!=COGNITIVE:raise ValueError('current cognitive endpoint differs')
    by_target={};science=[]
    for ordinal,row in edge_rows:
        if edge_owner(row)!=cid:raise ValueError('conflicting owner')
        if row['relation_type']=='about':
            if row['target_id'] in by_target:raise ValueError('duplicate target in source closure')
            by_target[row['target_id']]=(ordinal,row)
        else:science.append((ordinal,row))
    if set(by_target)!={md['subject_id'],COGNITIVE,OLD_COGNITIVE} or len(science)>1:
        raise ValueError('not the reviewed three-about closure')
    old_ordinal,old=by_target[OLD_COGNITIVE];kept_ordinal,kept=by_target[COGNITIVE]
    expected=dict(source_id=cid,target_id=COGNITIVE,relation_type='about',source='claim_extraction',confidence=md['confidence'],evidence_ref='',metadata={})
    if kept!=expected or old!={**expected,'target_id':OLD_COGNITIVE}:raise ValueError('obsolete branch payload differs')
    retained=[(o,r) for o,r in edge_rows if o!=old_ordinal]
    ordinary_edges(cid,original,original,retained)
    return retained,[dict(claim_id=cid,ordinal=old_ordinal,edge_sha256=digest(old),kept_ordinal=kept_ordinal,kept_edge_sha256=digest(kept),
        obsolete_target=OLD_COGNITIVE,current_target=COGNITIVE,nonendpoint_sha256=digest({k:v for k,v in old.items() if k not in {'source_id','target_id'}}))]


def reviewed_edges(cid,original,current,edge_rows):
    retained,_=split_owned(cid,original,edge_rows)
    return ordinary_edges(cid,original,current,retained)


def candidate_retirement_proof(cid,current,edge_rows,descriptor):
    """Reconstruct the deleted neutral branch from the unchanged kept about."""
    if cid not in BRANCH_CLAIMS or descriptor['claim_id']!=cid:raise ValueError('unreviewed candidate branch')
    ordinary_edges(cid,current,current,edge_rows)
    canonical=[r for _,r in edge_rows if r['relation_type']=='about' and r['target_id']==COGNITIVE]
    if len(canonical)!=1 or digest(canonical[0])!=descriptor['kept_edge_sha256']:
        raise ValueError('unchanged kept cognitive branch differs')
    old=deepcopy(canonical[0]);old['target_id']=OLD_COGNITIVE
    if digest(old)!=descriptor['edge_sha256']:raise ValueError('retired neutral branch cannot be reproduced')
    return True
