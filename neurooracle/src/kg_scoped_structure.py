"""Reviewed complete MRI literals, one collapsed about closure, one direction.

Evidence controls exact selected records. No fuzzy naming, sample extrapolation,
scientific-edge fabrication, blanket polarity flips, or detail-anchor deletion.
"""
from copy import deepcopy
import hashlib
from xml.etree import ElementTree as ET

from .claim_semantics import declared_type_atoms
from .kg_bulk_identity import change_claim as identity_change
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import (literal_node as imaging_node, edge_owner,
    reviewed_edges as ordinary_edges, apply_edge, GENES)
from .relation_evidence import name_key
from .schema import ConceptNode

VERSION='kg.scoped_structure.v1'
COLLISION='CLM:CASE1MAN:22306803:8568'
DIRECTION='CLM:ead63272cd396884'
INTERACTION='COMT genotype by externalizing behavior interaction'
ACTIVATION='dorsal anterior cingulate and lateral prefrontal interference-related activation'
MRI_SPECS={
 'CLM:CASE1MAN:11407273:11776':('11407273','bilateral pulvinar high signal on MRI',('MRI brain scans show bilateral pulvinar high signal',)),
 'CLM:CASE1MAN:17259350:11927':('17259350','contralateral amygdala T2 relaxation signal',('T2 relaxometry','contralateral amygdala showed lower signal')),
 'CLM:CASE1MAN:20070577:4804':('20070577','leftward dorsolateral prefrontal cortex fMRI lateralization for pleasant words',('patterns of fMRI activity','Both groups showed a leftward lateralization for pleasant words in DLPFC.')),
 'CLM:CASE1MAN:24248213:4066':('24248213','symmetrical increased T2 signal in posterior or posterior-lateral cervical and thoracic spinal cord columns',('symmetrical abnormally increased T2 signal intensity','posterior or posterior and lateral columns in the cervical and thoracic spinal cord')),
 'CLM:CASE1MAN:26280255:8658':('26280255','frontotemporal diffusion MRI laterality indexes',('diffusion MRI','Laterality indexes of DFA','Laterality indexes of DTTL and DNUM')),
}


def source_documents(contents):
    documents={}
    for content in contents:
        for a in ET.fromstring(content).findall('PubmedArticle'):
            pmid=a.findtext('./MedlineCitation/PMID')
            if pmid in documents:raise ValueError('duplicate owning PMID')
            title=a.find('./MedlineCitation/Article/ArticleTitle')
            if title is None:raise ValueError('missing owning title')
            abstract=' '.join(' '.join(''.join(e.itertext()).split()) for e in a.findall('./MedlineCitation/Article/Abstract/AbstractText'))
            documents[pmid]=dict(title=''.join(title.itertext()),abstract=abstract,
                dois=sorted({e.text for e in a.findall('./PubmedData/ArticleIdList/ArticleId') if e.get('IdType')=='doi' and e.text}))
    return documents


def source_proof(contents):
    documents=source_documents(contents)
    requirements={v[0]:v[2] for v in MRI_SPECS.values()}
    requirements['22306803']=('In 104 male participants','interference-related activation depended conjointly on externalizing',
        'dorsal anterior cingulate and lateral prefrontal cortex','val/val individuals with high trait externalizing')
    requirements['36847009']=('BACE1 concentration was negatively associated','hippocampal volume','in the MCI group')
    proof={}
    for pmid,fragments in requirements.items():
        if pmid not in documents:raise ValueError('missing own source')
        d=documents[pmid]
        if not all(f in d['abstract'] for f in fragments):raise ValueError('own source scientific support differs')
        proof[pmid]=dict(title=d['title'],dois=d['dois'],abstract_sha256=digest(d['abstract']),
            support_fragment_sha256=[digest(f) for f in fragments])
    if '10.1016/j.neuroimage.2012.01.097' not in proof['22306803']['dois'] or '10.3233/JAD-221174' not in proof['36847009']['dois']:
        raise ValueError('own DOI witness differs')
    return proof


def literal_node(name):
    name=name_key(name)
    if name==INTERACTION:
        key=hashlib.sha256((VERSION+'|individual_data|'+name).encode()).hexdigest()
        return ConceptNode(id='CLM_CONCEPT:interaction_mention_'+key,preferred_name=name,
            domain_tags=['dataset_variable'],source_vocab='claim_extraction').to_dict()
    if name not in {ACTIVATION,*[v[1] for v in MRI_SPECS.values()]}:raise ValueError('unreviewed full literal')
    return imaging_node(name)


def roles(value):return {a.value for a in declared_type_atoms(value)}


def reviewed_claim(record,changes,proof):
    cid=record['id'];md=record['metadata'];inner=md.get('metadata') or {}
    if cid==COLLISION:
        pmid='22306803';expected={'subject':(ACTIVATION,'imaging_marker'),'object':(INTERACTION,'individual_data')}
        if md['subject_id']!=md['object_id'] or md['subject_id']!='CUI:C1421437':raise ValueError('not the reviewed double-gene collapse')
        if md['predicate']!='is_associated_with' or md['evidence']['sample_size']!=104:raise ValueError('interaction scope differs')
    elif cid in MRI_SPECS:
        pmid,name,_=MRI_SPECS[cid];expected={'subject':(name,'imaging_marker')}
    elif cid==DIRECTION:
        pmid='36847009';expected={}
        if md.get('raw_text')!='Plasma BACE1 concentration was negatively associated with hippocampal volume in MCI due to AD.':raise ValueError('direction source statement differs')
        if md['predicate']!='is_associated_with' or md['evidence'].get('direction')!='positive':raise ValueError('not the reviewed direction conflict')
    else:raise ValueError('unreviewed claim')
    if str(md['source_paper'].get('pmid'))!=pmid or pmid not in proof:raise ValueError('owning source differs')
    if md.get('negated') is not False:raise ValueError('negation not approved for change')
    if not md.get('raw_text'):raise ValueError('missing original scientific text')
    if len(changes)!=len(expected) or {ch['side'] for ch in changes}!=set(expected):raise ValueError('complete side changes required')
    for ch in changes:
        side=ch['side'];name,role=expected[side]
        if ch['name']!=name_key(md.get(side+'_name')) or ch['name']!=name:raise ValueError('complete case-sensitive scope differs')
        if ch['old_id']!=md[side+'_id'] or ch['old_id'] not in GENES:raise ValueError('unreviewed old endpoint')
        if inner.get(side+'_id',ch['old_id'])!=ch['old_id']:raise ValueError('nested endpoint conflict')
        outer_type,inner_type=md.get(side+'_type'),inner.get(side+'_type')
        if outer_type and inner_type and roles(outer_type)!=roles(inner_type):raise ValueError('nested role conflict')
        if roles(outer_type or inner_type)!={role}:raise ValueError('explicit source role differs')
        if ch['target_id']!=literal_node(name)['id']:raise ValueError('whole-name target differs')
    event=dict(claim_id=cid,claim_sha256=digest(record),changes=changes,nonidentity_sha256=digest(nonidentity_claim(record)),
        source_pmid=pmid,public_source_proof=proof[pmid],direction_change=(dict(old='positive',new='negative') if cid==DIRECTION else None))
    out=change_claim(record,event);event['current_node_sha256']=digest(out)
    return event,out


def change_claim(record,event):
    out=identity_change(record,event)
    if event.get('direction_change'):
        change=event['direction_change']
        if record['id']!=DIRECTION or change!=dict(old='positive',new='negative') or out['metadata']['evidence'].get('direction')!='positive':raise ValueError('unapproved direction change')
        out['metadata']['evidence']['direction']='negative'
    if event.get('current_node_sha256') and digest(out)!=event['current_node_sha256']:raise ValueError('approved output changed')
    return out


def reverse_claim(record,event):
    if digest(record)!=event['current_node_sha256']:raise ValueError('changed output')
    original=deepcopy(record)
    if event.get('direction_change'):
        if record['id']!=DIRECTION or original['metadata']['evidence']['direction']!='negative':raise ValueError('direction inverse differs')
        original['metadata']['evidence']['direction']='positive'
    for ch in event['changes']:
        f=ch['side']+'_id'
        if original['metadata'][f]!=ch['target_id']:raise ValueError('endpoint inverse differs')
        original['metadata'][f]=ch['old_id']
        inner=original['metadata'].get('metadata') or {}
        if f in inner:
            if inner[f]!=ch['target_id']:raise ValueError('nested inverse differs')
            inner[f]=ch['old_id']
    if digest(original)!=event['claim_sha256'] or change_claim(original,event)!=record:raise ValueError('inverse source digest differs')
    return original


def collision_edge(original,current,edge_rows):
    before,after=original['metadata'],current['metadata']
    if original['id']!=COLLISION or before['subject_id']!=before['object_id'] or before['subject_id']!='CUI:C1421437':raise ValueError('not a reviewed collision')
    if len(edge_rows)!=1:raise ValueError('not exactly one neutral about')
    ordinal,row=edge_rows[0]
    expected=dict(source_id=COLLISION,target_id='CUI:C1421437',relation_type='about',source='claim_extraction',
        confidence=before['confidence'],evidence_ref='',metadata={})
    if row!=expected:raise ValueError('about contains unreviewed payload/owner')
    if after['subject_id']==after['object_id'] or any(after[s+'_id']!=literal_node(after[s+'_name'])['id'] for s in ('subject','object')):raise ValueError('incorrect complete collision endpoints')
    return ordinal,row


def reviewed_edges(cid,original,current,edge_rows):
    if cid!=COLLISION:return ordinary_edges(cid,original,current,edge_rows)
    ordinal,row=collision_edge(original,current,edge_rows)
    out=deepcopy(row);out['target_id']=current['metadata']['subject_id']
    return [dict(ordinal=ordinal,claim_id=cid,edge_sha256=digest(row),current_edge_sha256=digest(out),
        changes={'target_id':dict(old=row['target_id'],new=out['target_id'])})]


def reviewed_added_about(cid,original,current,edge_rows):
    if cid!=COLLISION:return []
    ordinal,row=collision_edge(original,current,edge_rows)
    added=deepcopy(row);added['target_id']=current['metadata']['object_id']
    return [dict(claim_id=cid,original_neutral_about_ordinal=ordinal,original_neutral_about_sha256=digest(row),
        endpoint_side='object',row=added,row_sha256=digest(added),not_a_new_scientific_assertion=True)]
