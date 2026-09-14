"""Exact source-backed closure for three reviewed R42 findings.

No generic scientific validity classifier, new extraction, global protein
synonym rule, or rewriting of an old audit seal is introduced here.
"""
from bisect import bisect_left
from copy import deepcopy
import hashlib
from xml.etree import ElementTree as ET
from .kg_bulk_identity import change_claim
from .kg_identity_pilot import digest,nonidentity_claim
from .kg_literal_endpoint_repair import reviewed_edges,reverse_claim,apply_edge,edge_owner
from .kg_scoped_structure import source_documents
from .schema import ConceptNode

VERSION='kg.source_scope_resolution.v1'
MYELIN='CLM:CASE1MAN:28526817:9725'
FRAILTY='CLM:593b77e75c8f89d8'
VIRUS='CLM:6984974f4fafa31e9b85adc9149cb732'
VIRAL_NAME='vaccinia virus complement control protein'
VIRAL_REUSE_ID='CLM_CONCEPT:vaccinia_virus_complement_control_protein_bb95bef7ca1f'
VIRAL_REUSE_SHA='6f7a72028f9c9a7396b9907066ad3aec87c6363e4cc9754b7287439ee41adc37'
VIRAL_INCIDENTS={
    'CLM:7ee398d42656898c':('80e4bc38e98741f3c54ea18025b82ec6ad2198e77d43f3bffa49fa703af9af6c','12165132','10.1089/08977150260139093'),
    'CLM:8e7e3673f69c7156':('a55b3a08830ec09ab89876a7568e285af69ce70c435745de32586b265204030c','12485887','10.1111/j.1749-6632.2002.tb04659.x'),
    'CLM:584a2b9d1bb1bbf3':('b13a5f0a8f8f81729fef6c324962caceacaff0019d18a38ca1125be873278499','18490064','10.1016/j.bbr.2008.03.038'),
}
CLAIM_HASHES={
    MYELIN:'43c32a093881ffede3d87b953d58063cc4d43e0652e5750d77bf25f4306d8551',
    FRAILTY:'061b71829c90db5bdf61191dd8c96ad883fe04613c5caf4d588c15c948f4d93e',
    VIRUS:'e7dba62e9d324c8ae85111be7687a1a5f4a3a129d56fe94c3e1c946996b5b5c1',
}
PMIDS={MYELIN:'28526817',FRAILTY:'41297452',VIRUS:'29534078'}
DOIS={'28526817':'10.1038/s41598-017-02062-y','41297452':'10.1016/j.archger.2025.106089','29534078':'10.1371/journal.pone.0193740'}


def normalized(text):return ' '.join(text.split()).replace('\u2212','-')


def public_source_proof(pubmed_xml,myelin_xml):
    docs=source_documents([pubmed_xml]);proof={}
    required={
        '28526817':('myelin in the LPFC was reduced in MDD participants who had experienced a greater number of depressive episodes',),
        '41297452':('Nine articles, including 10 studies','association was still significant','OR = 1.25'),
        '29534078':('vaccinia virus complement control protein (VCP)','cleared within 1-3 hours of delivery'),
    }
    for pmid,phrases in required.items():
        if pmid not in docs or DOIS[pmid].casefold() not in {d.casefold() for d in docs[pmid]['dois']}:raise ValueError('own PMID/DOI differs')
        if not all(normalized(s) in normalized(docs[pmid]['abstract']) for s in phrases):raise ValueError('own abstract scope differs')
        proof[pmid]=dict(doi=DOIS[pmid],title=docs[pmid]['title'],abstract_sha256=digest(docs[pmid]['abstract']),
            support_fragment_sha256=[digest(s) for s in phrases])
    root=ET.fromstring(myelin_xml)
    article_ids={e.get('pub-id-type'):e.text for e in root.findall('./front/article-meta/article-id')}
    if article_ids.get('pmid')!='28526817' or article_ids.get('doi')!=DOIS['28526817']:raise ValueError('own fulltext PMID/DOI differs')
    paragraphs=[normalized(''.join(e.itertext())) for e in root.findall('./body//p')]
    passages=[p for p in paragraphs if 'across all MDD participants LPFC R1 was not significantly related to depression severity' in p]
    if len(passages)!=1:raise ValueError('missing unique severity result')
    passage=passages[0]
    if not all(s in passage for s in ('BDI-II: r = -0.02, p = 0.918','BDI-II]: r = -0.07, p = 0.782')):raise ValueError('fulltext severity statistics differ')
    proof['28526817'].update(fulltext_xml_sha256=hashlib.sha256(myelin_xml.encode()).hexdigest(),
        severity_result_paragraph_sha256=digest(passage),severity_association_reported_as_nonsignificant=True)
    return proof


def source_claim(row,proof):
    cid=row['id'];md=row['metadata']
    if cid not in CLAIM_HASHES or digest(row)!=CLAIM_HASHES[cid]:raise ValueError('unreviewed claim or current hash differs')
    if str(md['source_paper'].get('pmid'))!=PMIDS[cid] or PMIDS[cid] not in proof:raise ValueError('claim own source differs')
    if md.get('negated') is not False:raise ValueError('negation differs')
    return md


def reviewed_myelin_removal(row,owned,proof):
    md=source_claim(row,proof)
    if row['id']!=MYELIN or md['predicate']!='correlates_with':raise ValueError('wrong myelin claim')
    if (md['subject_name'],md['object_name'])!=('lateral prefrontal cortex qMRI myelin content','major depressive disorder symptom severity'):raise ValueError('severity scope differs')
    if md['raw_text']!='Myelin in the LPFC was related to depression-relevant clinical measures.':raise ValueError('current assertion differs')
    if not proof['28526817'].get('severity_association_reported_as_nonsignificant'):raise ValueError('missing source severity result')
    if reviewed_edges(MYELIN,row,row,owned)!=[]:raise ValueError('unchanged removal closure differs')
    if len(owned)!=2 or any(e['relation_type']!='about' for _,e in owned):raise ValueError('reviewed claim-only representation differs')
    return dict(claim_id=MYELIN,claim_sha256=digest(row),pmid='28526817',
        reason='affirmative_symptom_severity_assertion_not_supported_by_own_explicit_fulltext_result',
        original_source_evidence_preserved=True,no_null_result_claim_synthesized=True,
        source_proof=proof['28526817'],owned_edges=[dict(ordinal=o,edge_sha256=digest(e)) for o,e in owned])


def reviewed_frailty_retirements(row,owned,proof):
    md=source_claim(row,proof)
    if row['id']!=FRAILTY or len(owned)!=6:raise ValueError('wrong six-edge frailty scope')
    canonical=(md['subject_id'],md['object_id'])
    if canonical!=('CLM_CONCEPT:white_matter_hyperintensity_burden','CLM_CONCEPT:frailty_09cfc9267861'):raise ValueError('current frailty anchors differ')
    old=('CLM_CONCEPT:white_matter_hyperintensity_burden_e206a534fc08','CLM_CONCEPT:frailty')
    replacements=dict(zip(old,canonical));kept=[];removed=[]
    for ordinal,edge in owned:
        if edge_owner(edge)!=FRAILTY:raise ValueError('frailty owner differs')
        if {edge['source_id'],edge['target_id']}&set(old):removed.append((ordinal,edge))
        else:kept.append((ordinal,edge))
    if len(kept)!=3 or len(removed)!=3 or reviewed_edges(FRAILTY,row,row,kept)!=[]:raise ValueError('canonical closure differs')
    result=[];matched=set()
    for ordinal,edge in removed:
        current=deepcopy(edge)
        for field in ('source_id','target_id'):current[field]=replacements.get(current[field],current[field])
        choices=[(o,e) for o,e in kept if current==e]
        if len(choices)!=1 or choices[0][0] in matched:raise ValueError('alternative payload is not one exact duplicate')
        retained,retained_row=choices[0];matched.add(retained)
        result.append(dict(ordinal=ordinal,claim_id=FRAILTY,edge_sha256=digest(edge),retained_ordinal=retained,
            retained_edge_sha256=digest(retained_row),retired_endpoint_values={f:edge[f] for f in ('source_id','target_id') if edge[f]!=current[f]}))
    if len(matched)!=3:raise ValueError('incomplete duplicate branch mapping')
    return sorted(result,key=lambda r:r['ordinal'])


def viral_node():
    # No global VCP alias or human/mouse gene identity is asserted by this node.
    key=hashlib.sha256((VERSION+'|source_expanded_protein|'+VIRAL_NAME).encode()).hexdigest()
    return ConceptNode(id='CLM_CONCEPT:protein_mention_'+key,preferred_name=VIRAL_NAME,
        domain_tags=['claim_concept'],source_vocab='claim_extraction').to_dict()


def viral_reuse_source_proof(xml):
    docs=source_documents([xml]);result={}
    if set(docs)!={v[1] for v in VIRAL_INCIDENTS.values()}:raise ValueError('viral reuse PMID set differs')
    for _,pmid,doi in VIRAL_INCIDENTS.values():
        d=docs[pmid]
        if doi.casefold() not in {s.casefold() for s in d['dois']} or VIRAL_NAME not in d['title'].casefold():raise ValueError('viral reuse source identity differs')
        result[pmid]=dict(title=d['title'],doi=doi,abstract_sha256=digest(d['abstract']))
    return result


def validate_viral_reuse_witness(witness):
    if not witness or (witness.get('node_id'),witness.get('node_sha256'),witness.get('preferred_name'))!=(VIRAL_REUSE_ID,VIRAL_REUSE_SHA,VIRAL_NAME):raise ValueError('unreviewed viral reuse target')
    if witness.get('global_claim_incidence_scan_complete') is not True:raise ValueError('incomplete viral incidence scan')
    incidents=witness['incidents'];sources=witness['public_identity_sources']
    if len(incidents)!=3 or {r['claim_id'] for r in incidents}!=set(VIRAL_INCIDENTS):raise ValueError('viral incident set differs')
    for r in incidents:
        sha,pmid,doi=VIRAL_INCIDENTS[r['claim_id']]
        if (r['claim_sha256'],r['pmid'],r['doi'],r['side'])!=(sha,pmid,doi,'subject'):raise ValueError('viral incident ownership differs')
        # These two complete sentence-capitalization variants are explicitly
        # source-verified here, not a generic case-insensitive identity merge.
        if r['name'] not in {VIRAL_NAME,'Vaccinia virus complement control protein'}:raise ValueError('viral incident full name differs')
        if pmid not in sources or sources[pmid]['doi']!=doi or VIRAL_NAME not in sources[pmid]['title'].casefold():raise ValueError('viral incident public title differs')
    return True


def reviewed_viral_reuse(node,incidents,public_identity_sources):
    if node['id']!=VIRAL_REUSE_ID or digest(node)!=VIRAL_REUSE_SHA or node.get('preferred_name')!=VIRAL_NAME:raise ValueError('viral reuse node changed')
    if node.get('semantic_types') or node.get('external_ids') or node.get('aliases'):raise ValueError('unreviewed external viral identity')
    if any(r.get('outer_type') not in (None,'') or r.get('inner_type') not in (None,'') for r in incidents):raise ValueError('viral incident roles require separate review')
    witness=dict(node_id=node['id'],node_sha256=digest(node),preferred_name=VIRAL_NAME,
        incidents=sorted([{k:r[k] for k in ('claim_id','claim_sha256','pmid','doi','side','name')} for r in incidents],key=lambda r:r['claim_id']),
        public_identity_sources=public_identity_sources,global_claim_incidence_scan_complete=True,
        existing_claim_scientific_conclusions_not_revalidated=True,all_existing_node_and_claim_values_unchanged=True)
    validate_viral_reuse_witness(witness);return witness


def reviewed_viral_identity(row,proof,target_id,reuse_witness=None):
    md=source_claim(row,proof)
    if row['id']!=VIRUS or (md['subject_id'],md['subject_name'])!=('CUI:C1421437','VCP'):raise ValueError('not reviewed viral VCP')
    if target_id==VIRAL_REUSE_ID:validate_viral_reuse_witness(reuse_witness)
    elif target_id!=viral_node()['id'] or reuse_witness is not None:raise ValueError('unreviewed viral target')
    if md['predicate']!='is_associated_with' or md['object_name']!='uptake in all retinal layers followed by clearance within 1-3 hours':raise ValueError('viral statement scope differs')
    if md['raw_text']!='After VCP was injected into the eye, it was taken up in all layers of the retina but was cleared within 1-3 hours of delivery.':raise ValueError('viral quoted statement differs')
    event=dict(claim_id=VIRUS,claim_sha256=digest(row),nonidentity_sha256=digest(nonidentity_claim(row)),
        changes=[dict(side='subject',old_id='CUI:C1421437',target_id=target_id,name='VCP')],
        source_pmid='29534078',public_source_proof=proof['29534078'],target_reuse_witness=reuse_witness)
    out=change_claim(row,event);event['current_node_sha256']=digest(out)
    return event,out


def compacted_ordinal(old,removed):
    if old in removed:raise ValueError('removed ordinal has no current position')
    return old-bisect_left(removed,old)


def original_ordinal(current,removed):
    if current<1:raise ValueError('invalid current ordinal')
    old=current
    for gone in removed:
        if gone<=old:old+=1
        else:break
    return old
