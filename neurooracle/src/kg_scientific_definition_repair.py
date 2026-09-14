"""R57 eight source-defined wording/measurement repairs; original evidence stays."""
from copy import deepcopy
from xml.etree import ElementTree as ET
from .kg_historic_literal_repair import literal_node as complete_literal
from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import edge_owner,reviewed_edges as source_edges
from .kg_scoped_structure import source_documents

VERSION='kg.scientific_definition_repair.v1'
BACE='CLM:ead63272cd396884'
ACC='CLM:10532161a8b2296ab45e58e9db202527'
DPD_A='CLM:1e89987996da15c2c8921884e85f0ce7'
DPD_B='CLM:0a6379df92d50ec2272354b3c523f2dc'
CLDN='CLM:case3_topup_20260730_manual_b0002_088_01'
CHILD='CLM:ff283248e7dd42deb59044449fc63dcd'
ENIGMA='CLM:CASE1MAN:22504417:9443'
FES='CLM:e228496be8a5'
OLD_BILATERAL='CLM_CONCEPT:bilateral_hippocampal_volume_889764a9fddb'
SEPARATE='left and right hippocampal volumes'
MEAN='mean bilateral hippocampal volume'
SUM='sum of left and right hippocampal volumes'
CONCENTRATION='plasma BACE1 concentration'
SEVERITY='depressive disorder severity in DPD'
NEW_NAMES={SUM,CONCENTRATION,SEVERITY}
TARGETS={SEPARATE:'CLM_CONCEPT:left_and_right_hippocampal_volumes_b21f1dc52b59',
    MEAN:'CLM_CONCEPT:imaging_mention_acdcaec52d1fb4a938dd707134a763d0e3f7281edf4957db41cee0084712bde8',
    **{name:complete_literal(name)['id'] for name in NEW_NAMES}}
# cid: owning PMID, old predicate, endpoint tuples (side, old complete name, old ID, corrected complete name).
SPECS={
    BACE:('36847009','is_associated_with',[('subject','BACE1 amyloid-processing pathway','CLM_CONCEPT:a29d53b54aa122f6',CONCENTRATION)]),
    ACC:('37721751','increases',[]),
    DPD_A:('38223081','correlates_with',[('subject','bilateral hippocampal volume',OLD_BILATERAL,SEPARATE),
        ('object','with increasing depressive-disorder severity in DPD','CLM_CONCEPT:with_increasing_depressive_disorder_severity_in_dpd_e88caaa999dc',SEVERITY)]),
    DPD_B:('38223081','correlates_with',[('subject','bilateral hippocampal volume',OLD_BILATERAL,SEPARATE),
        ('object','severity of depressive disorder in DPD in an inverse direction','CLM_CONCEPT:severity_of_depressive_disorder_in_dpd_in_an_inverse_direction_33fce4b23e25',SEVERITY)]),
    CLDN:('42300138','correlates_with',[('object','bilateral hippocampal volume',OLD_BILATERAL,SEPARATE)]),
    CHILD:('20735996','mediates',[('subject','bilateral hippocampal volume',OLD_BILATERAL,SUM)]),
    ENIGMA:('22504417','is_associated_with',[('subject','bilateral hippocampal volume','CUI:C1421437',MEAN)]),
    FES:('35145436','distinguishes',[('subject','bilateral hippocampal volume','CUI:C1421437',SEPARATE)]),
}
DOIS={'36847009':'10.3233/JAD-221174','37721751':'10.1001/jamanetworkopen.2023.34483','38223081':'10.21037/qims-23-919',
    '42300138':'10.1097/psy.0000000000001501','20735996':'10.1016/j.brainres.2010.08.049','22504417':'10.1038/ng.2250','35145436':'10.3389/fpsyt.2021.747386'}
FULL_FRAGMENTS={
    '37721751':('greater surface area','our findings cannot establish a causal relationship','our data were cross-sectional'),
    '38223081':('negative correlation between left and right hippocampal volumes and HAMD score in the DPD group','we employed a cross-sectional design'),
    '42300138':('six linear regressions (3 CLDN5 probes X 2 hemispheres)','left (B =','right (B ='),
    '20735996':('the sum of left and right hippocampal volumes','sum of left and right hippocampal volume'),
    '22504417':('mean bilateral hippocampal','rs7294919'),
    '35145436':('the corresponding volumes of eight regions from each hemisphere',
        'Left hippocampus/amygdala and right hippocampus/amygdala/thalamus volumes were significantly lower in FES compared with HCs'),
}


def source_proof(abstract_xml,fulltexts):
    docs=source_documents([abstract_xml]);proof={}
    if set(fulltexts)!=set(FULL_FRAGMENTS):raise ValueError('six own fulltexts required')
    for pmid,doi in DOIS.items():
        d=docs.get(pmid)
        if not d or doi.casefold() not in {x.casefold() for x in d['dois']}:raise ValueError('own abstract/DOI differs')
        p=dict(pmid=pmid,doi=doi,abstract_sha256=digest(d['abstract']),title=d['title'])
        if pmid=='36847009':
            fragments=('Plasma BACE1 concentrations were measured','BACE1 concentration was negatively associated','hippocampal volume','in the MCI group')
            if not all(f in d['abstract'] for f in fragments):raise ValueError('concentration source differs')
            p['concentration_not_pathway']=True
        else:
            root=ET.fromstring(fulltexts[pmid]);articles=[root] if root.tag=='article' else root.findall('article')
            if len(articles)!=1:raise ValueError('ambiguous owning article')
            a=articles[0];ids={n.get('pub-id-type'):n.text for n in a.findall('./front/article-meta/article-id')}
            if ids.get('pmid')!=pmid or str(ids.get('doi','')).casefold()!=doi.casefold():raise ValueError('wrong owning fulltext identity')
            body=a.find('body')
            if body is None:raise ValueError('missing own body')
            text=' '.join(''.join(body.itertext()).split());fragments=FULL_FRAGMENTS[pmid]
            if not all(f in text for f in fragments):raise ValueError('own reviewed definition differs: '+pmid)
            p.update(body_sha256=digest(text),scope_support_fragment_hashes=[digest(f) for f in fragments])
        proof[pmid]=p
    return proof


def literal_node(name):
    if name not in NEW_NAMES:raise ValueError('unreviewed new measurement/description')
    return complete_literal(name)


def changes_for(cid):
    if cid not in SPECS:raise ValueError('unreviewed claim')
    return [dict(side=side,name=old_name,new_name=name,old_id=old_id,target_id=TARGETS[name]) for side,old_name,old_id,name in SPECS[cid][2]]


def field_changes(record):
    """Only finite scalar fields; source text, statistics and audit are never copied or replaced."""
    cid=record['id'];md=record['metadata'];inner=md.get('metadata') or {};pmid,predicate,sides=SPECS[cid]
    changes=[]
    for side,old_name,old_id,name in sides:
        changes.extend([( ('metadata',side+'_id'),old_id,TARGETS[name]),(('metadata',side+'_name'),old_name,name)])
        for suffix,old,new in (('_id',old_id,TARGETS[name]),('_name',old_name,name)):
            if side+suffix in inner:changes.append((('metadata','metadata',side+suffix),old,new))
    if cid==BACE:
        changes.append((('metadata','metadata','subject_type'),'PATHWAY','IMAGING_MARKER'))
        if 'subject_type' in md:changes.append((('metadata','subject_type'),'PATHWAY','IMAGING_MARKER'))
    if cid==ACC:
        changes.extend([(('metadata','predicate'),'increases','is_associated_with'),(('metadata','evidence','direction'),'','positive')])
        if 'predicate' in inner:changes.append((('metadata','metadata','predicate'),'increases','is_associated_with'))
    elif cid in {DPD_A,DPD_B}:changes.append((('metadata','evidence','direction'),'','negative'))
    old_name=f"{md['subject_name']} {predicate} {md['object_name']}"
    labels={side:name for side,old_name,old_id,name in sides}
    new_name=f"{labels.get('subject',md['subject_name'])} {'is_associated_with' if cid==ACC else predicate} {labels.get('object',md['object_name'])}"
    changes.append((('preferred_name',),old_name,new_name))
    return changes


def at_path(record,path):
    for k in path:record=record[k]
    return record


def rewrite(record,fields,reverse=False):
    out=deepcopy(record)
    for path,old,new in fields:
        if at_path(out,path)!=(new if reverse else old):raise ValueError('finite original/current field differs: '+'.'.join(path))
        parent=out
        for k in path[:-1]:parent=parent[k]
        parent[path[-1]]=old if reverse else new
    return out


def protected_digest(record,fields):
    out=deepcopy(record)
    for path,old,new in fields:
        parent=out
        for k in path[:-1]:parent=parent[k]
        parent[path[-1]]='__EXACT_REVIEWED_FIELD__'
    return digest(out)


def reviewed_claim(record,changes,proof,reviews):
    cid=record['id'];md=record['metadata'];review=reviews.get(cid)
    if cid not in SPECS or not review or review['claim_sha256']!=digest(record):raise ValueError('not exact finite current claim')
    pmid,predicate,sides=SPECS[cid]
    if changes!=changes_for(cid) or str(md['source_paper'].get('pmid'))!=pmid or md['predicate']!=predicate or md.get('negated') is not False:
        raise ValueError('source/scope differs')
    doi=md['source_paper'].get('doi') or ''
    if (cid!=BACE and doi.casefold()!=DOIS[pmid].casefold()) or (cid==BACE and doi!=''):raise ValueError('original source bibliography differs')
    if pmid not in proof or proof[pmid]['doi']!=DOIS[pmid]:raise ValueError('own public proof missing')
    if not md.get('raw_text'):raise ValueError('original source text missing')
    fields=field_changes(record);out=rewrite(record,fields)
    if out['metadata']['subject_id']==out['metadata']['object_id']:raise ValueError('new endpoint collapse')
    protected=protected_digest(record,fields)
    if protected_digest(out,fields)!=protected:raise ValueError('unapproved science/audit changed')
    event=dict(claim_id=cid,claim_sha256=digest(record),changes=changes,field_changes=[dict(path=list(p),old=o,new=n) for p,o,n in fields],
        protected_fields_sha256=protected,original_audit_sha256=digest(md.get('scope_reaudit')),public_source_proof=proof[pmid],
        current_node_sha256=digest(out),original_raw_text_and_statistics_preserved=True,old_audit_not_revalidated=True)
    return event,out


def change_claim(record,event):
    if digest(record)!=event['claim_sha256'] or event['changes']!=changes_for(record['id']):raise ValueError('source/event differs')
    fields=field_changes(record)
    if [dict(path=list(p),old=o,new=n) for p,o,n in fields]!=event['field_changes']:raise ValueError('unapproved field scope')
    out=rewrite(record,fields)
    if digest(out)!=event['current_node_sha256'] or protected_digest(out,fields)!=event['protected_fields_sha256']:raise ValueError('output/protected fields differ')
    return out


def reverse_claim(record,event):
    if digest(record)!=event['current_node_sha256']:raise ValueError('current record differs')
    fields=[(tuple(r['path']),r['old'],r['new']) for r in event['field_changes']]
    original=rewrite(record,fields,reverse=True)
    if digest(original)!=event['claim_sha256'] or change_claim(original,event)!=record:raise ValueError('source inverse differs')
    return original


def reviewed_edges(cid,original,current,edge_rows):
    source_edges(cid,original,original,edge_rows)
    before=original['metadata'];after=current['metadata'];events=[]
    for ordinal,row in edge_rows:
        out=deepcopy(row)
        if row['relation_type']=='about':
            side=next(side for side in ('subject','object') if before[side+'_id']==row['target_id']);out['target_id']=after[side+'_id']
        else:
            edge_md=row.get('metadata') or {}
            extra=set(edge_md)-{'claim_case_study_ids','claim_id','negated','paper_case_study_ids'}
            if extra and not (cid==CLDN and extra=={'original_predicate'}
                and edge_md['original_predicate']==before['predicate']==after['predicate']=='correlates_with'):
                raise ValueError('unreviewed scientific edge payload')
            if (row.get('metadata') or {}).get('negated',before['negated'])!=before['negated']:
                raise ValueError('scientific edge negation disagrees')
            out.update(source_id=after['subject_id'],target_id=after['object_id'],relation_type=after['predicate'])
        if row!=out:
            if out['source_id']==out['target_id']:raise ValueError('new self loop')
            events.append(dict(ordinal=ordinal,claim_id=cid,edge_sha256=digest(row),current_edge_sha256=digest(out),
                changes={f:dict(old=row[f],new=out[f]) for f in ('source_id','target_id','relation_type') if row[f]!=out[f]}))
    source_edges(cid,current,current,[(o,apply_edge(r,next(e for e in events if e['ordinal']==o))) if any(e['ordinal']==o for e in events) else (o,r) for o,r in edge_rows])
    return events


def apply_edge(row,event,reverse=False):
    if digest(row)!=event['current_edge_sha256' if reverse else 'edge_sha256']:raise ValueError('edge source changed')
    if not set(event['changes'])<={'source_id','target_id','relation_type'}:raise ValueError('unapproved edge field')
    out=deepcopy(row)
    for field,ch in event['changes'].items():
        if row[field]!=ch['new' if reverse else 'old']:raise ValueError('edge field changed')
        out[field]=ch['old' if reverse else 'new']
    if digest(out)!=event['edge_sha256' if reverse else 'current_edge_sha256']:raise ValueError('edge evidence changed')
    return out


def update_scope_findings(scope,plan):
    """Separate repaired fields from unresolved source definitions and old audits."""
    out=deepcopy(scope);changed={r['claim_id']:r for r in plan['events']}
    if set(out['current_claim_hashes'])!={BACE,ACC,'CLM:CASE1MAN:36716140:8023'}:raise ValueError('unexpected source scientific register')
    for cid,h in list(out['current_claim_hashes'].items()):
        if cid in changed:
            if changed[cid]['claim_sha256']!=h:raise ValueError('source scope claim hash differs')
            out['current_claim_hashes'][cid]=changed[cid]['current_node_sha256']
    for r in out['findings']:
        if r['claim_id']==BACE:r.update(classification='hippocampal_aggregation_and_historical_audit_review',
            detail='The subject is now plasma BACE1 concentration, not a pathway. Available own abstract does not define bilateral aggregation; old genetic/pathway scope audit is preserved as historical evidence, not revalidated.')
        elif r['claim_id']==ACC:r.update(classification='historic_ACC_excerpt_retained_after_primary_source_repair',
            detail='The primary fulltext supports a positive observational association; predicate/direction are corrected. Original truncated raw_text and extraction span remain untouched for provenance; the old audit is not a new scientific approval.')
    out['current_claim_hashes'][ENIGMA]=changed[ENIGMA]['current_node_sha256']
    out['findings'].append(dict(claim_id=ENIGMA,classification='compound_variant_expression_object_requires_review',
        detail='Mean bilateral hippocampal volume identity is corrected. The object still combines rs7294919 and TESC expression, although the source tested variant-volume and variant-expression associations separately. No component was silently deleted.'))
    for cid,r in sorted(plan['unmodified_sample_size_reviews'].items()):
        if r['science']['evidence']['sample_size']!=170 or str(r['source_paper'].get('pmid'))!='30384145':raise ValueError('unmodified sample-denominator finding differs')
        out['current_claim_hashes'][cid]=r['claim_sha256']
        out['findings'].append(dict(claim_id=cid,classification='sex_specific_telomere_sample_denominator_unresolved',
            detail='Stored sample_size=170 conflicts with the own abstract describing 30 patients and 60 controls. Sex-specific analysis denominators are unavailable; neither 90 nor another guessed value was written.'))
    out.setdefault('resolved',{}).update(BACE1_concentration_subject_definition=BACE,ACC_positive_observational_relation=ACC,
        DPD_duplicate_relation_expressions_normalized=[DPD_A,DPD_B],source_defined_hippocampal_measurement_scopes=[ENIGMA,FES,CLDN,CHILD])
    out.update(hippocampal_scope='Mean, sum and separately reported hemispheres are now explicitly distinguished for the finite reviewed set. 15038994 and other unspecified aggregations still require evidence.',
        source_resolution_limits='Eight reviewed field corrections only. Original raw_text, statistical values and old audit records stay unchanged; these are not blanket scientific approvals.',
        issue_register_is_exhaustive=False,queue_counts_may_overlap=True,old_audits_not_revalidated=True)
    return out
