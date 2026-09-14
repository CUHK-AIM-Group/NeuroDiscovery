"""Three finite observation claims: precise existing evidence fields, no new schema."""
from copy import deepcopy
from .kg_identity_pilot import digest
from .kg_scoped_structure import source_documents
from .kg_scientific_definition_repair import at_path,rewrite,protected_digest,apply_edge
from .kg_literal_endpoint_repair import edge_owner,reviewed_edges as source_edges

VERSION='kg.observational_semantics.v1'
REBOX='CLM:a3ce9c23b246de1a'
INPATIENT='CLM:debb3aa8b8f253b0'
CONTINUATION='CLM:a5442a6df750ab2d'
SPECS={
    REBOX:dict(pmid='12490769',doi='10.1097/00004850-200301000-00002',predicate='treats',original_predicate='improves',direction='positive',
        study_type='randomized controlled trials',
        methodology='Within-reboxetine-group cognitive change at day 56 versus baseline; nonsignificant changes in the other arms do not establish a between-arm treatment-effect difference.',
        fragments=('randomized, double-blind, placebo- and active-treatment-controlled','at day 56 compared with baseline',
            'No significant changes or trends in this direction were seen among patients who received either placebo or paroxetine.')),
    INPATIENT:dict(pmid='12769242',doi='10.1037/0002-9432.73.2.212',predicate='treats',original_predicate='improves',direction='positive',
        study_type='prospective study',
        methodology='Functional improvement observed during comprehensive psychiatric inpatient care; the abstract does not report a randomized or controlled treatment-effect comparison.',
        fragments=('This prospective study examined psychosocial and cognitive functioning','under comprehensive psychiatric inpatient care',
            'Nonverbal and general cognitive performance, self-image, and overall psychosocial functioning improved in both groups.')),
    CONTINUATION:dict(pmid='15131518',doi='',predicate='reduces',original_predicate='reduces',direction='negative',
        study_type='nonrandomized unblinded continuation phase',
        methodology='Continuation after acute-phase full or partial remission; combined therapy compared with either monotherapy without randomized assignment, blinding or placebo, particularly among partial remitters.',
        fragments=('Continuation treatment assignment was not randomized or blinded. There was no placebo group.',
            'COMB was associated with less symptom re-emergence during the continuation phase than either monotherapy, particularly for partial remitters.')),
}


def source_proof(xml):
    docs=source_documents([xml]);out={}
    for spec in SPECS.values():
        pmid=spec['pmid'];d=docs.get(pmid)
        if not d or not all(f in d['abstract'] for f in spec['fragments']):raise ValueError('own qualified abstract missing or changed')
        if spec['doi'] and spec['doi'].casefold() not in {x.casefold() for x in d['dois']}:raise ValueError('own DOI differs')
        out[pmid]=dict(pmid=pmid,doi=spec['doi'],title=d['title'],abstract_sha256=digest(d['abstract']),
            reviewed_fragment_hashes=[digest(f) for f in spec['fragments']],scope='finite_observational_predicate_direction_and_design_not_full_audit')
    return out


def literal_node(name):raise ValueError('this batch adds no concept nodes')


def changes_for(cid):
    if cid not in SPECS:raise ValueError('unreviewed observation claim')
    return []


def field_changes(record):
    spec=SPECS[record['id']];md=record['metadata'];inner=md.get('metadata') or {}
    fields=[(('metadata','predicate'),spec['predicate'],'is_associated_with')]
    for field in ('direction','study_type','methodology'):fields.append((('metadata','evidence',field),'',spec[field]))
    if 'predicate' in inner:fields.append((('metadata','metadata','predicate'),spec['predicate'],'is_associated_with'))
    fields.append((('preferred_name',),f"{md['subject_name']} {spec['predicate']} {md['object_name']}",
        f"{md['subject_name']} is_associated_with {md['object_name']}"))
    return fields


def reviewed_claim(record,changes,proof,reviews):
    cid=record['id'];md=record['metadata'];spec=SPECS.get(cid);review=reviews.get(cid)
    if not spec or not review or review['claim_sha256']!=digest(record) or changes!=[]:raise ValueError('not finite current observation')
    source=md.get('source_paper') or {};inner=md.get('metadata') or {}
    if str(source.get('pmid'))!=spec['pmid'] or str(source.get('doi') or '').casefold()!=spec['doi'].casefold():raise ValueError('current source differs')
    if md.get('negated') is not False or md.get('predicate')!=spec['predicate'] or inner.get('original_predicate')!=spec['original_predicate']:
        raise ValueError('current predicate/negation/history differs')
    if not md.get('raw_text') or not proof.get(spec['pmid']) or proof[spec['pmid']]['doi']!=spec['doi']:raise ValueError('own scientific proof missing')
    fields=field_changes(record);out=rewrite(record,fields);protected=protected_digest(record,fields)
    if protected_digest(out,fields)!=protected:raise ValueError('unapproved evidence/audit changed')
    event=dict(claim_id=cid,claim_sha256=digest(record),changes=[],field_changes=[dict(path=list(p),old=o,new=n) for p,o,n in fields],
        protected_fields_sha256=protected,original_audit_sha256=digest(md.get('scope_reaudit')),public_source_proof=proof[spec['pmid']],
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
    events=[];spec=SPECS[cid]
    for ordinal,row in edge_rows:
        if row['relation_type']=='about':continue
        md=row.get('metadata') or {}
        if set(md)-{'claim_id','negated','claim_case_study_ids','paper_case_study_ids','original_predicate'}:
            raise ValueError('unreviewed scientific edge payload')
        if md.get('original_predicate')!=spec['original_predicate'] or md.get('negated') is not False:
            raise ValueError('scientific edge history or negation differs')
        out=deepcopy(row);out['relation_type']='is_associated_with'
        events.append(dict(ordinal=ordinal,claim_id=cid,edge_sha256=digest(row),current_edge_sha256=digest(out),
            changes={'relation_type':dict(old=spec['predicate'],new='is_associated_with')}))
    if len(events)!=1:raise ValueError('one observed scientific edge required')
    changed={e['ordinal']:e for e in events}
    source_edges(cid,current,current,[(o,apply_edge(r,changed[o]) if o in changed else r) for o,r in edge_rows])
    return events


def remaining_issues(issues,plan):
    changed={e['claim_id']:e for e in plan['events']};bycid={r['claim_id']:r for r in issues}
    if len(bycid)!=len(issues) or len(issues)!=17:raise ValueError('issue cardinality differs')
    if set(changed)!=set(SPECS) or not set(changed)<=set(bycid):raise ValueError('finite issue scope differs')
    for cid,event in changed.items():
        if bycid[cid]['current_node_sha256']!=event['claim_sha256']:raise ValueError('source issue hash differs')
    return [deepcopy(r) for r in issues if r['claim_id'] not in changed]
