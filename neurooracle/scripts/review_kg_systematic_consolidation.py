"""Prepare complete R73 rule hits for scientific review; not an apply gate."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from fetch_kg_systematic_sources import parsed
from neurooracle.src.claim_semantics import looks_like_concrete_imaging_measurement,looks_like_method_or_procedure_entity,looks_like_non_imaging_assay_entity
from neurooracle.src.kg_literal_endpoint_repair import MOLECULAR
from neurooracle.src.kg_gene_boundary_repair import word_interior_hits
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import title_key
from neurooracle.src.kg_systematic_consolidation import literal
from neurooracle.src.relation_evidence import name_key
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round73_systematic_relation_consolidation'
IMAGE_CONTEXT=re.compile(r'\b(?:mri|fmri|dti|pet|eeg|meg|bold|neuroimag\w*|magnetic resonance|diffusion tensor|optical coherence|neuroanatom\w*|brain imag\w*|white matter|grey matter|gray matter|functional connectivity|electrophysiolog\w*)\b',re.I)
NON_MEASURE=re.compile(r'\b(?:loci|haplotypes?|associations?|relationships?|mediation|moderation|intervention|absence|presence|mapping|microglial|recruitment)\b|COVID-related stress|physical activity|impaired .* syndrome',re.I)
ALLOWED_ROLES={'biomarker','imaging_marker','network'}
TYPE_DOMAINS={
 'gut microbiota':'microbial_community','symptom severity':'clinical_measure','fronto-limbic network':'imaging_feature',
 'autistic traits':'clinical_phenotype','worry severity':'clinical_measure','ASD symptom severity':'clinical_measure',
 'negative schizotypy':'clinical_phenotype','functional decline':'clinical_outcome','ventromedial prefrontal connectivity':'imaging_feature',
 'gray-white functional synchrony':'imaging_feature','childhood trauma exposure':'exposure','older age':'age_group',
 'structural-connectome-constrained coordinated deformation pattern':'imaging_feature','disorganization symptoms':'clinical_symptom',
 'inattentiveness':'clinical_measure','dynamic inter-network connectivity entropy':'imaging_feature','predicted age difference':'imaging_feature',
 'Klinefelter syndrome':'disease','regional-homogeneity pattern':'imaging_feature','autistic-trait severity':'clinical_measure',
 'OCD diagnosis':'diagnostic_classification','autistic adults':'patient_population','positive symptom severity':'clinical_measure',
 'frontotemporal network':'imaging_feature','ALPS index':'imaging_feature','amyloid-beta pathology':'pathology_state'}


def scientific_metadata(row):
    out=dict(row['scientific_fields']);out['metadata']=dict(row['nested_endpoint_fields']);return out


def measurement_spelling(name):
    # These full-word spelling variants do not remove side, quantity, modality,
    # direction, state, brain region, compound scope or an assay/model name.
    name=re.sub(r'\bgrey\b','gray',name)
    return re.sub(r'\b(gray|white)-matter\b',r'\1 matter',name)


def own_source(md,docs):
    paper=md.get('source_paper') or {};pmid=str(paper.get('pmid') or '')
    source=docs.get(pmid)
    if not source:return None,'owning_primary_record_unavailable'
    def title(value):return title_key(str(value or '').translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"'})))
    if title(paper.get('title'))!=title(source['title']):return source,'owning_title_needs_identity_review'
    if not source['abstract']:return source,'owning_abstract_unavailable'
    return source,None


def surface_gate(name):
    if MOLECULAR.search(name):return 'molecular_or_genetic_identity_requires_separate_review'
    if NON_MEASURE.search(name):return 'relationship_exposure_or_compound_result_not_a_measurement'
    if looks_like_method_or_procedure_entity(name) or looks_like_non_imaging_assay_entity(name):return 'method_or_nonimaging_assay_scope'
    if not looks_like_concrete_imaging_measurement(name):return 'measurement_definition_not_established'
    return None


def observation_gate(md,side,docs):
    name=md[side+'_name'];inner=md.get('metadata') or {}
    roles={str(x).casefold() for x in (md.get(side+'_type'),inner.get(side+'_type')) if x not in ('',None)}
    if not roles or not roles<=ALLOWED_ROLES:return 'explicit_measurement_role_missing_or_conflicting'
    if inner.get(side+'_id',md[side+'_id'])!=md[side+'_id']:return 'nested_endpoint_conflict'
    source,reason=own_source(md,docs)
    if reason:return reason
    text=source['title']+' '+' '.join(x['text'] for x in source['abstract'])
    if not IMAGE_CONTEXT.search(text) or not md.get('raw_text'):return 'source_imaging_or_electrophysiology_context_not_established'
    return None


def main():
    require((OUTPUT/'REVIEW_PROJECTION.json').exists(),'current scientific projections required')
    require(not (OUTPUT/'PLAN.json').exists(),'plan frozen; no review overwrite')
    baseline=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    require(baseline==j.read_json(j.OUTPUT/'CAMPAIGN.json'),'source advanced')
    previous=Path(baseline['current_acceptance']['path']).parent
    db=sqlite3.connect((OUTPUT/'WORKING_PROJECTIONS.sqlite').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    science={r['claim_id']:r for r in rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl')}
    witnesses={r['node_id']:r for r in rows(OUTPUT/'TARGET_WITNESSES.jsonl')}
    docs={}
    for folder in (previous/'source_review',OUTPUT/'source_review'):
        for path in sorted(folder.glob('*.xml')):
            if path.name.startswith('PMC'):continue
            for r in parsed(path):docs[r['pmid']]=r
    terms=VerifiedEntityTerms(j.read_json(baseline['current_entity_terms']['path']))
    genes=rows(baseline['current_gene_holds']['path']);gene_pairs={(r['claim_id'],r['side']) for r in genes}
    gene_names={r['name'] for r in genes};approved_names={n for n in gene_names if surface_gate(n) is None}
    spec={};decisions=[];families=[];new_nodes={};canonical_targets={}
    by_name=defaultdict(list)
    for nid,node in witnesses.items():by_name[name_key(node['preferred_name']).casefold()].append(node)
    def put(cid,side,target,review_id,new_name=None):
        r=spec.setdefault(cid,dict(replacements={},review_ids=[]))
        field=side+'_id'
        require(field not in r['replacements'] or r['replacements'][field]==target,'competing target')
        r['replacements'][field]=target
        if new_name is not None:r['replacements'][side+'_name']=new_name
        if review_id not in r['review_ids']:r['review_ids'].append(review_id)
    for name in sorted(approved_names):
        incidents=[dict(r) for r in db.execute('SELECT * FROM endpoint_incidents WHERE name=?',(name,))]
        require(all(r['cid'] in science for r in incidents),'full-name incident outside projection')
        eligible=[];rejected=[]
        for inc in incidents:
            md=scientific_metadata(science[inc['cid']]);reason=observation_gate(md,inc['side'],docs)
            current=witnesses[inc['nid']]
            if not reason and 'T028' in current['semantic_types']:
                labels={current['preferred_name'],*current['aliases']}
                if any(re.search(r'(?<!\w)'+re.escape(s.strip())+r'(?!\w)',name,re.I) for s in labels if s.strip()):
                    reason='whole_token_gene_alias_requires_source_disambiguation'
            elif not reason and current['semantic_types']:
                if name_key(current['preferred_name']).casefold()!=name_key(name).casefold():reason='canonical_target_has_different_complete_scope'
            if reason:rejected.append(dict(claim_id=inc['cid'],side=inc['side'],reason=reason))
            else:eligible.append(inc)
        if not eligible:
            families.append(dict(name=name,total_incidents=len(incidents),eligible=0,held=rejected));continue
        canonical_name=measurement_spelling(name)
        target,target_kind=canonical_targets.get(canonical_name,(None,None))
        term=terms.entries.get(name_key(name))
        if term and target is None:
            n=witnesses[term['target_id']]
            if (name_key(n['preferred_name']).casefold()==name_key(name).casefold() and
                set(n['semantic_types'])<={'T033','T034','T046','T190','T201'} and n['semantic_types']):
                target=n['node_id'];target_kind='full_name_authoritative_finding'
        if target is None:
            for n in by_name[name_key(name).casefold()]:
                if (n['node_id'].startswith('CUI:') and n['source_vocab']=='UMLS_2026AA' and
                    str(n['metadata_identity_fields'].get('audit_ref','')).startswith('cuis/') and
                    set(n['semantic_types'])<={'T033','T034','T046','T190','T201'} and n['semantic_types']):
                    target=n['node_id'];target_kind='own_source_supported_complete_UMLS_finding';break
        if target is None:
            candidates=by_name[name_key(name).casefold()]
            if canonical_name!=name:candidates=candidates+by_name[canonical_name.casefold()]
            for n in sorted(candidates,key=lambda n:(not n['node_id'].startswith('NCL_IMAGING:'),n['node_id'])):
                if n['semantic_types'] or n['external_ids'] or not set(n['domain_tags'])<= {'imaging_feature','biomarker','connectivity'} or not n['domain_tags']:continue
                if not n['node_id'].startswith(('CLM_CONCEPT:','NCL_IMAGING:')):continue
                all_inc=[dict(r) for r in db.execute('SELECT * FROM endpoint_incidents WHERE nid=?',(n['node_id'],))]
                if any(measurement_spelling(i['name'])!=canonical_name for i in all_inc):continue
                if any(i['cid'] not in science or observation_gate(scientific_metadata(science[i['cid']]),i['side'],docs) for i in all_inc):continue
                target=n['node_id'];target_kind='complete_full_name_literal_and_incident_review';break
        if target is None:
            node=literal(canonical_name);target=node['id'];target_kind='typed_full_mention_without_external_equivalence'
            require(db.execute('SELECT 1 FROM nodes WHERE nid=?',(target,)).fetchone() is None,'new literal already exists')
            new_nodes[target]=node
        canonical_targets[canonical_name]=(target,target_kind)
        review_id='IMAGE_FULL_MENTION_'+digest((name,target))[:16]
        for inc in eligible:
            md=scientific_metadata(science[inc['cid']])
            changed=inc['nid']!=target
            normalized=canonical_name if canonical_name!=name else None
            if changed or normalized:put(inc['cid'],inc['side'],target,review_id,normalized)
            decisions.append(dict(claim_id=inc['cid'],side=inc['side'],name=name,old_id=inc['nid'],target_id=target,
                review_id=review_id,source_pmid=str(md['source_paper'].get('pmid')),source=docs[str(md['source_paper']['pmid'])]['source'],
                original_was_gene='T028' in witnesses[inc['nid']]['semantic_types'],changed=changed or bool(normalized),
                definition_scope='complete named imaging/electrophysiological measure; all acquisition, region, model and statistical details remain observations',
                effect_or_causal_validity_reassessed=False))
        families.append(dict(name=name,target_id=target,target_kind=target_kind,total_incidents=len(incidents),eligible=len(eligible),held=rejected))
    type_decisions=[]
    for group in j.read_json(previous/'source_review/ROOT_TYPE_REVIEW.json'):
        name=group['complete_surface'];nodes={r['node_id'] for r in group['claims']}
        require(len(nodes)==1,'type surface has multiple unresolved nodes');nid=next(iter(nodes))
        incs=[dict(r) for r in db.execute('SELECT * FROM endpoint_incidents WHERE nid=?',(nid,))]
        expected={(x['claim_id'],x['side']) for x in group['claims']}
        require({(x['cid'],x['side']) for x in incs}==expected,'type review needs additional incident sources')
        require(all(i['name']==name for i in incs),'type definition differs by incident')
        for inc in incs:
            md=scientific_metadata(science[inc['cid']]);source,reason=own_source(md,docs)
            require(reason is None,'type source unavailable: '+inc['cid']+' '+str(reason))
        type_decisions.append(dict(node_id=nid,node_sha256=witnesses[nid]['node_sha256'],domains=[TYPE_DOMAINS[name]],
            complete_surface=name,review_id='TYPE_'+digest(name)[:16],current_incidents=incs,source_review=group,
            classification_only_no_scale_or_model_equivalence=True,all_current_incident_sources_checked=True))
    deferred=[]
    for group in rows(previous/'DEFERRED_GROUPS.jsonl'):
        route=({'object_id':'CUI:C0878589'} if group['subject_name']=='hippocampal silent synapse loss' else
               {'subject_id':'CUI:C0030567'} if group['subject_name']=="Parkinson's disease" else {})
        if route:
            for cid in group['complete_current_member_ids']:
                md=scientific_metadata(science[cid]);source,reason=own_source(md,docs);require(reason is None,'deferred source identity')
                for field,value in route.items():put(cid,field[:-3],value,'WHOLE_SOURCE_GROUP_'+group['subject_name'])
            group=dict(group,decision='APPROVED_WHOLE_GROUP_IDENTITY',source_scope_review='own article supports experimental acute SCI in rats or clinical PD diagnosis; retain role annotations and measured response distinction')
        else:group=dict(group,decision='EVIDENCE_REQUIRED_NOT_APPROVED',requirements=group['review_note'])
        deferred.append(group)
    cid='CLM:8ecd6f77479e';md=scientific_metadata(science[cid])
    require(md['source_paper']['pmid']=='35995794' and md['predicate']=='mediates','PRS source predicate changed')
    full=OUTPUT/'source_review/PMC9395379.xml';require(full.exists(),'own full text required')
    spec.setdefault(cid,dict(replacements={},review_ids=[]))['replacements']['predicate']='is_associated_with'
    spec[cid]['review_ids'].append('PRS_FT_SOURCE_ASSOCIATION_NOT_FORMAL_MEDIATION')
    # Remove pure no-ops before the separately reviewed group-closure gate.
    for cid,row in list(spec.items()):
        md=scientific_metadata(science[cid]);row['replacements']={k:v for k,v in row['replacements'].items() if md[k]!=v}
        if not row['replacements']:del spec[cid]
    write_rows(OUTPUT/'DRAFT_MEASUREMENT_DECISIONS.jsonl',decisions)
    j.atomic_json(OUTPUT/'DRAFT_MEASUREMENT_FAMILIES.json',families)
    j.atomic_json(OUTPUT/'DRAFT_TYPE_DECISIONS.json',type_decisions)
    j.atomic_json(OUTPUT/'DRAFT_DEFERRED_DECISIONS.json',deferred)
    j.atomic_json(OUTPUT/'DRAFT_REPLACEMENTS.json',spec)
    write_rows(OUTPUT/'DRAFT_NEW_NODES.jsonl',list(new_nodes.values()))
    ids={str(scientific_metadata(science[cid])['source_paper'].get('pmid') or '') for cid in spec}
    j.atomic_json(OUTPUT/'DRAFT_REVIEW_SUMMARY.json',dict(at=j.utc_now(),status='COMPLETE_RULE_HITS_DRAFT_REQUIRES_REVIEW_AND_GROUP_CLOSURE',
        total_scientific_projection_claims=len(science),measurement_names_considered=len(approved_names),families_with_eligible=sum(f.get('eligible',0)>0 for f in families),
        eligible_endpoints=len(decisions),proposed_claims=len(spec),proposed_gene_endpoints=sum(r['original_was_gene'] and r['changed'] for r in decisions),
        proposed_type_nodes=len(type_decisions),type_incidents=sum(len(r['current_incidents']) for r in type_decisions),
        owning_pmids_in_proposals=len(ids),primary_records_available=len(docs),remaining_source_pmids=sorted(ids-set(docs)),
        held_measurement_reasons=dict(Counter(x['reason'] for f in families for x in f['held'])),graph_mutations=0))
    db.close();print(json.dumps(j.read_json(OUTPUT/'DRAFT_REVIEW_SUMMARY.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
