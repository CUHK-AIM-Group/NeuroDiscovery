"""R74 finite, owning-paper-scoped non-gene mention review.

An exact mention is detached from a demonstrably different individual gene.
The literal carries the owning PMID; scales, assays, interventions, and protein
species are not made equivalent across papers. No names or role annotations
are rewritten. Each scientific family still needs its primary-source card.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from prune_current_kg import rows,write_rows
from kg_accepted_candidate_lineage import require
from review_kg_systematic_consolidation import scientific_metadata,own_source
from neurooracle.src.kg_reviewed_relation_repair import role_values
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_semantic_measurement_repair import literal

OUTPUT=j.OUTPUT/'round74_semantic_measurement_batch'

# Finite review scope, not an inference rule for arbitrary future mentions.
FAMILIES={
 'clinical_measure':(
 'obsessive-compulsive symptom severity','general psychopathology factor',
 'depression severity','diminished expression','individualized compulsions',
 'neurocognitive performance','fine motor function','intraindividual reaction-time variability',
 'obsessive-compulsive disorder symptoms','processing speed deficits',
 'age at onset of obsessive-compulsive disorder','mental state decoding',
 'Functional Assessment Questionnaire','depressive symptom severity'),
 'clinical_phenotype':(
 'childhood irritability','preadolescent irritability','somatic-symptom adolescent depression',
 'auditory-verbal hallucinations','poorer social awareness','somatic depression'),
 'exposure':('childhood maltreatment exposure','childhood maltreatment',
 'coarse particulate matter exposure during pregnancy'),
 'imaging_feature':(
 'module switching rate flexibility','expected white matter glucose metabolism',
 'atypical white matter glucose metabolism','hippocampal atrophy','subcortical brain volumes',
 'error-related negativity amplitude','locus coeruleus integrity',
 'structural magnetic resonance imaging phenotype profile in schizophrenia','subcortical volume',
 'ALPS glymphatic index reduction','MRI indices of glymphatic system','error-related negativity',
 'anterior cingulate myo-inositol concentration','basal ganglia tissue iron',
 'hippocampal subfield volumes','tau positron emission tomography signal'),
 'biochemical_measure':(
 'DNA methylation GDF15','plasma GDF15','triglyceride principal component scores',
 'maternal interleukin-6 concentration during pregnancy','CSF amyloid-beta',
 'cerebrospinal fluid beta-amyloid','CSF amyloid-beta42',
 'CSF amyloid-beta 42','CSF amyloid-beta 1-42','CSF amyloid-beta42 concentration'),
 'biological_process':('oxidative stress','neurodegeneration','glymphatic dysfunction',
 'glymphatic system dysfunction','inflammation','systemic inflammation'),
 'pathology_state':('amyloid-beta plaques','cerebrovascular changes'),
 'molecular_mention':('amyloid-beta','amyloid-beta peptides','brain-derived neurotrophic factor',
 'tenascin-C','magnesium'),
 'intervention':('creatine supplementation','electroconvulsive therapy',
 'transcranial temporal interference stimulation','repetitive transcranial magnetic stimulation'),
}
DOMAINS={name:domain for domain,names in FAMILIES.items() for name in names}

TITLE_VARIANTS={
 '21889115':'Impact of apolipoprotein ɛ4–cerebrospinal fluid beta‐amyloid interaction on hippocampal volume loss over 1 year in mild cognitive impairment',
 '29592889':'Dissociable influences of <i>APOE</i> ε4 and polygenic risk of AD dementia on amyloid and cognition',
 '36030541':'Associations of neural error‐processing with symptoms and traits in a dimensional sample recruited across the obsessive–compulsive spectrum',
}
HELD_CLAIMS={
 'CLM:0eae8bfb6a41':'source_separates_amyloid_deposition_and_neurodegeneration_but_extracted_sentence_conflates_them',
 'CLM:b688aa544f0a':'declared_gene_target_role_conflicts_with_source_amyloid_pathology_requires_role_review',
}
SOURCE_REVIEW_NOTES={
 '15593001':'Intracisternal BDNF treatment in infant rats concerns the administered factor; gene locus equivalence is not asserted.',
 '19838819':'Extracellular matrix molecule and its protein function are explicit; do not infer gene-level causal results.',
 '21889115':'Own NCBI PMID, matching complete title with epsilon/beta typography, and the conclusion independently support CSF Aβ and neurodegeneration.',
 '29592889':'Only inherited italic markup changes the title; longitudinal hippocampal atrophy is explicit in the own abstract.',
 '36030541':'Hyphen/en-dash typography only; ERN amplitudes and symptom factors are separately described.',
 '36047604':'Hold the extracted Aβ/neurodegeneration conflation; the own methods distinguish amyloid PET from cortical thickness/FDG.',
 '40145494':'Source reviews plasma, CSF and PET outcomes separately; the exact source mention is not a claim of assay interchangeability.',
 '40155270':'Source explicitly concerns N-terminal pyroglutamate-modified amyloid forms; preserve this qualifier in the observation, without general protein or APP gene equivalence.',
 '40647320':'Magnesium mineral, deficiency and levels are explicit; no particular salt, supplement dose or MRS2 gene is inferred.',
 '41274863':'DNAm GDF15 and plasma GDF15 remain two distinct exact source mentions; neither is the GDF1 locus.',
 '41280460':'Creatine supplementation is the reviewed intervention, distinct from the molecular substrate and creatine kinase system.',
 '41851083':'Expected and atypical WM metabolic patterns are distinct measures with opposite reported associations; retain both complete names.',
}

def source_gate(md,docs):
    doc,reason=own_source(md,docs)
    pmid=str(md['source_paper'].get('pmid') or '')
    if reason=='owning_title_needs_identity_review' and md['source_paper'].get('title')==TITLE_VARIANTS.get(pmid):
        require(doc['own_article_ids'].get('pubmed')==pmid,'owning article identity differs')
        return doc,None
    return doc,reason

def decide():
    require(not (OUTPUT/'PLAN.json').exists(),'plan frozen')
    base=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    science={r['claim_id']:scientific_metadata(r) for r in rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl')}
    witnesses={r['node_id']:r for r in rows(OUTPUT/'TARGET_WITNESSES.jsonl')}
    docs={r['pmid']:r for r in rows(OUTPUT/'PRIMARY_SOURCES.jsonl')}
    specs={};decisions=[];nodes={};families=defaultdict(list);source_cards={}
    selected=rows(OUTPUT/'SELECTED_GENE_ENDPOINTS.jsonl')
    gene_pairs={(r['claim_id'],r['side']) for r in selected}
    completeness=[]
    for inc in rows(OUTPUT/'COMPLETE_SELECTED_NAME_INCIDENTS.jsonl'):
        cid,side=inc['claim_id'],inc['side'];md=science[cid];w=witnesses[md[side+'_id']]
        is_gene='T028' in w['semantic_types']
        require(not is_gene or (cid,side) in gene_pairs,'unreviewed additional gene occurrence')
        completeness.append(dict(**inc,current_node_id=w['node_id'],current_node_sha256=w['node_sha256'],
            decision='SOURCE_IDENTITY_REVIEW' if is_gene else 'RETAIN_EXISTING_NONGENE_TARGET',
            nongene_equivalence_not_reassessed=not is_gene))
    for r in selected:
        cid,side=r['claim_id'],r['side'];md=science[cid];name=md[side+'_name'];pmid=str(md['source_paper'].get('pmid') or '')
        doc,reason=source_gate(md,docs);reason=HELD_CLAIMS.get(cid) or reason
        roles=role_values(md,side)
        if not reason and (not roles or len({v.casefold() for v in roles})!=1):reason='declared_endpoint_role_missing_or_conflicting'
        if not reason and (md.get('metadata') or {}).get(side+'_id',md[side+'_id'])!=md[side+'_id']:reason='nested_endpoint_identity_conflict'
        if reason:
            families[name].append(dict(claim_id=cid,side=side,reason=reason));continue
        node=literal(name,DOMAINS[name],pmid);nodes[node['id']]=node
        review_id='R74_SOURCE_MENTION_'+digest((pmid,name,DOMAINS[name]))[:20]
        spec=specs.setdefault(cid,dict(replacements={},review_ids=[]));spec['replacements'][side+'_id']=node['id']
        if review_id not in spec['review_ids']:spec['review_ids'].append(review_id)
        decisions.append(dict(claim_id=cid,side=side,name=name,old_id=md[side+'_id'],target_id=node['id'],
            domain=DOMAINS[name],review_id=review_id,source_pmid=pmid,source=doc['source'],
            original_was_gene=True,changed=True,declared_roles=roles,definition_scope='owning_paper_exact_mention_only',
            cross_paper_equivalence_asserted=False,effect_or_causal_validity_reassessed=False))
        source_cards[pmid]=dict(pmid=pmid,source=doc['source'],source_title=doc['title'],abstract_sha256=digest(doc['abstract']),
            own_article_ids=doc['own_article_ids'],review_scope='finite_non_gene_entity_identity_only',
            note=SOURCE_REVIEW_NOTES.get(pmid,'Own abstract describes the named clinical construct, imaging quantity, exposure or biochemical/pathological entity; retain all source-specific methods and assertions.'),
            full_text_reviewed=False,all_effect_directions_validated=False)
    write_rows(OUTPUT/'COMPLETE_NAME_INCIDENT_DISPOSITIONS.jsonl',completeness)
    write_rows(OUTPUT/'APPROVED_SOURCE_IDENTITY_CARDS.jsonl',list(source_cards.values()))
    write_rows(OUTPUT/'DRAFT_MEASUREMENT_DECISIONS.jsonl',decisions)
    j.atomic_json(OUTPUT/'DRAFT_MEASUREMENT_FAMILIES.json',[dict(name=n,eligible=sum(d['name']==n for d in decisions),held=families[n]) for n in sorted(DOMAINS)])
    j.atomic_json(OUTPUT/'DRAFT_TYPE_DECISIONS.json',[])
    prior=Path(base['current_acceptance']['path']).parent
    j.atomic_json(OUTPUT/'DRAFT_DEFERRED_DECISIONS.json',[dict(g,decision='EVIDENCE_REQUIRED_NOT_APPROVED') for g in rows(prior/'DEFERRED_GROUPS.jsonl')])
    j.atomic_json(OUTPUT/'DRAFT_REPLACEMENTS.json',specs);write_rows(OUTPUT/'DRAFT_NEW_NODES.jsonl',list(nodes.values()))
    j.atomic_json(OUTPUT/'DRAFT_REVIEW_SUMMARY.json',dict(at=j.utc_now(),status='SOURCE_IDENTITY_REVIEW_REQUIRES_COMPLETE_GROUP_AND_EDGE_GATE',
        proposed_claims=len(specs),proposed_gene_endpoints=len(decisions),proposed_type_nodes=0,remaining_source_pmids=[],
        source_cards=len(source_cards),held_reasons=dict(Counter(h['reason'] for f in families.values() for h in f)),
        all_selected_name_incidents_accounted_for=len(completeness),graph_mutations=0))
    print(json.dumps(j.read_json(OUTPUT/'DRAFT_REVIEW_SUMMARY.json')),flush=True)

def prepare():
    base=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    require(base==j.read_json(j.OUTPUT/'CAMPAIGN.json'),'current graph advanced')
    science={r['claim_id']:scientific_metadata(r) for r in rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl')}
    gene=rows(base['current_gene_holds']['path'])
    chosen=[r for r in gene if r['name'] in DOMAINS]
    all_hits=[]
    for cid,md in science.items():
        for side in ('subject','object'):
            if md[side+'_name'] in DOMAINS:
                all_hits.append(dict(claim_id=cid,side=side,name=md[side+'_name'],pmid=str(md['source_paper'].get('pmid') or '')))
    ids=sorted({str(science[r['claim_id']]['source_paper'].get('pmid') or '') for r in chosen})
    j.atomic_json(OUTPUT/'SOURCE_REQUEST_PMIDS.json',[p for p in ids if p.isdigit()])
    write_rows(OUTPUT/'SELECTED_GENE_ENDPOINTS.jsonl',chosen)
    write_rows(OUTPUT/'COMPLETE_SELECTED_NAME_INCIDENTS.jsonl',all_hits)
    j.atomic_json(OUTPUT/'WORK_SCOPE.json',dict(at=j.utc_now(),graph=base['current_graph'],
        status='FINITE_SOURCE_REVIEW_REQUIRED',families=FAMILIES,
        selected_gene_endpoints=len(chosen),selected_claims=len({r['claim_id'] for r in chosen}),
        complete_named_incidents=len(all_hits),owning_source_ids=len(ids),
        old_shared_groups_may_not_split=True,no_new_cross_paper_scale_equivalence=True,
        inherited_complete_name_discovery=j.read_json(OUTPUT/'REVIEW_PROJECTION.json')['reused_discovery'],
        structural_followup='Review actual edge inventory; missing/different edges stay held without role guessing.',
        protected_formal_sources=base['formal_sources'],graph_mutations=0))
    print(json.dumps(j.read_json(OUTPUT/'WORK_SCOPE.json'),ensure_ascii=False),flush=True)

def cards():
    science={r['claim_id']:scientific_metadata(r) for r in rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl')}
    docs={r['pmid']:r for r in rows(OUTPUT/'PRIMARY_SOURCES.jsonl')}
    grouped=defaultdict(list)
    for r in rows(OUTPUT/'SELECTED_GENE_ENDPOINTS.jsonl'):
        md=science[r['claim_id']];grouped[str(md['source_paper'].get('pmid') or '')].append(r)
    result=[]
    for pmid,rr in sorted(grouped.items()):
        doc=docs.get(pmid)
        result.append(dict(pmid=pmid,names=sorted({r['name'] for r in rr}),endpoints=len(rr),
            source_title=doc['title'] if doc else '',abstract=doc['abstract'] if doc else [],
            publication_types=doc['publication_types'] if doc else [],source=doc['source'] if doc else None,
            incidents=[dict(claim_id=r['claim_id'],side=r['side'],name=r['name'],
                roles=role_values(science[r['claim_id']],r['side']),
                raw_text=science[r['claim_id']].get('raw_text'),
                title_gate=own_source(science[r['claim_id']],docs)[1]) for r in rr]))
    write_rows(OUTPUT/'SOURCE_REVIEW_CARDS.jsonl',result)
    print(json.dumps(dict(cards=len(result),available=sum(bool(r['abstract']) for r in result),
        title_gates=dict(Counter(i['title_gate'] for r in result for i in r['incidents'])))),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
