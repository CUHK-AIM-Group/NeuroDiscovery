"""R75 finite, owning-paper-scoped non-gene mention review.

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
from neurooracle.src.kg_source_scoped_sets_repair import literal,reviewed_literal,WORK_NAME,WORK_PMIDS

OUTPUT=j.OUTPUT/'round75_source_scoped_sets'

# Finite review scope, not an inference rule for arbitrary future mentions.
FAMILIES={
 'gene_set':(
 'psychiatric risk genes','polygenic-modulated sensorimotor subregion genes',
 'oligogenic-modulated sensorimotor subregion genes','sensorimotor subregion genes',
 'bipolar disorder GWAS risk genes','dopamine-related genes','genes',
 'genes related to glutamatergic synapse and calcium/cAMP signaling',
 'grey matter volume reduction associated genes','ADHD regional-vulnerability implicated gene sets',
 'ADHD, autism, and intellectual-disability risk genes','brain development genes',
 'genes associated with both schizophrenia and cortical MRI metrics',
 'genes determining synaptic biology and glutamate, gamma-aminobutyric acid, dopamine, and serotonin neurotransmitter systems',
 'iron transport and storage genes','mid-line axon development genes','myelin-related genes',
 'psychiatric-disorder-enriched maturation gene networks','schizophrenia-associated genes','synapse-related genes'),
 'expression_profile':(
 'gene expression','grey matter volume reduction associated gene expression',
 'immune and neurological gene-expression transcripts','microglial and neuronal transcriptional signatures',
 'normative expression profiles of disorder risk genes','CA1 pyramidal cell, astrocyte and microglia gene-expression profiles',
 'aging-related gene expression in schizophrenia','aging-related gene expression pattern',
 'anterior insula immune inflammasome and neurodevelopmental gene-expression modules',
 'apoptosis autophagy and neurodevelopment gene-set expression',
 'apoptosis, autophagy and neurodevelopmental gene-set expression',
 'astrocyte and excitatory-neuron transcriptome signatures',
 'autism-related spatial gene-expression enrichment',
 'autophagy, apoptosis and neurodevelopmental gene-set expression',
 'bipolar-disorder associated gene expression','dopamine pathway gene expression',
 'downregulated gene expression','excitatory neuron gene expression','expression of psychosis-related genes',
 'gene expression patterns','gene expression profiles','gene expression spatial profiles','gene transcriptomic profiles',
 'higher prefrontal-cortex TYW5 transcription','inhibitory receptor and inflammation-related gene profiles',
 'limbic network gene-expression signature','local serotonergic receptor gene expression',
 'major depressive disorder gene expression','mood-disorder morbidity and suicide mortality gene-expression signature',
 'neuroimaging-guided transcriptomic signatures','oligodendrocyte gene expression signature',
 'reduced hippocampal GABAergic neurodevelopmental and synaptic gene expression',
 'somatostatin and parvalbumin cortical transcript distributions','spatial gene-expression pattern',
 'synaptic-signaling and neurodegenerative gene-expression enrichment',
 'transmembrane transport and ion channel gene expression'),
 'genetic_variant_mention':(
 'DISC1-TRAX risk haplotypes','13q schizophrenia-risk haplotype homozygosity',
 'CACNA1C rs1006737 risk allele','RBFOX1 common-variant carrier status','SNAP25 MnlI genotype','rs10994336',
 '5-HTTLPR and CRH-related genetic variation','CACNA1C rs1006737 genotype in schizophrenia',
 'CNTNAP2 autism-risk genotype','DBH genotype','DISC1 rs2738880 genotype','DISC1 rs821616 genotype','DISC1 rs821617 genotype',
 'ErbB4 schizophrenia-risk haplotype','FKBP5 rs1360780 genotype in major depressive disorder',
 'GBA variant carrier status','GRIN2B genotype','HTR3A genotype','NEUROG3 rs144643855 genotype',
 'PPM1F genotype','PPM1F rs9610608 genotype','SLC6A15 rs1545843 A+ genotype',
 'SLC6A4 5-HTTLPR genotype','SLC6A4 rs16965628 genotype',
 'TMPRSS15-linked variants','TRAM1L1-downstream rs34043524-linked variants',
 'VMAT1 genotype in major depressive disorder',
 'ZNF804A psychosis-risk genotype across schizophrenia and bipolar disorder',
 'anxiety-disorder genetic-risk variants','balanced translocation carrier genotype',
 'dopamine-related ADHD candidate gene variants','miR-137 schizophrenia risk genotype',
 'potentially causative copy-number variants','risk genotypes','rs203772 genotype',
 'white matter hyperintensity volume genetic risk variants'),
 'transcript_set':('RNA transcripts',),
 'genetic_construct':(
 'bipolar disorder genetic risk','genetic factors','genetically inferred major depression liability',
 'heritability','genetic covariation between schizophrenia and brain phenotypes',
 'shared psychiatric-subcortical polygenic architecture',
 'anxiety disorder polygenic risk score','attention-deficit hyperactivity disorder genetic risk',
 'attention-deficit/hyperactivity disorder polygenic risk score','autism spectrum disorder genetic liability',
 'autism spectrum disorder polygenic risk score','bipolar disorder polygenic risk score',
 'cognitive and non-cognitive educational-attainment polygenic scores',
 'common-variant genetic architecture of schizophrenia risk',
 'cross-disorder polygenic risk score','cross-disorder psychiatric polygenic risk',
 'depression polygenic liability','depression shared genetic architecture','educational-attainment polygenic score',
 'genetic architecture of longitudinal brain structural change','genetic architecture of organ aging',
 'genetic architecture of sociability','genetic heritability','genetic liability to psychiatric disorders',
 'genetic risk for obsessive-compulsive disorder','genetically determined psychosis susceptibility',
 'genetically predicted major depressive disorder','genetically predicted metabolic syndrome components',
 'hippocampal and medial prefrontal gray matter volume immune-gene scores','hippocampal volume genetic architecture',
 'intracranial and subcortical brain volume genetic architecture','levels of genetic correlation across psychiatric disorders',
 'major depression and psychiatric cross-disorder genetic risk','major depression and schizophrenia genetic liability',
 'major depressive disorder genetic risk','neurodevelopmental and internalizing polygenic risk scores',
 'neurodevelopmental internalizing compulsive and psychotic polygenic risk scores',
 'polygenic risk for major depression, post-traumatic stress disorder, or schizophrenia in middle childhood',
 'polygenic risk for problematic alcohol use in substance-naive children',
 'schizophrenia and cross-disorder polygenic scores','shared genetic components between psychiatric disorders',
 'shared genetic variance between schizophrenia and bipolar disorder','subcortical-volume common-variant genetic architecture'),
 'biomarker_panel':(
 'plasma biomarkers','neurodegeneration biomarkers','AD biomarkers','CSF biomarkers','Core 2 biomarkers',
 'amyloid biomarkers','biochemical biomarkers','biological biomarkers','blood biomarkers',
 'cerebrospinal fluid biomarkers','cerebrospinal fluid metabolites','cerebrovascular biomarkers',
 'cerebrovascular disease biomarkers','inflammatory biomarkers','metabolic biomarkers',
 'multimodal biomarker signature','multimodal biomarkers','peripheral biomarkers',
 'plasma biomarkers of neurodegeneration','small vessel disease biomarkers'),
 'microbiome_measure':('gut microbiota composition','gut microbiota alpha diversity','infant gut microbiome alpha diversity'),
 'molecular_set':('pro-inflammatory cytokines',),
 'biochemical_measure':('peripheral pro-inflammatory cytokine levels',
 'sIL-6R and sTNF-R1 inflammatory cytokine levels','DNA-methylation CRP inflammation signature'),
 'computational_model':('spatial-temporal co-attention learning model',),
 'risk_profile':('substance use-related risk profile',),
 'intervention':('transcranial temporal interference stimulation',),
}
DOMAINS={name:domain for domain,names in FAMILIES.items() for name in names}

TITLE_VARIANTS={
 '29947313':'Significant concordance of genetic variation that increases both the risk for obsessive–compulsive disorder and the volumes of the nucleus accumbens and putamen',
 '32853481':"<scp><i>GBA</i></scp> Variants in Parkinson's Disease: Clinical, Metabolomic, and Multimodal Neuroimaging Phenotypes",
}
HELD_CLAIMS={
 'CLM:CASE1MAN:39173920:1967':'source_separates_inhibitory_receptor_maps_and_inflammatory_transcription_but_endpoint_conflates_levels',
 'CLM:CASE1MAN:41619402:4239':'owning_anorexia_paper_does_not_support_mood_disorder_insula_transcriptomic_sentence_source_attribution_requires_review',
}
SOURCE_REVIEW_NOTES={}


def source_gate(md,docs):
    doc,reason=own_source(md,docs)
    pmid=str(md['source_paper'].get('pmid') or '')
    if reason=='owning_title_needs_identity_review' and md['source_paper'].get('title')==TITLE_VARIANTS.get(pmid):
        require(doc['own_article_ids'].get('pubmed')==pmid,'owning article identity differs')
        return doc,None
    return doc,reason

def verified_work_review(base,science,docs):
    pre,final,withdrawn=(docs[p] for p in ('40463528','41167554','40835066'))
    def links(doc,kind,pmid):
        return any(r['ref_type']==kind and r['pmid']==pmid for r in doc['comments_corrections'])
    require(links(pre,'UpdateIn','41167554') and links(final,'UpdateOf','40463528'),'missing reciprocal preprint/final linkage')
    require(links(final,'UpdateOf','40835066') and links(withdrawn,'UpdateIn','41167554'),'intermediate version linkage differs')
    require(withdrawn['title'].startswith('WITHDRAWN:') and 'Retraction Notice' in withdrawn['publication_types'],'withdrawal evidence differs')
    require('Systematic Review' in final['publication_types'],'final source role differs')
    sentence='Transcranial temporal interference stimulation (tTIS) is a novel, non-invasive method developed to selectively modulate deep brain regions and associated neural circuits.'
    for doc in (pre,final):
        require(sentence in ' '.join(a['text'] for a in doc['abstract']),'method sentence differs across versions')
        require('CRD42024559678' in ' '.join(a['text'] for a in doc['abstract']),'review registration differs')
    rid='REL:45ed1797a7f27d353f75cfece1f73b23ee75d38104f1e25630df56c01a4222b0'
    group=next(g for g in rows(base['current_shared_relations']['path']) if g['id']==rid)
    ids=sorted(m['claim_id'] for m in group['members'])
    require(ids==sorted(['CLM:6f9f487c3c4d','CLM:78f9d81c35dd','CLM:ae6ac9e1a05f']),'whole work group changed')
    for cid in ids:
        md=science[cid]
        require(md['raw_text']==sentence and not md['negated'] and md['predicate']=='modulates'
            and md['subject_name']==WORK_NAME and md['object_name']=='deep brain regions'
            and str(md['source_paper']['pmid']) in WORK_PMIDS,'work observation outside reviewed scope')
    review=dict(status='VERIFIED_TWO_VERSION_METHOD_IDENTITY',graph=base['current_graph'],at=j.utc_now(),
        complete_current_member_ids=ids,relation_id=rid,source_pmids=list(WORK_PMIDS),
        evidence={d['pmid']:dict(source=d['source'],own_article_ids=d['own_article_ids'],links=d['comments_corrections'],
            title=d['title'],abstract_sha256=digest(d['abstract']),publication_types=d['publication_types']) for d in (pre,final,withdrawn)},
        withdrawn_intermediate_excluded='40835066',withdrawal_reason='publisher process error',
        statement_sha256=digest(sentence),source_records_preserved=True,paper_identity_registry_unchanged=True,
        independent_primary_studies=0,therapeutic_efficacy_not_inferred=True,full_text_reviewed=False)
    j.atomic_json(OUTPUT/'VERIFIED_WORK_VERSION_REVIEW.json',review)
    return dict(relation_id=rid,complete_current_member_ids=ids,decision='APPROVED_WHOLE_GROUP_IDENTITY',
        review_id='R75_VERIFIED_TTIS_WORK_VERSIONS',review=j.fingerprint(OUTPUT/'VERIFIED_WORK_VERSION_REVIEW.json'))


def decide():
    require(not (OUTPUT/'PLAN.json').exists(),'plan frozen')
    base=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    science={r['claim_id']:scientific_metadata(r) for r in rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl')}
    witnesses={r['node_id']:r for r in rows(OUTPUT/'TARGET_WITNESSES.jsonl')}
    docs={r['pmid']:r for r in rows(OUTPUT/'PRIMARY_SOURCES.jsonl')}
    reviews=j.read_json(OUTPUT/'MANUAL_SOURCE_REVIEW.json')
    work=verified_work_review(base,science,docs)
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
        if not reason and name not in reviews.get(pmid,{}).get('approved_names',[]):reason='finite_owning_source_identity_review_not_approved'
        roles=role_values(md,side)
        if not reason and (not roles or len({v.casefold() for v in roles})!=1):reason='declared_endpoint_role_missing_or_conflicting'
        if not reason and (md.get('metadata') or {}).get(side+'_id',md[side+'_id'])!=md[side+'_id']:reason='nested_endpoint_identity_conflict'
        if reason:
            families[name].append(dict(claim_id=cid,side=side,reason=reason));continue
        node=reviewed_literal(name,DOMAINS[name],pmid);nodes[node['id']]=node
        review_id='R75_SOURCE_MENTION_'+digest((pmid,name,DOMAINS[name]))[:20]
        spec=specs.setdefault(cid,dict(replacements={},review_ids=[]));spec['replacements'][side+'_id']=node['id']
        if review_id not in spec['review_ids']:spec['review_ids'].append(review_id)
        decisions.append(dict(claim_id=cid,side=side,name=name,old_id=md[side+'_id'],target_id=node['id'],
            domain=DOMAINS[name],review_id=review_id,source_pmid=pmid,source=doc['source'],
            original_was_gene=True,changed=True,declared_roles=roles,
            definition_scope='verified_two_version_work_mention' if name==WORK_NAME and pmid in WORK_PMIDS else 'owning_paper_exact_mention_only',
            source_pmids_scope=list(WORK_PMIDS) if name==WORK_NAME and pmid in WORK_PMIDS else [pmid],
            cross_work_equivalence_asserted=False,effect_or_causal_validity_reassessed=False))
        source_cards[pmid]=dict(pmid=pmid,source=doc['source'],source_title=doc['title'],abstract_sha256=digest(doc['abstract']),
            own_article_ids=doc['own_article_ids'],review_scope='finite_non_gene_entity_identity_only',
            note=reviews[pmid]['note'],
            full_text_reviewed=False,all_effect_directions_validated=False)
    write_rows(OUTPUT/'COMPLETE_NAME_INCIDENT_DISPOSITIONS.jsonl',completeness)
    write_rows(OUTPUT/'APPROVED_SOURCE_IDENTITY_CARDS.jsonl',list(source_cards.values()))
    write_rows(OUTPUT/'DRAFT_MEASUREMENT_DECISIONS.jsonl',decisions)
    j.atomic_json(OUTPUT/'DRAFT_MEASUREMENT_FAMILIES.json',[dict(name=n,eligible=sum(d['name']==n for d in decisions),held=families[n]) for n in sorted(DOMAINS)])
    j.atomic_json(OUTPUT/'DRAFT_TYPE_DECISIONS.json',[])
    prior=Path(base['current_acceptance']['path']).parent
    j.atomic_json(OUTPUT/'DRAFT_DEFERRED_DECISIONS.json',[*[dict(g,decision='EVIDENCE_REQUIRED_NOT_APPROVED') for g in rows(prior/'DEFERRED_GROUPS.jsonl')],work])
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
    j.atomic_json(OUTPUT/'SOURCE_REQUEST_PMIDS.json',sorted({p for p in ids if p.isdigit()}|{'40835066'}))
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
