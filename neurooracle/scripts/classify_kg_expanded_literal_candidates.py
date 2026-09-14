"""Classify R52 exact non-molecular full-mention candidates, without graph writes."""
from collections import Counter
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round52_expanded_literal_review'
VERSION='kg.nonmolecular_literal_review.v1'
MOLECULAR=re.compile(r'\b(?:gen(?:e|es|etic|etically|otype|otypes|omic|omics)|polygenic|oligogenic|alleles?|haplotypes?|polymorphisms?|mutations?|variants?|SNPs?|rs\d+|RNA|DNA|proteins?|proteom\w*|receptors?|enzymes?|kinases?|transcripts?|transcription\w*|expression|methylation|phosphorylation|signaling|pathways?|subunits?|cytokines?|interleukins?|amyloid|tau|BACE1|APOE|COMT|GRN|BDNF|GABA|NMDA|serotonin|dopamine|glutamate|calbindin|molecular|neurotransmitter\w*)\b',re.I)
IMAGING=re.compile(r'\b(?:MRI|fMRI|sMRI|DTI|EEG|BOLD|cort(?:ex|ical|ico\w*)|subcortical|hippocamp\w*|amygdal\w*|thalam\w*|cerebell\w*|brain|white[ -]matter|gr[ae]y[ -]matter|connectivity|network|volume|volumes|thickness|surface[ -]area|anisotropy|diffusivity|perfusion|hemodynamic|radiomic\w*|activation|oscillations?|amplitude|frequency|electric[ -]field)\b',re.I)
CLINICAL=re.compile(r'\b(?:symptoms?|severity|performances?|impairments?|function(?:ing)?|dysfunction|scores?|cognition|cognitive|memory|attention|recovery|relapse|remission|response|treatment|hospitalization|suicid\w*|psychos\w*|psychotic|psychopatholog\w*|disorders?|disease|depress\w*|anxiety|schizophren\w*|bipolar|autis\w*|dementia|parkinson\w*|alzheimer\w*|behavio\w*|loneliness|quality|mortality|disability|atrophy|decline|improvement|learning|processing|risk|phenotypes?|trauma\w*|health|wellbeing|well-being)\b',re.I)
TASK_OR_DATA=re.compile(r'\b(?:score|scores|task|tasks|performance|status|duration|frequency|questionnaire|processing|learning|attention|integration)\b',re.I)


def candidate_gate(review,overlap):
    if review['claim_id'] in overlap:return 'R50_changed_claim_requires_fresh_separate_review'
    if review['structural_holds']:return 'structural_or_whole_alias_hold'
    if review['existing_name_or_alias_candidates']:return 'existing_full_name_or_case_alias_requires_reuse_review'
    roles=review['declared_roles'];name=name_key(review['name'])
    if len(roles)!=1:return 'undeclared_or_mixed_role'
    if roles[0] in {'gene_target','drug'}:return 'molecular_or_drug_identity_scope'
    if len(name)<8 or len(re.findall(r'[A-Za-z]+',name))<2:return 'short_or_symbolic_mention'
    if MOLECULAR.search(name):return 'molecular_or_genetic_scope'
    rule=IMAGING if roles[0]=='imaging_marker' else CLINICAL if roles[0] in {'outcome','disease'} else TASK_OR_DATA if roles[0] in {'individual_data','cognitive_task'} else None
    if rule is None or not rule.search(name):return 'outside_finite_nonmolecular_surfaces'
    return None


def main():
    require(not (OUTPUT/'CANDIDATE_CLASSIFICATION.json').exists(),'classification exists')
    fp=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json');source=j.read_json(fp['path'])
    require(source['status']=='INSPECTED_NOT_APPLIED' and source['full_source_sha_verified'],'source incomplete')
    for item in source['artifacts'].values():require(j.fingerprint(item['path'])==item,'source projection changed')
    reviews=rows(source['artifacts']['CURRENT_ENDPOINT_REVIEWS.jsonl']['path']);overlap=set(source['pending_changed_claim_overlap'])
    classified=[dict(r,literal_candidate_gate=candidate_gate(r,overlap),no_new_science_or_ontology_type_inferred=True) for r in reviews]
    eligible=[r for r in classified if r['literal_candidate_gate'] is None]
    names=sorted({r['name'] for r in eligible})
    roles=Counter(tuple(r['declared_roles']) for r in eligible);reasons=Counter(r['literal_candidate_gate'] or 'candidate_for_fresh_current_plan' for r in classified)
    write_rows(OUTPUT/'CLASSIFIED_ENDPOINTS.jsonl',classified)
    write_rows(OUTPUT/'CANDIDATE_NAMES.jsonl',[dict(name=name,endpoint_count=sum(r['name']==name for r in eligible),
        declared_roles=sorted({v for r in eligible if r['name']==name for v in r['declared_roles']})) for name in names])
    result=dict(status='CLASSIFIED_NOT_PLANNED_OR_APPLIED',version=VERSION,at=j.utc_now(),source_inspection=fp,code=j.fingerprint(Path(__file__)),
        endpoints=len(classified),candidate_endpoints=len(eligible),candidate_claims=len({r['claim_id'] for r in eligible}),candidate_names=len(names),
        candidate_role_counts={','.join(k):v for k,v in roles.items()},reasons=dict(reasons),
        classified=j.fingerprint(OUTPUT/'CLASSIFIED_ENDPOINTS.jsonl'),candidate_names_file=j.fingerprint(OUTPUT/'CANDIDATE_NAMES.jsonl'),
        graph_modified=False,record_preimages_saved=False,scientific_truth_not_inferred=True,requires_current_source_plan_and_complete_inverse=True)
    j.atomic_json(OUTPUT/'CANDIDATE_CLASSIFICATION.json',result);print({k:result[k] for k in ('candidate_endpoints','candidate_claims','candidate_names','candidate_role_counts','reasons')},flush=True)


if __name__=='__main__':main()
