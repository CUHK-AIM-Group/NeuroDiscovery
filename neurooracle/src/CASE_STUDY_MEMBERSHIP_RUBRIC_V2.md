# KG v3 expansion Case Study membership rubric (current)

Rubric version: `2026-08-10.peer17.v2`

Every formal KG claim is reviewed independently against all 17 peer Case Study
IDs. Membership is non-exclusive. `general` is implicit for every retained KG
claim and is never written as a Case Study ID. `hindcasting` is a validation
protocol and is never written as a Case Study ID. An empty membership list is a
valid decision: it means the claim remains useful only to the general corpus.

Use only the supplied claim evidence, paper title, abstract when available, and
other claims extracted from the same paper. Do not infer an unreported result.
A paper can help several Case Studies. A claim receives a label only when the
claim itself is direct evidence for that scope; broad topic similarity is not
enough.

Case Study membership describes evidence directly useful to the research task;
it does not require one paper to complete the entire task. This is the same
principle used by Case 1: a single-diagnosis neural result may contribute to a
transdiagnostic analysis. For Case 2, one directly supported chain component may
contribute to a pathway-mediation analysis, while the KG connects components
across papers.

## Formal IDs and decision conditions

1. `case1_transdiagnostic`: the claim reports a brain-region, brain-network,
   neural-system, or neuroimaging change/association in at least one disease or
   diagnosis. One diagnosis is sufficient; the paper need not itself compare
   multiple diagnoses. Exclude peripheral-only markers, generic methods, and
   non-neural disease facts.
2. `case2_pathway_mediation`: the claim directly supports either a
   genetic/gene/molecular-pathway → brain imaging or neural-physiology relation,
   or a baseline brain imaging/neural-physiology → later clinical or cognitive
   outcome relation. One directly supported link is sufficient; the paper need
   not contain both links or perform a mediation analysis. This membership is
   derived deterministically whenever the claim qualifies for
   `imaging_genetics`, `progression_prediction`, or `prognosis`. Exclude
   genetic-only disease associations without a neural phenotype, cross-sectional
   neural associations without a genetic/pathway input or later outcome, and
   merely parallel measurements with no reported relation.
3. `biomarker_discovery`: an imaging/neural marker distinguishes, detects,
   associates with, or predicts a disease/diagnostic state in a way directly
   useful for discovering a disease biomarker. A generic non-neural biomarker
   does not qualify.
4. `disease_subtyping`: imaging or multivariate patient features identify,
   characterize, or predict subgroups/subtypes within a disease. Ordinary
   case-control classification is not subtyping.
5. `progression_prediction`: baseline imaging/neural features predict future
   conversion or disease progression. A longitudinal/future outcome is
   mandatory; cross-sectional severity associations do not qualify.
6. `imaging_genetics`: a genotype, gene, variant, molecular pathway, or related
   molecular target is directly linked to an imaging or neural phenotype.
7. `differential_diagnosis`: imaging/neural features distinguish disease A from
   disease B (or two clinically competing diagnoses). Disease-versus-healthy
   control alone is not differential diagnosis.
8. `drug_response_prediction`: pretreatment patient/imaging/genetic profile plus
   a named drug or pharmacological treatment predicts later response, remission,
   or response magnitude. Treatment effects without prediction do not qualify.
9. `personalised_treatment`: patient-specific disease/imaging/genetic features
   directly inform selection, recommendation, or comparative choice among drugs
   or treatments. A single-treatment response association alone is insufficient.
10. `drug_repurposing`: an existing drug is supported for a novel disease
    indication through a relevant mechanism or evidence path. This ID remains
    valid even though no current top-up campaign is planned.
11. `adverse_event_prediction`: a drug or intervention, optionally with patient
    features, predicts or causes a defined adverse event, toxicity, or treatment
    complication.
12. `neuromodulation_target`: disease/symptom/task evidence identifies or
    validates a concrete stimulation site, circuit, or neuromodulation target.
    Mentioning a brain mechanism or therapy without a target is insufficient.
13. `functional_localization`: a stimulus, cognitive operation, behaviour, or
    experimental task maps to a brain region, network, activation, or other
    localized neural response.
14. `cognitive_decoding`: brain activity/imaging patterns are used to predict or
    decode a stimulus, cognitive state, or mental-state label. Mere association
    or localization is not decoding.
15. `connectome_behavior`: functional/structural connectivity or graph/network
    organization is directly related to or predicts a behavioural, cognitive,
    symptom, or trait measure.
16. `brain_age`: imaging/neural features produce or validate a biological/
    chronological brain-age estimate, or a brain-age gap is directly related to
    an outcome. Ordinary age-related brain change without a brain-age construct
    does not qualify.
17. `prognosis`: an acute/baseline disease and imaging/neural state predicts a
    later clinical, functional, cognitive, survival, recovery, or recurrence
    outcome. A future/longitudinal outcome is mandatory.

## Mandatory gates

The reviewer must explicitly return these booleans. A corresponding label is
invalid unless its gate is true. Case 2 has no independent model gate: its
membership is derived from the validated component labels below.

- `case1_transdiagnostic` → `neural_disease_change_verified`
- `progression_prediction` and `prognosis` → `longitudinal_verified`
- `imaging_genetics` → `genetic_to_neural_verified`
- `differential_diagnosis` → `disease_vs_disease_verified`
- `cognitive_decoding` → `decoding_verified`
- `brain_age` → `brain_age_construct_verified`
- `drug_response_prediction` → `pretreatment_drug_response_prediction_verified`
- `personalised_treatment` → `treatment_selection_verified`
- `neuromodulation_target` → `concrete_stimulation_target_verified`

## Mutation gate

During KG v3 collection, every newly extracted claim must carry one validated
final review under this epoch before formal KG mutation. Existing claims sealed
under the prior rubric remain historical and are not silently relabelled. The
whole graph will be projected to this epoch only through the explicitly planned
post-collection full re-audit.

## Immutable result contract

This rubric and the ordered 17-ID registry define a new audit epoch. A claim that
reaches `final_complete` under this epoch is a canonical decision, not a model
suggestion to be regenerated later.

- Every final decision is content-addressed by an `audit_key` over the claim's
  evidence-bearing fields, canonical paper identity, rubric SHA-256, ordered
  Case Study registry SHA-256, and audit-contract version.
- The final labels and mandatory gates are sealed by `decision_sha256`.
- If the evidence and contract hashes still match, the decision must be reused
  exactly. A different model, prompt, reasoning effort, concurrency setting, or
  session is not a reason to review it again.
- New claims from an already represented paper are audited once as new claims;
  they do not invalidate finalized claims. Paper membership is recomputed as the
  deterministic union of all current claim decisions.
- A changed claim evidence payload or a changed rubric/registry starts a new,
  explicitly versioned audit epoch. The prior decision remains historical and
  must never be overwritten in place.
- The prior `2026-08-02.peer17.v1` epoch remains valid for claims already sealed
  under it until the planned full-KG re-audit projects all retained evidence to
  this epoch.
