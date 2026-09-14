# Full-graph Case Study membership re-audit rubric

Rubric version: `2026-08-02.peer17.v1`

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

## Formal IDs and decision conditions

1. `case1_transdiagnostic`: the claim reports a brain-region, brain-network,
   neural-system, or neuroimaging change/association in at least one disease or
   diagnosis. One diagnosis is sufficient; the paper need not itself compare
   multiple diagnoses. Exclude peripheral-only markers, generic methods, and
   non-neural disease facts.
2. `case2_pathway_mediation`: the paper as a whole provides a complete
   genetic/pathway → brain imaging/physiology → longitudinal clinical or
   cognitive outcome mediation/causal chain. All three stages, longitudinal
   outcome, and the chain/mediation relation must be supported. Label only
   claims that form or state part of that verified chain.
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
invalid unless its gate is true:

- `case1_transdiagnostic` → `neural_disease_change_verified`
- `case2_pathway_mediation` → `case2_full_chain_verified`
- `progression_prediction` and `prognosis` → `longitudinal_verified`
- `imaging_genetics` → `genetic_to_neural_verified`
- `differential_diagnosis` → `disease_vs_disease_verified`
- `cognitive_decoding` → `decoding_verified`
- `brain_age` → `brain_age_construct_verified`
- `drug_response_prediction` → `pretreatment_drug_response_prediction_verified`
- `personalised_treatment` → `treatment_selection_verified`
- `neuromodulation_target` → `concrete_stimulation_target_verified`

## Mutation gate

No formal KG file may be changed until all 299,461 graph claims have exactly one
validated final review, paper membership has been recomputed as the union of its
claim memberships, every graph/extracted/edge copy is projected consistently,
and the complete projection passes dry-run validation.

## Immutable result contract

This rubric and the ordered 17-ID registry define one frozen audit epoch. A
claim that reaches `final_complete` under this epoch is a canonical decision,
not a model suggestion to be regenerated later.

- Every final decision is content-addressed by an `audit_key` over the claim's
  evidence-bearing fields, canonical paper identity, rubric SHA-256, ordered
  Case Study registry SHA-256, and audit-contract version.
- The final labels and mandatory gates are sealed by `decision_sha256`.
- If the evidence and contract hashes still match, the decision must be reused
  exactly. A different model, prompt, reasoning effort, concurrency setting, or
  session is not a reason to review it again.
- New claims from an already represented paper are audited once as new claims;
  they do not invalidate finalized claims. Paper membership is recomputed as
  the deterministic union of all current claim decisions.
- A changed claim evidence payload or a changed rubric/registry starts a new,
  explicitly versioned audit epoch. The prior decision remains historical and
  must never be overwritten in place.
- Rebuilding a ledger that already contains finalized decisions is refused by
  default. Deliberate destruction requires the explicit `--discard-reviews`
  flag and therefore constitutes a new audit epoch.
