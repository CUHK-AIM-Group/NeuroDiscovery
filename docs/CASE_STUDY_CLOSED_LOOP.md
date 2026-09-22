# Unified Case-Study Closed Loop

## Scope

The completed experiment registry contains nine lines:

1. `case1_transdiagnostic`
2. `case2_pathway_mediation`
3. `differential_diagnosis`
4. `disease_subtyping`
5. `connectome_behavior`
6. `brain_age`
7. `progression_prediction`
8. `prognosis`
9. `imaging_genetics`

`biomarker_discovery` is the executable task implementation of
`case1_transdiagnostic`; it is not counted as a separate experiment line.

## Shared Contract

- Task semantics live in `core/scripts/case_study_feedback_adapters.py`.
- Batch selection and feedback reuse live in
  `core/scripts/case_study_closed_loop_engine.py`.
- Each seed and trial owns an isolated, append-only experimental overlay.
- The formal literature KG is never mutated by an experiment.
- Feedback has four states: `supported`, `contradicted`, `inconclusive`, and
  `execution_failed`.
- A failed or non-significant experiment is not automatically scientific
  contradiction.
- The next batch queries supported and contradicted semantic factors and factor
  pairs from the overlay before ranking candidates.

Case 2 writes two assertions per candidate and requires the complete
gene/pathway to imaging marker to outcome chain. Other tasks use the relation
templates registered for their candidate unit.

## Supplemental Seven Run Contract

- The canonical input is release `full_v2_20260825_092909z`; the three artifact
  hashes are pinned by `core/scripts/canonical_kg_release.py`.
- Development uses five fixed seeds and five paired outer folds per seed.
- Final evaluation uses ten fixed seeds and five paired outer folds per seed.
- Differential diagnosis, connectome behavior, brain age, progression,
  prognosis, and imaging genetics require frozen external validation.
- Disease subtyping is the only registered external-validation exception; its
  completion gates are internal cluster stability and held-out continuous
  symptom separation.
- Brain-age discovery uses HCP-YA only. HCP-Aging is not mixed into discovery.
- Formal execution requires a full-hash code lock produced and verified by
  `core/scripts/run_supplemental_case_studies_formal.py`.
- API-backed framework jobs use DeepSeek V4 Pro through the finite local router;
  DeepSeek's official API is never an automatic fallback.

## Frozen Score Components

NeuroDiscovery can consume three optional outcome-blind columns:

- `score_kge`
- `score_novelty`
- `score_critic`

They may be embedded in the frozen public registry or supplied as a separate
CSV plus `case-study-score-components.v1` manifest. The manifest pins candidate
IDs, table hashes, model or prompt provenance, and confirms that scores were
frozen before any experimental outcome was read. KGE scores additionally
require a checkpoint hash and a `formal_release` KG snapshot hash. A partial KG
re-audit must not be used to train or silently activate KGE scores.

## Hindcasting

Future evidence is filtered by `claim_case_study_ids`, not by the broader paper
scope. The primary hit requires an exact future endpoint relation or a complete
future-supported path. For Case 2, both mediation steps are mandatory. A hit on
only one path edge is exported as a secondary diagnostic and is not counted as
primary future support.

