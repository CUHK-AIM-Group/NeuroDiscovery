# T385_reproducibility_checklist: Reproducibility Checklist Audit
## Task Description

Audit included papers against a reproducibility checklist (data availability, code, seeds, environment, hyperparameters), producing per-paper scores.

## Input Requirement

Required input(s):

- Paper list or corpus (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Checklist items fixed in a YAML schema.

- Score = items met / total; evidence quoted per item.

- Save all generated artifacts to:
  - benchmark_results/T385_reproducibility_checklist/


## Expected Output

Expected output artifact(s):

- `repro_scores.csv`

- `evidence_notes.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
