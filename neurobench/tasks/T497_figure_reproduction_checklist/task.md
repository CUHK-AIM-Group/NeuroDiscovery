# T497_figure_reproduction_checklist: Figure Reproduction Checklist
## Task Description

Write the reproduction checklist: which script + config + seed regenerates every results figure/table, and verify one end-to-end re-run.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- One row per artifact: script, config, seed, output hash.

- One artifact fully re-run as proof.

- Save artefacts under `models/benchmark_results/T497_figure_reproduction_checklist/`.


## Expected Output

Expected output artifact(s):

- `REPRODUCE.md`

- `rerun_verification.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
