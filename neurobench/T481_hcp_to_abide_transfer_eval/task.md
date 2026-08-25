# T481_hcp_to_abide_transfer_eval: HCP->ABIDE Transfer Evaluation
## Task Description

Evaluate HCP-pretrained models on ABIDE (zero-shot and linear-probe): transfer curve vs. amount of ABIDE finetuning data.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Finetuning fractions: 0/10/25/50/100%.

- Same atlas and preprocessing.

- Save artefacts under `models/benchmark_results/T481_hcp_to_abide_transfer_eval/`.


## Expected Output

Expected output artifact(s):

- `transfer_curve.csv`

- `transfer_curve.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
