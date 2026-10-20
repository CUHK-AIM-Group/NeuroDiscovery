# T473_noise_robustness_comparison: Noise Robustness Comparison
## Task Description

Evaluate all models under FC noise injection (sigma grid); plot degradation curves and rank models by robustness.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Same noise realizations across models (seeded).

- AUC-of-degradation-curve as the robustness score.

- Save artefacts under `models/benchmark_results/T473_noise_robustness_comparison/`.


## Expected Output

Expected output artifact(s):

- `noise_robustness.csv`

- `degradation_curves.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
