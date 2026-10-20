# T470_saliency_method_comparison: Saliency Method Comparison
## Task Description

For one GNN, compare saliency methods (gradients, IG, attention, occlusion): agreement + a sanity check (model parameter randomization).

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Cascading randomization sanity check included.

- Methods ranked by sanity-check pass.

- Save artefacts under `models/benchmark_results/T470_saliency_method_comparison/`.


## Expected Output

Expected output artifact(s):

- `saliency_comparison.csv`

- `sanity_check.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
