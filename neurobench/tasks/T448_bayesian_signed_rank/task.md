# T448_bayesian_signed_rank: Bayesian Signed-Rank Comparison
## Task Description

Compare model pairs with the Bayesian signed-rank test; report posterior probabilities of practical equivalence (ROPE).

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- ROPE justified.

- Plot posterior distributions per pair.

- Save artefacts under `models/benchmark_results/T448_bayesian_signed_rank/`.


## Expected Output

Expected output artifact(s):

- `bayesian_pairwise.csv`

- `posterior_plots.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
