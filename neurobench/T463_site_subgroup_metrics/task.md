# T463_site_subgroup_metrics: Site-Subgroup Metrics (ABIDE)
## Task Description

Report per-site classification metrics for every model on ABIDE; identify sites where all models fail (systematic site effects).

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Small-N sites flagged.

- Per-site N shown next to metrics.

- Save artefacts under `models/benchmark_results/T463_site_subgroup_metrics/`.


## Expected Output

Expected output artifact(s):

- `site_metrics.csv`

- `site_heatmap.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
