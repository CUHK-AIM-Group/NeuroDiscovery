# T453_atlas_model_anova: Atlas x Model Interaction ANOVA
## Task Description

Two-way ANOVA (model x atlas) on 5-fold metrics: quantify main effects and interaction; simple-effects follow-up where interaction is significant.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Assumptions checked (normality of residuals, sphericity note).

- Effect sizes (partial eta^2) reported.

- Save artefacts under `models/benchmark_results/T453_atlas_model_anova/`.


## Expected Output

Expected output artifact(s):

- `anova_table.csv`

- `interaction_plot.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
