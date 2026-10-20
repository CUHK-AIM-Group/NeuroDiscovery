# T468_shap_vs_attention_agreement: SHAP vs. Attention Agreement
## Task Description

Quantify agreement between SHAP-based and attention-based ROI importance for the same predictions: rank correlation per subject, aggregated.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Same subjects for both methods.

- Per-subject Spearman + distribution.

- Save artefacts under `models/benchmark_results/T468_shap_vs_attention_agreement/`.


## Expected Output

Expected output artifact(s):

- `method_agreement.csv`

- `agreement_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
