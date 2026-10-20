# T467_roi_importance_topk_overlap: ROI Importance Top-k Overlap
## Task Description

Compare per-model ROI importance rankings (from attention/SHAP/IG tasks): top-20 overlap coefficients between model pairs.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Importance sources documented per model.

- Overlap = Szymkiewicz-Simpson coefficient.

- Save artefacts under `models/benchmark_results/T467_roi_importance_topk_overlap/`.


## Expected Output

Expected output artifact(s):

- `topk_overlap_matrix.csv`

- `overlap_heatmap.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
