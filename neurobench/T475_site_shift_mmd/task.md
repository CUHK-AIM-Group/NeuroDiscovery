# T475_site_shift_mmd: Site Shift Quantification (MMD)
## Task Description

Quantify distribution shift between ABIDE sites in FC space via MMD with a Gaussian kernel; relate shift magnitude to LOSO performance.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Bandwidth via median heuristic.

- Correlation of MMD vs. LOSO drop reported.

- Save artefacts under `models/benchmark_results/T475_site_shift_mmd/`.


## Expected Output

Expected output artifact(s):

- `site_mmd_matrix.csv`

- `mmd_vs_loso.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
