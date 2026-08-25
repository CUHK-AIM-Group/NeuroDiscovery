# T136_combat_harmonization_protocol: ComBat Harmonization Across Sites
## Task Description

Apply ComBat batch-effect harmonization to multi-site FC feature matrices
(ABIDE sites), then quantify how site effects are reduced and how much
biological signal (diagnosis, age) is preserved.

## Input Requirement

Required input(s):

- Per-subject FC feature matrices (NPZ) with site labels (required)
- Covariates CSV (site, diagnosis, age, sex) (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use neuroComBat (Python port) or `neuroHarmonize`; protect diagnosis and
  age as biological covariates.
- Harmonization fitted on the training split only, then applied to test
  (no leakage).
- Save artefacts under `models/benchmark_results/T136_combat/`.

## Expected Output

- Pre/post-harmonization site-effect quantification (e.g. per-feature
  variance explained by site, or kBET-style metric) as CSV
- A downstream ABIDE dx model (logistic regression) evaluated pre vs. post
  harmonization: accuracy / AUC / F1 comparison table
- `harmonization_report.md` with method, parameters, and interpretation
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- Site-effect metric must decrease post-harmonization.
- Downstream diagnosis performance must not collapse (AUC drop <= 0.05).
- Leakage-free protocol must be evident from the logs.
