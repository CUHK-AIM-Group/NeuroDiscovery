# T107_ridge_baseline: Ridge / Logistic Regression Baseline on FC

## Task Description

Train classical Ridge regression (HCP age) and L2-penalised logistic regression (ABIDE) on the vectorised upper triangle of FC matrices as a strong, simple baseline.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Subject list and labels CSV

If any required input is missing, return:

- Missing required input

## Constraints

- Vectorise upper triangle (excluding diagonal); apply Fisher-z transform.
- 5-fold deterministic split shared with T101-T106.
- Tune `alpha` via nested CV on the train fold only.
- Use `models/ridge/train.py`.
- Save artefacts under `models/benchmark_results/T107_ridge/<setting>/`.

## Expected Output

- Per-fold test metrics CSV (regression + classification)
- Top-K edge coefficients (interpretable feature ranking)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- This is the floor baseline - all deep models in T101-T106 should beat it on at least one of MAE / AUC.
- Coefficient ranking must be stable across folds (top-100 edge overlap >= 50%).
