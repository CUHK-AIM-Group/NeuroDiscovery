# T114_roi_mlp_baseline: ROI-FC MLP Baseline

## Task Description

Train a simple MLP (2-3 hidden layers, dropout, BatchNorm) on vectorised FC upper-triangle features. Strong "deep but graph-blind" baseline.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Subject list and labels CSV

If any required input is missing, return:

- Missing required input

## Constraints

- Vectorise FC upper triangle; Fisher-z transform; standardise per fold.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T114_roi_mlp/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Sits between ridge (T107) and GNN models (T101/T104/T105) - confirms graph inductive bias actually helps.
