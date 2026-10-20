# T110_fbnetgen_train_eval: FBNetGen (Learnable FC) Training and Evaluation

## Task Description

Train FBNetGen (Kan et al.) - an end-to-end model that learns subject-specific FC from BOLD time-series via a 1D-CNN encoder and a graph generator, jointly with the downstream classifier.

## Input Requirement

Required input(s):

- ROI BOLD time-series (NPZ)
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Joint training of FC generator + classifier; report both task loss and graph-sparsity regulariser.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T110_fbnetgen/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Generated FC adjacency for one held-out subject (visualised vs Pearson-FC)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Beats Pearson-FC + ridge (T107) on at least one metric.
- Generated FC must remain sparse (mean degree < 0.5 x N).
