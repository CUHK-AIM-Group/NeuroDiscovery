# T105_lggnn_train_eval: LGGNN (Local-Global GNN) Training and Evaluation

## Task Description

Train and evaluate LGGNN - a Local-Global graph neural network combining local message passing with a global attention readout - on ROI-level FC.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/lggnn/` net + training script.
- 5-fold deterministic split shared with T101-T104.
- Save artefacts under `models/benchmark_results/T105_lggnn/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Global attention weights per ROI averaged across test set
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age MAE comparable to BNT baseline.
- ABIDE AUC >= 0.60.
- Attention weights non-degenerate (entropy > 0.5 x log N).
