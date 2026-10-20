# T109_bolt_train_eval: BolT (BOLD Transformer) Training and Evaluation

## Task Description

Train and evaluate BolT (Bedel et al.) - a transformer with fused window-level local attention and global cross-window attention on ROI BOLD time-series.

## Input Requirement

Required input(s):

- ROI BOLD time-series (NPZ)
- Atlas spec; subject list; labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Window size and number of fused-window layers per BolT defaults.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T109_bolt/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Cross-window attention heatmap for one subject
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age MAE comparable to BNT (T102).
- ABIDE AUC >= 0.65 on at least one atlas.
