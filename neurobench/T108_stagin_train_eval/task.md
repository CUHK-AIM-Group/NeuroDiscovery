# T108_stagin_train_eval: STAGIN (Spatio-Temporal Attention GIN) Training and Evaluation

## Task Description

Train and evaluate STAGIN (Kim et al., NeurIPS 2021) on ROI BOLD time-series. STAGIN forms dynamic FC graphs over sliding windows, applies GIN per snapshot, and aggregates with READOUT-SERO / GARO temporal attention.

## Input Requirement

Required input(s):

- ROI BOLD time-series (NPZ: `[n_subjects, n_rois, n_timepoints]`)
- Atlas spec; subject list; labels CSV (HCP age + ABIDE dx)
- Sliding-window length and stride

If any required input is missing, return:

- Missing required input

## Constraints

- Window length 50 TRs, stride 3 TRs (defaults from paper) - configurable.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T108_stagin/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Temporal attention curves for one held-out subject
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Outperforms ridge baseline (T107) on at least one metric.
- Temporal attention must not collapse to uniform (entropy < 0.95 x log T).
