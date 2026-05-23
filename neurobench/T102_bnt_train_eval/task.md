# T102_bnt_train_eval: BrainNetworkTransformer (BNT) Training and Evaluation

## Task Description

Train and evaluate BrainNetworkTransformer (BNT) on ROI-level FC matrices for HCP age regression and ABIDE classification. BNT treats each ROI row of the FC matrix as a token and uses multi-head self-attention with orthonormal cluster readout.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ) for a chosen atlas
- Subject list and labels CSV (HCP age + ABIDE dx)
- Atlas dimensionality matching the model input

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py --model bnt`.
- 5-fold deterministic split, identical to T101 splits for paired comparison.
- z-score label standardisation for regression; class-balanced sampling for ABIDE.
- Save artefacts under `models/benchmark_results/T102_bnt/<setting>/`.

## Expected Output

- Per-fold test metrics CSV (MAE / R^2 for regression; ACC / AUC / F1 for classification)
- Attention head visualisation for one held-out subject
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age test MAE <= ~5.0 years on Schaefer-200.
- ABIDE AUC >= 0.65 on at least one atlas.
- Splits must match T101 (verifiable from `splits.json` hash).
