# T117_swift_train_eval: SwiFT (4D Swin Transformer) Training and Evaluation

## Task Description

Train and evaluate SwiFT (Kim et al., NeurIPS 2023) - a 4D Swin-Transformer for fMRI volumes - on HCP age regression and ABIDE classification.

## Input Requirement

Required input(s):

- Preprocessed 4D BOLD NIfTI per subject
- Brain mask
- Subject list and labels CSV
- Optional pretrained SwiFT checkpoint

If any required input is missing, return:

- Missing required input

## Constraints

- Patch size 4x4x4x4 (defaults from paper); 3 stages of Swin blocks.
- 5-fold deterministic split shared with T101.
- Mixed-precision training required.
- Save artefacts under `models/benchmark_results/T117_swift/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Attention rollout map for one subject (3D visualisation)
- GPU memory + wall-time log
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age MAE competitive with NeuroSTORM (T116).
- Attention rollout map non-degenerate (entropy check).
