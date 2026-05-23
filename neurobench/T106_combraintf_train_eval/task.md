# T106_combraintf_train_eval: ComBrainTF Training and Evaluation

## Task Description

Train and evaluate ComBrainTF - a community-aware brain transformer that injects atlas community structure into multi-head attention - on FC matrices. Tests both single-task and multi-task heads (joint age + sex).

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Atlas community labels (e.g. Yeo-7 / Yeo-17 network assignment per ROI)
- Subject list and labels CSV (HCP age + sex; ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/combraintf/` net + training script.
- Atlas-community lookup must match atlas in use.
- 5-fold deterministic split shared with T101-T105.
- Save artefacts under `models/benchmark_results/T106_combraintf/<setting>/`.

## Expected Output

- Per-fold test metrics for each head
- Multi-task vs single-task head comparison
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Multi-task head non-degenerate (both tasks > random baseline).
- Single-task HCP age MAE within ~10% of BNT.
- Community injection verifiable in attention pattern.
