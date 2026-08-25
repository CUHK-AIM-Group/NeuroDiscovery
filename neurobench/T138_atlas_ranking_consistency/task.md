# T138_atlas_ranking_consistency: Cross-Atlas Model Ranking Consistency
## Task Description

Quantify whether model rankings are stable across parcellations: evaluate
three models (braingnn, gcn, roi_mlp_baseline) on three atlases
(`aal_116`, `schaefer_200_7net`, `schaefer_400_7net`) for ABIDE
classification, then compute ranking-consistency statistics across atlases.

## Input Requirement

Required input(s):

- Per-atlas FC matrices (NPZ) for the three atlases (required)
- Subject list and labels CSV (required)

If any required input is missing, return:

- Missing required input

## Constraints

- 5-fold deterministic split shared with T101/T134/T135.
- Same hyperparameters per model across atlases (only input dimension
  changes).
- Save artefacts under `models/benchmark_results/T138_atlas_consistency/`.

## Expected Output

- Model x atlas performance matrix CSV (mean AUC across folds)
- Kendall's W (or Spearman rank correlation per atlas pair) for model
  rankings, with a short interpretation
- Heatmap PNG of the model x atlas matrix
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- All 9 (model, atlas) cells completed.
- Ranking statistics computed and interpreted correctly.
- Manually scored for completeness and correctness of the protocol.
