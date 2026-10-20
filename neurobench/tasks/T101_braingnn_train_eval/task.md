# T101_braingnn_train_eval: BrainGNN Training and Evaluation

## Task Description

Train and evaluate BrainGNN on preprocessed functional connectivity (FC) graphs for two settings: HCP age regression and ABIDE diagnosis classification. BrainGNN uses ROI-as-node graphs with R-pool / S-pool layers and TopK pooling.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)
- Subject list file (`ready_subjects.txt`)
- Labels CSV
  - HCP age: `data/hcp_age_labels.csv` (continuous age in years)
  - ABIDE dx: `data/abide_dx_labels.csv` (binary: ASD vs control)
- Atlas name and ROI count (must match FC dimension)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py --model braingnn` for both settings.
- 5-fold split with deterministic `--seed`; report fold 0..4 test metrics.
- Loss includes `consist_loss + topk_loss + unit_loss` per the BrainGNN paper.
- Save artefacts under `models/benchmark_results/T101_braingnn/<setting>/`.
- Save checkpoints under `models/checkpoints/braingnn/<atlas>/fold{k}.pt`.

## Expected Output

- Per-fold test metrics CSV (regression: MAE / RMSE / R^2; classification: accuracy / AUC / F1)
- Aggregated 5-fold mean +/- std
- One representative ROI-importance plot (TopK pooled nodes) per setting
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- Test MAE within published BrainGNN range on HCP age (<= ~5.5 years).
- ABIDE binary AUC >= 0.65 on at least one atlas.
- Manually scored for reproducibility (seed + atlas + fold all logged).
