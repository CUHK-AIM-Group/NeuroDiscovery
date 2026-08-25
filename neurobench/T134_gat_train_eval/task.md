# T134_gat_train_eval: GAT Training and Evaluation
## Task Description

Train and evaluate a Graph Attention Network (GAT) baseline on preprocessed
functional connectivity (FC) graphs for two settings: HCP age regression and
ABIDE diagnosis classification. ROI-as-node graphs, attention coefficients
exported for interpretability.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas
  (e.g. `schaefer_200_7net`, `aal_116`)
- Subject list file (`ready_subjects.txt`)
- Labels CSV
  - HCP age: `data/hcp_age_labels.csv` (continuous age in years)
  - ABIDE dx: `data/abide_dx_labels.csv` (binary: ASD vs control)
- Atlas name and ROI count (must match FC dimension)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py --model gat` for both settings.
- 5-fold split with deterministic `--seed`; report fold 0..4 test metrics.
- 2-layer GAT, 4 attention heads, hidden dim 64 (document any deviation).
- Save artefacts under `models/benchmark_results/T134_gat/<setting>/`.
- Save checkpoints under `models/checkpoints/gat/<atlas>/fold{k}.pt`.

## Expected Output

- Per-fold test metrics CSV (regression: MAE / RMSE / R^2; classification:
  accuracy / AUC / F1)
- Aggregated 5-fold mean +/- std
- One attention-weight visualization (ROI x ROI attention matrix) per setting
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- HCP age MAE within a reasonable baseline range (<= ~6.0 years).
- ABIDE binary AUC >= 0.62 on at least one atlas.
- Manually scored for reproducibility (seed + atlas + fold all logged).
