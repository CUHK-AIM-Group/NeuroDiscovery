# T417_elasticnet_baseline: ElasticNet Baseline
## Task Description

Train and evaluate the ElasticNet Baseline on vectorized FC features for the standard settings. ElasticNet with inner-CV alpha/l1_ratio selection.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `models/train_unified.py --model elasticnet` for both settings.

- 5-fold split with deterministic `--seed`; report fold 0..4 test metrics.

- Feature scaling documented; inner CV protocol kept nested (no test leakage).

- Save artefacts under `models/benchmark_results/T417_elasticnet_baseline/<setting>/`.

- Save checkpoints under `models/checkpoints/elasticnet/<atlas>/fold{k}.pt`.


## Expected Output

Expected output artifact(s):

- Per-fold test metrics CSV (regression: MAE / RMSE / R^2; classification: accuracy / AUC / F1)

- Aggregated 5-fold mean +/- std

- `result_YYYYMMDD_HHMMSS.json` metadata file


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- Results within a reasonable baseline range for the model class (document the reference used).

- Manually scored for reproducibility (seed + atlas + fold all logged).
