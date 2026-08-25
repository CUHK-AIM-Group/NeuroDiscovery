# T420_optuna_hparam_sweep: Optuna Hyperparameter Sweep (BrainGNN)
## Task Description

Run an Optuna hyperparameter sweep for BrainGNN (lr, hidden dim, pooling ratio, weight decay): 30 trials, pruned, with the search space and best config recorded.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)

- Base config to start from (optional)


If any required input is missing, return:

- Missing required input


## Constraints

- Nested protocol: sweep on train/val only; final eval on untouched test folds.

- Save artefacts under `models/benchmark_results/T420_optuna_hparam_sweep/`.

- Study persisted (sqlite) so it can be resumed.


## Expected Output

Expected output artifact(s):

- `optuna_study.db`

- `best_config.json`

- `importance_plot.png`

- Final 5-fold metrics CSV with the best config


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- Best-config final metrics must not reuse val-tuned information.

- This test case is manually evaluated.
