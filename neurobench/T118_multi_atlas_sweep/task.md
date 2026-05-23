# T118_multi_atlas_sweep: Multi-Atlas Sweep and Leaderboard

## Task Description

Run a single model across the 10-atlas grid (`aal_116`, `basc_122`, `destrieux_148`, `harvard_oxford_cort`, `harvard_oxford_sub`, `msdl_39`, `power_264`, `schaefer_100_7net`, `schaefer_200_7net`, `schaefer_400_7net`) for both HCP age regression and ABIDE classification. Output a unified leaderboard.

## Input Requirement

Required input(s):

- Per-atlas FC matrices (NPZ) - 10 atlases
- Subject list and labels CSV
- Choice of model (default BrainGNN, configurable)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/scripts/run_benchmark.py --models <m> --atlases all`.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T118_multi_atlas/<model>/<setting>/`.
- Report leaderboard CSV with one row per (atlas, fold).

## Expected Output

- Aggregated leaderboard CSV (mean +/- std across folds, per atlas)
- Best-atlas selection per setting
- Atlas x metric heatmap
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- All 10 atlases x 5 folds completed (no missing cells).
- Best atlas matches conventional findings (Schaefer-200/400 usually wins for HCP age).
- Variance across atlases must be reported and discussed.
