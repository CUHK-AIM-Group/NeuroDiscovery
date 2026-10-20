# T137_leave_one_site_out: Leave-One-Site-Out Generalization
## Task Description

Run a leave-one-site-out (LOSO) evaluation on ABIDE: train a model on all
sites but one, test on the held-out site, rotate over all sites, and produce
a per-site generalization matrix.

## Input Requirement

Required input(s):

- Per-subject FC matrices (NPZ) with site labels (required)
- Labels CSV (`data/abide_dx_labels.csv`) (required)
- Choice of model (default `gcn`, configurable)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/train_unified.py` with a LOSO split mode (or an equivalent
  documented script); deterministic seed shared with T101/T134/T135.
- Sites with fewer than 20 subjects after filtering are excluded and logged.
- Save artefacts under `models/benchmark_results/T137_loso/<model>/`.

## Expected Output

- Per-site test metrics CSV (accuracy / AUC / F1 per held-out site)
- Site x site generalization summary (mean/std across held-out sites)
- Comparison table vs. random-split performance from the same model
- `result_YYYYMMDD_HHMMSS.json` metadata file

## Evaluation

- All eligible sites covered (no silently skipped sites).
- LOSO performance must be reported honestly even when below random split.
- Discussion of site heterogeneity (sample size, scanner) included.
