# Benchmark Test Case 14: DWI ROI Statistics

## Task Description

Compute ROI statistics for any subset of DTI metrics among:

- FA
- MD
- AD
- RD

Input includes one label ROI/atlas NIfTI in the same space as DWI.
For each selected metric, compute per-label statistics:

- `n_vox` (minimum threshold default 10)
- `mean`
- `median`
- `std`
- `p05`
- `p95`

## Input Requirement

Required inputs:

- ROI/atlas label NIfTI (same space as metrics)
- One or more metric maps from FA/MD/AD/RD

If required input is missing, return:

- `Missing required input`

## Output Requirement

Save CSV outputs to:

- `benchmark_results/T14_dwi_roi/`

Expected file naming:

- `roi_stats_FA.csv`
- `roi_stats_MD.csv`
- `roi_stats_AD.csv`
- `roi_stats_RD.csv`

(Any non-empty subset is acceptable, according to chosen metric subset.)

Output structure rule:

- Write one CSV per selected metric.
- Do not collapse multiple metrics into a single combined CSV.

CSV requirements:

- Sorted by `label`
- Required columns:
  - `label`, `n_vox`, `mean`, `median`, `std`, `p05`, `p95`
