# T126_abide_download_organize: ABIDE I Fetch and Per-Site Organization
## Task Description

Fetch ABIDE I preprocessed resting-state fMRI (CPAC pipeline, filt_global)
for a given list of acquisition sites, and organize the derivatives into a
per-site directory layout with a unified subject manifest.

## Input Requirement

Required input(s):

- Site list file (e.g. `sites.txt` with ABIDE site IDs, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `nilearn.datasets.fetch_abide_pcp` (or direct HTTP mirrors) with
  `pipeline='cpac'`, `band_pass_filtering=True`, `global_signal_regression=True`.
- Organize as `derivatives/site-<ID>/sub-<label>_ses-1_task-rest_...nii.gz`.
- Phenotypic data merged into `participants.tsv` (subject, site, dx, age, sex).
- Save all generated artifacts to:
  - benchmark_results/T126_abide_download_organize/

## Expected Output

Expected output artifact(s):

- Organized derivatives tree per site
- `participants.tsv` + `download_log.json` (per-file status, resume-safe)
- `manifest_summary.md` (counts per site, failed downloads listed)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
