# T184_fsl_vbm_pipeline: FSL-VBM Pipeline
## Task Description

Run the FSL-VBM protocol: brain extraction, tissue segmentation, study-specific GM template, nonlinear normalization, modulation, smoothing, and permutation stats with `randomise`.

## Input Requirement

Required input(s):

- T1w NIfTIs for the group (>= 10, required)

- Design matrix + contrasts for `randomise` (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Follow the standard fslvbm_1/2/3 steps; document FSL version.

- Smoothing 3 mm (sigma-equivalent); 5000 permutations with TFCE.

- Save all generated artifacts to:
  - benchmark_results/T184_fsl_vbm_pipeline/


## Expected Output

Expected output artifact(s):

- `stats/` randomise outputs (TFCE corrected)

- `gm_template.nii.gz`

- `vbm_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
