# T194_hcp_icafix_only: HCP ICA+FIX Stage Only
## Task Description

Run only the ICA+FIX denoising stage of the HCP functional pipeline on a preprocessed rfMRI run, using a trained FIX classifier.

## Input Requirement

Required input(s):

- HCP minimal-preprocessing outputs for the subject (required)

- FIX training file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Start from PostFreeSurfer/fMRIVolume outputs.

- Document the FIX training file used and threshold.

- Save all generated artifacts to:
  - benchmark_results/T194_hcp_icafix_only/


## Expected Output

Expected output artifact(s):

- Cleaned dtseries (FIX-denoised)

- `fix_classification_report.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
