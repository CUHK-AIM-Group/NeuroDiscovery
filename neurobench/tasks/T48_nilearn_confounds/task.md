# Benchmark Test Case 48: Nilearn Confound Processing

## Task Description

Load local fMRIPrep confounds TSV and preprocessed BOLD, apply standard denoising regressors (motion + WM/CSF, etc.), and output denoised confounds and sample_mask.

## Input Requirement

Required input(s):

- Preprocessed BOLD image (required)
- fMRIPrep confounds TSV (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T71_nilearn_confounds/

## Expected Output

Expected output artifact(s):

- Denoised confounds table
- sample_mask indices

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
