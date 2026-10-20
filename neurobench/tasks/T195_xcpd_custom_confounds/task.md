# T195_xcpd_custom_confounds: XCP-D with Custom Confound File
## Task Description

Run XCP-D with a user-supplied custom confounds TSV merged with the standard motion parameters.

## Input Requirement

Required input(s):

- fMRIPrep derivatives directory for the subject(s) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Custom confound columns must be documented (name, origin, units).

- Verify the final regression design includes the custom columns.

- Save all generated artifacts to:
  - benchmark_results/T195_xcpd_custom_confounds/


## Expected Output

Expected output artifact(s):

- Denoised BOLD + design matrix TSV

- `confound_merge_report.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
