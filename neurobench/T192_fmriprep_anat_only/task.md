# T192_fmriprep_anat_only: fMRIPrep Anatomical-Only Stage
## Task Description

Run fMRIPrep with `--anat-only`: surface reconstruction and anatomical normalization only, stopping before functional preprocessing.

## Input Requirement

Required input(s):

- fMRIPrep derivatives directory for the subject(s) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `--anat-only`; confirm no func outputs are produced.

- This is a stage split of the full fMRIPrep pipeline.

- Save all generated artifacts to:
  - benchmark_results/T192_fmriprep_anat_only/


## Expected Output

Expected output artifact(s):

- `anat/` derivatives tree

- Anatomical QC HTML


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
