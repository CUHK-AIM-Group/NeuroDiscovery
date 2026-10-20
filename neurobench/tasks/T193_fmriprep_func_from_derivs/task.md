# T193_fmriprep_func_from_derivs: fMRIPrep Functional Stage from Existing Anat
## Task Description

Run fMRIPrep functional preprocessing reusing existing anatomical derivatives via `--anat-derivatives`.

## Input Requirement

Required input(s):

- fMRIPrep derivatives directory for the subject(s) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `--anat-derivatives` pointing at a completed anat run.

- Verify anatomical outputs are NOT recomputed (log evidence).

- Save all generated artifacts to:
  - benchmark_results/T193_fmriprep_func_from_derivs/


## Expected Output

Expected output artifact(s):

- `func/` derivatives tree

- Functional QC HTML


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
