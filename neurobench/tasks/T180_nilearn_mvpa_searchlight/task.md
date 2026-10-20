# T180_nilearn_mvpa_searchlight: Nilearn MVPA Searchlight Pipeline
## Task Description

Run a whole-brain searchlight decoding analysis with nilearn: decode the given condition labels from task fMRI with an SVM, cross-validated per subject.

## Input Requirement

Required input(s):

- Task fMRI 4D NIfTI (required)

- Condition labels / events file (required)

- Brain mask (required)


If any required input is missing, return:

- Missing required input


## Constraints

- `nilearn.decoding.SearchLight` with radius 6 mm, SVC linear.

- Report chance level and permutation-based significance map.

- Save all generated artifacts to:
  - benchmark_results/T180_nilearn_mvpa_searchlight/


## Expected Output

Expected output artifact(s):

- `searchlight_accuracy.nii.gz`

- `permutation_threshold.json`

- `searchlight_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
