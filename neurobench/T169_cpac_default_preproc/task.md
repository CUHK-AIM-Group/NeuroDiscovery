# T169_cpac_default_preproc: C-PAC Default Preprocessing Pipeline
## Task Description

Run C-PAC's default preprocessing pipeline (anatomical + functional, nuisance regression with the default strategy) for one subject.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- FreeSurfer license file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use the C-PAC container with the default pipeline configuration; keep the generated pipeline YAML.

- Document the C-PAC version.

- Save all generated artifacts to:
  - benchmark_results/T169_cpac_default_preproc/


## Expected Output

Expected output artifact(s):

- C-PAC output directory (anat + func derivatives)

- `pipeline_config_used.yaml`

- `cpac_run_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
