# T170_cpac_custom_strategy: C-PAC Custom Nuisance Strategy
## Task Description

Run C-PAC with a custom pipeline configuration: aCompCor nuisance regression, no GSR, band-pass 0.01-0.1 Hz, scrubbing at FD 0.5 mm.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- FreeSurfer license file (required)

- Base pipeline YAML to modify (optional)


If any required input is missing, return:

- Missing required input


## Constraints

- Start from a default C-PAC pipeline config and edit ONLY the nuisance/filter/scrubbing sections; show a diff of the YAML.

- Scrubbing must be reported as percent volumes removed.

- Save all generated artifacts to:
  - benchmark_results/T170_cpac_custom_strategy/


## Expected Output

Expected output artifact(s):

- C-PAC output directory

- `pipeline_config_diff.txt`

- `scrubbing_summary.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
