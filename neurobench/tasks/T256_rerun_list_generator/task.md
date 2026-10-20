# T256_rerun_list_generator: Re-run List Generator
## Task Description

From QC and completeness reports, generate the definitive list of subject/session/modality units that must be re-run, formatted for the target pipeline (fMRIPrep participant labels, SLURM array indices).

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Output formats: participant-label list + SLURM array spec.

- Deduplicated and sorted.

- Save all generated artifacts to:
  - benchmark_results/T256_rerun_list_generator/


## Expected Output

Expected output artifact(s):

- `rerun_participants.txt`

- `rerun_array_spec.txt`

- `rerun_rationale.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
