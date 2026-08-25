# T288_snakemake_slurm_profile: Snakemake SLURM Profile
## Task Description

Create a Snakemake SLURM profile (resources per rule, job naming, log paths) and demonstrate submission of a small workflow.

## Input Requirement


- No interactive input.


## Constraints

- Profile as `slurm/config.yaml`.

- Per-rule mem/time in the profile, not inline.

- Save all generated artifacts to:
  - benchmark_results/T288_snakemake_slurm_profile/


## Expected Output

Expected output artifact(s):

- Profile directory

- `submission_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
