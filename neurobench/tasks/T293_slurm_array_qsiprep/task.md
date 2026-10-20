# T293_slurm_array_qsiprep: SLURM Array: QSIPrep Batch
## Task Description

SLURM array job running QSIPrep over a subject list, one subject per array task, with per-subject logs and failure collection.

## Input Requirement


- No interactive input.


## Constraints

- Array size derived from subject list file.

- Dry-run mode prints commands without submitting.

- Save all generated artifacts to:
  - benchmark_results/T293_slurm_array_qsiprep/


## Expected Output

Expected output artifact(s):

- `run_qsiprep_array.slurm`

- `dry_run_output.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
