# T305_cron_pipeline_monitor: Cron Pipeline Completion Monitor
## Task Description

Cron-driven monitor that checks a SLURM job list and a derivatives directory, then emails/writes a digest of completed/failed subjects.

## Input Requirement


- No interactive input.


## Constraints

- Digest format: counts + failed list.

- Idempotent; no duplicate alerts for the same state.

- Save all generated artifacts to:
  - benchmark_results/T305_cron_pipeline_monitor/


## Expected Output

Expected output artifact(s):

- `pipeline_monitor.sh`

- `sample_digest.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
