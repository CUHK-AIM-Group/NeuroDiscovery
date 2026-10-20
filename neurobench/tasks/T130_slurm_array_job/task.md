# T130_slurm_array_job: SLURM Array Job for Batch fMRIPrep
## Task Description

Write a SLURM array job that runs fMRIPrep over a list of BIDS subjects, one
array task per subject, with per-subject logs and a final aggregation step.

## Input Requirement

Required input(s):

- BIDS dataset path + subject list file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Array size derived from the subject list (`#SBATCH --array=...`).
- Per-subject resource requests (cpus/mem/time) justified in comments.
- Stdout/stderr captured per subject (`logs/sub-<ID>_%j.out`).
- A `--dry-run` mode that prints the commands without submitting must be
  supported for evaluation on machines without SLURM.
- Save all generated artifacts to:
  - benchmark_results/T130_slurm_array_job/

## Expected Output

Expected output artifact(s):

- `run_fmriprep_array.slurm`
- `aggregate_qc.slurm` (or equivalent post-step) that collects per-subject
  success/failure into `qc_summary.csv`
- `dry_run_output.txt` from the dry-run mode

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
