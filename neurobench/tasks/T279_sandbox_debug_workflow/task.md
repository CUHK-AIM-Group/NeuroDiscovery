# T279_sandbox_debug_workflow: Apptainer Sandbox Debug Workflow
## Task Description

Debug a failing containerized pipeline by rebuilding the SIF as a writable sandbox, applying a fix interactively, and re-sealing the image, documenting the whole flow.

## Input Requirement


- No interactive input.


## Constraints

- Document every interactive change.

- Final image rebuilt cleanly (no sandbox leftovers).

- Save all generated artifacts to:
  - benchmark_results/T279_sandbox_debug_workflow/


## Expected Output

Expected output artifact(s):

- `debug_notes.md`

- `changes.diff`

- `rebuild_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
