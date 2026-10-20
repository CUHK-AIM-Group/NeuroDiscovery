# T291_make_pipeline_driver: Makefile Pipeline Driver
## Task Description

Provide a Makefile that drives the common local pipeline operations (setup, convert, qc, clean-derivatives) with documented targets.

## Input Requirement


- No interactive input.


## Constraints

- Every target has a `## comment` so `make help` works.

- No destructive target without a confirm prompt.

- Save all generated artifacts to:
  - benchmark_results/T291_make_pipeline_driver/


## Expected Output

Expected output artifact(s):

- `Makefile`

- `make_help_output.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
