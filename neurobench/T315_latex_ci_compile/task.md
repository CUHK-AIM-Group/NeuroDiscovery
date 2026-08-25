# T315_latex_ci_compile: LaTeX CI Compile
## Task Description

CI job that compiles the manuscript with tectonic (or latexmk) on push, uploads the PDF artifact, and fails on LaTeX errors.

## Input Requirement


- No interactive input.


## Constraints

- Warnings summarized in the job log.

- Artifact retention documented.

- Save all generated artifacts to:
  - benchmark_results/T315_latex_ci_compile/


## Expected Output

Expected output artifact(s):

- Workflow YAML

- `compile_log_sample.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
