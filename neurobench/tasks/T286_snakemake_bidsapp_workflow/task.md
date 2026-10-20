# T286_snakemake_bidsapp_workflow: Snakemake BIDS-App Workflow
## Task Description

Wrap a BIDS app (e.g. fMRIPrep) in a Snakemake workflow: one rule per subject, wildcards from the BIDS tree, logs per rule.

## Input Requirement


- No interactive input.


## Constraints

- Subject list discovered from the BIDS dataset (pybids or glob).

- `snakemake -n` dry-run output included.

- Save all generated artifacts to:
  - benchmark_results/T286_snakemake_bidsapp_workflow/


## Expected Output

Expected output artifact(s):

- `Snakefile`

- `dry_run.txt`

- `dag.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
