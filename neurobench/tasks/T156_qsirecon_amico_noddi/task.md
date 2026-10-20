# T156_qsirecon_amico_noddi: QSIRecon Reconstruction: AMICO NODDI
## Task Description

Run QSIRecon on QSIPrep derivatives using `--recon-spec amico_noddi` producing NDI/ODI/ISOVF maps. Produce the reconstructed outputs plus a per-stage log.

## Input Requirement

Required input(s):

- QSIPrep derivatives directory for one subject (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Container execution (Docker or Singularity) with pinned version.

- Document the exact recon-spec JSON/YAML used.

- Save all generated artifacts to:
  - benchmark_results/T156_qsirecon_amico_noddi/


## Expected Output

Expected output artifact(s):

- Reconstruction outputs per spec (tractogram and/or scalar maps)

- `recon_spec_used.json`

- `qsirecon_log.txt` (runtime per stage)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
