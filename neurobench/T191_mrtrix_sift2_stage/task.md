# T191_mrtrix_sift2_stage: MRtrix SIFT2 Stage (stage split of T190)
## Task Description

Stage split of T190: given a whole-brain tractogram and FOD, run SIFT2 to produce streamline weights, with convergence and QC reporting.

## Input Requirement

Required input(s):

- Tractogram `.tck` (required)

- FOD image `wmfod.mif` (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `tcksift2` default regularisation; document iterations.

- Report the proportionality-coefficient QC plots.

- Save all generated artifacts to:
  - benchmark_results/T191_mrtrix_sift2_stage/


## Expected Output

Expected output artifact(s):

- `weights.csv` (per-streamline weights)

- `sift2_mu.txt` + QC PNG

- `sift2_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
