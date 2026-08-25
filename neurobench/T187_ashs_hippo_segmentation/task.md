# T187_ashs_hippo_segmentation: ASHS Hippocampal Subfield Segmentation
## Task Description

Run ASHS automatic segmentation of hippocampal subfields using a high-resolution T2 template package, extract volumes, and export label QC snapshots.

## Input Requirement

Required input(s):

- T1w + high-resolution T2w (oblique coronal) NIfTIs (required)

- ASHS template package (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Document ASHS version and atlas package.

- Keep intermediate registration QC.

- Save all generated artifacts to:
  - benchmark_results/T187_ashs_hippo_segmentation/


## Expected Output

Expected output artifact(s):

- ASHS label maps (native space)

- `subfield_volumes.csv`

- `ashs_qc.pdf`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
