# T144_mrtrix_mrdegibbs: MRtrix mrdegibbs Ringing Removal
## Task Description

Remove Gibbs ringing artefacts from a raw DWI dataset using MRtrix3 `mrdegibbs` and quantify residual ringing on a high-b-value shell.

## Input Requirement

Required input(s):

- Raw DWI NIfTI/MIF with bval/bvec (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Run `mrdegibbs` BEFORE any other preprocessing step (state this ordering in the log).

- Keep original data untouched; write corrected output as a new file.

- Save all generated artifacts to:
  - benchmark_results/T144_mrtrix_mrdegibbs/


## Expected Output

Expected output artifact(s):

- `dwi_degibbs.mif`

- `degibbs_qc.png` (before/after axial slice comparison)

- `mrdegibbs_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
