# T171_oxford_asl_pipeline: Oxford ASL (BASIL) Perfusion Pipeline
## Task Description

Run oxford_asl/BASIL on a pCASL dataset: motion correction, calibration with the M0 image, and CBF quantification in native + MNI space.

## Input Requirement

Required input(s):

- ASL NIfTI (label/control pairs, required)

- M0 calibration image (required)

- T1w NIfTI for registration (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `oxford_asl` with structural + calibration inputs; document acquisition parameters (PLD, labeling duration) from the BIDS sidecar.

- Register CBF to MNI152 and report both spaces.

- Save all generated artifacts to:
  - benchmark_results/T171_oxford_asl_pipeline/


## Expected Output

Expected output artifact(s):

- `cbf_native.nii.gz`, `cbf_mni.nii.gz`

- `basil_report.txt` (parameters + fit statistics)

- `cbf_qc.png` overlay


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
